"""``quark.Engine`` — standalone Waypoint-1.5 inference.

Drop-in replacement for the ``world_engine.WorldEngine`` API on the
quark side: ``append_frame`` / ``gen_frame`` / ``reset`` /
``set_prompt``. The DiT runs on quark kernels (``Waypoint15`` +
``GenerateFrame``); the VAE runs on the vendored torch ``TAEHV``
(``quark.models.vae``).

Quantization is configured via ``Waypoint15Config.quant`` (see
``QuantConfig``) — no env vars. The ``quant`` constructor kwarg is a
shorthand: ``"fp8"`` (default) → end-to-end fp8 on sm_89+, ``"bf16"``
→ ``QuantConfig.all_bf16()`` safe fallback.

Not yet supported (raises): ``set_prompt`` / prompt cross-attention
(``CrossAttention`` not ported), ``get_state`` / ``load_state`` (KV
ring-buffer round-trip not implemented).
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

from quark.models.config import _resolve_path, load_yaml_config
from quark.models.vae import ChunkedStreamingTAEHV
from quark.models.waypoint_15 import (
    CtrlInput,
    GenerateFrame,
    QuantConfig,
    Waypoint15,
    Waypoint15Config,
    remap_state_dict,
)


def _qt_borrow(t: torch.Tensor, dtype: str = "bf16"):
    """Zero-copy wrap a contiguous CUDA torch tensor as a ``QuarkTensor``.

    Holds a reference via ``owner`` so the torch storage outlives the
    wrapper.
    """
    from quark.runtime.tensor import QuarkTensor

    assert t.is_cuda, "quark.Engine requires CUDA tensors"
    assert t.is_contiguous(), "borrowed tensor must be contiguous"
    return QuarkTensor.borrow(
        t.data_ptr(),
        t.numel() * t.element_size(),
        tuple(t.shape),
        dtype,
        owner=t,
    )


def _resolve_quant(quant: str | QuantConfig | None) -> QuantConfig:
    if quant is None or quant == "fp8":
        return QuantConfig()
    if quant == "bf16":
        return QuantConfig.all_bf16()
    if isinstance(quant, QuantConfig):
        return quant
    raise ValueError(
        f"quark.Engine: quant must be 'fp8' / 'bf16' / QuantConfig(...), got {quant!r}"
    )


class Engine:
    """Standalone inference engine for a Waypoint-1.5 checkpoint.

    Parameters
    ----------
    model_uri:
        Path to a model directory or HF repo id. Must contain
        ``config.yaml`` and ``model.safetensors``.
    quant:
        ``"fp8"`` (default) for end-to-end fp8 on sm_89+, ``"bf16"`` for
        the safe fallback, or a ``QuantConfig`` for per-component
        control. Replaces the old ``QUARK_NO_FP8`` / ``QUARK_MOE_NO_FP8``
        env vars.
    config_overrides:
        Optional mapping merged into the YAML config before
        ``Waypoint15Config.from_dict`` (e.g. swap ``ae_uri`` to a local
        path during development).
    device:
        ``torch.device`` or string. Defaults to the current CUDA device.
    dtype:
        Latent dtype on the VAE side. ``torch.bfloat16`` matches the
        DiT residual stream and avoids a cast at the borrow boundary.
    load_weights:
        Set ``False`` to construct an architecture-only model (useful
        for testing the graph capture without a real checkpoint).
    """

    def __init__(
        self,
        model_uri: str,
        *,
        quant: str | QuantConfig | None = None,
        config_overrides: dict[str, Any] | None = None,
        device=None,
        dtype=torch.bfloat16,
        load_weights: bool = True,
    ):
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype

        # ── Config ──────────────────────────────────────────────
        raw_cfg = load_yaml_config(model_uri)
        if config_overrides:
            raw_cfg = {**raw_cfg, **config_overrides}
        self.raw_cfg = raw_cfg

        if raw_cfg.get("prompt_conditioning") is not None:
            warnings.warn(
                "quark.Engine: prompt_conditioning is enabled on this config but "
                "prompt cross-attention has not been ported to quark yet. The "
                "prompt will be ignored — outputs will be degraded.",
                RuntimeWarning,
                stacklevel=2,
            )

        cfg = Waypoint15Config.from_dict(raw_cfg, quant=_resolve_quant(quant))
        self.cfg = cfg
        self.model = Waypoint15(cfg)

        # ── Weights ─────────────────────────────────────────────
        if load_weights:
            from quark.nn.io import load_safetensors

            sf_path = _resolve_path(model_uri, filename="model.safetensors")
            raw = load_safetensors(sf_path, dtype="bf16")
            self.model.load_state_dict(remap_state_dict(raw, cfg), strict=False)

        self.model.prepare()
        self.gen = GenerateFrame(self.model)
        self._graph_ready = False

        # Persistent ctrl input — stable bf16 device buffer + host packer.
        self._ctrl_dev, self._ctrl_fill = self.model.make_ctrl_buffer()

        # ── VAE ─────────────────────────────────────────────────
        pH, pW = cfg.patch
        if not raw_cfg.get("taehv_ae", False):
            raise NotImplementedError(
                "quark.Engine currently supports the TAEHV VAE only. "
                "Set ``taehv_ae: true`` in the model config."
            )
        ae_uri = raw_cfg["ae_uri"]
        # ``height``/``width`` are passed through so the AE can lock the
        # encode/decode resize sizes at init (otherwise it's inferred
        # from the first call).
        self.vae = ChunkedStreamingTAEHV.from_pretrained(
            ae_uri,
            auto_aspect_ratio=raw_cfg.get("auto_aspect_ratio", True),
            dtype=dtype,
            device=self.device,
            height=cfg.height * pH,
            width=cfg.width * pW,
        )

        # ── Frame plumbing ──────────────────────────────────────
        # quark.Waypoint15 consumes a flat (1, C*H*W) latent — Patchify
        # handles the spatial reshape internally. The VAE wants
        # (1, C, H, W).
        pixel_h, pixel_w = cfg.height * pH, cfg.width * pW
        self._pixel_shape = (1, cfg.channels, pixel_h, pixel_w)
        self._flat_shape = (1, cfg.channels * pixel_h * pixel_w)
        self.frm_shape = (1, 1, cfg.channels, pixel_h, pixel_w)

        latent_fps = raw_cfg["inference_fps"] / raw_cfg["temporal_compression"]
        assert raw_cfg["base_fps"] % latent_fps == 0
        self.ts_mult = int(raw_cfg["base_fps"] // latent_fps)
        self._frame_counter = 0

        # Stable noise + output buffers for the graph hot path. torch
        # owns the memory; quark borrows the pointer. Refilling noise
        # via ``torch.randn(out=...)`` keeps the borrowed pointer
        # graph-stable.
        self._noise_torch = torch.empty(self._flat_shape, device=self.device, dtype=self.dtype)
        self._noise_qt = _qt_borrow(self._noise_torch)
        self._latent_out_torch = torch.empty(
            self._pixel_shape, device=self.device, dtype=self.dtype
        )

    # ── Public API ──────────────────────────────────────────────

    def reset(self) -> None:
        """Reset KV caches, frame counter, and VAE streaming state."""
        self.gen.reset()
        self._frame_counter = 0
        self.vae.reset()

    def set_prompt(self, prompt: str) -> None:
        raise NotImplementedError("quark.Engine.set_prompt: prompt cross-attention not yet ported.")

    @torch.inference_mode()
    def append_frame(self, img: torch.Tensor, ctrl: CtrlInput | None = None) -> torch.Tensor:
        """VAE-encode ``img``, run a commit-only DiT pass to write this
        frame's KV ring entry, then VAE-decode and return the image."""
        x0_pixel = self.vae.encode(img)  # (1, C, H, W)
        x0_flat = x0_pixel.reshape(self._flat_shape).contiguous()
        x0_qt = _qt_borrow(x0_flat)
        ctrl_qt = self._encode_ctrl(ctrl)
        ctrl_emb = self.model.encode_ctrl(ctrl_qt)
        self.model(
            x0_qt,
            sigma_idx=len(self.cfg.scheduler_sigmas) - 1,
            frame_t=self.gen.frame_t,
            ctrl_emb=ctrl_emb,
            frozen=False,
        )
        self.gen.frame_t.increment()
        self._frame_counter += 1
        return self.vae.decode(x0_pixel)

    @torch.inference_mode()
    def gen_frame(self, ctrl: CtrlInput | None = None, return_img: bool = True):
        """Refill the noise buffer, run ``GenerateFrame`` (denoise +
        commit + ft++) via a captured CUDA graph, D2D-copy the latent
        for VAE decode. First call captures the graph."""
        torch.randn(self._flat_shape, out=self._noise_torch)
        ctrl_qt = self._encode_ctrl(ctrl)

        if not self._graph_ready:
            self.gen.prepare_graph(self._noise_qt, ctrl_qt, start_frame_t=self._frame_counter)
            self._graph_ready = True

        latent_qt = self.gen(self._noise_qt, ctrl_qt)
        self._frame_counter += 1

        # QuarkTensor → torch: D2D memcpy into the pre-allocated torch
        # output buffer. No host round-trip.
        latent_qt.copy_into_ptr(self._latent_out_torch.data_ptr())

        return self.vae.decode(self._latent_out_torch) if return_img else self._latent_out_torch

    def get_state(self):
        raise NotImplementedError(
            "quark.Engine.get_state: KV ring-buffer round-trip not implemented."
        )

    def load_state(self, state):
        raise NotImplementedError(
            "quark.Engine.load_state: KV ring-buffer round-trip not implemented."
        )

    # ── Internals ───────────────────────────────────────────────

    def _encode_ctrl(self, ctrl: CtrlInput | None):
        """Pack a ``CtrlInput`` into the stable ``[1, padded_in]`` bf16
        device buffer. Returns ``None`` when ``ctrl_conditioning`` is
        off; ``ctrl_fill`` writes in place on the pre-allocated device
        buffer so the graph-captured input pointer stays stable."""
        if self._ctrl_fill is None:
            return None
        if ctrl is None:
            ctrl = CtrlInput()
        return self._ctrl_fill(ctrl)
