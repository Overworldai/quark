"""``Engine`` — CUDA backend.

Torch and quark share the same CUDA allocator, so the latent
boundary is a zero-copy ``QuarkTensor.borrow`` over the torch
storage. The DiT runs through ``GenerateFrame`` graph capture (one
captured frame, replayed every call); the VAE is the torch-side
``ChunkedStreamingTAEHV``. Two-thread submit/drain isn't useful on
this path because there's no asynchronous decode worker — the CUDA
graph + decode both run synchronously on the device thread; the
two-thread API is kept only for cross-platform parity with
:class:`EngineMetal`.
"""

from __future__ import annotations

from typing import Any

import torch

from quark.engine.base import Engine, _qt_borrow, _resolve_quant
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


class EngineCUDA(Engine):
    """Engine for CUDA / non-Apple-Silicon hosts. Mirror of the legacy
    monolithic ``Engine`` pre-split — see :class:`Engine` (in
    ``quark.engine.base``) for the full ctor docstring."""

    def __init__(
        self,
        model_uri: str,
        *,
        quant: str | QuantConfig | None = None,
        config_overrides: dict[str, Any] | None = None,
        device=None,
        dtype=torch.bfloat16,
        load_weights: bool = True,
        # Accepted for cross-platform API parity with EngineMetal; the
        # torch VAE doesn't consume a CoreML cache so the kwarg is a
        # no-op on CUDA.
        taehv_cache_dir: str | None = None,
    ):
        import warnings

        if device is None:
            # Quark's DiT runs on the local accelerator regardless of
            # which torch device this attribute names — that's only
            # used for the VAE side. Pick the best torch backend for
            # the current OS: CUDA on Linux/Windows, MPS on Apple
            # Silicon (rare on this subclass — it's selected only when
            # the platform check in ``Engine.__new__`` lands on CUDA),
            # CPU otherwise.
            if torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            elif (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
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

        # Only override ``quant`` when the caller explicitly asked for
        # one. Default falls through to ``Waypoint15Config.quant``'s
        # ``default_factory`` (``QuantConfig.all_bf16`` per the standalone
        # config) — passing ``QuantConfig()`` (all-fp8) here unconditionally
        # made ``Engine`` diverge from the WE backend on the same model.
        if quant is None:
            cfg = Waypoint15Config.from_dict(raw_cfg)
        else:
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

        # Staging buffer for ``submit_frame`` / ``next_pixels``. CUDA has
        # no worker thread — submit runs inline and stashes the decoded
        # tensor here for the matching next_pixels() to pop.
        self._pending_pixels: torch.Tensor | None = None

    # ── Public API ──────────────────────────────────────────────

    def reset(self) -> None:
        """Reset KV caches, frame counter, and VAE streaming state."""
        # Drop any frame staged by ``submit_frame`` but never collected.
        self._pending_pixels = None
        self.gen.reset()
        self._frame_counter = 0
        self.vae.reset()

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
        """Synchronous one-frame inference: denoise + commit + decode.

        Runs the full ``GenerateFrame`` graph + torch decode
        synchronously. The two-thread submit/drain pattern works on
        CUDA but doesn't gain anything over plain ``gen_frame`` here
        (no async decode worker); it's kept for parity with
        :class:`EngineMetal`.
        """
        self.submit_frame(ctrl)
        if not return_img:
            self.next_pixels()
            return None
        return self.next_pixels()

    def flush_pixels(self) -> torch.Tensor | None:
        """Drain the last in-flight pipelined decode (if any) and return
        its pixels. Use after the final ``submit_frame`` call (e.g. on
        session end) to collect the trailing frame. Returns ``None`` if
        nothing is pending."""
        return self.next_pixels()

    @torch.inference_mode()
    def submit_frame(self, ctrl: CtrlInput | None = None) -> None:
        """Begin one frame of inference; pair with ``next_pixels``.

        On CUDA there's no worker thread — this runs the full
        ``gen_frame`` body synchronously and stages the resulting
        torch tensor for ``next_pixels`` to pop. The two-thread
        pattern still works (it just doesn't gain anything over
        ``gen_frame``); see :class:`EngineMetal` for the path where
        the worker actually overlaps.

        Raises ``RuntimeError`` if a previous ``submit_frame`` hasn't
        been drained — pipeline depth is bounded at 1.
        """
        if self._pending_pixels is not None:
            raise RuntimeError(
                "quark.Engine.submit_frame: previous frame not drained — "
                "call next_pixels() before submitting another."
            )
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
        # Decode runs synchronously on the same thread — same semantics
        # as the pre-split ``gen_frame``. Pixels are staged for
        # ``next_pixels()`` to pop.
        self._pending_pixels = self.vae.decode(self._latent_out_torch)

    def next_pixels(self) -> torch.Tensor | None:
        """Block for the previously-submitted frame's pixels.

        Returns the same ``(T, H, W, 3) uint8`` torch tensor
        ``gen_frame`` returns. Returns ``None`` if no submit is
        pending.
        """
        pixels = self._pending_pixels
        self._pending_pixels = None
        return pixels
