"""Base class + per-platform factory for ``quark.Engine``.

``Engine`` is constructed as ``quark.Engine(model_uri, ...)``; the
``__new__`` factory below picks the right concrete subclass for the
current platform. Public attribute names + the ``gen_frame`` /
``append_frame`` / ``submit_frame`` / ``next_pixels`` /
``flush_pixels`` / ``reset`` API match the legacy
``world_engine.WorldEngine`` shape so existing consumers (Biome,
benches, scripts) don't change at the call site.

Today the two backends are:

  * :class:`quark.engine.cuda.EngineCUDA` — torch tensors at the
    latent boundary (zero-copy ``QuarkTensor.borrow``), CUDA graph
    capture via ``GenerateFrame``, torch-side
    ``ChunkedStreamingTAEHV`` VAE.
  * :class:`quark.engine.metal.EngineMetal` — Apple Silicon. DiT
    runs eagerly under ``quark.lazy()`` (Metal has no graph capture),
    VAE runs on the Apple Neural Engine via ``quark.taehv``, and the
    latent boundary is plain numpy.

Adding a new backend is one new ``EngineX(Engine)`` subclass plus a
branch in ``__new__``; nothing in ``Engine`` itself or in callers
needs to change.

Quantization shorthand: the ``quant`` ctor kwarg is normalised by
:func:`_resolve_quant` to a ``QuantConfig``. ``"fp8"`` / ``None`` →
``QuantConfig()`` (end-to-end fp8 on sm_89+); ``"bf16"`` →
``QuantConfig.all_bf16()``. Subclasses may further constrain the
result (Metal forces all-bf16 — no native fp8 in MSL).

Not yet supported (raises): ``set_prompt`` / prompt cross-attention,
``get_state`` / ``load_state``.
"""

from __future__ import annotations

import sys
from typing import Any

import torch

from quark.models.waypoint_15 import CtrlInput, QuantConfig

_IS_METAL = sys.platform == "darwin"


def _qt_from_torch(t: torch.Tensor, dtype: str = "bf16"):
    """Bring a torch tensor into a quark-managed buffer.

    On CUDA: zero-copy wrap via ``QuarkTensor.borrow`` (torch and
    quark share the device allocator, so ``data_ptr()`` is the same
    address quark kernels read).

    On Apple Silicon: torch MPS and quark's Metal-cpp pool use
    different MTLDevice allocators, so a borrow won't work. Round-
    trip through host: ``t.cpu().numpy()`` → ``QuarkTensor.from_numpy``
    (which does the host→Metal-pool copy via ``pin_buffer``). Per-
    frame cost on a 360p latent (8×16 = 128 tokens × 32 channels =
    16 KB) is sub-millisecond and not on the DiT hot path — only
    fires at the latent boundary (post-VAE-encode, post-DiT-forward).

    Holds a reference via ``owner`` so the torch storage outlives
    the wrapper on the CUDA path; on Metal the wrapper owns its own
    pool buffer so ``owner`` is unused.
    """
    from quark.runtime.tensor import QuarkTensor

    assert t.is_contiguous(), "tensor must be contiguous"
    if t.is_cuda:
        return QuarkTensor.borrow(
            t.data_ptr(),
            t.numel() * t.element_size(),
            tuple(t.shape),
            dtype,
            owner=t,
        )
    # MPS / CPU torch tensor — copy through host. ``.cpu()`` is a
    # no-op for CPU tensors and a one-shot copy for MPS.
    arr = t.detach().cpu().contiguous().numpy()
    qt = QuarkTensor.from_numpy(arr, dtype=dtype)
    return qt.reshape(*t.shape) if qt.shape != tuple(t.shape) else qt


# Backwards-compat alias — ``_qt_borrow`` used to assert ``is_cuda``;
# callers in this module now go through ``_qt_from_torch``.
_qt_borrow = _qt_from_torch


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

    Constructing ``Engine(model_uri, ...)`` returns a platform-
    appropriate concrete subclass — :class:`EngineCUDA` on Linux /
    Windows / CUDA Macs, :class:`EngineMetal` on Apple Silicon. The
    ``__new__`` factory below does the dispatch; ``__init__`` is
    implemented per-subclass so the bodies don't carry an
    ``if self._is_metal`` ladder.

    Parameters
    ----------
    model_uri:
        Path to a model directory or HF repo id. Must contain
        ``config.yaml`` and ``model.safetensors``.
    quant:
        ``"fp8"`` (default) for end-to-end fp8 on sm_89+, ``"bf16"``
        for the safe fallback, or a ``QuantConfig`` for per-component
        control. Apple Silicon forces all-bf16 (no native fp8 in MSL).
    config_overrides:
        Optional mapping merged into the YAML config before
        ``Waypoint15Config.from_dict`` (e.g. swap ``ae_uri`` to a local
        path during development).
    device:
        ``torch.device`` or string. Defaults to the current CUDA device
        on CUDA, MPS or CPU on Apple Silicon. The DiT runs on quark
        kernels regardless; ``device`` only steers the torch-side VAE
        on the CUDA path. On Apple Silicon the attribute is
        informational (the VAE runs on the ANE through ``quark.taehv``).
    dtype:
        Latent dtype on the VAE side. ``torch.bfloat16`` matches the
        DiT residual stream and avoids a cast at the borrow boundary.
    load_weights:
        ``False`` builds an architecture-only model — useful for
        graph-capture testing without a real checkpoint.
    taehv_cache_dir:
        Apple Silicon only. Root directory for the local CoreML
        ``.mlpackage`` mirror that ``quark.taehv.load_taehv`` populates
        from HF on first use. ``None`` falls through to the
        ``$QUARK_TAEHV_CACHE`` env var, then to ``~/.cache/quark/taehv``.
        Host applications (Biome, etc.) that want artifact storage
        under their own data dir should pass an in-app path here.
        CUDA callers ignore the kwarg (the torch VAE doesn't use the
        CoreML mirror).
    """

    # Subclasses set this in ``__init__`` after dispatch.
    raw_cfg: dict
    cfg: Any
    model: Any
    device: torch.device
    dtype: torch.dtype

    def __new__(cls, *_args, **_kwargs):
        if cls is Engine:
            if _IS_METAL:
                from quark.engine.metal import EngineMetal

                cls = EngineMetal
            else:
                from quark.engine.cuda import EngineCUDA

                cls = EngineCUDA
        return object.__new__(cls)

    # ── Shared API surface ──────────────────────────────────────

    @property
    def model_cfg(self):
        """``world_engine.WorldEngine``-compatible config view.

        Biome (and other WE consumers) read attributes off
        ``engine.model_cfg`` (``model_type``, ``prompt_conditioning``,
        ``temporal_compression``, ``n_frames``, ``inference_fps``, …)
        from the upstream YAML. Quark stores that mapping verbatim on
        ``self.raw_cfg``; wrap it in a SimpleNamespace so attribute
        access keeps working without callers reaching into the dict.
        """
        from types import SimpleNamespace

        return SimpleNamespace(**self.raw_cfg)

    def set_prompt(self, prompt: str) -> None:
        raise NotImplementedError("quark.Engine.set_prompt: prompt cross-attention not yet ported.")

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
