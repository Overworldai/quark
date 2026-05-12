"""``Engine`` — Apple Silicon backend.

EXEMPT FROM 500-LINE RULE — see ``docs/CONVENTIONS.md`` §file layout.
The Metal path is a single coherent unit (init + DiT eager dispatch
under ``quark.lazy()`` + ANE TAEHV decode + pipelined-decode worker
glue); splitting it further would just shuffle private helpers across
files for no readability gain.

Layout differs from the CUDA path:

  * No graph capture. Metal has no equivalent of CUDA graphs, so the
    DiT runs eagerly under ``quark.lazy()`` (one command buffer per
    frame). Denoise runs ``sync=True`` so the latent is host-readable
    for the ANE handoff; commit runs ``sync=False`` so its KV-write
    kernels overlap with the next frame's encode work.

  * No torch tensors at the latent boundary. torch MPS and quark's
    Metal-cpp pool use disjoint ``MTLDevice`` allocators — a borrow
    won't work, so we route through numpy carriers (host pointer copy
    only — no device round-trip).

  * VAE on the ANE via ``quark.taehv.load_taehv`` / CoreMLTAEHV. The
    decode work runs in a single-worker ``PipelinedDecoder`` thread
    on the two-thread submit/drain path, and inline on the main thread
    on the synchronous ``gen_frame`` path (the inline path saves the
    ~0.4 ms thread-handoff cost — see ``gen_frame`` docstring).

Quant is forced to all-bf16 here regardless of the caller's request:
Metal has no native fp8 (no e4m3 type in MSL).
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

from quark.engine._pin import _pin_params_to_device
from quark.engine.base import Engine, _resolve_quant
from quark.models.config import _resolve_path, load_yaml_config
from quark.models.waypoint_15 import (
    CtrlInput,
    QuantConfig,
    Waypoint15,
    Waypoint15Config,
    remap_state_dict,
)


class EngineMetal(Engine):
    """Engine for Apple Silicon hosts. Mirror of the legacy
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
        taehv_cache_dir: str | None = None,
    ) -> None:
        # Device on Darwin is informational — the DiT runs on Metal-cpp,
        # the VAE runs on the ANE. The torch ``device`` attribute is
        # only kept for API parity with the CUDA path (Biome reads it
        # for logging).
        if device is None:
            if (
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

        # Force all-bf16 quant on Apple Silicon — Metal has no native
        # fp8 (no e4m3 type in MSL). Caller-supplied ``quant`` can
        # still override fields individually if e.g. they want a
        # custom QuantConfig with ``moe="bf16"`` only.
        resolved_quant = _resolve_quant(quant)
        if resolved_quant != QuantConfig.all_bf16():
            from dataclasses import replace as _dc_replace

            resolved_quant = _dc_replace(
                resolved_quant,
                linear="bf16",
                kv_cache="bf16",
                attn_compute="bf16",
                moe="bf16",
            )
            warnings.warn(
                "quark.Engine: forcing QuantConfig.all_bf16() on Apple Silicon "
                "(Metal has no native fp8). Pass quant=QuantConfig.all_bf16() "
                "explicitly to silence this warning.",
                RuntimeWarning,
                stacklevel=3,
            )

        cfg = Waypoint15Config.from_dict(raw_cfg, quant=resolved_quant)
        self.cfg = cfg
        self.model = Waypoint15(cfg)

        # ── Weights ─────────────────────────────────────────────
        if load_weights:
            from quark.nn.io import load_safetensors

            sf_path = _resolve_path(model_uri, filename="model.safetensors")
            raw = load_safetensors(sf_path, dtype="bf16")
            self.model.load_state_dict(remap_state_dict(raw, cfg), strict=False)

        self.model.prepare()
        # Pin every weight + persistent state buffer into the Metal pool
        # so the dispatcher's input cache survives ``quark.lazy()`` eval
        # boundaries. Without this every ``gen_frame`` re-uploads the
        # ~3.7 GB of bf16 weights from numpy carriers into fresh pool
        # buffers — measured 2× model-forward slowdown vs the bench
        # script which already pins. Lifted from
        # ``scripts/bench_quark_world_engine.py::_move_params_to_device``.
        _pin_params_to_device(self.model)
        # No GenerateFrame on Darwin — Metal has no graph capture.
        # ``gen_frame`` runs the denoise + commit eagerly under
        # ``quark.lazy()``. Stub the attribute so external callers that
        # type-check ``hasattr(engine, "gen")`` don't break.
        self.gen = None
        self._graph_ready = False

        # ── Manual euler loop state ─────────────────────────────
        # ``GenerateFrame`` on CUDA owns these inside the graph; we own
        # them at the engine level and pass them to the model each call.
        from quark.nn.module import _tensor as _qk_tensor

        sigmas = cfg.scheduler_sigmas
        self._n_denoise = len(sigmas) - 1
        self._dsig_tensors = [
            _qk_tensor([float(sigmas[i + 1] - sigmas[i])], dtype="f32")
            for i in range(self._n_denoise)
        ]
        from quark import nn as _qnn

        self._euler_steps = _qnn.ModuleList([_qnn.EulerStep() for _ in range(self._n_denoise)])

        # Persistent ctrl input — same helper the CUDA path uses.
        self._ctrl_dev, self._ctrl_fill = self.model.make_ctrl_buffer()

        # ── VAE (CoreML/ANE — no torch on the hot path) ─────────
        if not raw_cfg.get("taehv_ae", False):
            raise NotImplementedError(
                "quark.Engine on Darwin currently supports the TAEHV VAE only. "
                "Set ``taehv_ae: true`` in the model config."
            )
        # The CoreML TAEHV repo is by-convention ``<ae_uri>-coreml`` —
        # holds pre-built .mlpackage artifacts at every supported latent
        # resolution. Resolution order:
        #
        #   1. ``QUARK_TAEHV_COREML_URI`` env var — for ad-hoc redirects
        #      (e.g. testing against a personal / staging HF repo before
        #      publishing to the canonical Overworld-Models namespace).
        #   2. ``coreml_uri`` in the model config — for permanent
        #      per-model overrides (forks or private repos).
        #   3. ``<ae_uri>-coreml`` — default convention.
        #
        # ``coreml_revision`` pins the HF revision for reproducibility;
        # default ``main`` tracks the latest publish-script run.
        import os as _os

        from quark.taehv import load_taehv

        ae_uri = raw_cfg["ae_uri"]
        coreml_uri = (
            _os.environ.get("QUARK_TAEHV_COREML_URI")
            or raw_cfg.get("coreml_uri")
            or f"{ae_uri}-coreml"
        )
        coreml_revision = raw_cfg.get("coreml_revision")
        # ``load_taehv``'s ``latent_height`` / ``latent_width`` are the
        # PRE-patchify latent dims (the encoder's CoreML input spatial
        # shape). ``cfg.height * cfg.patch[0]`` / ``cfg.width * cfg.patch[1]``
        # gives those — the publish script exports artifacts keyed on
        # exactly this pair.
        ae_lat_h = cfg.height * cfg.patch[0]
        ae_lat_w = cfg.width * cfg.patch[1]
        self._taehv = load_taehv(
            coreml_uri,
            latent_height=ae_lat_h,
            latent_width=ae_lat_w,
            revision=coreml_revision,
            cache_dir=taehv_cache_dir,
            compute_units="CPU_AND_NE",
        )
        # Stub ``self.vae`` so external callers that introspect
        # ``engine.vae`` see the taehv handle (the methods don't match
        # the torch ``ChunkedStreamingTAEHV`` API but the attribute
        # exists for parity).
        self.vae = self._taehv

        # ── Frame plumbing ──────────────────────────────────────
        pH, pW = cfg.patch
        pixel_h, pixel_w = cfg.height * pH, cfg.width * pW
        self._pixel_shape = (1, cfg.channels, pixel_h, pixel_w)
        self._flat_shape = (1, cfg.channels * pixel_h * pixel_w)
        self.frm_shape = (1, 1, cfg.channels, pixel_h, pixel_w)
        self._half_dt = "bf16"

        latent_fps = raw_cfg["inference_fps"] / raw_cfg["temporal_compression"]
        assert raw_cfg["base_fps"] % latent_fps == 0
        self.ts_mult = int(raw_cfg["base_fps"] // latent_fps)
        self._frame_counter = 0

        # Pre-allocate the per-frame scratch tensors as STABLE Metal pool
        # buffers. Re-creating these each frame via ``QuarkTensor.from_numpy``
        # / ``_qk_tensor`` was the long-running stability bug that surfaced
        # in Biome's hot loop after a few hundred frames: each call pushed
        # a new entry onto the dispatcher's ``g_lazy_buffers`` handle
        # table (the table never shrinks, only NULLs out), and the pool
        # eventually wedged silently. Holding stable buffers + writing
        # new bytes via ``ctypes.memmove`` keeps the per-frame allocation
        # surface zero.
        import numpy as _np_init

        from quark.runtime.tensor import QuarkTensor as _QT

        self._noise_qt = _QT.zeros(*self._flat_shape, dtype="bf16")
        self._noise_staged = _np_init.zeros(self._flat_shape, dtype=_np_init.uint16)
        self._frame_t_qt = _QT.zeros(1, dtype="s32")

        # Pipelined decoder — single-worker thread that runs the ANE/CoreML
        # decode while the main thread is encoding the next frame's
        # denoise. The synchronous ``gen_frame`` path goes through the
        # same pipe (``submit_frame`` + immediate ``next_pixels``); the
        # worker still does the host snapshot off-thread, just nothing
        # else overlaps with the wait. ``submit_frame`` + drain-on-
        # another-thread is the lag-1 pipelined path.
        from quark.taehv import PipelinedDecoder as _PipelinedDecoder

        self._pipe = _PipelinedDecoder(self._taehv)
        self._submit_in_flight = False

    # ── Public API ──────────────────────────────────────────────

    def reset(self) -> None:
        """Reset model KV caches + taehv decoder streaming state.

        Drains any in-flight pipelined decode first so a stale frame
        from before the reset can't surface as the next ``next_pixels``
        result. The drained pixels are discarded — by definition the
        caller is throwing away the current world state."""
        if self._submit_in_flight:
            import contextlib

            with contextlib.suppress(Exception):
                self._pipe.flush()
            self._submit_in_flight = False
        self.model.reset()
        self._frame_counter = 0
        self._taehv.reset()

    def append_frame(self, img, ctrl: CtrlInput | None = None):
        """Encode ``img`` via the ANE TAEHV encoder, drive a single
        commit-only DiT pass to write this frame's KV ring entry, then
        decode the same latent for the return image. Mirrors the
        CUDA-path semantics: KV cache advances by one, frame counter
        increments, returns the post-VAE-roundtrip image."""
        import numpy as _np

        latent_np = self._taehv_encode(img)  # [1, 32, latH, latW] f32
        latent_qt = self._latent_np_to_qt(latent_np)

        ctrl_qt = self._encode_ctrl(ctrl)
        ctrl_emb = self.model.encode_ctrl(ctrl_qt) if ctrl_qt is not None else None

        from quark.nn.module import _tensor as _qk_tensor

        ft = _qk_tensor([self._frame_counter], dtype="s32")
        self.model(
            latent_qt,
            sigma_idx=self._n_denoise,
            frame_t=ft,
            ctrl_emb=ctrl_emb,
            frozen=False,
        )
        self._frame_counter += 1

        # Decode the just-encoded latent through the streaming taehv —
        # this also primes the decoder state on first call. Return the
        # full 4-frame temporal stack to match the WE-shaped contract
        # (Biome and other consumers slice ``result[i]`` for each of
        # the ``temporal_compression`` subframes).
        pixels = self._taehv.decode(latent_np.astype(_np.float16))  # [4, H_pix, W_pix, 3]
        return torch.from_numpy(pixels)

    def gen_frame(self, ctrl: CtrlInput | None = None, return_img: bool = True):
        """Synchronous one-frame inference: denoise + commit + decode,
        returns pixels for *this* frame.

        Routes through ``submit_frame`` + ``next_pixels`` so the ANE
        decode runs on the ``PipelinedDecoder`` worker thread (CoreML
        asyncPrediction-backed). The decode overlaps with caller-side
        work between frames (host-side numpy conversion, video encoding,
        etc.) — that's the implicit pipelining; the call still blocks
        until pixels are ready.

        For two-thread submit/drain (real overlap of decode N-1 with
        denoise N) call ``submit_frame`` / ``next_pixels`` directly
        — see ``scripts/repro_pipelined_engine.py``. Pipeline depth is
        bounded at 1, so ``gen_frame`` and the two-thread API are
        mutually exclusive within one session.
        """
        self.submit_frame(ctrl)
        pixels = self.next_pixels()
        return pixels if return_img else None

    def flush_pixels(self) -> torch.Tensor | None:
        """Drain the last in-flight pipelined decode (if any) and return
        its pixels. Use after the final ``submit_frame`` call (e.g. on
        session end) to collect the trailing frame. Returns ``None`` if
        nothing is pending."""
        return self.next_pixels()

    def submit_frame(self, ctrl: CtrlInput | None = None) -> None:
        """Pipelined-decode submit: compute the latent and hand it to
        the worker thread for ANE decode in the background. Returns
        immediately. Pair with ``next_pixels`` to collect the result
        on a separate thread.

        Intended use is the two-thread submit/drain pattern: one
        thread calls ``submit_frame`` in a loop, another calls
        ``next_pixels`` in a loop, with a depth-1 semaphore between
        them for backpressure (see ``scripts/repro_pipelined_engine.py``).

        For single-threaded synchronous use, prefer ``gen_frame`` —
        it uses an inline-decode path that skips the worker handoff
        and is ~0.4 ms faster per frame.

        Raises ``RuntimeError`` if a previous ``submit_frame`` hasn't
        been drained — pipeline depth is bounded at 1.
        """
        if self._submit_in_flight:
            raise RuntimeError(
                "quark.Engine.submit_frame: previous frame not drained — "
                "call next_pixels() before submitting another."
            )
        cur = self._compute_latent(ctrl)
        # Hand the live latent QuarkTensor to the decoder worker.
        # ``PipelinedDecoder`` does the host snapshot + decode on its
        # own thread; the latent's storage stays alive thanks to the
        # dispatcher's refcount-aware lazy-buffer lifetime.
        self._pipe.submit(cur)
        self._submit_in_flight = True

    def next_pixels(self) -> torch.Tensor | None:
        """Block on the previously-submitted decode and return the
        4-frame temporal stack as a torch tensor.

        Returns ``None`` if no submit is pending (e.g. directly after
        ``reset()``, or after a previous ``next_pixels`` already
        drained the only in-flight frame)."""
        if not self._submit_in_flight:
            return None
        pixels_np = self._pipe.next()  # [4, H_pix, W_pix, 3] uint8
        self._submit_in_flight = False
        if pixels_np is None:
            return None
        # Match the legacy ``gen_frame`` return contract: full 4-frame
        # temporal stack as a torch tensor. Biome slices ``result[i]``
        # for each of the ``temporal_compression`` subframes.
        return torch.from_numpy(pixels_np)

    # ── Internals ───────────────────────────────────────────────

    def _compute_latent(self, ctrl: CtrlInput | None):
        """Run denoise + commit on the GPU and return the live latent
        ``QuarkTensor``. Shared by ``submit_frame`` (which hands the
        latent to a worker for pipelined decode) and ``gen_frame``
        (which decodes inline on the main thread).

        Two-phase lazy dispatch:

        1. Denoise (n euler steps) inside ``quark.lazy()`` (default
           ``sync=True``). The trailing sync makes the latent
           host-readable so the taehv decode below can ship it to
           the ANE without a fresh wait.

        2. Commit (one model call writing this frame's KV ring entry)
           inside ``quark.lazy(sync=False)`` — fire-and-forget. The
           commit's GPU work then overlaps with the ANE decode and
           the next frame's Python encoding. Cross-frame ordering
           is via the dispatcher's cross-encoder fence machinery
           (next frame's attention reads fence-wait on commit's KV
           writes automatically).

        An earlier version of this file collapsed the two phases into
        a single ``quark.lazy()`` block as a workaround for the
        dispatcher-side data race that wedged metal-cpp after 200-1500
        frames. Once the underlying ``g_buf_to_fence`` race was fixed
        (5f7962d, refined into a deferred-cleanup queue at 541b1e5),
        the split is safe again.
        """
        import ctypes as _ctypes

        import numpy as _np

        import quark
        from quark.runtime.sync import synchronize as _drain_lazy_queue

        ctrl_qt = self._encode_ctrl(ctrl)
        ctrl_emb = self.model.encode_ctrl(ctrl_qt) if ctrl_qt is not None else None

        # Refill the STABLE noise buffer in place. Using a fresh
        # ``QuarkTensor.from_numpy`` per frame leaks ``g_lazy_buffers``
        # handle slots over long-running consumers (the dispatcher's
        # handle table never shrinks; pool fragmentation eventually
        # wedges Metal silently — surfaced as Biome's mid-session
        # process death after ~hundreds of frames).
        rng = _np.random.default_rng()
        noise_f32 = rng.standard_normal(self._flat_shape).astype(_np.float32)
        self._noise_staged[:] = (noise_f32.view(_np.uint32) >> 16).astype(_np.uint16)
        _drain_lazy_queue()
        _ctypes.memmove(
            self._noise_qt.data_ptr(),
            self._noise_staged.ctypes.data,
            self._noise_staged.nbytes,
        )
        x = self._noise_qt

        # Same in-place pattern for the s32 frame counter — fresh
        # ``_qk_tensor([fi])`` per call hits the same handle-table
        # leak that the noise buffer used to.
        ft_bytes = (int(self._frame_counter) & 0xFFFFFFFF).to_bytes(4, "little")
        _ctypes.memmove(self._frame_t_qt.data_ptr(), ft_bytes, 4)
        ft = self._frame_t_qt

        # Phase 1: denoise. Sync at exit so ``cur`` is host-readable
        # for the taehv decode below.
        with quark.lazy():
            cur = x
            for si in range(self._n_denoise):
                v = self.model(
                    cur,
                    sigma_idx=si,
                    frame_t=ft,
                    ctrl_emb=ctrl_emb,
                    frozen=True,
                )
                cur = self._euler_steps[si](cur, v, self._dsig_tensors[si])

        # Phase 2: commit. ``sync=False`` queues the KV-write kernels
        # without waiting — they run concurrently with the taehv
        # decode and the next frame's Python encoding.
        with quark.lazy(sync=False):
            self.model(
                cur,
                sigma_idx=self._n_denoise,
                frame_t=ft,
                ctrl_emb=ctrl_emb,
            )

        self._frame_counter += 1
        return cur

    def _taehv_encode(self, img):
        """Coerce ``img`` into the ``[4, H_pix, W_pix, 3] uint8`` numpy
        layout the ANE TAEHV encoder expects.

        Accepted inputs:
          * ``torch.Tensor`` (uint8 or [-1, 1] / [0, 1] float),
            ``np.ndarray``, or ``PIL.Image`` shaped ``[H, W, 3]``
            (single-frame seed) — broadcast to 4 frames.
          * ``torch.Tensor`` / ``np.ndarray`` shaped
            ``[4, H, W, 3]`` (a real 4-frame temporal seed, the
            ``ChunkedStreamingTAEHV.encode`` shape Biome / WE
            already produce) — pass straight through.

        TAEHV's encoder is temporal: it needs exactly 4 frames per
        call. The 1-frame broadcast is the cold-start case; the
        4-frame stack is what callers that already know the cadence
        produce.
        """
        import numpy as _np

        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        elif hasattr(img, "convert"):  # PIL.Image — single frame
            arr = _np.asarray(img.convert("RGB"))
        else:
            arr = _np.asarray(img)

        if arr.dtype != _np.uint8:
            # Floats coming from torch are typically in [-1, 1] (the WE
            # taehv path's range) or [0, 1]; collapse both via a
            # non-negativity check.
            arr_f = arr.astype(_np.float32)
            if arr_f.min() < -0.01:
                arr_f = (arr_f * 0.5 + 0.5) * 255.0
            else:
                arr_f = arr_f * 255.0
            arr = _np.clip(arr_f, 0, 255).round().astype(_np.uint8)

        if arr.ndim == 4 and arr.shape[0] == 4 and arr.shape[-1] == 3:
            # Already a 4-frame stack — pass through.
            arr_4 = _np.ascontiguousarray(arr)
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            # Single frame — broadcast to 4.
            arr_4 = _np.broadcast_to(arr, (4, *arr.shape)).copy()
        else:
            raise ValueError(
                "_taehv_encode: expected [H, W, 3] (single frame) "
                f"or [4, H, W, 3] (temporal stack) uint8, got shape {arr.shape}"
            )
        return self._taehv.encode(arr_4)

    def _latent_np_to_qt(self, latent_np):
        """``[1, 32, latH, latW] f32`` → flat ``QuarkTensor`` (bf16) the
        Waypoint15 model expects."""
        import numpy as _np

        from quark.runtime.tensor import QuarkTensor

        flat_f32 = _np.ascontiguousarray(latent_np, dtype=_np.float32).reshape(1, -1)
        carrier = (flat_f32.view(_np.uint32) >> 16).astype(_np.uint16)
        qt = QuarkTensor.from_numpy(carrier, dtype="bf16")
        return qt.reshape(*self._flat_shape)

    def _qt_to_latent_np_f32(self, qt):
        """Inverse of ``_latent_np_to_qt``. Reshape to the taehv
        decoder's expected ``[1, 32, latH, latW]``."""
        import numpy as _np

        raw = qt.to_numpy()  # uint16 bf16-carrier
        f32 = (raw.astype(_np.uint32) << 16).view(_np.float32)
        pH, pW = self.cfg.patch
        pixel_h, pixel_w = self.cfg.height * pH, self.cfg.width * pW
        return f32.reshape(1, self.cfg.channels, pixel_h, pixel_w)
