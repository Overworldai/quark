"""``Engine`` — Intel iGPU/Arc backend (OpenCL + IGC).

EXEMPT FROM 500-LINE RULE — see ``docs/CONVENTIONS.md`` §file layout.
The Intel path is a single coherent unit (init + DiT eager dispatch
under ``quark.lazy()`` + OpenVINO TAEHV decode + pipelined-decode
worker glue); splitting it further would just shuffle private
helpers across files for no readability gain.

Layout differs from the CUDA path (same way EngineMetal does):

  * No graph capture. The DiT runs eagerly under ``quark.lazy()``
    (one batched submit per frame). Denoise runs ``sync=True`` so
    the latent is host-readable for the OpenVINO handoff; commit
    runs ``sync=False`` so its KV-write kernels overlap with the
    next frame's encode work.

  * No torch tensors at the latent boundary. Torch has no usable
    Intel GPU device on Linux (no IPEX in our stack, no SYCL
    interop); we route through numpy carriers (host pointer copy
    only — no device round-trip).

  * VAE on Intel via ``quark.taehv.load_taehv`` / OpenVINOTAEHV. The
    decode work runs in a single-worker ``PipelinedDecoder`` thread
    that does the host snapshot off-thread; the OpenVINO GPU plugin
    consumes the decode call directly. On hosts where the GPU
    plugin isn't reachable (CPU plugin only), ``_vae_compute_units``
    captures the fallback selection so callers can introspect.

Quant is forced to all-bf16 here regardless of the caller's request:
IGC doesn't expose fp8 dpas, so e4m3 paths can't compile. The s8
quant path (``gemm_int`` / ``owl_attn_int8``) is a separate kernel
family selected at model-construction time — not gated by this.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

from quark.engine._pin import _pin_params_to_device
from quark.engine.base import Engine, _resolve_quant_for_family
from quark.models.config import _resolve_path, load_yaml_config
from quark.models.waypoint_15 import (
    CtrlInput,
    QuantConfig,
    Waypoint15,
    Waypoint15Config,
    remap_state_dict,
)


class EngineIntel(Engine):
    """Engine for Intel iGPU/Arc hosts (Linux + OpenCL + IGC). Mirror
    of :class:`EngineMetal` retargeted to the OCL backend + OpenVINO
    TAEHV VAE — see :class:`Engine` (in ``quark.engine.base``) for
    the full ctor docstring."""

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
        # Torch device on Intel is informational — the DiT runs on
        # OCL, the VAE runs on OpenVINO. The torch ``device`` attribute
        # is only kept for API parity with the CUDA path (Biome reads
        # it for logging).
        if device is None:
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

        from quark.device import DeviceFamily

        resolved_quant = _resolve_quant_for_family(quant, DeviceFamily.INTEL_GPU)
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
        # Pin every weight + persistent state buffer onto the OCL USM
        # pool so the dispatcher's input cache survives ``quark.lazy()``
        # eval boundaries. Without this, every ``gen_frame`` would
        # re-upload the ~3.7 GB of bf16 weights from numpy carriers
        # into fresh pool buffers — same recycle-on-lazy-eval shape as
        # the Metal pool, ~2× model-forward slowdown without.
        _pin_params_to_device(self.model)
        # No GenerateFrame on Intel — OCL has no graph capture. Stub
        # the attribute so external callers that type-check
        # ``hasattr(engine, "gen")`` don't break.
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

        # Pre-allocate the per-frame scratch tensors as STABLE OCL pool
        # buffers. Re-creating these each frame via
        # ``QuarkTensor.from_numpy`` / ``_qk_tensor`` would leak the
        # dispatcher's handle table over long-running sessions (same
        # shape as the Metal stability bug). Holding stable buffers +
        # writing new bytes via ``ctypes.memmove`` keeps the per-frame
        # allocation surface zero.
        import numpy as _np_init

        from quark.runtime.tensor import QuarkTensor as _QT

        self._noise_qt = _QT.zeros(*self._flat_shape, dtype="bf16")
        self._noise_staged = _np_init.zeros(self._flat_shape, dtype=_np_init.uint16)
        self._frame_t_qt = _QT.zeros(1, dtype="s32")

        # ── VAE (OpenVINO — Intel GPU plugin preferred) ─────────
        # ``QUARK_SKIP_VAE=1`` skips VAE load + pipeline construction.
        # Intended for tests / smoke runs on hosts without OpenVINO
        # (Mac CI). The engine is unusable for gen_frame in that mode;
        # ``self._taehv`` and ``self._pipe`` stay ``None``.
        import os as _os_skip

        if _os_skip.environ.get("QUARK_SKIP_VAE") == "1":
            self._taehv = None
            self.vae = None
            self._pipe = None
            self._vae_compute_units = None
            self._submit_in_flight = False
            return

        if not raw_cfg.get("taehv_ae", False):
            raise NotImplementedError(
                "quark.Engine on Intel currently supports the TAEHV VAE only. "
                "Set ``taehv_ae: true`` in the model config."
            )
        # The OpenVINO TAEHV repo is by-convention ``<ae_uri>-openvino`` —
        # holds pre-built IR (.xml + .bin) artifacts at every supported
        # latent resolution. Resolution order:
        #
        #   1. ``QUARK_TAEHV_OPENVINO_URI`` env var — for ad-hoc redirects
        #      (e.g. testing against a personal / staging HF repo before
        #      publishing to the canonical Overworld-Models namespace).
        #   2. ``openvino_uri`` in the model config — for permanent
        #      per-model overrides (forks or private repos).
        #   3. ``<ae_uri>-openvino`` — default convention.
        #
        # ``openvino_revision`` pins the HF revision for reproducibility;
        # default ``main`` tracks the latest publish-script run.
        import os as _os

        from quark.taehv import load_taehv

        ae_uri = raw_cfg["ae_uri"]
        openvino_uri = (
            _os.environ.get("QUARK_TAEHV_OPENVINO_URI")
            or raw_cfg.get("openvino_uri")
            or f"{ae_uri}-openvino"
        )
        openvino_revision = raw_cfg.get("openvino_revision")
        # ``load_taehv``'s ``latent_height`` / ``latent_width`` are the
        # PRE-patchify latent dims (the encoder's input spatial shape).
        # ``cfg.height * cfg.patch[0]`` / ``cfg.width * cfg.patch[1]``
        # gives those — the publish script exports artifacts keyed on
        # exactly this pair.
        ae_lat_h = cfg.height * cfg.patch[0]
        ae_lat_w = cfg.width * cfg.patch[1]
        # Prefer the OpenVINO GPU plugin; fall back to CPU when the GPU
        # plugin isn't reachable on this host (e.g. the apt-blocked
        # `intel-opencl-icd-runtime` install on some devkit images per
        # `project_openvino_taehv.md`). The fallback is honest about
        # what it loaded — exposed via ``self._vae_compute_units`` for
        # tests / logging.
        try:
            self._taehv = load_taehv(
                openvino_uri,
                latent_height=ae_lat_h,
                latent_width=ae_lat_w,
                revision=openvino_revision,
                cache_dir=taehv_cache_dir,
                compute_units="GPU",
            )
            self._vae_compute_units = "GPU"
        except Exception as exc:
            warnings.warn(
                "quark.Engine: OpenVINO GPU plugin unavailable for TAEHV "
                f"({type(exc).__name__}: {exc}); falling back to CPU. "
                "Install intel-opencl-icd + the OpenVINO GPU plugin to "
                "enable the GPU path.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._taehv = load_taehv(
                openvino_uri,
                latent_height=ae_lat_h,
                latent_width=ae_lat_w,
                revision=openvino_revision,
                cache_dir=taehv_cache_dir,
                compute_units="CPU",
            )
            self._vae_compute_units = "CPU"
        # Stub ``self.vae`` so external callers that introspect
        # ``engine.vae`` see the taehv handle (the methods don't match
        # the torch ``ChunkedStreamingTAEHV`` API but the attribute
        # exists for parity).
        self.vae = self._taehv

        # Pipelined decoder — single-worker thread that runs the
        # OpenVINO decode while the main thread is encoding the next
        # frame's denoise. The synchronous ``gen_frame`` path goes
        # through the same pipe (``submit_frame`` + immediate
        # ``next_pixels``); the worker still does the host snapshot
        # off-thread, just nothing else overlaps with the wait.
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
        """Encode ``img`` via the OpenVINO TAEHV encoder, drive a
        single commit-only DiT pass to write this frame's KV ring
        entry, then decode the same latent for the return image.
        Mirrors the CUDA-path semantics: KV cache advances by one,
        frame counter increments, returns the post-VAE-roundtrip
        image."""
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

        Routes through ``submit_frame`` + ``next_pixels`` so the VAE
        decode runs on the ``PipelinedDecoder`` worker thread. The
        decode overlaps with caller-side work between frames
        (host-side numpy conversion, video encoding, etc.) — that's
        the implicit pipelining; the call still blocks until pixels
        are ready.

        For two-thread submit/drain (real overlap of decode N-1 with
        denoise N) call ``submit_frame`` / ``next_pixels`` directly.
        Pipeline depth is bounded at 1, so ``gen_frame`` and the
        two-thread API are mutually exclusive within one session.
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
        the worker thread for OpenVINO decode in the background.
        Returns immediately. Pair with ``next_pixels`` to collect the
        result on a separate thread.

        Intended use is the two-thread submit/drain pattern: one
        thread calls ``submit_frame`` in a loop, another calls
        ``next_pixels`` in a loop, with a depth-1 semaphore between
        them for backpressure.

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
           host-readable so the OpenVINO decode below can ship it
           without a fresh wait.

        2. Commit (one model call writing this frame's KV ring entry)
           inside ``quark.lazy(sync=False)`` — fire-and-forget. The
           commit's GPU work then overlaps with the OpenVINO decode
           and the next frame's Python encoding.

        Today the OCL launcher syncs per-launch, so the
        ``quark.lazy()`` boundaries are effectively no-ops; the win
        kicks in once batched submit lands (Phase 4 of the OCL-E2E
        plan).
        """
        import ctypes as _ctypes

        import numpy as _np

        import quark
        from quark.runtime.sync import synchronize as _drain_lazy_queue

        ctrl_qt = self._encode_ctrl(ctrl)
        ctrl_emb = self.model.encode_ctrl(ctrl_qt) if ctrl_qt is not None else None

        # Refill the STABLE noise buffer in place. Using a fresh
        # ``QuarkTensor.from_numpy`` per frame would leak dispatcher
        # handle slots over long-running sessions (same shape as the
        # Metal stability bug).
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
        # ``_qk_tensor([fi])`` per call hits the same handle-table leak.
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
        # without waiting — they run concurrently with the OpenVINO
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
        layout the OpenVINO TAEHV encoder expects.

        Accepted inputs:
          * ``torch.Tensor`` (uint8 or [-1, 1] / [0, 1] float),
            ``np.ndarray``, or ``PIL.Image`` shaped ``[H, W, 3]``
            (single-frame seed) — broadcast to 4 frames.
          * ``torch.Tensor`` / ``np.ndarray`` shaped
            ``[4, H, W, 3]`` (a real 4-frame temporal seed) — pass
            straight through.

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
            arr_4 = _np.ascontiguousarray(arr)
        elif arr.ndim == 3 and arr.shape[-1] == 3:
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
