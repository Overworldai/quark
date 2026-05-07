"""Pipelined TAEHV decoder — overlap ANE decode with the next frame's
GPU forward.

The ANE and the Apple GPU are distinct compute pools sharing unified
memory: a decode dispatched to the Neural Engine doesn't contend
with kernels enqueued on the Metal command queue *for compute*.
Running them serially costs ~22 ms / latent on 360p; running them
in parallel via a single-worker ``ThreadPoolExecutor`` hides most
of that behind the next frame's DiT forward.

Note: full hiding (0 ms tax) is bounded by SoC-wide contention,
not the per-engine separation. Under heavy GPU load the same
``decode()`` call inflates from ~12 ms isolated to ~60 ms — same
inflation regardless of which engine routes the decoder
(``CPU_AND_NE`` vs ``CPU_AND_GPU``), so it's a system budget /
fabric limit, not a compute-pool collision. See the docstring on
:class:`~quark.taehv.coreml.CoreMLTAEHV` for the diagnostic notes.

The earlier inline implementation in
``scripts/bench_quark_world_engine.py`` did the host-side latent
memcpy on the main thread before submitting the decode task. That
puts the ``ctypes.memmove`` from a freshly GPU-written buffer
(needs cache-coherence sync on M-series UMA) and the
``torch.from_numpy + .float32 + .numpy + .float16`` format dance
on the main thread, where they steal time from the next frame's
``pcf.gemm`` Python dispatch. We measured that as ~5 ms / frame of
forward-time inflation.

This wrapper moves both the host snapshot AND the decode call into
the worker thread, so they overlap with the main thread's next-
frame dispatch. The latent buffer is read via the QuarkTensor's
``data_ptr`` from inside the worker — safe because the
refcount-aware lazy-buffer lifetime
(``src/quark/drivers/_metal_dispatch.cpp``) keeps the storage
alive as long as a Python wrapper exists, and the worker thread
holds a reference until the future resolves.
"""

from __future__ import annotations

import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quark.taehv.coreml import CoreMLTAEHV


class PipelinedDecoder:
    """One-decode-in-flight wrapper around :class:`CoreMLTAEHV`.

    Single-worker pool; the API is intentionally small:

      * :meth:`submit` — start decoding a latent; non-blocking.
        At most one decode is in flight; a second ``submit`` before
        the first completes will block (by design — we pipeline one
        ahead, not many).
      * :meth:`next` — return the previous frame's image (or None
        if no prior submit). Blocks if the decode hasn't finished.
      * :meth:`flush` — drain the last pending submit and return
        its image (or None if nothing pending). Call after the
        last :meth:`submit` to collect the tail.

    The expected loop is:

        for fi, latent in enumerate(latents):
            pipe.submit(latent)
            img = pipe.next()              # None on iter 0
            if img is not None:
                handle(img)
        img = pipe.flush()
        if img is not None:
            handle(img)
    """

    def __init__(self, ae: CoreMLTAEHV):
        self._ae = ae
        # Cache the decoder input shape so ``submit`` can reshape a
        # flat-2D ``QuarkTensor`` (the world-model commit returns
        # ``(1, C*H*W)``) into the ``(1, 32, latH, latW)`` the
        # decoder wants — without forcing the caller to reshape on
        # the main thread.
        self._decoder_input_shape: tuple[int, int, int, int] = (
            1,
            32,
            ae._lat_h,
            ae._lat_w,
        )
        # ``max_workers=1`` is load-bearing: we only ever pipeline
        # one decode ahead. A larger pool would let multiple decodes
        # race for the same MemBlock state, which is shared / order-
        # dependent.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="taehv-decode",
        )
        self._pending: Future | None = None

    # ── Public API ─────────────────────────────────────────────────

    def submit(self, latent: Any) -> None:
        """Start decoding ``latent`` in the background.

        ``latent`` may be:
          * a ``QuarkTensor`` — host snapshot happens on the worker
            thread (uses ``data_ptr`` + ``ctypes.memmove``).
          * a ``np.ndarray`` of shape ``[1, 32, latH, latW]`` —
            handed straight to the decoder.

        If a previous ``submit`` is still in flight, this raises
        ``RuntimeError`` rather than queuing — pipeline depth is
        intentionally bounded at 1 (one ahead) so memory stays
        bounded and ordering w.r.t. the decoder's MemBlock state
        is unambiguous.
        """
        if self._pending is not None:
            raise RuntimeError(
                "PipelinedDecoder.submit: a previous submit is still in "
                "flight — call .next() (or .flush()) to drain it first."
            )
        self._pending = self._executor.submit(self._snapshot_and_decode, latent)

    def next(self) -> np.ndarray | None:
        """Collect the previously-submitted frame and return it.

        Returns ``None`` if no submit is pending (e.g. on the first
        loop iter, or after a manual ``flush``).
        """
        if self._pending is None:
            return None
        result = self._pending.result()
        self._pending = None
        return result

    def flush(self) -> np.ndarray | None:
        """Drain the last pending submit, then return its frame.
        ``None`` if nothing was pending."""
        return self.next()

    def shutdown(self) -> None:
        """Stop the worker thread. Called automatically on ``__exit__``
        / ``__del__`` but exposed for explicit cleanup."""
        if self._pending is not None:
            with contextlib.suppress(Exception):
                self._pending.result(timeout=5.0)
            self._pending = None
        self._executor.shutdown(wait=True)

    def __enter__(self) -> PipelinedDecoder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def __del__(self) -> None:
        # Best-effort cleanup; don't raise from __del__.
        with contextlib.suppress(Exception):
            self.shutdown()

    # ── Internals (run on the worker thread) ───────────────────────

    def _snapshot_and_decode(self, latent: Any) -> np.ndarray:
        """Worker-thread body: convert ``latent`` to a numpy
        ``[1, 32, latH, latW] f16`` array, then decode.

        Doing the host snapshot here (rather than on the main
        thread before submit) overlaps the cache-coherence-sync'd
        memcpy with the main thread's next-frame Python dispatch.

        We explicitly ``del latent`` right after the snapshot so
        the QuarkTensor reference held by this Future drops *before*
        the (slow, ~25 ms) CoreML decode runs. That lets the
        dispatcher's refcount-aware buffer pool recycle the
        latent's Metal buffer on the next ``eval()`` call —
        otherwise the buffer would stay live for the whole 25 ms
        of decode and the next frame's auto-allocations would have
        to ``newBuffer`` instead of pool-popping.
        """
        latent_np = self._latent_to_numpy(latent)
        del latent  # release storage refcount before the slow decode
        return self._ae.decode(latent_np)

    def _latent_to_numpy(self, latent: Any) -> np.ndarray:
        """Coerce a QuarkTensor or a numpy array to the
        ``[1, 32, latH, latW] f16`` layout the decoder wants.

        QuarkTensor path: snapshot the buffer via ``ctypes.memmove``
        from ``data_ptr()``, widen bf16 → f32 → f16, then reshape
        to ``self._decoder_input_shape``. The host snapshot is the
        cache-coherence-sync step that costs ~3 ms on M-series UMA
        right after a GPU-write — doing it here (worker thread)
        instead of on the main thread overlaps it with the next
        frame's GPU forward dispatch.

        Numpy path: trust the caller's shape and dtype; ``decode``
        will refuse anything that doesn't match.
        """
        if isinstance(latent, np.ndarray):
            return latent

        # QuarkTensor path. We avoid importing QuarkTensor at module
        # load to keep the import graph torch-free; this branch only
        # runs in user processes that already have quark loaded.
        import ctypes

        shape = tuple(int(s) for s in latent.shape)
        dtype = getattr(latent, "dtype", None) or getattr(
            latent,
            "quark_dtype",
            None,
        )
        if dtype not in ("bf16", "f16"):
            raise TypeError(
                f"PipelinedDecoder: latent dtype {dtype!r} not supported (expected 'bf16' or 'f16')"
            )

        n_elems = 1
        for s in shape:
            n_elems *= s
        # Both bf16 and f16 are 2 bytes / element; carrier dtype is
        # ``np.uint16`` for the raw bit copy.
        carrier = np.empty(shape, dtype=np.uint16)
        ctypes.memmove(carrier.ctypes.data, latent.data_ptr(), n_elems * 2)

        if dtype == "bf16":
            # bf16 stored in upper 16 bits of f32; widen by shifting.
            f32 = (carrier.astype(np.uint32) << 16).view(np.float32)
        else:  # f16
            f32 = carrier.view(np.float16).astype(np.float32)

        # Decoder expects ``self._decoder_input_shape`` (typically
        # ``(1, 32, latH, latW)``). The world-model commit returns
        # the latent flat as ``(1, C*H*W)``, so reshape if needed.
        target = self._decoder_input_shape
        target_n = 1
        for s in target:
            target_n *= s
        if n_elems != target_n:
            raise ValueError(
                f"PipelinedDecoder: latent has {n_elems} elements, "
                f"decoder expected {target_n} for shape {target}"
            )
        return f32.reshape(target).astype(np.float16)
