"""Pipelined OpenVINO TAEHV decoder — overlap Intel-GPU AE decode with
the next frame's SPV DiT forward.

Same threadpool architecture as :mod:`quark.taehv.coreml.pipeline` —
single worker, one decode in flight, host snapshot done on the
worker thread so the cache-coherence-sync'd memcpy from the
SPV-side host-coherent Vulkan buffer overlaps with the main thread's
next-frame dispatch.

On Intel Battlemage the DiT runs on the SPV/Vulkan compute queue
and the AE runs on the OpenVINO GPU plugin (Level Zero queue). Both
queues compete for the same GPU compute units, so the overlap win is
smaller than on Apple's split ANE+GPU silicon — but the host-side
work (memcpy, dtype conversion, Python dispatch) still benefits from
running off the main thread.
"""

from __future__ import annotations

import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quark.taehv.openvino.runtime import OpenVINOTAEHV


class PipelinedDecoder:
    """One-decode-in-flight wrapper around :class:`OpenVINOTAEHV`.

    API matches the CoreML pipeline's: ``submit`` / ``next`` /
    ``flush`` / ``shutdown``.
    """

    def __init__(self, ae: OpenVINOTAEHV):
        self._ae = ae
        self._decoder_input_shape: tuple[int, int, int, int] = (
            1, 32, ae._lat_h, ae._lat_w,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="taehv-openvino-decode",
        )
        self._pending: Future | None = None

    def submit(self, latent: Any) -> None:
        if self._pending is not None:
            raise RuntimeError(
                "PipelinedDecoder.submit: previous submit still in flight — "
                "call .next() / .flush() to drain it first."
            )
        self._pending = self._executor.submit(self._snapshot_and_decode, latent)

    def next(self) -> np.ndarray | None:
        if self._pending is None:
            return None
        result = self._pending.result()
        self._pending = None
        return result

    def flush(self) -> np.ndarray | None:
        return self.next()

    def shutdown(self) -> None:
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
        with contextlib.suppress(Exception):
            self.shutdown()

    # ── Worker-thread body ─────────────────────────────────────────

    def _snapshot_and_decode(self, latent: Any) -> np.ndarray:
        latent_np = self._latent_to_numpy(latent)
        del latent  # release storage ref before the slow AE call
        return self._ae.decode(latent_np)

    def _latent_to_numpy(self, latent: Any) -> np.ndarray:
        """Coerce QuarkTensor / numpy ndarray → ``[1, 32, latH, latW] f32``.

        On SPV the QuarkTensor's storage is host-coherent Vulkan
        memory; ``data_ptr()`` returns a host-readable pointer. The
        caller MUST have already synced the SPV queue (typically via
        ``quark.eval()`` at end of frame) before calling
        :meth:`submit`. We deliberately do NOT call ``synchronize()``
        on the worker thread — that would block the main thread from
        submitting the next frame's SPV work while we wait, which
        adds ~80 ms / frame of pipeline-mode DiT slowdown.
        """
        if isinstance(latent, np.ndarray):
            arr = latent
            if arr.dtype == np.uint16:  # bf16 carrier
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            return arr.astype(np.float32, copy=False).reshape(self._decoder_input_shape)

        # QuarkTensor path
        import ctypes

        shape = tuple(int(s) for s in latent.shape)
        dtype = getattr(latent, "dtype", None) or getattr(latent, "quark_dtype", None)
        n_elems = 1
        for s in shape:
            n_elems *= s

        if dtype == "bf16":
            carrier = np.empty(shape, dtype=np.uint16)
            ctypes.memmove(carrier.ctypes.data, latent.data_ptr(), n_elems * 2)
            f32 = (carrier.astype(np.uint32) << 16).view(np.float32)
        elif dtype == "f16":
            carrier = np.empty(shape, dtype=np.float16)
            ctypes.memmove(carrier.ctypes.data, latent.data_ptr(), n_elems * 2)
            f32 = carrier.astype(np.float32)
        elif dtype == "f32":
            carrier = np.empty(shape, dtype=np.float32)
            ctypes.memmove(carrier.ctypes.data, latent.data_ptr(), n_elems * 4)
            f32 = carrier
        else:
            raise TypeError(
                f"PipelinedDecoder: latent dtype {dtype!r} not supported "
                "(expected 'bf16' / 'f16' / 'f32')."
            )

        target = self._decoder_input_shape
        target_n = 1
        for s in target:
            target_n *= s
        if n_elems != target_n:
            raise ValueError(
                f"PipelinedDecoder: latent has {n_elems} elements, "
                f"decoder expected {target_n} for shape {target}"
            )
        return f32.reshape(target)
