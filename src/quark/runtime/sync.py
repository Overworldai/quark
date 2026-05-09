"""Backend-neutral sync / zero / dtype-probe helpers.

Used by the autotune inner loop (bounded search + full genetic) to
keep the per-config dance portable between CUDA (QuarkTensor) and
Metal (np.ndarray) without reaching into ``quark.backend.PT`` — which
is being retired in the numpy-refs migration.
"""

from __future__ import annotations

import sys
from typing import Any

from quark.ir import DType

import os as _os

_IS_METAL = sys.platform == "darwin"
_IS_SPV = _os.environ.get("QUARK_BACKEND", "").lower() in ("spv", "intel")


def synchronize() -> None:
    """Wait for all device work to complete on the default stream.

    On Metal: drain BOTH the lazy ``queue_launch`` queue (functional
    fast-paths like NAX GEMM enqueue here without auto-flushing) and
    the eager encoder. The pre-migration "PyObjC dispatch is
    synchronous" assumption no longer holds — without this drain,
    callers that ``synchronize()`` between phases outside a
    ``quark.lazy()`` block (e.g. the bench's seed commit) silently
    leave their dispatches sitting in the queue, and later phases
    (warmup, generation) execute them out-of-order with stale state.

    On SPV: ``_sd.sync`` calls ``vkDeviceWaitIdle`` which drains
    everything in flight on the compute queue. Async-mode launches
    (``sync=False``) accumulate in the queue until this fires.
    """
    if _IS_METAL:
        from quark.drivers import _metal_dispatch as _md

        if _md.has_lazy_pending():
            _md.eval_queue()
        if _md.has_pending():
            _md.eval()
        return
    if _IS_SPV:
        from quark.drivers import _spv_dispatch as _sd

        _sd.sync()
        return
    from quark.runtime.cuda import CudaRuntime

    CudaRuntime.instance().stream_synchronize(0)


def zero_buffer(buf: Any) -> Any:
    """Zero ``buf`` in place (or return a fresh zero-filled copy).

    QuarkTensor → in-place memset. np.ndarray → returns a new
    ``np.zeros(...)``."""
    if hasattr(buf, "zero_"):
        return buf.zero_()
    if _IS_METAL:
        import numpy as np

        return np.zeros(buf.shape, dtype=buf.dtype)
    # Fallback — shouldn't hit in practice.
    raise TypeError(f"zero_buffer: don't know how to zero {type(buf).__name__}")


_PC_TO_IR: dict[str, DType] = {
    "f32": DType.F32,
    "f16": DType.F16,
    "bf16": DType.BF16,
    "e4m3": DType.E4M3,
    "e5m2": DType.E5M2,
    "s32": DType.S32,
    "s64": DType.S64,
    "u8": DType.U8,
    "s8": DType.S8,
    "u16": DType.U16,
    "u32": DType.U32,
}


def ir_dtype_of(buf: Any) -> DType:
    """Resolve ``buf.dtype`` to a quark ``DType`` enum.

    QuarkTensor stores dtype as a short string; np.ndarray carries an
    ``np.dtype`` we map via its ``.name`` attribute.
    """
    dt = getattr(buf, "dtype", None)
    if isinstance(dt, str):
        ir = _PC_TO_IR.get(dt)
        if ir is None:
            raise TypeError(f"ir_dtype_of: QuarkTensor dtype {dt!r} has no quark.ir.DType")
        return ir
    # numpy dtype has a ``.name`` like "float32", "uint16".
    name = getattr(dt, "name", None)
    _NP_NAME_TO_IR = {
        "float32": DType.F32,
        "float16": DType.F16,
        "int32": DType.S32,
        "int64": DType.S64,
        "uint8": DType.U8,
        "int8": DType.S8,
        "uint16": DType.U16,
        "uint32": DType.U32,
    }
    if name in _NP_NAME_TO_IR:
        return _NP_NAME_TO_IR[name]
    raise TypeError(f"ir_dtype_of: cannot resolve {type(buf).__name__}.dtype={dt!r}")
