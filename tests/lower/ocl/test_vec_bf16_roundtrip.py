"""vec_load + vec_store of BF16 width=8 — narrows packed-load/store path.

ValueResidualPacked + AdaGateResidual share the pattern:
   c_vec = qk.vec_load(stage.curr, ..., width=8, dtype=bf16)
   ... (process per lane via vec_extract / convert / build) ...
   qk.vec_store(g.Out, qk.vec_build(out_elems), ...)

Each scalar component visitor (Split/Merge, Bitcast, Convert, scalar
FMA, fma_bf16x2 expansion) round-trips identity in isolation.

This test exercises only:
   tmp = vec_load(in, tid*8, width=8)
   vec_store(out, tmp, tid*8)

If this fails, the bug is in OCL's vec_load or vec_store visitor.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from quark.ir import DType
from quark.ir.builder import Builder
from quark.ir.tensor import GlobalTensor
from quark.ir.types import BufferType
from quark.lower._common.spirv_assemble import text_to_binary
from quark.lower.legalize import legalize
from quark.lower.ocl import OpenClSpirVLowerer


def _ocl_driver_or_skip():
    try:
        from quark.drivers.ocl import OclDriver, is_available
    except Exception:
        pytest.skip("OCL driver not importable")
    if not is_available():
        pytest.skip("No OCL device available")
    return OclDriver()


def _build_vec_roundtrip_ir(rows: int = 4, width: int = 8):
    """One workitem per row: vec_load width=8 BF16, vec_store same."""
    b = Builder("vec_bf16_module")
    fn = b.begin_function("vec_bf16_roundtrip")
    b.param("X", BufferType(DType.BF16))
    b.param("Z", BufferType(DType.BF16))
    g_x = GlobalTensor(dtype=DType.BF16, shape=(rows, width), stride=(width, 1),
                       name="X", param=fn.params[0])
    g_z = GlobalTensor(dtype=DType.BF16, shape=(rows, width), stride=(width, 1),
                       name="Z", param=fn.params[1])
    tid = b.thread_idx("x")
    col0 = b.const(DType.U32, 0)
    v = b.vec_load(g_x, tid, col0, width=width)
    b.vec_store(g_z, v, tid, col0)
    b.end_function()
    return b.module


def test_vec_bf16_width8_roundtrip():
    """``vec_load(BF16, w=8) → vec_store`` must round-trip identity."""
    driver = _ocl_driver_or_skip()

    rows = 4
    width = 8
    n = rows * width
    rng = np.random.default_rng(0xFAE0001)
    # Avoid NaN/Inf/subnormal — just normal BF16 patterns.
    X = np.empty(n, dtype=np.uint16)
    i = 0
    while i < n:
        x = int(rng.integers(0, 2**16, dtype=np.uint16))
        exp = (x >> 7) & 0xFF
        if exp == 0 or exp == 0xFF:
            continue
        X[i] = x
        i += 1
    X = X.reshape(rows, width)

    module = _build_vec_roundtrip_ir(rows, width)
    legalize(module, driver.caps)
    result = OpenClSpirVLowerer(local_size=(rows, 1, 1)).lower_module(module)
    binary = text_to_binary(result.source, target_env="opencl2.0")

    nbytes = n * 2
    x_h, x_map = driver.allocate_buffer(nbytes)
    z_h, z_map = driver.allocate_buffer(nbytes)
    ctypes.memmove(x_map, X.ctypes.data, X.nbytes)

    compiled = driver.compile(
        binary,
        entry=result.entry_name,
        n_buffers=result.n_buffers,
        subgroup_size=result.subgroup_size,
        local_size=result.local_size,
    )
    driver.launch(compiled, grid=(1, 1, 1),
                  buffers=[x_h, z_h], sync=True)

    got = np.empty(n, dtype=np.uint16).reshape(rows, width)
    ctypes.memmove(got.ctypes.data, z_map, n * 2)
    np.testing.assert_array_equal(got, X)
