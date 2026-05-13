"""fma_bf16x2 identity round-trip via the legalizer expansion.

This is the smallest test that actually triggers the full
``_expand_fma_bf16x2`` chain on the OCL backend. Operands:
  a = X[i] (B32, two packed BF16 values from the input)
  b = packed BF16(1.0)x2 = 0x3F803F80
  c = packed BF16(0.0)x2 = 0x00000000

Math: a * 1.0 + 0.0 = a, so out[i] must equal X[i] for every finite
BF16 pair in X.

Previous tests have shown that:
- SplitB32 + MergeB32 round-trip identity.
- Bitcast B16↔BF16 round-trips identity.
- Convert BF16↔F32 round-trips identity.
- The full Split→Bitcast→Convert→Convert→Bitcast→Merge composition
  without the F32 FMA round-trips identity.

So if THIS test fails, the bug is either in:
  (a) the F32 fma visitor, or
  (b) how the legalizer wires its expansion result Value to downstream
      stores when MergeB32 reuses ``out_b32`` (the ArithOp's result Value).
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


def _build_fma_identity_ir(n: int = 16):
    """``out[i] = fma_bf16x2(in[i], 1.0_x2, 0.0_x2)`` — must equal in[i]."""
    b = Builder("fma_identity_module")
    fn = b.begin_function("fma_identity")
    b.param("X", BufferType(DType.B32))
    b.param("Z", BufferType(DType.B32))
    g_x = GlobalTensor(dtype=DType.B32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_z = GlobalTensor(dtype=DType.B32, shape=(n,), stride=(1,),
                       name="Z", param=fn.params[1])
    tid = b.thread_idx("x")
    a = b.load(g_x, tid)
    ones = b.const(DType.B32, 0x3F803F80)   # bf16(1.0) twice
    zeros = b.const(DType.B32, 0x00000000)  # bf16(0.0) twice
    d = b.fma_bf16x2(a, ones, zeros)
    b.store(g_z, d, tid)
    b.end_function()
    return b.module


def _make_finite_bf16x2_inputs(n: int, seed: int = 0xFA1DEAD0) -> np.ndarray:
    """n uint32 patterns where both halves decode to finite, non-zero,
    non-subnormal BF16. Subnormals are excluded because FMA on Battlemage
    may flush them, breaking the round-trip identity."""
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.uint32)
    i = 0
    while i < n:
        x = int(rng.integers(0, 2**32, dtype=np.uint32))
        ok = True
        for shift in (0, 16):
            half = (x >> shift) & 0xFFFF
            exp = (half >> 7) & 0xFF
            if exp == 0 or exp == 0xFF:   # subnormal, zero, NaN, or Inf
                ok = False
                break
        if ok:
            out[i] = np.uint32(x)
            i += 1
    return out


def test_fma_bf16x2_identity():
    """``fma_bf16x2(x, 1.0_x2, 0.0_x2)`` must equal ``x`` for every
    finite, normal BF16 pair packed in a B32."""
    driver = _ocl_driver_or_skip()

    n = 16
    X = _make_finite_bf16x2_inputs(n)

    # Production path goes through Launcher._lower which runs legalize()
    # before lower_module; reproduce that here so fma_bf16x2 expands.
    module = _build_fma_identity_ir(n)
    legalize(module, driver.caps)
    result = OpenClSpirVLowerer(local_size=(n, 1, 1)).lower_module(module)
    binary = text_to_binary(result.source, target_env="opencl2.0")

    nbytes = n * 4
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

    got = np.empty(n, dtype=np.uint32)
    ctypes.memmove(got.ctypes.data, z_map, got.nbytes)
    np.testing.assert_array_equal(got, X)
