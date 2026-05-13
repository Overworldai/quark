"""Full fma_bf16x2 legalizer chain WITHOUT the FMA — narrows to composition.

The fma_bf16x2 expansion is:
  Split B32 → 2× Bitcast B16→BF16 → 2× Convert BF16→F32 → 2× F32 FMA
  → 2× Convert F32→BF16 → 2× Bitcast BF16→B16 → Merge B32.

Individual scalar visitors (Split/Merge, Bitcast B16↔BF16, Convert
BF16↔F32) all round-trip identity. This test composes them in the
same order the legalizer does, but skips the FMA — both lanes go
F32 → F32 unchanged. The whole chain must round-trip identity on
every (finite BF16-pair) packed into a B32.

If this passes, the bug is in the F32 FMA (or its operand wiring).
If this fails, the bug is in how the visitors compose at vec-2 scope.
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
from quark.lower.ocl import OpenClSpirVLowerer


def _ocl_driver_or_skip():
    try:
        from quark.drivers.ocl import OclDriver, is_available
    except Exception:
        pytest.skip("OCL driver not importable")
    if not is_available():
        pytest.skip("No OCL device available")
    return OclDriver()


def _build_full_chain_no_fma_ir(n: int = 16):
    """For each B32 in[i]:
        lo_b16, hi_b16 = split_b32(in[i])
        lo_bf = bitcast(lo_b16, bf16); hi_bf = bitcast(hi_b16, bf16)
        lo_f  = convert(lo_bf, f32);   hi_f  = convert(hi_bf, f32)
        lo_bf'= convert(lo_f,  bf16);  hi_bf'= convert(hi_f,  bf16)
        lo_b'= bitcast(lo_bf', b16);   hi_b'= bitcast(hi_bf', b16)
        out[i] = merge_b32(lo_b', hi_b')
    """
    b = Builder("full_chain_module")
    fn = b.begin_function("full_chain")
    b.param("X", BufferType(DType.B32))
    b.param("Z", BufferType(DType.B32))
    g_x = GlobalTensor(dtype=DType.B32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_z = GlobalTensor(dtype=DType.B32, shape=(n,), stride=(1,),
                       name="Z", param=fn.params[1])
    tid = b.thread_idx("x")
    v = b.load(g_x, tid)

    lo_b16, hi_b16 = b.split_b32(v)
    lo_bf = b.bitcast(lo_b16, DType.BF16)
    hi_bf = b.bitcast(hi_b16, DType.BF16)
    lo_f = b.convert(lo_bf, DType.F32)
    hi_f = b.convert(hi_bf, DType.F32)
    lo_bf_out = b.convert(lo_f, DType.BF16)
    hi_bf_out = b.convert(hi_f, DType.BF16)
    lo_b_out = b.bitcast(lo_bf_out, DType.B16)
    hi_b_out = b.bitcast(hi_bf_out, DType.B16)
    merged = b.merge_b32(lo_b_out, hi_b_out)

    b.store(g_z, merged, tid)
    b.end_function()
    return b.module


def _make_finite_bf16x2_inputs(n: int, seed: int = 0xCAFE0000) -> np.ndarray:
    """n uint32 patterns where both halves decode to finite BF16."""
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.uint32)
    i = 0
    while i < n:
        x = int(rng.integers(0, 2**32, dtype=np.uint32))
        lo = x & 0xFFFF
        hi = (x >> 16) & 0xFFFF
        if ((lo >> 7) & 0xFF) == 0xFF or ((hi >> 7) & 0xFF) == 0xFF:
            continue  # skip NaN/Inf in either lane
        out[i] = np.uint32(x)
        i += 1
    return out


def test_full_chain_no_fma_roundtrip():
    """Composing Split/Bitcast/Convert↔Convert/Bitcast/Merge in the
    same order as the fma_bf16x2 expansion (minus the FMA) must be
    the identity on every finite-BF16-pair B32."""
    driver = _ocl_driver_or_skip()

    n = 16
    X = _make_finite_bf16x2_inputs(n)

    result = OpenClSpirVLowerer(local_size=(n, 1, 1)).lower_module(
        _build_full_chain_no_fma_ir(n)
    )
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
