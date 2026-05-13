"""Convert BF16↔F32 round-trip — narrows the legalizer-chain bug further.

Following the chain:
  Split B32 → Bitcast B16→BF16 → Convert BF16→F32 → FMA →
  Convert F32→BF16 → Bitcast BF16→B16 → Merge B32.

- SplitB32 + MergeB32 verified identity (test_b32_roundtrip).
- Bitcast B16↔BF16 verified identity (test_bitcast_roundtrip).

This test checks Convert BF16→F32→BF16 in isolation. BF16→F32 is
lossless (zero-extend the mantissa low bits); F32→BF16 with default
"rn" rounding should round-to-nearest-even — so BF16→F32→BF16 must
be the identity on every finite BF16 value.
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


def _build_bf16_f32_bf16_ir(n: int = 16):
    """``out[i] = convert(convert(bitcast(in[i] as bf16), f32), bf16)``."""
    b = Builder("convert_module")
    fn = b.begin_function("convert_chain")
    b.param("X", BufferType(DType.B16))
    b.param("Z", BufferType(DType.B16))
    g_x = GlobalTensor(dtype=DType.B16, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_z = GlobalTensor(dtype=DType.B16, shape=(n,), stride=(1,),
                       name="Z", param=fn.params[1])
    tid = b.thread_idx("x")
    v_b16 = b.load(g_x, tid)
    v_bf = b.bitcast(v_b16, DType.BF16)
    v_f32 = b.convert(v_bf, DType.F32)
    v_bf_out = b.convert(v_f32, DType.BF16)
    v_b16_out = b.bitcast(v_bf_out, DType.B16)
    b.store(g_z, v_b16_out, tid)
    b.end_function()
    return b.module


def _make_finite_bf16_inputs(n: int, seed: int = 0xC0FFEE) -> np.ndarray:
    """Generate n uint16 patterns that decode to finite, non-NaN BF16."""
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.uint16)
    i = 0
    while i < n:
        x = rng.integers(0, 2**16, dtype=np.uint16)
        # Reject NaN/Inf: BF16 has 8-bit exp, all-ones exp = NaN/Inf.
        exp = (int(x) >> 7) & 0xFF
        if exp == 0xFF:
            continue
        out[i] = x
        i += 1
    return out


def test_convert_bf16_f32_bf16_roundtrip():
    """BF16 → F32 → BF16 must be the identity on every finite BF16."""
    driver = _ocl_driver_or_skip()

    n = 16
    X = _make_finite_bf16_inputs(n)

    result = OpenClSpirVLowerer(local_size=(n, 1, 1)).lower_module(
        _build_bf16_f32_bf16_ir(n)
    )
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

    got = np.empty(n, dtype=np.uint16)
    ctypes.memmove(got.ctypes.data, z_map, got.nbytes)
    np.testing.assert_array_equal(got, X)
