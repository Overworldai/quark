"""Bitcast B16↔BF16 round-trip — narrows the legalizer-chain bug.

The fma_bf16x2 expansion in ``quark.lower.legalizations`` chains:
  Split B32 → Bitcast B16→BF16 → Convert BF16→F32 → FMA →
  Convert F32→BF16 → Bitcast BF16→B16 → Merge B32.

SplitB32 + MergeB32 verified identity (test_b32_roundtrip).
ValueResidualPacked / AdaGateResidual produce wrong numerics.
This test checks the Bitcast B16↔BF16 portion in isolation.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from quark.ir import DType
from quark.ir.builder import Builder
from quark.ir.op import BitcastOp
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


def _build_b16_bf16_b16_ir(n: int = 16):
    """``out[i] = bitcast(bitcast(in[i] as b16, bf16), b16)``."""
    from quark.ir.value import Value
    from quark.ir.types import ValueShape

    b = Builder("bitcast_module")
    fn = b.begin_function("bitcast_chain")
    b.param("X", BufferType(DType.B16))
    b.param("Z", BufferType(DType.B16))
    g_x = GlobalTensor(dtype=DType.B16, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_z = GlobalTensor(dtype=DType.B16, shape=(n,), stride=(1,),
                       name="Z", param=fn.params[1])
    tid = b.thread_idx("x")
    v_b16 = b.load(g_x, tid)
    # bitcast B16 → BF16, then BF16 → B16 — must be identity.
    v_bf = fn.fresh_value(ValueShape(DType.BF16))
    bc1 = BitcastOp(results=(v_bf,), operands=(v_b16,),
                    attrs={"dst_dtype": DType.BF16})
    fn.body.blocks[0].ops.append(bc1)
    v_b16_2 = fn.fresh_value(ValueShape(DType.B16))
    bc2 = BitcastOp(results=(v_b16_2,), operands=(v_bf,),
                    attrs={"dst_dtype": DType.B16})
    fn.body.blocks[0].ops.append(bc2)
    b.store(g_z, v_b16_2, tid)
    b.end_function()
    return b.module


def test_bitcast_b16_bf16_b16_roundtrip():
    """B16 → BF16 → B16 must be the identity on all 16-bit patterns."""
    driver = _ocl_driver_or_skip()

    n = 16
    rng = np.random.default_rng(0xCAFEBABE)
    X = rng.integers(0, 2**16 - 1, size=(n,), dtype=np.uint16)

    result = OpenClSpirVLowerer(local_size=(n, 1, 1)).lower_module(
        _build_b16_bf16_b16_ir(n)
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
