"""SplitB32 / MergeB32 round-trip — minimal reproducer.

Builds a kernel ``out[i] = merge(split(in[i]))`` and verifies the
round-trip is identity. If round-trip is identity but kernels using
the split/merge chain (ValueResidualPacked, AdaGateResidual) produce
wrong output, the bug is in those kernels' emit. If round-trip
itself is broken, the bug is in the OCL visitors.
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


def _build_split_merge_roundtrip_ir(n: int = 16):
    """``out[i] = merge_b32(split_b32(in[i]))`` for each work-item."""
    b = Builder("split_merge_module")
    fn = b.begin_function("split_merge")
    b.param("X", BufferType(DType.B32))
    b.param("Z", BufferType(DType.B32))
    g_x = GlobalTensor(dtype=DType.B32, shape=(n,), stride=(1,),
                       name="X", param=fn.params[0])
    g_z = GlobalTensor(dtype=DType.B32, shape=(n,), stride=(1,),
                       name="Z", param=fn.params[1])
    tid = b.thread_idx("x")
    v = b.load(g_x, tid)
    lo, hi = b.split_b32(v)
    merged = b.merge_b32(lo, hi)
    b.store(g_z, merged, tid)
    b.end_function()
    return b.module


def test_split_merge_roundtrip_is_identity():
    """The composition of SplitB32 + MergeB32 must be the identity
    function on every 32-bit pattern."""
    driver = _ocl_driver_or_skip()

    n = 16
    rng = np.random.default_rng(0xDEADBEEF)
    X = rng.integers(0, 2**32 - 1, size=(n,), dtype=np.uint32)

    result = OpenClSpirVLowerer(local_size=(n, 1, 1)).lower_module(
        _build_split_merge_roundtrip_ir(n)
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
    # Every position must round-trip exactly.
    np.testing.assert_array_equal(got, X)
