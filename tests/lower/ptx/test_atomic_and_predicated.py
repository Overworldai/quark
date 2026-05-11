"""Tests for atomic RMW and predicated load/store lowering."""

import re

from quark.ir import BufferType, DType, GlobalTensor
from tests.lower.ptx.conftest import lower


def _g1d(b, dtype=DType.F32, n=128):
    b.param("X", BufferType(dtype))
    return GlobalTensor(
        dtype=dtype,
        shape=(n,),
        stride=(1,),
        name="X",
        param=b.function.params[-1],
    )


class TestAtomicRmw:
    def test_add_f32(self, fresh_builder):
        g = _g1d(fresh_builder)
        idx = fresh_builder.const(DType.U32, 0)
        v = fresh_builder.const(DType.F32, 1.0)
        fresh_builder.atomic_rmw(g, "add", v, idx)
        out = lower(fresh_builder)
        assert re.search(r"atom\.global\.add\.f32 %f\d+, \[%rd\d+\], %f\d+;", out)

    def test_min_u32(self, fresh_builder):
        g = _g1d(fresh_builder, dtype=DType.U32)
        idx = fresh_builder.const(DType.U32, 0)
        v = fresh_builder.const(DType.U32, 5)
        fresh_builder.atomic_rmw(g, "min", v, idx)
        out = lower(fresh_builder)
        assert "atom.global.min.u32" in out


class TestPredicatedScalarMem:
    def test_predicated_gmem_load(self, fresh_builder):
        g = _g1d(fresh_builder)
        idx = fresh_builder.const(DType.U32, 0)
        a = fresh_builder.const(DType.F32, 1.0)
        z = fresh_builder.const(DType.F32, 0.0)
        p = fresh_builder.cmp("lt", z, a)
        fresh_builder.load(g, idx, pred=p)
        out = lower(fresh_builder)
        assert re.search(r"@%p\d+ ld\.global\.f32", out)

    def test_predicated_gmem_store(self, fresh_builder):
        g = _g1d(fresh_builder)
        idx = fresh_builder.const(DType.U32, 0)
        a = fresh_builder.const(DType.F32, 1.0)
        z = fresh_builder.const(DType.F32, 0.0)
        p = fresh_builder.cmp("lt", z, a)
        fresh_builder.store(g, a, idx, pred=p)
        out = lower(fresh_builder)
        assert re.search(r"@%p\d+ st\.global\.f32", out)


class TestPredicatedVecMem:
    def test_predicated_vec_load(self, fresh_builder):
        b = fresh_builder
        b.param("X", BufferType(DType.F32))
        g = GlobalTensor(
            dtype=DType.F32,
            shape=(16, 32),
            stride=(32, 1),
            name="X",
            param=b.function.params[-1],
        )
        row = b.const(DType.U32, 0)
        col = b.const(DType.U32, 0)
        z = b.const(DType.F32, 0.0)
        one = b.const(DType.F32, 1.0)
        p = b.cmp("lt", z, one)
        b.vec_load(g, row, col, width=4, pred=p)
        out = lower(b)
        assert re.search(r"@%p\d+ ld\.global\.v4\.f32", out)
