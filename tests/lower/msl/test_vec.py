"""Tests for vector build/extract/split/merge MSL lowering."""

from popcorn.ir import DType
from tests.lower.msl.conftest import lower


class TestVecBuild:
    def test_vec_build_aliases(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        v = b.vec_build([a, c])
        # VecBuild is a zero-emit alias — it doesn't add new lines.
        # VecExtract should resolve to the original names.
        b.vec_extract(v, 0)
        b.vec_extract(v, 1)
        out = lower(b)
        # The output should still contain the original consts.
        assert "float _pc_f32_0 =" in out
        assert "float _pc_f32_1 =" in out


class TestSplitMerge:
    def test_split_b32(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.B32, 0x00010002)
        lo, hi = b.split_b32(v)
        out = lower(b)
        assert "as_type<ushort2>" in out
        assert ".x;" in out
        assert ".y;" in out

    def test_merge_b32(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.B16, 1)
        c = b.const(DType.B16, 2)
        b.merge_b32(a, c)
        out = lower(b)
        assert "as_type<uint>(ushort2(" in out
