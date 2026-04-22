"""Tests for PTX lowering of vec build/extract and split/merge_b32."""

import re

from quark.ir import DType
from tests.lower.ptx.conftest import body, lower


class TestVecBuildExtract:
    def test_vec_build_is_pure_aliasing(self, fresh_builder):
        """VecBuildOp emits no PTX instructions — the result shares
        the inputs' registers."""
        b = fresh_builder
        a0 = b.const(DType.F32, 1.0)
        a1 = b.const(DType.F32, 2.0)
        a2 = b.const(DType.F32, 3.0)
        a3 = b.const(DType.F32, 4.0)
        b.vec_build([a0, a1, a2, a3])
        text = body(lower(b))
        # Only the four consts + ret should be emitted — no extra movs.
        movs = re.findall(r"mov\.f32 ", text)
        assert len(movs) == 4

    def test_vec_extract_is_pure_aliasing(self, fresh_builder):
        b = fresh_builder
        a0 = b.const(DType.F32, 1.0)
        a1 = b.const(DType.F32, 2.0)
        v = b.vec_build([a0, a1])
        b.vec_extract(v, 1)
        text = body(lower(b))
        # Only two consts.
        movs = re.findall(r"mov\.f32 ", text)
        assert len(movs) == 2


class TestSplitMergeB32:
    def test_split_emits_braced_mov(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.B32, 0)
        b.split_b32(x)
        text = body(lower(b))
        assert re.search(r"mov\.b32 \{%h\d+, %h\d+\}, %b\d+;", text)

    def test_merge_emits_braced_mov(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.B32, 0)
        lo, hi = b.split_b32(x)
        b.merge_b32(lo, hi)
        text = body(lower(b))
        assert re.search(r"mov\.b32 %b\d+, \{%h\d+, %h\d+\};", text)

    def test_split_merge_round_trip_reuses_halves(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.B32, 0)
        lo, hi = b.split_b32(x)
        b.merge_b32(lo, hi)
        text = body(lower(b))
        # lo/hi appear in both the split and merge lines — the same
        # register name on both sides.
        split = re.search(r"mov\.b32 \{(%h\d+), (%h\d+)\}, %b\d+;", text)
        merge = re.search(r"mov\.b32 %b\d+, \{(%h\d+), (%h\d+)\};", text)
        assert split and merge
        assert split.group(1) == merge.group(1)
        assert split.group(2) == merge.group(2)
