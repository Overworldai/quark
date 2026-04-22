"""Tests for ShuffleOp / SubgroupReduceOp / SubgroupBroadcastOp lowering."""

import re

import pytest

from quark.ir import DType
from tests.lower.ptx.conftest import lower


class TestShuffle:
    @pytest.mark.parametrize(
        "kind,ptx_mode,clamp",
        [
            ("bfly", "bfly", 31),
            ("xor", "bfly", 31),  # xor → bfly
            ("up", "up", 0),
            ("down", "down", 31),
            ("idx", "idx", 31),
        ],
    )
    def test_mode_and_clamp_per_kind(self, fresh_builder, kind, ptx_mode, clamp):
        b = fresh_builder
        x = b.const(DType.F32, 1.0)
        b.shuffle(kind, x, 16)
        out = lower(b)
        assert re.search(rf"shfl\.sync\.{ptx_mode}\.b32 %f\d+, %f\d+, 16, {clamp}, -1;", out)

    def test_shuffle_preserves_dtype(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.F32, 1.0)
        y = b.shuffle("bfly", x, 8)
        # Shuffles are b32 but the Value stays f32; the downstream arith
        # should find an f32 register class for the result.
        assert y.dtype is DType.F32


class TestSubgroupReduce:
    def test_sum_emits_5_stage_butterfly_chain(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.F32, 1.0)
        b.subgroup_reduce("sum", x)
        out = lower(b)
        # Five bfly shuffles over xor-offsets 16, 8, 4, 2, 1
        # plus five add.f32 combines.
        for offset in (16, 8, 4, 2, 1):
            assert re.search(rf"shfl\.sync\.bfly\.b32 %f\d+, %f\d+, {offset}, 31, -1;", out), (
                f"missing butterfly shuffle at offset {offset}"
            )
        assert out.count("shfl.sync.bfly.b32") == 5
        assert out.count("add.f32") == 5

    def test_max_uses_max_instruction(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.F32, 1.0)
        b.subgroup_reduce("max", x)
        out = lower(b)
        assert out.count("shfl.sync.bfly.b32") == 5
        assert out.count("max.f32") == 5
        assert "add.f32" not in out

    def test_min_uses_min_instruction(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.F32, 1.0)
        b.subgroup_reduce("min", x)
        out = lower(b)
        assert out.count("min.f32") == 5

    def test_u32_sum(self, fresh_builder):
        """Integer reductions use the dtype-native add."""
        b = fresh_builder
        x = b.const(DType.U32, 1)
        b.subgroup_reduce("sum", x)
        out = lower(b)
        assert out.count("shfl.sync.bfly.b32") == 5
        assert out.count("add.u32") == 5

    def test_bitwise_and_uses_bit_suffix(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.U32, 1)
        b.subgroup_reduce("and", x)
        out = lower(b)
        assert out.count("and.b32") == 5

    def test_unknown_reduce_op_rejected(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.F32, 1.0)
        # The IR op-construction layer catches unknown reduce ops.
        with pytest.raises(ValueError, match="unknown op"):
            b.subgroup_reduce("wut", x)


class TestSubgroupBroadcast:
    def test_broadcast_from_lane_zero(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.F32, 1.0)
        b.subgroup_broadcast(x, lane=0)
        out = lower(b)
        assert re.search(r"shfl\.sync\.idx\.b32 %f\d+, %f\d+, 0, 31, -1;", out)

    def test_broadcast_from_arbitrary_lane(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.F32, 1.0)
        b.subgroup_broadcast(x, lane=7)
        out = lower(b)
        assert re.search(r"shfl\.sync\.idx\.b32 %f\d+, %f\d+, 7, 31, -1;", out)
