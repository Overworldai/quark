"""Tests for shuffle / subgroup ops MSL lowering."""

from quark.ir import DType
from tests.lower.msl.conftest import lower


class TestShuffle:
    def test_shuffle_xor(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.shuffle("xor", v, 1)
        out = lower(b)
        assert "simd_shuffle_xor(_pc_f32_0, 1u)" in out

    def test_shuffle_idx(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.shuffle("idx", v, 0)
        out = lower(b)
        assert "simd_shuffle(_pc_f32_0, 0u)" in out

    def test_shuffle_up(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.shuffle("up", v, 1)
        out = lower(b)
        assert "simd_shuffle_up(_pc_f32_0, 1u)" in out

    def test_shuffle_down(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.shuffle("down", v, 2)
        out = lower(b)
        assert "simd_shuffle_down(_pc_f32_0, 2u)" in out


class TestSubgroupReduce:
    def test_reduce_sum(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.subgroup_reduce("sum", v)
        out = lower(b)
        assert "simd_sum(_pc_f32_0)" in out

    def test_reduce_max(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.subgroup_reduce("max", v)
        out = lower(b)
        assert "simd_max(_pc_f32_0)" in out


class TestSubgroupBroadcast:
    def test_broadcast_lane_0(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.subgroup_broadcast(v, lane=0)
        out = lower(b)
        assert "simd_broadcast(_pc_f32_0, 0u)" in out
