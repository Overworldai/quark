"""Tests for thread/block/lane indexing and barrier MSL lowering."""

from tests.lower.msl.conftest import lower


class TestThreadIdx:
    def test_thread_idx_x(self, fresh_builder):
        b = fresh_builder
        b.thread_idx("x")
        out = lower(b)
        assert "thread_position_in_threadgroup.x" in out

    def test_thread_idx_y(self, fresh_builder):
        b = fresh_builder
        b.thread_idx("y")
        out = lower(b)
        assert "thread_position_in_threadgroup.y" in out


class TestBlockIdx:
    def test_block_idx_x(self, fresh_builder):
        b = fresh_builder
        b.block_idx("x")
        out = lower(b)
        assert "threadgroup_position_in_grid.x" in out


class TestBlockDim:
    def test_block_dim_x(self, fresh_builder):
        b = fresh_builder
        b.block_dim("x")
        out = lower(b)
        assert "threads_per_threadgroup.x" in out


class TestGridDim:
    def test_grid_dim_x(self, fresh_builder):
        b = fresh_builder
        b.grid_dim("x")
        out = lower(b)
        assert "threadgroups_per_grid.x" in out


class TestLaneId:
    def test_lane_id(self, fresh_builder):
        b = fresh_builder
        b.lane_id()
        out = lower(b)
        assert "thread_index_in_simdgroup" in out


class TestSubgroupId:
    def test_subgroup_id(self, fresh_builder):
        b = fresh_builder
        b.subgroup_id()
        out = lower(b)
        assert "simdgroup_index_in_threadgroup" in out


class TestGroupId:
    def test_group_id(self, fresh_builder):
        b = fresh_builder
        b.group_id()
        out = lower(b)
        assert "thread_index_in_simdgroup >> 2" in out


class TestThreadIdInGroup:
    def test_thread_id_in_group(self, fresh_builder):
        b = fresh_builder
        b.thread_id_in_group()
        out = lower(b)
        assert "thread_index_in_simdgroup & 3" in out


class TestBarrier:
    def test_block_barrier(self, fresh_builder):
        b = fresh_builder
        b.barrier("block")
        out = lower(b)
        assert "threadgroup_barrier(metal::mem_flags::mem_threadgroup)" in out

    def test_subgroup_barrier(self, fresh_builder):
        b = fresh_builder
        b.barrier("subgroup")
        out = lower(b)
        assert "simdgroup_barrier(metal::mem_flags::mem_none)" in out

    def test_system_barrier(self, fresh_builder):
        b = fresh_builder
        b.barrier("system")
        out = lower(b)
        assert "threadgroup_barrier(metal::mem_flags::mem_device)" in out
