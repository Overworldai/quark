"""Tests for thread-identity and barrier lowering."""

import re

import pytest

from tests.lower.ptx.conftest import lower


@pytest.mark.parametrize(
    "method,dim,sreg",
    [
        ("thread_idx", "x", "%tid.x"),
        ("thread_idx", "y", "%tid.y"),
        ("thread_idx", "z", "%tid.z"),
        ("block_idx", "x", "%ctaid.x"),
        ("block_idx", "y", "%ctaid.y"),
        ("block_dim", "x", "%ntid.x"),
        ("grid_dim", "x", "%nctaid.x"),
    ],
)
def test_thread_identity_mov(fresh_builder, method, dim, sreg):
    b = fresh_builder
    getattr(b, method)(dim)
    out = lower(b)
    assert f"mov.u32 %r0, {sreg};" in out


def test_lane_id(fresh_builder):
    b = fresh_builder
    b.lane_id()
    out = lower(b)
    assert "mov.u32 %r0, %laneid;" in out


def test_subgroup_id_shifts_tid_x(fresh_builder):
    b = fresh_builder
    b.subgroup_id()
    out = lower(b)
    # subgroup = tid.x >> 5 — emitted as a mov + shr pair.
    assert "mov.u32" in out and "%tid.x" in out
    assert "shr.u32" in out and ", 5;" in out


def test_group_id_shifts_laneid_right_by_two(fresh_builder):
    b = fresh_builder
    b.group_id()
    out = lower(b)
    # groupID = laneid >> 2 — mov.u32 tmp, %laneid; shr.b32 dst, tmp, 2.
    assert re.search(r"mov\.u32 %r\d+, %laneid;", out)
    assert re.search(r"shr\.b32 %r\d+, %r\d+, 2;", out)


def test_thread_id_in_group_masks_laneid_with_three(fresh_builder):
    b = fresh_builder
    b.thread_id_in_group()
    out = lower(b)
    # tidIG = laneid & 3 — mov.u32 tmp, %laneid; and.b32 dst, tmp, 3.
    assert re.search(r"mov\.u32 %r\d+, %laneid;", out)
    assert re.search(r"and\.b32 %r\d+, %r\d+, 3;", out)


class TestBarrier:
    def test_block_barrier(self, fresh_builder):
        b = fresh_builder
        b.barrier("block")
        out = lower(b)
        assert "bar.sync 0;" in out

    def test_subgroup_barrier(self, fresh_builder):
        b = fresh_builder
        b.barrier("subgroup")
        out = lower(b)
        assert "bar.warp.sync 0xffffffff;" in out

    def test_system_barrier(self, fresh_builder):
        b = fresh_builder
        b.barrier("system")
        out = lower(b)
        assert "membar.sys;" in out
