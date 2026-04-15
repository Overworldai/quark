"""Tests for cp.async lowering."""

import re

import pytest

from popcorn.ir import BufferType, DType, GlobalTensor
from tests.lower.ptx.conftest import lower


def _setup(b):
    b.param("X", BufferType(DType.BF16))
    g = GlobalTensor(
        dtype=DType.BF16,
        shape=(64, 64),
        stride=(64, 1),
        name="X",
        param=b.function.params[-1],
    )
    A = b.smem_alloc("A", DType.BF16, (64, 64))
    return g, A


class TestAsyncCopy:
    def test_basic_16b_copy(self, fresh_builder):
        g, A = _setup(fresh_builder)
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.async_copy(A, g, dst_idx=(row, col), src_idx=(row, col), count=16)
        out = lower(fresh_builder)
        assert re.search(r"cp\.async\.ca\.shared\.global.L2::256B \[%r\d+\], \[%rd\d+\], 16;", out)

    @pytest.mark.parametrize("count", [4, 8, 16])
    def test_count_sizes(self, fresh_builder, count):
        g, A = _setup(fresh_builder)
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.async_copy(A, g, dst_idx=(row, col), src_idx=(row, col), count=count)
        out = lower(fresh_builder)
        assert f", {count};" in out

    def test_invalid_count_rejected(self, fresh_builder):
        g, A = _setup(fresh_builder)
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.async_copy(A, g, dst_idx=(row, col), src_idx=(row, col), count=12)
        with pytest.raises(NotImplementedError, match="count"):
            lower(fresh_builder)

    def test_commit_and_wait(self, fresh_builder):
        fresh_builder.async_commit()
        fresh_builder.async_wait(0)
        fresh_builder.async_wait(2)
        out = lower(fresh_builder)
        assert "cp.async.commit_group;" in out
        assert "cp.async.wait_group 0;" in out
        assert "cp.async.wait_group 2;" in out

    def test_dynamic_src_index_goes_through_u64_math(self, fresh_builder):
        g, A = _setup(fresh_builder)
        dyn_row = fresh_builder.thread_idx("x")
        col0 = fresh_builder.const(DType.U32, 0)
        dst_row = fresh_builder.const(DType.U32, 0)
        fresh_builder.async_copy(A, g, dst_idx=(dst_row, col0), src_idx=(dyn_row, col0), count=16)
        out = lower(fresh_builder)
        # gmem dynamic offset: cvt.u64.u32 + mul.lo.u64 with stride*elem_bytes = 64*2 = 128.
        assert "cvt.u64.u32" in out
        assert re.search(r"mul\.lo\.u64 %rd\d+, %rd\d+, 128;", out)

    def test_predicated_async_copy(self, fresh_builder):
        g, A = _setup(fresh_builder)
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        hi = fresh_builder.const(DType.U32, 1)
        p = fresh_builder.cmp("lt", row, hi)
        fresh_builder.async_copy(A, g, dst_idx=(row, col), src_idx=(row, col), count=16, pred=p)
        out = lower(fresh_builder)
        assert re.search(r"@%p\d+ cp\.async\.ca\.shared\.global", out)
