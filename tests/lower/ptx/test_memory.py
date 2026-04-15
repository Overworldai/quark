"""Tests for PTX lowering of memory ops (smem alloc, scalar load/store)."""

import re

from popcorn.ir import BufferType, DType, GlobalTensor
from tests.lower.ptx.conftest import lower, lower_full


class TestSmemAlloc:
    """SmemAllocOp compiles the declared tiles into a single dynamic
    `.extern .shared` pool. The byte total is exposed on the returned
    `LoweredKernel` so the runtime knows how much smem to request at
    launch; the PTX pool declaration itself is unsized."""

    def test_pool_declared_as_extern_shared_with_no_size(self, fresh_builder):
        b = fresh_builder
        b.smem_alloc("A", DType.F32, (8, 16))  # 8*16*4 = 512 bytes
        lowered = lower_full(b)
        assert ".extern .shared .align 16 .b8 _smem_pool[];" in lowered.ptx
        assert lowered.smem_bytes == 512

    def test_pool_size_sums_across_allocs(self, fresh_builder):
        b = fresh_builder
        b.smem_alloc("A", DType.F32, (4, 4))  # 64 bytes, aligned → 64
        b.smem_alloc("B", DType.F32, (4, 4))  # 64 bytes, aligned → 64
        lowered = lower_full(b)
        assert lowered.smem_bytes == 128

    def test_pool_size_respects_alignment(self, fresh_builder):
        b = fresh_builder
        b.smem_alloc("A", DType.F32, (1, 5))  # 20 bytes, rounds to 20
        b.smem_alloc("B", DType.F32, (1, 4))  # 16 bytes starting at 32 (align 16)
        lowered = lower_full(b)
        # Total: 32 + 16 = 48
        assert lowered.smem_bytes == 48

    def test_base_reg_set_to_pool_symbol(self, fresh_builder):
        b = fresh_builder
        b.smem_alloc("A", DType.F32, (4, 4))
        out = lower(b)
        assert "mov.u32 %r0, _smem_pool;" in out


class TestSmemLoadStore:
    def test_scalar_smem_load_static_offset(self, fresh_builder):
        b = fresh_builder
        A = b.smem_alloc("A", DType.F32, (8, 16))
        row = b.const(DType.U32, 2)
        col = b.const(DType.U32, 3)
        b.load(A, row, col)
        out = lower(b)
        # Static offset = 2*16*4 + 3*4 = 140
        assert re.search(r"ld\.shared\.f32 %f\d+, \[%r\d+ \+ 140\];", out)

    def test_scalar_smem_store_static_offset(self, fresh_builder):
        b = fresh_builder
        A = b.smem_alloc("A", DType.F32, (4, 4))
        row = b.const(DType.U32, 1)
        col = b.const(DType.U32, 2)
        v = b.load(A, row, col)
        b.store(A, v, row, col)
        out = lower(b)
        # Static offset = 1*4*4 + 2*4 = 24
        assert re.search(r"st\.shared\.f32 \[%r\d+ \+ 24\]", out)

    def test_dynamic_index_uses_mul_add(self, fresh_builder):
        b = fresh_builder
        A = b.smem_alloc("A", DType.F32, (8, 16))
        dyn_row = b.thread_idx("x")
        col = b.const(DType.U32, 0)
        b.load(A, dyn_row, col)
        out = lower(b)
        # Expect mul.lo.u32 with factor 64 (stride 16 * 4 bytes)
        assert re.search(r"mul\.lo\.u32 %r\d+, %r\d+, 64;", out)


class TestGmemLoadStore:
    def _make_g(self, b, shape=(64, 128)):
        b.param("X", BufferType(DType.F32))
        return GlobalTensor(
            dtype=DType.F32,
            shape=shape,
            stride=(shape[1], 1),
            name="X",
            param=b.function.params[-1],
        )

    def test_param_loaded_with_ld_param(self, fresh_builder):
        g = self._make_g(fresh_builder)
        row = fresh_builder.const(DType.U32, 0)
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.load(g, row, col)
        out = lower(fresh_builder)
        assert "ld.param.u64 %rd0, [X];" in out

    def test_static_gmem_load_offset(self, fresh_builder):
        g = self._make_g(fresh_builder)
        row = fresh_builder.const(DType.U32, 2)
        col = fresh_builder.const(DType.U32, 1)
        fresh_builder.load(g, row, col)
        out = lower(fresh_builder)
        # row=2,col=1: 2*128*4 + 1*4 = 1028
        assert re.search(r"ld\.global\.f32 %f\d+, \[%rd0 \+ 1028\];", out)

    def test_dynamic_gmem_index_widens_to_u64(self, fresh_builder):
        g = self._make_g(fresh_builder)
        row = fresh_builder.thread_idx("x")
        col = fresh_builder.const(DType.U32, 0)
        fresh_builder.load(g, row, col)
        out = lower(fresh_builder)
        # Must see cvt.u64.u32 before pointer arith.
        assert "cvt.u64.u32" in out
        # And a mul.lo.u64 with factor 512 (128 * 4 bytes).
        assert re.search(r"mul\.lo\.u64 %rd\d+, %rd\d+, 512;", out)
