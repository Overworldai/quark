"""Tests for the Builder API on arithmetic, memory, smem, and intrinsics."""

import pytest

from popcorn.ir import (
    Builder,
    DType,
    MmaShape,
    ValueShape,
    validate_module,
)


def _fresh() -> Builder:
    b = Builder("test")
    b.begin_function("f")
    return b


class TestArithmetic:
    def test_const_and_add(self):
        b = _fresh()
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        s = b.add(a, c)
        b.end_function()
        assert s.dtype is DType.F32
        assert s.shape == ValueShape(DType.F32)

    def test_shape_mismatch_rejected(self):
        b = _fresh()
        x = b.const(DType.F32, 1.0)
        y = b.const(DType.F16, 2.0)
        with pytest.raises(TypeError, match="share shape"):
            b.add(x, y)

    def test_fma(self):
        b = _fresh()
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        d = b.const(DType.F32, 3.0)
        out = b.fma(a, c, d)
        assert out.dtype is DType.F32

    def test_cmp_returns_pred(self):
        b = _fresh()
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        p = b.cmp("lt", a, c)
        assert p.dtype is DType.PRED

    def test_select_requires_pred(self):
        b = _fresh()
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        p = b.cmp("lt", a, c)
        r = b.select(p, a, c)
        assert r.dtype is DType.F32

    def test_convert(self):
        b = _fresh()
        a = b.const(DType.F32, 1.0)
        bf = b.convert(a, DType.BF16)
        assert bf.dtype is DType.BF16

    def test_bitcast_preserves_width(self):
        b = _fresh()
        x = b.const(DType.F32, 1.0)
        y = b.bitcast(x, DType.B32)
        assert y.dtype is DType.B32
        assert y.width == 1


class TestMath:
    @pytest.mark.parametrize(
        "method",
        ["rcp_approx", "rsqrt_approx", "ex2_approx", "sqrt", "exp2", "log2", "tanh"],
    )
    def test_math_builders_exist(self, method: str):
        b = _fresh()
        x = b.const(DType.F32, 1.0)
        out = getattr(b, method)(x)
        assert out.dtype is DType.F32


class TestThreadIdentity:
    def test_block_thread_and_lane(self):
        b = _fresh()
        bid = b.block_idx("x")
        tid = b.thread_idx("y")
        ntid = b.block_dim("z")
        lane = b.lane_id()
        warp = b.subgroup_id()
        for v in (bid, tid, ntid, lane, warp):
            assert v.dtype is DType.U32

    def test_bad_dim_rejected(self):
        b = _fresh()
        with pytest.raises(ValueError):
            b.block_idx("q")

    def test_mma_group_id_and_tidIG(self):
        """group_id / thread_id_in_group are first-class u32 ops that
        every mma fragment loader uses. They name-default to "groupID"
        and "tidIG" to match the PTX ISA glossary."""
        b = _fresh()
        g = b.group_id()
        t = b.thread_id_in_group()
        assert g.dtype is DType.U32 and g.width == 1
        assert t.dtype is DType.U32 and t.width == 1
        assert g.name == "groupID"
        assert t.name == "tidIG"

    def test_group_id_with_custom_name(self):
        b = _fresh()
        v = b.group_id(name="g_upper")
        assert v.name == "g_upper"


class TestSharedMemoryAndLoadStore:
    def test_smem_alloc_and_scalar_load_store(self):
        b = _fresh()
        A = b.smem_alloc("A", DType.F32, (16, 32))
        row = b.const(DType.U32, 0)
        col = b.const(DType.U32, 0)
        v = b.load(A, row, col)
        b.store(A, v, row, col)
        b.end_function()
        validate_module(b.module)

    def test_smem_scalar_load_requires_correct_rank(self):
        b = _fresh()
        A = b.smem_alloc("A", DType.F32, (16, 32))
        row = b.const(DType.U32, 0)
        with pytest.raises(ValueError, match="rank"):
            b.load(A, row)  # missing col

    def test_smem_stride_with_pad(self):
        b = _fresh()
        A = b.smem_alloc("A", DType.F32, (16, 32), pad=4)
        assert A.stride == (36, 1)
        b.end_function()

    def test_vec_load_stores(self):
        b = _fresh()
        A = b.smem_alloc("A", DType.F32, (16, 32))
        row = b.const(DType.U32, 0)
        col = b.const(DType.U32, 0)
        v = b.vec_load(A, row, col, width=4)
        assert v.width == 4
        b.vec_store(A, v, row, col)
        b.end_function()
        validate_module(b.module)


class TestCrossLane:
    def test_shuffle_and_reduce(self):
        b = _fresh()
        x = b.const(DType.F32, 1.0)
        y = b.shuffle("bfly", x, 16)
        z = b.subgroup_reduce("sum", x)
        w = b.subgroup_broadcast(x, lane=0)
        for v in (y, z, w):
            assert v.dtype is DType.F32


class TestVecOpsViaBuilder:
    def test_vec_build_and_extract(self):
        b = _fresh()
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        v = b.vec_build([a, c])
        assert v.width == 2
        e = b.vec_extract(v, 0)
        assert e.width == 1 and e.dtype is DType.F32

    def test_split_merge_b32(self):
        b = _fresh()
        x = b.const(DType.B32, 0)
        lo, hi = b.split_b32(x)
        merged = b.merge_b32(lo, hi)
        assert merged.dtype is DType.B32


class TestMatmul:
    def test_register_and_use_shape(self):
        b = Builder("m")
        b.register_shape(
            MmaShape(
                name="m16n8k16_bf16",
                m=16,
                n=8,
                k=16,
                a_dtype=DType.BF16,
                b_dtype=DType.BF16,
                acc_dtype=DType.F32,
            )
        )
        b.begin_function("f")
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        a = b.load_matrix(A, "m16n8k16_bf16", which="a", row=0, col=0)
        c = b.load_matrix(A, "m16n8k16_bf16", which="b", row=0, col=0)
        cc = b.load_matrix(A, "m16n8k16_bf16", which="c", row=0, col=0)
        d = b.mma("m16n8k16_bf16", a, c, cc)
        b.end_function()
        validate_module(b.module)
        # D carrier dtype mirrors acc_dtype (F32 here) — not B32.
        assert d.dtype is DType.F32

    def test_unregistered_shape_id_rejected(self):
        b = _fresh()
        A = b.smem_alloc("A", DType.BF16, (16, 16))
        with pytest.raises(KeyError, match="not registered"):
            b.load_matrix(A, "unknown_shape", which="a")
