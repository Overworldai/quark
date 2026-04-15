"""Phase 2: structured control flow tests.

Covers ForLoopOp / IfRegionOp / BarrierOp / YieldOp through the
Builder context-manager API, plus a worked K-loop example that
mirrors the proposal's §8 example.
"""

import pytest

from popcorn.ir import (
    BufferType,
    Builder,
    DType,
    MmaShape,
    print_module,
    validate_module,
)


def _begin() -> Builder:
    b = Builder("cf")
    b.begin_function("f")
    return b


class TestForLoop:
    def test_empty_for_loop_no_carry(self):
        b = _begin()
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        with b.for_loop(lo, hi, step) as (k, _):
            # Use the induction variable to ensure it's in scope.
            one = b.const(DType.U32, 1)
            b.add(k, one)
        b.end_function()
        validate_module(b.module)

    def test_for_loop_with_accumulator(self):
        b = _begin()
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 16)
        step = b.const(DType.U32, 1)
        acc0 = b.const(DType.F32, 0.0)
        with b.for_loop(lo, hi, step, iv_name="k", carried=[acc0]) as (k, (acc_in,)):
            one = b.const(DType.F32, 1.0)
            acc_next = b.add(acc_in, one)
            b.yield_(acc_next)
        (final,) = b.last_results
        b.end_function()
        validate_module(b.module)
        assert final.dtype is DType.F32

    def test_yield_shape_mismatch_detected_at_build(self):
        b = _begin()
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        acc = b.const(DType.F32, 0.0)
        with pytest.raises(TypeError, match=r"for_loop: yield\[0\]"):
            with b.for_loop(lo, hi, step, carried=[acc]) as (k, (acc_in,)):
                wrong = b.const(DType.BF16, 0.0)
                b.yield_(wrong)

    def test_missing_yield_with_carried_rejected(self):
        b = _begin()
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        acc = b.const(DType.F32, 0.0)
        with pytest.raises(RuntimeError, match="must end with builder.yield_"):
            with b.for_loop(lo, hi, step, carried=[acc]):
                pass

    def test_mixed_dtype_bounds_rejected(self):
        b = _begin()
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.S32, 4)
        step = b.const(DType.U32, 1)
        with pytest.raises(TypeError, match="share a dtype"):
            with b.for_loop(lo, hi, step):
                pass

    def test_nested_for_loops(self):
        b = _begin()
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        acc0 = b.const(DType.F32, 0.0)
        with b.for_loop(lo, hi, step, iv_name="i", carried=[acc0]) as (i, (acc_i,)):
            with b.for_loop(lo, hi, step, iv_name="j", carried=[acc_i]) as (
                j,
                (acc_j,),
            ):
                one = b.const(DType.F32, 1.0)
                nxt = b.add(acc_j, one)
                b.yield_(nxt)
            (acc_after_inner,) = b.last_results
            b.yield_(acc_after_inner)
        b.end_function()
        validate_module(b.module)


class TestIfRegion:
    def test_if_branches_yield_matching(self):
        b = _begin()
        a = b.const(DType.F32, 1.0)
        z = b.const(DType.F32, 0.0)
        p = b.cmp("lt", z, a)
        with b.if_(p, carried=[a]) as (then_in, else_in, arms):
            with arms.then_():
                one = b.const(DType.F32, 1.0)
                r = b.add(then_in[0], one)
                b.yield_(r)
            with arms.else_():
                b.yield_(else_in[0])
        b.end_function()
        validate_module(b.module)
        (res,) = b.last_results
        assert res.dtype is DType.F32

    def test_if_zero_carried_autocloses_arms(self):
        b = _begin()
        a = b.const(DType.F32, 1.0)
        z = b.const(DType.F32, 0.0)
        p = b.cmp("lt", z, a)
        with b.if_(p) as (_, _, arms):
            with arms.then_():
                b.const(DType.F32, 5.0)
            with arms.else_():
                b.const(DType.F32, 6.0)
        b.end_function()
        validate_module(b.module)

    def test_missing_else_arm_rejected(self):
        b = _begin()
        a = b.const(DType.F32, 1.0)
        z = b.const(DType.F32, 0.0)
        p = b.cmp("lt", z, a)
        with pytest.raises(RuntimeError, match="else_.*never entered"):
            with b.if_(p, carried=[a]) as (then_in, _, arms):
                with arms.then_():
                    b.yield_(then_in[0])

    def test_yield_shape_mismatch_across_arms(self):
        b = _begin()
        a = b.const(DType.F32, 1.0)
        z = b.const(DType.F32, 0.0)
        p = b.cmp("lt", z, a)
        with pytest.raises(TypeError, match="yield.*shape mismatch"):
            with b.if_(p, carried=[a]) as (then_in, else_in, arms):
                with arms.then_():
                    b.yield_(then_in[0])
                with arms.else_():
                    wrong = b.const(DType.BF16, 0.0)
                    b.yield_(wrong)

    def test_pred_required(self):
        b = _begin()
        a = b.const(DType.F32, 1.0)
        with pytest.raises(TypeError, match="pred must be"):
            with b.if_(a):  # not a PRED
                pass


class TestBarrier:
    def test_barrier_scopes(self):
        b = _begin()
        b.barrier("block")
        b.barrier("subgroup")
        b.barrier("system")
        b.end_function()
        validate_module(b.module)

    def test_unknown_scope_rejected(self):
        b = _begin()
        with pytest.raises(ValueError):
            b.barrier("entire_datacenter")


class TestWorkedKLoop:
    def test_k_pipeline_style_loop_builds_and_validates(self):
        b = Builder("gemm_example")
        b.register_shape(
            MmaShape(
                name="m16n8k16_bf16",
                m=16,
                n=8,
                k=16,
                a_dtype=DType.BF16,
                b_dtype=DType.BF16,
                acc_dtype=DType.F32,
                ptx="mma.sync.aligned.m16n8k16",
            )
        )
        b.begin_function("gemm")
        b.param("X", BufferType(DType.BF16))
        b.param("W", BufferType(DType.BF16))
        b.param("Y", BufferType(DType.F32))
        a_tile = b.smem_alloc("A", DType.BF16, (32, 16))
        b_tile = b.smem_alloc("B", DType.BF16, (16, 16))

        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 8)
        step = b.const(DType.U32, 1)

        with b.for_loop(lo, hi, step, iv_name="k") as (k, _):
            b.barrier("block")
            afrag = b.load_matrix(a_tile, "m16n8k16_bf16", which="a", row=0, col=0)
            bfrag = b.load_matrix(b_tile, "m16n8k16_bf16", which="b", row=0, col=0)
            cfrag = b.load_matrix(a_tile, "m16n8k16_bf16", which="c", row=0, col=0)
            b.mma("m16n8k16_bf16", afrag, bfrag, cfrag)
            b.barrier("block")

        b.end_function()
        validate_module(b.module)

        dump = print_module(b.module)
        assert "for_loop" in dump
        assert "mma %" in dump
        assert "barrier" in dump
