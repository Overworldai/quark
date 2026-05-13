"""Tests for PTX lowering of structured control flow.

Covers ForLoopOp (empty + with carry), IfRegionOp (with and without
carried values), nested loops, and yield coalescing.
"""

import re

from quark.ir import DType
from tests.lower.ptx.conftest import body, lower


class TestForLoop:
    def test_empty_loop_has_label_and_backedge(self, fresh_builder):
        b = fresh_builder
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        with b.for_loop(lo, hi, step, iv_name="k"):
            pass
        out = lower(b)
        text = body(out)
        # iv = lo
        assert re.search(r"mov\.u32 %r\d+, %r\d+;", text)
        # label line
        assert re.search(r"\$L_\d+:", text)
        # backedge: add + setp.lt + @p bra
        assert re.search(r"add\.u32 ", text)
        assert re.search(r"setp\.lt\.u32", text)
        assert re.search(r"@%p\d+ bra \$L_\d+;", text)

    def test_loop_with_carry_coalesces_yield(self, fresh_builder):
        b = fresh_builder
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 16)
        step = b.const(DType.U32, 1)
        acc0 = b.const(DType.F32, 0.0)
        with b.for_loop(lo, hi, step, iv_name="k", carried=[acc0]) as (k, (acc_in,)):
            one = b.const(DType.F32, 1.0)
            acc_next = b.add(acc_in, one)
            b.yield_(acc_next)
        out = lower(b)
        text = body(out)
        # Pre-loop seed: mov.f32 acc_result, acc0
        # and after the add, a mov.f32 acc_result, acc_next (the yield coalesce).
        # Count f32 movs — we expect at least two (init + coalesce).
        f32_movs = re.findall(r"mov\.f32 %f\d+, %f\d+;", text)
        assert len(f32_movs) >= 2

    def test_unroll_python_inlines_n_iters_no_loop_label(self, fresh_builder):
        """``ForLoopOp(unroll=True)`` Python-unrolls the body at lower
        time — N inlined iterations with iv bound to ``mov.u32 iv,
        <i>;`` per iter. Produces no loop label and no backedge — the
        result is the same instruction stream as today's Python-side
        ``for i in range(N)`` emit. Lets the kernel cohort migrate
        copy loops to IR while keeping CUDA perf unchanged."""
        b = fresh_builder
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        with b.for_loop(lo, hi, step, iv_name="k", unroll=True) as (k, _):
            pass
        out = lower(b)
        text = body(out)
        # No loop label, no backedge — fully inlined.
        assert not re.search(r"\$L_\d+:", text), (
            f"unrolled loop should have no loop label, got:\n{text}"
        )
        assert not re.search(r"@%p\d+ bra \$L_\d+;", text), (
            "unrolled loop should have no backedge"
        )
        # 4 iv-init movs (one per iter, with literal const value).
        iv_movs = re.findall(r"mov\.u32 %r\d+, \d+;", text)
        assert len(iv_movs) >= 4, (
            f"expected ≥4 iv-init movs for 4 iters, got {len(iv_movs)}:\n{text}"
        )

    def test_unroll_rejects_non_const_bounds(self, fresh_builder):
        """``unroll=True`` requires statically-resolvable lo/hi/step.
        Today only ConstOp-traced bounds are supported (BlockDimOp-
        folding via FunctionAttrs is a planned follow-up). A runtime
        bound (e.g. derived from ``thread_idx``) raises
        NotImplementedError with a clear message."""
        import pytest as _pytest
        b = fresh_builder
        tid = b.thread_idx("x")
        # Use tid as the upper bound — genuinely runtime, can't fold.
        lo = b.const(DType.U32, 0)
        step = b.const(DType.U32, 1)
        with b.for_loop(lo, tid, step, iv_name="k", unroll=True) as (k, _):
            pass
        with _pytest.raises(NotImplementedError, match="statically-resolvable"):
            lower(b)

    def test_nested_loops_have_two_labels(self, fresh_builder):
        b = fresh_builder
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        with b.for_loop(lo, hi, step, iv_name="i"):
            with b.for_loop(lo, hi, step, iv_name="j"):
                pass
        out = lower(b)
        text = body(out)
        labels = re.findall(r"\$L_\d+:", text)
        assert len(labels) == 2


class TestIfRegion:
    def test_if_emits_pred_branch_and_labels(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        z = b.const(DType.F32, 0.0)
        p = b.cmp("lt", z, a)
        with b.if_(p) as (_, _, arms):
            with arms.then_():
                b.const(DType.F32, 2.0)
            with arms.else_():
                b.const(DType.F32, 3.0)
        out = lower(b)
        text = body(out)
        assert re.search(r"@!%p\d+ bra \$ELSE_\d+;", text)
        assert re.search(r"\$ELSE_\d+:", text)
        assert re.search(r"\$ENDIF_\d+:", text)
        # The then body must have a forward branch to the endif.
        assert re.search(r"bra \$ENDIF_\d+;", text)

    def test_if_with_carry_coalesces(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        z = b.const(DType.F32, 0.0)
        p = b.cmp("lt", z, a)
        with b.if_(p, carried=[a]) as (then_in, else_in, arms):
            with arms.then_():
                one = b.const(DType.F32, 2.0)
                r = b.add(then_in[0], one)
                b.yield_(r)
            with arms.else_():
                b.yield_(else_in[0])
        out = lower(b)
        text = body(out)
        # then branch emits a coalesce mov from `r` into the result slot.
        # The else yield of `else_in[0]` is aliased to the result slot
        # so no mov is needed there.
        assert "add.f32" in text


class TestTopLevelYield:
    def test_yield_at_function_body_is_noop(self, fresh_builder):
        b = fresh_builder
        b.const(DType.F32, 1.0)
        b.yield_()
        out = lower(b)
        text = body(out)
        # No spurious mov from yield at top level — just the const and ret.
        assert "mov.f32 %f0, 0f3f800000" in text
        assert text.strip().endswith("ret;")
