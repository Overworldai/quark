"""Tests for control flow (for/if/yield) MSL lowering."""

from popcorn.ir import DType
from tests.lower.msl.conftest import lower


class TestForLoop:
    def test_simple_for_loop(self, fresh_builder):
        b = fresh_builder
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 10)
        step = b.const(DType.U32, 1)
        with b.for_loop(lo, hi, step, iv_name="k"):
            pass
        out = lower(b)
        assert "for (uint" in out
        assert "{" in out
        assert "}" in out

    def test_for_loop_with_carry(self, fresh_builder):
        b = fresh_builder
        init_val = b.const(DType.F32, 0.0)
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        with b.for_loop(lo, hi, step, iv_name="k", carried=[init_val]) as (k, (acc_in,)):
            one = b.const(DType.F32, 1.0)
            acc_next = b.add(acc_in, one)
            b.yield_(acc_next)
        out = lower(b)
        # Carry variable should be declared before the loop.
        assert "float" in out
        # For loop structure
        assert "for (uint" in out


class TestIfRegion:
    def test_simple_if(self, fresh_builder):
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
        assert "if (" in out
        assert "} else {" in out
        assert "}" in out

    def test_if_with_carry(self, fresh_builder):
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
        assert "if (" in out
        assert "} else {" in out
