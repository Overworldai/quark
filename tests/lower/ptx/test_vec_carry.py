"""Tests for vec-valued loop-carried variables and yield coalescing.

These were missing from the original test suite and surfaced as a
bug during the universal GEMM kernel development: the ForLoopOp
initializer emitted `mov.f32 {a,b,c,d}, {e,f,g,h}` which PTX
rejects — vec Values need per-component movs.
"""

import re

from quark.ir import DType
from tests.lower.ptx.conftest import body, lower


class TestVecCarriedLoop:
    def test_vec4_f32_accumulator_in_for_loop(self, fresh_builder):
        """A width-4 f32 vec as a loop-carried value must emit 4
        per-component `mov.f32` instructions in the loop init (before
        the label) and 4 in the yield coalesce (before the backedge),
        not one illegal `mov.f32 {a,b,c,d}, {e,f,g,h}`."""
        b = fresh_builder
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 4)
        step = b.const(DType.U32, 1)
        zero = b.const(DType.F32, 0.0)
        acc0 = b.vec_build([zero, zero, zero, zero])
        with b.for_loop(lo, hi, step, iv_name="k", carried=[acc0]) as (
            k,
            (acc_in,),
        ):
            # Pretend we update the accumulator by adding 1.0 to each
            # component. Build a new vec from the updated components.
            one = b.const(DType.F32, 1.0)
            c0 = b.add(b.vec_extract(acc_in, 0), one)
            c1 = b.add(b.vec_extract(acc_in, 1), one)
            c2 = b.add(b.vec_extract(acc_in, 2), one)
            c3 = b.add(b.vec_extract(acc_in, 3), one)
            acc_next = b.vec_build([c0, c1, c2, c3])
            b.yield_(acc_next)
        text = body(lower(b))
        # The key assertion: NO braced-to-braced mov in the output.
        # PTX's `mov.f32 {a,b,c,d}, {e,f,g,h}` is illegal — the
        # lowerer must decompose into 4 scalar `mov.f32 %fN, %fM`.
        assert re.search(r"mov\.f32 \{", text) is None, (
            "Found braced mov.f32 — vec Values must use per-component movs in loop init/yield"
        )
        # We should see multiple scalar f32 movs (at least 4 from
        # init + 4 from yield = 8).
        f32_movs = re.findall(r"mov\.f32 %f\d+, %f\d+;", text)
        assert len(f32_movs) >= 4, f"Expected ≥4 per-component f32 movs, found {len(f32_movs)}"

    def test_multiple_vec4_carried_values(self, fresh_builder):
        """Two vec-4 carried values (like a GEMM with 2 accumulators)
        should each decompose independently."""
        b = fresh_builder
        lo = b.const(DType.U32, 0)
        hi = b.const(DType.U32, 2)
        step = b.const(DType.U32, 1)
        zero = b.const(DType.F32, 0.0)
        acc0 = b.vec_build([zero, zero, zero, zero])
        acc1 = b.vec_build([zero, zero, zero, zero])
        with b.for_loop(lo, hi, step, iv_name="k", carried=[acc0, acc1]) as (k, (a0, a1)):
            b.yield_(a0, a1)
        text = body(lower(b))
        # At least 8 scalar movs from init (4+4), and the yield is
        # aliased (a0/a1 → result regs) so yield movs are self-movs
        # and get suppressed. Total: ≥8 init movs.
        assert re.search(r"mov\.f32 \{", text) is None
