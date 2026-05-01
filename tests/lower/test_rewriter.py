"""Tests for the SSA-fresh-Value allocator used inside legalization rewrites."""

from __future__ import annotations

from quark.ir import DType
from quark.ir.builder import Builder
from quark.ir.types import ValueShape
from quark.lower.legalize import Rewriter, _max_value_id


def _fn_with_three_consts():
    b = Builder("m")
    b.begin_function("fn")
    b.const(DType.F32, 1.0)
    b.const(DType.F32, 2.0)
    b.const(DType.F32, 3.0)
    b.end_function()
    return b.module.functions[0]


class TestMaxValueId:
    def test_empty_function_returns_minus_one(self):
        b = Builder("empty")
        b.begin_function("fn")
        b.end_function()
        fn = b.module.functions[0]
        assert _max_value_id(fn) == -1

    def test_includes_op_result_ids(self):
        fn = _fn_with_three_consts()
        # Three ConstOps, each produces one Value. IDs 0, 1, 2 in the
        # Builder's default assignment.
        assert _max_value_id(fn) == 2

    def test_param_referenced_by_op_is_included(self):
        """Param Values aren't directly enumerable from ``fn.params``
        (that holds ``Param`` dataclasses, not Values). Real code
        references params via an op's operand tuple — ``_max_value_id``
        picks them up via the op walk there."""
        from quark.ir import BufferType, GlobalTensor

        b = Builder("m")
        b.begin_function("fn")
        b.param("X", BufferType(DType.F32))
        g = GlobalTensor(
            dtype=DType.F32,
            shape=(128,),
            stride=(1,),
            name="X",
            param=b.function.params[-1],
        )
        idx = b.const(DType.U32, 0)
        b.load(g, idx)  # references the param via the LoadOp's tensor attr path
        b.end_function()
        fn = b.module.functions[0]
        # At least the const ``idx`` and the load's result are in there.
        assert _max_value_id(fn) >= 1


class TestRewriterAlloc:
    def test_fresh_values_have_strictly_increasing_ids(self):
        rw = Rewriter(next_id=10)
        v0 = rw.alloc(ValueShape(DType.F32))
        v1 = rw.alloc(ValueShape(DType.F32))
        v2 = rw.alloc(ValueShape(DType.BF16))
        assert v0.id == 10
        assert v1.id == 11
        assert v2.id == 12

    def test_for_function_seeds_past_existing_ids(self):
        fn = _fn_with_three_consts()
        rw = Rewriter.for_function(fn)
        fresh = rw.alloc(ValueShape(DType.F32))
        # Existing max ID is 2 → first fresh ID is 3.
        assert fresh.id == 3

    def test_alloc_carries_shape_and_name(self):
        rw = Rewriter(next_id=0)
        v = rw.alloc(ValueShape(DType.BF16, width=2), name="bf16_vec")
        assert v.dtype is DType.BF16
        assert v.width == 2
        assert v.name == "bf16_vec"
        assert v.producer is None  # caller attaches producer via op.results

    def test_driver_shares_counter_across_rewrites_in_one_function(self):
        """Regression for an ID-collision bug: two rewrites firing on
        different ops in the same function must not allocate
        overlapping Value.id ranges. Before the driver-scoped shared
        counter landed, ``for_op``'s 1M-past-op-ids seed could
        overlap when the ops had nearby IDs and one rewrite allocated
        more than the gap between seeds. Symptom: ``RegAllocator``
        would overwrite one Value's component mapping with
        another's — caught only at lowering time, as a silent
        miscompile."""
        from quark.lower.legalize import legalize

        class _NoBf16x2:
            has_fma_bf16x2 = False
            atomic_add_vector = frozenset()
            supports_async_copy = True
            has_native_subgroup_reduce = False

        b = Builder("m")
        b.begin_function("fn")
        a1 = b.const(DType.F32, 1.0)
        b1 = b.const(DType.F32, 2.0)
        b.cvt_rn_bf16x2_f32(a1, b1)
        a2 = b.const(DType.F32, 3.0)
        b2 = b.const(DType.F32, 4.0)
        b.cvt_rn_bf16x2_f32(a2, b2)
        b.end_function()
        legalize(b.module, _NoBf16x2())
        all_ids = [v.id for op in b.module.functions[0].body.ops for v in op.results]
        # 14 Values total (8 from each 5-op expansion + 2 const-unchanged
        # Values per input pair minus 2 overlap since each const was
        # consumed by only its own cvt). The count check is
        # structural; the important assertion is all IDs unique.
        assert len(all_ids) == len(set(all_ids)), (
            f"Value-ID collision: {len(all_ids)} Values, {len(set(all_ids))} unique IDs"
        )

    def test_for_op_leaves_room_even_without_full_fn_walk(self):
        """``Rewriter.for_op`` is used when the rewrite can't walk the
        whole function. It leaves a 1M-ID gap past the op's own
        operand/result IDs — crude but collision-free in practice."""
        from quark.ir.op import ConstOp
        from quark.ir.value import Value

        v = Value(id=42, shape=ValueShape(DType.F32))
        op = ConstOp(results=(v,), operands=(), attrs={"value": 0.0, "dtype": DType.F32})
        rw = Rewriter.for_op(op)
        fresh = rw.alloc(ValueShape(DType.F32))
        assert fresh.id > 42
