"""Tests for the backend-neutral legalization pass.

Exercises the registry/driver itself. Concrete rewrites (async_copy,
fma_bf16x2, vector atomic, subgroup_reduce) get their own per-rewrite
golden tests alongside the rewrites themselves.
"""

from __future__ import annotations

import pytest

from quark.ir import DType
from quark.ir.builder import Builder
from quark.ir.op import ArithOp, ConstOp
from quark.lower.legalize import (
    _LEGALIZATIONS,
    clear_legalizations_for,
    legalizations_for,
    legalize,
    register_legalization,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot / restore the global registry around every test so
    legalizations registered here don't leak into later suites."""
    snapshot = {k: list(v) for k, v in _LEGALIZATIONS.items()}
    yield
    _LEGALIZATIONS.clear()
    _LEGALIZATIONS.update({k: list(v) for k, v in snapshot.items()})


def _tiny_module():
    """Build a module with a couple of const + fma ops for the driver
    to walk. Returns ``(module, builder)``."""
    b = Builder("legalize_test")
    b.begin_function("fn")
    a = b.const(DType.F32, 1.0)
    c = b.const(DType.F32, 2.0)
    b.fma(a, c, a)
    b.end_function()
    return b.module


class TestEmptyRegistry:
    def test_noop_leaves_module_unchanged(self):
        m = _tiny_module()
        before = list(m.functions[0].body.ops)
        legalize(m, caps=None)
        after = list(m.functions[0].body.ops)
        assert before == after

    def test_empty_registry_reports_no_rewrites(self):
        # ConstOp has no rewrite registered. (ArithOp does, because
        # the bf16x2-family rewrite in `quark.lower.legalizations`
        # claims ArithOp at import time — it's a keep-path for every
        # other ArithOp kind.)
        assert legalizations_for(ConstOp) == ()

    def test_legalize_on_module_with_no_functions(self):
        """Pin the degenerate case: a Module with no functions walks
        cleanly and returns the same module. Future edits must not
        assume ``module.functions[0]`` exists."""
        from quark.ir import Module

        m = Module(name="empty")
        result = legalize(m, caps=None)
        assert result is m
        assert m.functions == []

    def test_legalize_on_function_with_no_ops(self):
        """Pin the degenerate case: a function with no ops walks
        cleanly — the per-function fresh-ID counter seeds from
        ``_max_value_id(fn) == -1`` → next_id = 0, but no allocs
        fire since no rewrites match."""
        b = Builder("empty")
        b.begin_function("fn")
        b.end_function()
        result = legalize(b.module, caps=None)
        assert result is b.module
        assert b.module.functions[0].body.ops == []


class TestRegistryMechanics:
    def test_register_and_query(self):
        @register_legalization(ConstOp)
        def _keep(op, caps):
            return None

        assert _keep in legalizations_for(ConstOp)

    def test_clear_for_op_type_empties_slot(self):
        @register_legalization(ConstOp)
        def _keep(op, caps):
            return None

        assert legalizations_for(ConstOp)
        clear_legalizations_for(ConstOp)
        assert legalizations_for(ConstOp) == ()

    def test_multiple_rewrites_first_wins(self):
        """First rewrite returning non-None wins — later rewrites
        registered for the same op type never see the original op."""
        call_log: list[str] = []

        @register_legalization(ConstOp)
        def _first(op, caps):
            call_log.append("first")
            return []

        @register_legalization(ConstOp)
        def _second(op, caps):
            call_log.append("second")
            return []

        m = _tiny_module()
        legalize(m, caps=None)
        # Two ConstOps in the module → first() ran twice, second() never.
        assert call_log == ["first", "first"]

    def test_multiple_rewrites_fallthrough_when_first_returns_none(self):
        """If the first rewrite returns None, the second rewrite gets
        a chance."""
        call_log: list[str] = []

        @register_legalization(ConstOp)
        def _skip(op, caps):
            call_log.append("skip")
            return None

        @register_legalization(ConstOp)
        def _handle(op, caps):
            call_log.append("handle")
            return []

        m = _tiny_module()
        legalize(m, caps=None)
        # 2 ConstOps → each visited by both rewrites in order.
        assert call_log == ["skip", "handle", "skip", "handle"]


class TestDriverRewriteSplicing:
    def test_replacement_list_spliced_at_position(self):
        """A rewrite returning N ops replaces the original at its
        position. Order of surrounding ops preserved."""

        @register_legalization(ArithOp)
        def _split_fma(op, caps):
            # Replace the single FMA with two fresh ConstOps.
            if op.attrs.get("kind") != "fma":
                return None
            from quark.ir.op import ConstOp as _ConstOp
            from quark.ir.types import ValueShape
            from quark.ir.value import Value

            # Re-use the op's result Value slot via a new ConstOp so
            # the module stays self-consistent in terms of SSA id space.
            return [
                _ConstOp(
                    results=(Value(id=-1, shape=ValueShape(DType.F32)),),
                    operands=(),
                    attrs={"value": 42.0, "dtype": DType.F32},
                ),
                _ConstOp(
                    results=(Value(id=-2, shape=ValueShape(DType.F32)),),
                    operands=(),
                    attrs={"value": 43.0, "dtype": DType.F32},
                ),
            ]

        m = _tiny_module()
        legalize(m, caps=None)
        ops = m.functions[0].body.ops
        # Before: ConstOp, ConstOp, ArithOp(fma).
        # After: ConstOp, ConstOp, ConstOp(42), ConstOp(43).
        assert len(ops) == 4
        assert all(isinstance(o, ConstOp) for o in ops)

    def test_rewrite_returning_empty_list_deletes_op(self):
        @register_legalization(ConstOp)
        def _delete(op, caps):
            return []

        m = _tiny_module()
        legalize(m, caps=None)
        ops = m.functions[0].body.ops
        # Both ConstOps deleted; only the ArithOp remains.
        assert len(ops) == 1
        assert isinstance(ops[0], ArithOp)

    def test_rewrite_does_not_re_match_its_own_output(self):
        """Replacement ops sit at the rewritten position; the driver
        advances past them before matching resumes. Protects against
        infinite-loop rewrites that substitute one instance of the same
        op type for another."""
        rewrite_calls = 0

        @register_legalization(ConstOp)
        def _clone_once(op, caps):
            nonlocal rewrite_calls
            rewrite_calls += 1
            if rewrite_calls > 10:
                raise RuntimeError("legalize driver looped on own output")
            return [op]  # return the SAME op — no-op rewrite, one per input

        m = _tiny_module()
        legalize(m, caps=None)
        # 2 ConstOps, each matched once. The ArithOp isn't matched.
        assert rewrite_calls == 2


class TestCapsPassthrough:
    def test_caps_forwarded_to_rewrite_fn(self):
        """The driver forwards ``caps`` verbatim — rewrites read hardware
        facts from it without any internal coercion."""
        seen_caps = []

        @register_legalization(ConstOp)
        def _peek(op, caps):
            seen_caps.append(caps)
            return None

        sentinel = object()
        m = _tiny_module()
        legalize(m, caps=sentinel)
        # Two ConstOps → rewrite fires twice, each time with the
        # same caps object we passed in.
        assert seen_caps == [sentinel, sentinel]
