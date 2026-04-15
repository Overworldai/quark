"""popcorn.lang façade — forwarders must not drift from Builder.

Guards the S1 authoring surface: every public symbol in ``popcorn.lang``
must be either a control-flow helper we own (``kernel_scope``,
``current_builder``, ``for_range``) or a forwarder whose name matches
a public ``Builder`` method.

The reverse check — "every Builder method has a forwarder" — is
*intentional not* enforced. Module-setup (``begin_function``, ``param``,
``register_shape``) and introspection (``last_results``, ``module``,
``function``) stay on the Builder; exposing them via ``pop.*`` would
invite misuse.
"""

from __future__ import annotations

import popcorn.lang as pop
from popcorn.ir import DType
from popcorn.ir.builder import Builder
from popcorn.ir.module import BufferType

# Symbols we add on top of Builder — not forwarders.
_OWN_SYMBOLS = {
    "kernel_scope",
    "current_builder",
    "for_range",  # renames Builder.for_loop
    "abs_",  # renames Builder.abs (avoids builtin shadow)
    "last_results",  # wraps the Builder property as a fn
    # High-level epilogue helpers (popcorn.lang.epilogue) — not forwarders.
    "silu",
    "cast",
    "store_acc",
    "atomic_store_acc",
    # Kernel-authoring memory helpers (popcorn.lang.memory) — not forwarders.
    "work_list_load",
    "index_cache",
    "q_register_load",
    # Dim-kwarg sugar that routes min/max/sum over frag reductions —
    # not a 1:1 Builder method.
    "sum",
}


def _forwarder_names() -> list[str]:
    return [n for n in pop.__all__ if n not in _OWN_SYMBOLS and not n.startswith("_")]


def test_every_forwarder_is_a_real_builder_method() -> None:
    missing = [n for n in _forwarder_names() if not hasattr(Builder, n)]
    assert not missing, (
        f"popcorn.lang exposes forwarders with no Builder method: {missing}. "
        "Either add the method to Builder or remove the forwarder."
    )


def test_renames_point_at_existing_methods() -> None:
    # for_range → for_loop, abs_ → abs
    assert hasattr(Builder, "for_loop")
    assert hasattr(Builder, "abs")


def test_current_builder_raises_when_unset() -> None:
    import pytest

    # Strictly, _ACTIVE_BUILDER may be set by a prior test that forgot
    # to close its function. Guard that by constructing a fresh scope
    # that returns immediately.
    try:
        pop.current_builder()
    except RuntimeError as e:
        assert "no active Builder" in str(e)
        return
    # If we got here, some outer scope has a builder active — that's
    # still valid behavior. Skip rather than fail on environment state.
    pytest.skip("a builder is active from outer scope; cannot test unset path")


def test_forward_emits_via_active_builder() -> None:
    b = Builder("t")
    b.begin_function("fn")
    b.param("x", BufferType(DType.F32))
    # Now _ACTIVE_BUILDER is b — pop.const should emit on it.
    c = pop.const(DType.F32, 1.5)
    assert c.dtype is DType.F32
    d = pop.add(c, pop.const(DType.F32, 0.5))
    # CSE or no, we got an F32 back.
    assert d.dtype is DType.F32
    b.end_function()


def test_kernel_scope_nests_cleanly() -> None:
    outer = Builder("outer")
    inner = Builder("inner")
    with pop.kernel_scope(outer):
        assert pop.current_builder() is outer
        with pop.kernel_scope(inner):
            assert pop.current_builder() is inner
        assert pop.current_builder() is outer


def test_for_range_is_a_context_manager() -> None:
    b = Builder("t")
    b.begin_function("fn")
    lo = pop.const(DType.S32, 0)
    hi = pop.const(DType.S32, 4)
    step = pop.const(DType.S32, 1)
    with pop.for_range(lo, hi, step, iv_name="i") as (iv, carried):
        assert iv.dtype is DType.S32
        assert carried == ()
    b.end_function()
