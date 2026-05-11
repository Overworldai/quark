"""Backend-neutral IR legalization pass.

Sits between IR construction and lowering. For each backend, rewrites
ops the target can't emit natively into equivalent IR that it can.

    Module (target-agnostic)
       ↓ legalize(module, caps)
    Module' (same IR — ops the target can't do natively expanded)
       ↓ lower
    Target code

The goal is to stop duplicating fallback logic across backends. Today
MSL handles "no ``async_copy`` / no ``fma.rn.bf16x2``" with inline
fallbacks in its visitors; once the legalization pass owns that job,
adding SPIR-V v1 inherits the same fallbacks for free — its lowerer
just never sees ``AsyncCopyOp`` or ``ArithOp(kind="fma_bf16x2")``.

Scope:

  * Framework + registry — this module.
  * Concrete rewrites — registered in follow-up commits. The first
    cohort: ``AsyncCopyOp`` / ``fma_bf16x2`` / ``cvt_rn_bf16x2_f32`` /
    vector-atomic ``AtomicRmwOp`` / ``SubgroupReduceOp``.

Non-scope: nested-region walks. Every ported rewrite so far operates
on straight-line function bodies — control-flow regions (``ForLoopOp``,
``IfOp``) come in when the first rewrite needs them, not as a
speculative plumbing layer here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quark.ir import Module
    from quark.ir.op import Op

# Per-function fresh-ID counter. The driver sets this (via
# ``_active_counter``) before walking a function's ops so every
# rewrite that fires on the same function shares one monotonic
# allocator. Without this, two rewrites firing on different ops can
# allocate overlapping ID ranges (each uses ``for_op``'s
# 1M-past-op-ids seed, and those seeds can overlap).
_ACTIVE_FRESH_ID: ContextVar[list[int] | None] = ContextVar(
    "_quark_legalize_active_fresh_id", default=None
)


# -----------------------------------------------------------------------------
# Rewrite registry
# -----------------------------------------------------------------------------

# Rewrite signature: ``(op, caps) -> RewriteResult``.
#
# ``None`` — no rewrite; leave the op as-is. ``[]`` — delete the op
# entirely. ``[new_op, ...]`` — splice the returned ops into the
# region in place of the original. The driver inserts them in order
# at the original op's position.
#
# Rewrites are pure IR-to-IR — they MUST NOT touch target-specific
# state. Everything they need comes via ``caps``.
RewriteResult = list["Op"] | None
RewriteFn = Callable[["Op", Any], RewriteResult]

# ``op_type -> [rewrite fn, ...]``. Multiple rewrites per op type run
# in registration order; the first one to return non-``None`` wins and
# later rewrites do not see the original op.
_LEGALIZATIONS: dict[type, list[RewriteFn]] = {}


def register_legalization(op_type: type) -> Callable[[RewriteFn], RewriteFn]:
    """Register ``fn`` as a legalization pattern for ``op_type``.

    Usage:

        @register_legalization(AsyncCopyOp)
        def legalize_async_copy(op, caps):
            if caps.supports_async_copy:
                return None  # keep the op
            return expand_to_vec_loop(op)
    """

    def _deco(fn: RewriteFn) -> RewriteFn:
        _LEGALIZATIONS.setdefault(op_type, []).append(fn)
        return fn

    return _deco


def clear_legalizations_for(op_type: type) -> None:
    """Drop all rewrites registered for ``op_type``. Exists for tests —
    production code should never need to call this."""
    _LEGALIZATIONS.pop(op_type, None)


def legalizations_for(op_type: type) -> tuple[RewriteFn, ...]:
    """Return the tuple of rewrites registered for ``op_type`` (possibly
    empty). Exposed for tests and dashboards."""
    return tuple(_LEGALIZATIONS.get(op_type, ()))


# -----------------------------------------------------------------------------
# Pass driver
# -----------------------------------------------------------------------------


def legalize(module: Module, caps: Any) -> Module:
    """Run every registered rewrite over ``module`` under ``caps``.

    The module is rewritten in place and returned — callers that want
    the original untouched must ``copy.deepcopy`` it first. The driver
    is idempotent once ``caps`` is pinned: running ``legalize`` twice
    with the same caps produces the same module as running it once
    (rewrites return ``None`` the second time because the IR no longer
    contains any op patterns they match).

    Walks nested regions (``ForLoopOp`` bodies, ``IfOp`` branches,
    ``WhileOp`` cond/body) via each op's generic ``regions`` tuple.
    Rewrites that emit ops inside a child region don't need to opt in —
    the driver recurses uniformly.

    Zero overhead when no legalizations are registered — the empty
    ``_LEGALIZATIONS`` dict short-circuits the walk.
    """
    if not _LEGALIZATIONS:
        return module
    for fn in module.functions:
        # Seed a shared fresh-ID counter from the function's current
        # max Value.id. Every rewrite that fires on this function —
        # top-level or nested — pulls fresh IDs from this counter
        # via ``Rewriter.for_op``, so different rewrites can't
        # allocate overlapping ranges. Uses a 1-element list as a
        # mutable cell so the context-var binding stays stable.
        counter = [_max_value_id(fn) + 1]
        tok = _ACTIVE_FRESH_ID.set(counter)
        try:
            _rewrite_region(fn.body.ops, caps)
        finally:
            _ACTIVE_FRESH_ID.reset(tok)
    return module


def _rewrite_region(ops: list[Op], caps: Any) -> None:
    """Rewrite ``ops`` in place. Each op is matched against its type
    (exact match, no MRO-walk) and the first matching rewrite that
    returns non-``None`` fires. The rewrite's replacement list is
    spliced into ``ops`` at the original position; the driver then
    resumes after the last replacement op so a rewrite can't match
    its own replacements.

    After matching the current op, recurses into any child regions
    it carries (``op.regions``). Replacement ops produced by a
    rewrite are walked the same way — a rewrite can legitimately
    emit a new ``ForLoopOp`` whose body still contains ops the
    pass needs to expand."""
    i = 0
    while i < len(ops):
        op = ops[i]
        rewrites = _LEGALIZATIONS.get(type(op))
        replacement: RewriteResult = None
        if rewrites:
            for fn in rewrites:
                replacement = fn(op, caps)
                if replacement is not None:
                    break
        if replacement is None:
            # Op stays; descend into its child regions before moving on.
            for region in op.regions:
                _rewrite_region(region.ops, caps)
            i += 1
            continue
        # Splice: replace ops[i:i+1] with replacement, walk each
        # replacement op's child regions, advance past them all.
        ops[i : i + 1] = replacement
        for new_op in replacement:
            for region in new_op.regions:
                _rewrite_region(region.ops, caps)
        i += len(replacement)


# -----------------------------------------------------------------------------
# Public re-exports
# -----------------------------------------------------------------------------


def _registered_op_types() -> Iterable[type]:
    """Iterable of op types that currently have at least one rewrite
    registered. Tests use this to assert the expected set is present."""
    return tuple(_LEGALIZATIONS.keys())


# -----------------------------------------------------------------------------
# Rewriter — SSA-safe fresh-Value allocation inside a legalization rewrite.
# -----------------------------------------------------------------------------


class Rewriter:
    """Helper for rewrites that need to emit new ops with fresh Values.

    A legalization rewrite returns ``[new_op, ...]``. Each new op's
    ``results`` tuple must contain SSA Values with IDs that don't
    collide with any Value already in the surrounding function, and
    the *final* op's result typically reuses the original op's
    result Value so downstream consumers' operand tuples keep
    working without a separate use-replacement pass.

    ``Rewriter`` wraps both operations:

      * :meth:`alloc` — return a fresh ``Value`` with a unique ID
        scoped to the function. IDs seed from ``max(existing) + 1``
        so the rewriter can't collide with params or existing ops.
      * :meth:`keep` — register a pre-existing Value (typically the
        original op's result) so subsequent ``alloc`` calls skip
        its ID.

    The rewriter doesn't own op construction — rewrites still
    instantiate :class:`ArithOp` / :class:`ConvertOp` / etc. directly
    so they can populate backend-specific ``attrs``. The rewriter's
    job is specifically Value-ID bookkeeping.

    Typical use:

        def my_rewrite(op: Op, caps: Any) -> list[Op] | None:
            if not caps.some_flag:
                rw = Rewriter.for_op(op)
                mid = rw.alloc(ValueShape(DType.F32))
                # ... construct ops that produce `mid` and then the
                #     original op's result as the final result ...
                return [mid_op, final_op]
            return None
    """

    def __init__(self, next_id: int) -> None:
        self._next_id = next_id
        # When non-None, ``alloc`` bumps this shared counter cell
        # instead of the instance-local ``_next_id``. Set by
        # ``Rewriter.for_op`` when a driver-scoped counter is active.
        self._shared: list[int] | None = None

    @classmethod
    def for_function(cls, fn: Any) -> Rewriter:
        """Seed the rewriter from the max Value ID currently present in
        ``fn`` (params + every op's results). Safe to call repeatedly;
        the rewriter keeps its own monotonic counter after that."""
        return cls(_max_value_id(fn) + 1)

    @classmethod
    def for_op(cls, op: Op) -> Rewriter:
        """Allocate against the driver's per-function counter when one
        is active; otherwise seed from the op's local IDs + 1M gap.

        Inside ``legalize(module, caps)`` the driver sets a shared
        fresh-ID cell before walking each function. Every rewrite that
        fires on that function — for any op — picks up the same cell
        via ``_ACTIVE_FRESH_ID``, so two rewrites can't allocate
        overlapping ranges.

        Outside a ``legalize(...)`` call (unit tests that exercise a
        rewrite directly), falls back to the op-local 1M-past-max
        seed. Imperfect standalone, but the shared-counter path is
        what runs in production."""
        counter = _ACTIVE_FRESH_ID.get()
        if counter is not None:
            rw = cls(counter[0])
            rw._shared = counter
            return rw

        max_id = 0
        for v in op.results:
            max_id = max(max_id, v.id)
        for v in op.operands:
            max_id = max(max_id, v.id)
        return cls(max_id + 1_000_000)

    def alloc(self, shape: Any, name: str = "") -> Any:
        """Allocate a fresh SSA Value with the given ``ValueShape``.

        The Value's ``producer`` is left ``None`` — the caller attaches
        it when constructing the op that produces the Value by passing
        the Value into that op's ``results`` tuple. (Post-construction,
        ``Op.__post_init__`` or the caller can set ``value.producer``
        if the validator needs it — today it does not.)

        When the rewriter shares a driver-scoped counter (see
        ``Rewriter.for_op``), the bump happens on the shared cell so
        later rewrites pick up where this one left off."""
        from quark.ir.value import Value

        if self._shared is not None:
            next_id = self._shared[0]
            self._shared[0] = next_id + 1
            self._next_id = self._shared[0]
        else:
            next_id = self._next_id
            self._next_id += 1
        return Value(id=next_id, shape=shape, producer=None, name=name)


def _max_value_id(fn: Any) -> int:
    """Return the largest ``Value.id`` reachable from the function's
    op graph, or ``-1`` if there are no ops / no Values.

    Recurses into nested regions (if / for-loop bodies). The MoE
    kernels wrap their whole body in a sentinel-skip ``IfRegionOp``,
    so every ConstOp / AsyncCopy lives one region down — a top-level
    walk would return ``-1`` and a subsequent ``_legalize_async``
    rewrite would mint VecLoad Values starting at id 0, colliding
    with the existing nested ids."""
    max_id = -1

    def walk(ops: list[Op]) -> None:
        nonlocal max_id
        for op in ops:
            for v in op.results:
                if v.id > max_id:
                    max_id = v.id
            for v in op.operands:
                if v.id > max_id:
                    max_id = v.id
            for region in op.regions:
                walk(region.ops)

    walk(fn.body.ops)
    return max_id
