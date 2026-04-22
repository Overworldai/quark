"""L2: run_pipeline — closure-based pipeline emitter.

Replaces the factory
quartet (``smem_factory``, ``a_loader_factory``, ``b_loader_factory``,
``lane_transform``) used by the legacy :class:`Pipeline` /
:class:`KLoop` pair with a closure body: the caller describes one
logical iteration's work as ``produce`` / ``consume`` lambdas over
an :class:`IterCtx`, and this module orchestrates the scheduling.

Same schedule as the legacy KLoop, collapsed into one code path:

* ``n_stages=1`` — synchronous loop, barrier-separated iterations.
* ``n_stages=2`` — software-pipelined double buffer (prologue
  prefetches iters 0, 1 into stages 0, 1; steady state rotates).

The legacy path stays in place for kernels not yet ported — deletion
is step 6 of the S2 migration. No ``emit_raw`` above the lowerer; no
new IR ops.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import quark.lang as qk
from quark.blocks.dsl import BlockContext, Carry
from quark.ir import DType, Value


@dataclass
class IterCtx:
    """One logical iteration's context, passed to produce / consume.

    ``iter_idx`` is the *logical* iteration number as an S32 Value —
    the number the producer/consumer use for addressing, not the
    physical stage slot. The pipeline scheduler decides which physical
    stage slot receives the iteration's data and routes it via
    ``stage`` / ``stage_idx``.

    ``carry`` is the loop-carried tuple at the start of ``consume``.
    ``consume`` returns the new carry. For ``produce`` the carry is
    read-only (carrying through a prefetch is never useful).
    """

    iter_idx: Value
    stage: Any
    stage_idx: int
    # ``carry`` is a ``Carry`` instance when the kernel declared one
    # (attribute-access on named slots), else a flat ``tuple[Value, ...]``
    # for simple GEMM-shape kernels.
    carry: Any
    bctx: BlockContext
    is_tail: bool = False


@dataclass
class PipelineBody:
    """Describes one logical iteration of a pipelined loop.

    * ``stages`` — exactly ``n_stages`` physical slots (typically
      :class:`SmemPlan` instances). The scheduler rotates through them.
    * ``produce(ictx)`` — emit gmem→smem loads for ``ictx.iter_idx``
      into ``ictx.stage``. Called once per logical iteration (plus the
      prologue prefetches for n_stages=2).
    * ``consume(ictx)`` — emit compute on ``ictx.stage`` using
      ``ictx.carry``; return the new carry as a tuple.
    * ``carry`` — initial loop-carried values. Three accepted forms:
        - an :class:`Accumulators` (normalized via ``.emit_init(bld)``
          at run time — the common case for GEMM-shaped kernels),
        - a tuple of Values (passed through as-is),
        - a callable ``(bctx) -> tuple[Value, ...]`` (legacy escape
          hatch for kernels that need per-loop emission).
      Defaults to ``()`` — the loop carries nothing.
    * ``epilogue(bctx, final_carry)`` — optional post-loop hook for
      kernels that write output (GEMM scatter, attention normalize).
      Not required; many kernels handle epilogue at the call site.
    * ``consume_tail(ictx)`` — optional override for the last
      iteration(s) when running at ``n_stages=2`` (the prefetch can
      legally alias past end-of-K, but some kernels want a distinct
      tail body). Defaults to ``consume`` with ``is_tail=True``.

    ``async_commit`` / ``async_wait`` emissions are driven
    automatically: ``run_pipeline`` checks ``bctx.bld.async_emissions``
    before and after every ``produce`` call and only issues the pair
    if the counter advanced. Scalar-only loaders stay clean.
    """

    stages: list[Any]
    produce: Callable[[IterCtx], None]
    # ``consume`` returns the next carry — either a ``Carry`` instance
    # (when ``body.carry`` is a Carry; ``run_pipeline`` flattens on yield)
    # or a tuple/list of Values for the simple GEMM-shape path.
    consume: Callable[[IterCtx], Any]
    carry: Any = ()  # Accumulators | Carry | tuple[Value, ...] | Callable[[BlockContext], tuple[Value, ...]]
    epilogue: Callable[[BlockContext, tuple[Value, ...]], None] | None = None
    consume_tail: Callable[[IterCtx], Any] | None = None

    def run(self, *, n_iters: int | Value, n_stages: int = 1) -> Any:
        """Emit the pipeline loop and return the final carry.

        Equivalent to ``run_pipeline(n_iters=..., body=self,
        n_stages=...)`` but lets kernel bodies fluently chain::

            PipelineBody(
                stages=stages, produce=produce, consume=mma, carry=acc,
            ).run(n_iters=K_outer, n_stages=c.n_stages)

        ``bctx`` is implicit via ``active_bctx()``.
        """
        return run_pipeline(n_iters=n_iters, body=self, n_stages=n_stages)


def _produce_with_async_commit(body: PipelineBody, ictx: IterCtx, bctx: BlockContext) -> bool:
    """Run ``body.produce(ictx)``; return True iff it emitted any
    ``async_copy``. Emits ``async_commit`` after produce when it did,
    so the caller can pair the matching ``async_wait`` without
    tracking the flag itself.
    """
    before = bctx.bld.async_emissions
    body.produce(ictx)
    emitted = bctx.bld.async_emissions > before
    if emitted:
        qk.async_commit()
    return emitted


def _resolve_init_carry(body: PipelineBody, bctx: BlockContext) -> tuple[Value, ...]:
    """Normalize ``body.carry`` to an init tuple.

    Accepts:
    - ``Accumulators`` (calls ``emit_init``),
    - ``Carry`` (named multi-subset; calls ``.init()``),
    - a plain tuple (pass-through),
    - a legacy ``(bctx) -> tuple`` callable.
    """
    from quark.blocks.dsl import Accumulators, Carry

    c = body.carry
    if isinstance(c, Accumulators):
        return tuple(c.init())
    if isinstance(c, Carry):
        return c.init()
    if callable(c):
        return tuple(c(bctx))
    return tuple(c)


def run_pipeline(
    bctx: BlockContext | None = None,
    *,
    n_iters: int | Value,
    body: PipelineBody,
    n_stages: int = 1,
) -> Carry | tuple[Value, ...]:
    """Emit the pipelined loop and return the final carry.

    ``bctx`` defaults to the active BlockContext via ``active_bctx()``
    — same convention as the rest of the DSL free-functions. Kernels
    call ``run_pipeline(n_iters=..., body=..., n_stages=...)`` without
    threading the context by hand.

    ``n_iters`` accepts a compile-time ``int`` (bulk of GEMM / attn
    cases) or a runtime U32 Value (owl_attn's inner loop runs a
    segment-variable chunk count). Runtime ``n_iters`` requires
    ``n_stages=1`` — the double-buffer emitter needs the half-iter
    count at compile time to place its prologue / epilogue.

    Invariants:
    - ``len(body.stages) == n_stages``.
    - ``n_stages ∈ {1, 2}``.
    - For ``n_stages=2`` with ``int`` n_iters, falls back to
      ``n_stages=1`` when ``n_iters // 2 < 2`` (the software pipeline
      needs ≥2 steady-state rounds to be worth the prologue cost).
    """
    if bctx is None:
        from quark.blocks.dsl import active_bctx

        bctx = active_bctx()
    if n_stages not in (1, 2):
        raise ValueError(f"run_pipeline: n_stages must be 1 or 2, got {n_stages}")
    if len(body.stages) != n_stages:
        raise ValueError(
            f"run_pipeline: body.stages has {len(body.stages)} entries but n_stages={n_stages}"
        )
    if isinstance(n_iters, Value):
        if n_stages != 1:
            raise NotImplementedError(
                "run_pipeline: runtime (Value) n_iters only supported at n_stages=1; "
                "double-buffer needs compile-time half_iters for its prologue/epilogue"
            )
        return _run_single_stage(bctx, n_iters=n_iters, body=body)
    if n_stages == 2 and (n_iters // 2) < 2:
        # Fall back — matches legacy KLoop semantics.
        return _run_single_stage(bctx, n_iters=n_iters, body=body)
    if n_stages == 1:
        return _run_single_stage(bctx, n_iters=n_iters, body=body)
    return _run_double_buffer(bctx, n_iters=n_iters, body=body)


# ---------------------------------------------------------------
# Single-stage synchronous loop.
# ---------------------------------------------------------------


def _run_single_stage(
    bctx: BlockContext,
    *,
    n_iters: int | Value,
    body: PipelineBody,
) -> Carry | tuple[Value, ...]:
    is_carry = isinstance(body.carry, Carry)
    init = _resolve_init_carry(body, bctx)
    stage = body.stages[0]
    hi = n_iters if isinstance(n_iters, Value) else bctx.const(DType.U32, n_iters)

    with qk.for_range(
        bctx.const(DType.U32, 0),
        hi,
        bctx.const(DType.U32, 1),
        iv_name="i",
        carried=init,
    ) as (i, carried):
        # When the kernel declared a Carry, pass it to consume with
        # its slots rebound to this iteration's values. The consume
        # body reads/writes named slots (carry.o, carry.m, ...) and
        # returns a Carry; we flatten before yield_.
        carry_in: Any = body.carry.rebind(tuple(carried)) if is_carry else tuple(carried)
        ictx = IterCtx(
            iter_idx=i,
            stage=stage,
            stage_idx=0,
            carry=carry_in,
            bctx=bctx,
        )
        emitted_async = _produce_with_async_commit(body, ictx, bctx)
        if emitted_async:
            qk.async_wait(0)
        qk.barrier("block")

        returned = body.consume(ictx)
        new_carry = returned.flatten() if is_carry else tuple(returned)
        qk.barrier("block")
        qk.yield_(*new_carry)

    final = tuple(bctx.bld.last_results)
    _stash_results_on_carry(body, final)
    if body.epilogue is not None:
        body.epilogue(bctx, final)
    if is_carry:
        return body.carry.rebind(final)
    return final


def _stash_results_on_carry(body: PipelineBody, final: tuple[Value, ...]) -> None:
    """When ``body.carry`` is an Accumulators, write the loop's final
    Values onto ``carry.results`` so epilogue helpers can take just the
    Accumulators instance and pull both layout (MT/NT) + values.

    For a ``Carry`` wrapper, do the same per Accumulators slot — each
    Accumulators-backed slot gets ``.results`` populated from the
    rebound Carry so ``qk.store_acc(o_acc, ...)`` works without the
    kernel having to repack.
    """
    from quark.blocks.dsl import Accumulators, Carry

    if isinstance(body.carry, Accumulators):
        body.carry.results = list(final)
        return
    if isinstance(body.carry, Carry):
        body.carry.rebind(final)
        # Walk the original slot specs (not the rebound values) to find
        # Accumulators instances and stash their per-tile results.
        # ``Carry.__init__`` keeps the raw specs alive in a parallel
        # dict so post-hoc lookups like this remain cheap.
        for name, spec in _iter_accumulators(body.carry):
            spec.results = list(getattr(body.carry, name))


def _iter_accumulators(carry: Any) -> Any:
    """Yield (name, Accumulators) for every Accumulators slot the Carry
    was constructed with. Walks the hidden ``_raw_specs`` stashed on
    Carry so the scan doesn't depend on slot-spec introspection heuristics.
    """
    from quark.blocks.dsl import Accumulators

    raw = getattr(carry, "_raw_specs", None)
    if raw is None:
        return
    for name, spec in raw.items():
        if isinstance(spec, Accumulators):
            yield name, spec


# ---------------------------------------------------------------
# Double-buffer software pipeline.
# ---------------------------------------------------------------


def _run_double_buffer(
    bctx: BlockContext,
    *,
    n_iters: int,
    body: PipelineBody,
) -> Carry | tuple[Value, ...]:
    """Prologue: prefetch iters 0, 1 into stages 0, 1.
    Steady state: half_iters - 1 full rounds, each computing a pair
    and prefetching the next pair. Epilogue: drain the final pair.

    Matches the legacy KLoop._emit_double_buffer layout exactly so
    perf stays neutral on port.
    """
    is_carry = isinstance(body.carry, Carry)
    stage0, stage1 = body.stages[0], body.stages[1]
    half_iters = n_iters // 2

    def _wrap_carry(flat: tuple[Value, ...]) -> Any:
        """Consume expects a Carry when body.carry is Carry; else tuple."""
        return body.carry.rebind(flat) if is_carry else flat

    def _flatten_carry(out: Any) -> tuple[Value, ...]:
        return out.flatten() if is_carry else tuple(out)

    raw_consume = body.consume
    raw_tail = body.consume_tail or raw_consume

    def consume(ictx: IterCtx) -> tuple[Value, ...]:
        return _flatten_carry(raw_consume(ictx))

    def consume_tail(ictx: IterCtx) -> tuple[Value, ...]:
        return _flatten_carry(raw_tail(ictx))

    # Prologue: prefetch chunks 0 and 1. ``_produce_with_async_commit``
    # also tells us whether produce emits cp.async so we can gate the
    # matching async_wait calls throughout the steady state + epilogue.
    ictx0 = IterCtx(
        iter_idx=bctx.const(DType.U32, 0),
        stage=stage0,
        stage_idx=0,
        carry=(),
        bctx=bctx,
    )
    has_async = _produce_with_async_commit(body, ictx0, bctx)

    ictx1 = IterCtx(
        iter_idx=bctx.const(DType.U32, 1),
        stage=stage1,
        stage_idx=1,
        carry=(),
        bctx=bctx,
    )
    _produce_with_async_commit(body, ictx1, bctx)

    init = _resolve_init_carry(body, bctx)

    # Steady state: half_iters - 1 rounds.
    with qk.for_range(
        bctx.const(DType.U32, 0),
        bctx.const(DType.U32, half_iters - 1),
        bctx.const(DType.U32, 1),
        iv_name="k_pair",
        carried=init,
    ) as (k_pair, carried):
        carry = tuple(carried)

        # iter indices consumed this round:  a = k_pair*2,   b = k_pair*2+1
        # iter indices prefetched this round: a = k_pair*2+2, b = k_pair*2+3
        base = k_pair * 2
        iter_a = base
        iter_b = base + 1
        iter_a_next = base + 2
        iter_b_next = base + 3

        # ── Compute stage 0 with iter_a ──
        if has_async:
            qk.async_wait(1)
        qk.barrier("block")
        ictx = IterCtx(
            iter_idx=iter_a,
            stage=stage0,
            stage_idx=0,
            carry=_wrap_carry(carry),
            bctx=bctx,
        )
        carry = consume(ictx)
        qk.barrier("block")

        # ── Prefetch stage 0 with iter_a + 2 ──
        ictx_pf = IterCtx(
            iter_idx=iter_a_next,
            stage=stage0,
            stage_idx=0,
            carry=(),
            bctx=bctx,
        )
        _produce_with_async_commit(body, ictx_pf, bctx)

        # ── Compute stage 1 with iter_b ──
        if has_async:
            qk.async_wait(1)
        qk.barrier("block")
        ictx = IterCtx(
            iter_idx=iter_b,
            stage=stage1,
            stage_idx=1,
            carry=_wrap_carry(carry),
            bctx=bctx,
        )
        carry = consume(ictx)
        qk.barrier("block")

        # ── Prefetch stage 1 with iter_b + 2 ──
        ictx_pf = IterCtx(
            iter_idx=iter_b_next,
            stage=stage1,
            stage_idx=1,
            carry=(),
            bctx=bctx,
        )
        _produce_with_async_commit(body, ictx_pf, bctx)

        qk.yield_(*carry)

    loop_results = tuple(bctx.bld.last_results)

    # ── Epilogue: drain final pair (iters n_iters-2, n_iters-1) ──
    if has_async:
        qk.async_wait(1)
    qk.barrier("block")
    ictx = IterCtx(
        iter_idx=bctx.const(DType.U32, n_iters - 2),
        stage=stage0,
        stage_idx=0,
        carry=_wrap_carry(loop_results),
        bctx=bctx,
        is_tail=True,
    )
    carry = consume_tail(ictx)
    qk.barrier("block")

    if has_async:
        qk.async_wait(0)
    qk.barrier("block")
    ictx = IterCtx(
        iter_idx=bctx.const(DType.U32, n_iters - 1),
        stage=stage1,
        stage_idx=1,
        carry=_wrap_carry(carry),
        bctx=bctx,
        is_tail=True,
    )
    carry = consume_tail(ictx)
    qk.barrier("block")

    final = tuple(carry)
    _stash_results_on_carry(body, final)
    if body.epilogue is not None:
        body.epilogue(bctx, final)
    if is_carry:
        return body.carry.rebind(final)
    return final
