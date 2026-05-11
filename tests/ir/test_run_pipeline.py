"""run_pipeline (S2 closure-based pipeline emitter) — structural tests.

Verifies the emitter produces the right schedule (load / commit / wait /
barrier / compute ordering) and threads carry values correctly. Deep
correctness lives in the kernel smokes + bench (GEMM is already ported
and cos_sim-validated on Metal).
"""

from __future__ import annotations

import pytest

import quark.lang as qk
from quark.blocks.dsl import BlockContext
from quark.blocks.l2.run_pipeline import IterCtx, PipelineBody, run_pipeline
from quark.ir import DType, Value
from quark.ir.builder import Builder
from quark.ir.module import BufferType


def _new_bctx(name: str = "t") -> tuple[Builder, BlockContext]:
    b = Builder(name)
    b.begin_function("fn")
    b.param("x", BufferType(DType.F32))
    bctx = BlockContext(b, n_threads=32, mma_cfg=None)
    return b, bctx


def test_single_stage_emits_expected_ops() -> None:
    b, bctx = _new_bctx()
    produce_calls, consume_calls = [], []

    def produce(ictx: IterCtx) -> None:
        produce_calls.append((ictx.stage_idx, ictx.is_tail))
        qk.barrier("block")  # stand-in for a load

    def consume(ictx: IterCtx) -> tuple[Value, ...]:
        consume_calls.append((ictx.stage_idx, ictx.is_tail, ictx.carry))
        return ictx.carry

    def init_carry(_bctx: BlockContext) -> tuple[Value, ...]:
        return (qk.const(DType.F32, 0.0),)

    body = PipelineBody(
        stages=["stage0"],
        produce=produce,
        consume=consume,
        carry=init_carry,
    )
    final = run_pipeline(bctx, n_iters=4, body=body, n_stages=1)
    b.end_function()

    # produce + consume are each called once (inside one ForLoopOp).
    # The carry inside the body is a fresh body-carried Value, distinct
    # from the for-op's result Value returned here — so compare only
    # structural fields.
    assert produce_calls == [(0, False)]
    assert [(s, t) for (s, t, _c) in consume_calls] == [(0, False)]
    assert len(final) == 1


def test_double_buffer_fires_prologue_and_epilogue() -> None:
    b, bctx = _new_bctx()
    produce_iters: list[int] = []
    consume_iters: list[tuple[int, bool]] = []

    def produce(ictx: IterCtx) -> None:
        # Our abstraction passes iter_idx as a Value; for prologue /
        # epilogue these are compile-time consts. Record stage_idx
        # order instead, which is deterministic.
        produce_iters.append(ictx.stage_idx)
        qk.barrier("block")

    def consume(ictx: IterCtx) -> tuple[Value, ...]:
        consume_iters.append((ictx.stage_idx, ictx.is_tail))
        return ictx.carry

    body = PipelineBody(
        stages=["s0", "s1"],
        produce=produce,
        consume=consume,
        carry=lambda _: (qk.const(DType.F32, 0.0),),
    )
    run_pipeline(bctx, n_iters=8, body=body, n_stages=2)
    b.end_function()

    # Prologue: produce stage 0, produce stage 1.
    # Steady state (half_iters-1 = 3 rounds), each fires:
    #   consume(0) produce(0) consume(1) produce(1)
    # Epilogue: consume(0, tail), consume(1, tail).
    assert produce_iters[0:2] == [0, 1]  # prologue prefetches
    # Tail consumes must have is_tail=True.
    assert consume_iters[-2:] == [(0, True), (1, True)]


def test_double_buffer_falls_back_to_single_stage_for_small_n() -> None:
    b, bctx = _new_bctx()
    consume_invocations = 0

    def produce(_ictx: IterCtx) -> None:
        qk.barrier("block")

    def consume(ictx: IterCtx) -> tuple[Value, ...]:
        nonlocal consume_invocations
        consume_invocations += 1
        return ictx.carry

    body = PipelineBody(
        stages=["s0", "s1"],
        produce=produce,
        consume=consume,
        carry=lambda _: (),
    )
    # half_iters = 1 < 2 → falls back to single-stage.
    # Single-stage calls consume once (inside the for_range body).
    run_pipeline(bctx, n_iters=2, body=body, n_stages=2)
    b.end_function()
    assert consume_invocations == 1


def test_invalid_stage_count_rejected() -> None:
    b, bctx = _new_bctx()
    body = PipelineBody(
        stages=["only_one"],
        produce=lambda _i: None,
        consume=lambda i: i.carry,
    )
    with pytest.raises(ValueError, match="n_stages must be 1 or 2"):
        run_pipeline(bctx, n_iters=4, body=body, n_stages=3)
    with pytest.raises(ValueError, match="body.stages has"):
        run_pipeline(bctx, n_iters=4, body=body, n_stages=2)
    b.end_function()


def test_epilogue_called_with_final_carry() -> None:
    b, bctx = _new_bctx()
    epi_calls: list[tuple] = []

    def epilogue(_bctx, carry):
        epi_calls.append(carry)

    body = PipelineBody(
        stages=["s"],
        produce=lambda _i: None,
        consume=lambda i: i.carry,
        carry=lambda _b: (qk.const(DType.F32, 7.0),),
        epilogue=epilogue,
    )
    final = run_pipeline(bctx, n_iters=3, body=body, n_stages=1)
    b.end_function()
    assert len(epi_calls) == 1
    assert epi_calls[0] == final
