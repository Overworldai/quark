"""Shared DSL helpers for the three MoE router kernels.

The router kernels (``moe_router``, ``moe_router_correct``,
``moe_router_shared``) all do the same per-token primitives:

  * insertion-sort the E logits to keep the top-P values + indices
  * softmax over the top-K of those sorted scores

Each one re-derives the same IR. Centralising the patterns here keeps
the three kernel ``build()`` bodies focused on what differs (their
routing-decision phases) instead of the boilerplate.

These are *kernel-build* helpers — they emit IR via ``qk.lang`` calls
and must run inside an active ``BlockContext``. They are not torch ops
or numpy callables.
"""

from __future__ import annotations

import quark.lang as qk
from quark.ir import DType, Value


def insertion_sort_topp(
    bctx,
    scores: list[Value],
    *,
    top_p: int,
    neg_inf: Value,
    zero_i32: Value,
) -> tuple[list[Value], list[Value]]:
    """Stable insertion-sort over ``scores`` (length E) returning the
    top-P (score, idx) pairs in descending order.

    Each pass is fully unrolled by the Python loop — the IR sees a
    chain of ``cmp``/``select`` ops only, no runtime control flow.

    Returns ``(pri_score, pri_idx)`` — both Python lists of length
    ``top_p`` of IR Values.
    """
    pri_score: list[Value] = [neg_inf] * top_p
    pri_idx: list[Value] = [zero_i32] * top_p
    for e, score in enumerate(scores):
        e_const = bctx.c(e, dtype=DType.S32)
        greater = [qk.cmp("gt", score, pri_score[p]) for p in range(top_p)]
        new_score: list[Value] = []
        new_idx: list[Value] = []
        for p in range(top_p):
            if p == 0:
                s_p = qk.select(greater[0], score, pri_score[0])
                i_p = qk.select(greater[0], e_const, pri_idx[0])
            else:
                # Insertion shifts pri[p-1] → p when ``greater[p-1]`` (i.e.
                # the score outranks p-1 too); else the score itself lands here.
                inner_s = qk.select(greater[p - 1], pri_score[p - 1], score)
                inner_i = qk.select(greater[p - 1], pri_idx[p - 1], e_const)
                s_p = qk.select(greater[p], inner_s, pri_score[p])
                i_p = qk.select(greater[p], inner_i, pri_idx[p])
            new_score.append(s_p)
            new_idx.append(i_p)
        pri_score = new_score
        pri_idx = new_idx
    return pri_score, pri_idx


def softmax_topk(scores: list[Value], *, top_k: int, log2e: Value) -> list[Value]:
    """Softmax over the first ``top_k`` entries of ``scores``.

    Uses ``ex2_approx`` (faster than ``exp_approx``) after rebasing
    against the per-call max. The max is computed via a select chain
    over ``scores[:top_k]`` so this works whether or not the caller
    pre-sorted them descending.

    Returns the K weights as IR Values (sums to 1 modulo float
    rounding).
    """
    max_score = scores[0]
    for k in range(1, top_k):
        gt = qk.cmp("gt", scores[k], max_score)
        max_score = qk.select(gt, scores[k], max_score)
    exp_vals = [qk.ex2_approx(qk.mul(qk.sub(scores[k], max_score), log2e)) for k in range(top_k)]
    sum_exp = exp_vals[0]
    for k in range(1, top_k):
        sum_exp = qk.add(sum_exp, exp_vals[k])
    rcp_sum = qk.rcp_approx(sum_exp)
    return [qk.mul(exp_vals[k], rcp_sum) for k in range(top_k)]
