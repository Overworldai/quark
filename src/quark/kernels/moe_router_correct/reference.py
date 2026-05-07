"""moe_router_correct numpy reference — purely-correct routing.

Each token goes to its actual top-K experts (no capacity, no substitution).
Slots are sorted by expert and per-expert padded to a multiple of BM=32.
Chunks past the last filled slot get expert=-1 (sentinel) so inproj/outproj
can skip them.
"""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import to_f32_numpy


def moe_router_correct_reference_numpy(
    spec,
    *,
    logits,
    token_ids=None,
    slot_weights=None,
    counts=None,
    work_list=None,
    offsets=None,
    token_slot_table=None,
):
    del token_ids, slot_weights, counts, work_list, offsets, token_slot_table
    M = spec.M
    E = spec.E
    K = spec.top_k
    total_slots = spec.total_slots
    BM = 32

    L = to_f32_numpy(logits, dtype_hint="f32").reshape(M, E)

    # Per-token top-K (sorted descending so we know which is rank 0..K-1).
    pri_idx_full = np.argsort(-L, axis=1)  # [M, E]
    topk_expert = pri_idx_full[:, :K].astype(np.int32)  # [M, K]
    topk_score = np.take_along_axis(L, topk_expert.astype(np.int64), axis=1)  # [M, K]

    # Softmax over top-K → per-token slot weights.
    shift = topk_score - topk_score.max(axis=-1, keepdims=True)
    exps = np.exp(shift)
    weights = exps / exps.sum(axis=-1, keepdims=True)  # [M, K]

    # Per-expert counts.
    out_counts = np.zeros(E, dtype=np.int32)
    for t in range(M):
        for k in range(K):
            out_counts[topk_expert[t, k]] += 1

    # Per-expert offsets with BM padding.
    out_offsets = np.zeros(E + 1, dtype=np.int32)
    cum = 0
    for e in range(E):
        out_offsets[e] = cum
        padded = ((int(out_counts[e]) + BM - 1) // BM) * BM
        cum += padded
    out_offsets[E] = cum
    total_active = cum

    if total_active > total_slots:
        raise ValueError(
            f"moe_router_correct: total active slots ({total_active}) "
            f"exceeds buffer capacity ({total_slots})"
        )

    # Scatter: for each (token, k), fill the next available slot in
    # expert e's range. Use a per-expert local cursor so we deterministically
    # match the kernel's atomic-add allocation order (within an expert).
    out_token_ids = np.zeros(total_slots, dtype=np.int32)
    out_slot_weights = np.zeros(total_slots, dtype=np.float32)
    out_token_slot_table = np.zeros((M, K), dtype=np.int32)
    cursors = np.zeros(E, dtype=np.int32)
    for t in range(M):
        for k in range(K):
            e = int(topk_expert[t, k])
            slot_in_e = int(cursors[e])
            cursors[e] += 1
            final = int(out_offsets[e]) + slot_in_e
            out_token_ids[final] = t
            out_slot_weights[final] = weights[t, k]
            out_token_slot_table[t, k] = final

    # Build work_list. One (grp_start, expert) per BM-chunk. Chunks past
    # total_active get expert = -1 (sentinel).
    n_chunks = total_slots // BM
    out_work_list = np.zeros(2 * n_chunks, dtype=np.int32)
    for i in range(n_chunks):
        grp_start = i * BM
        out_work_list[2 * i] = grp_start
        if grp_start >= total_active:
            out_work_list[2 * i + 1] = -1  # sentinel
            continue
        # Find which expert e this chunk belongs to.
        for e in range(E):
            if out_offsets[e] <= grp_start < out_offsets[e + 1]:
                out_work_list[2 * i + 1] = e
                break

    return {
        "token_ids": out_token_ids,
        "slot_weights": out_slot_weights,
        "counts": out_counts,
        "work_list": out_work_list,
        "offsets": out_offsets,
        "token_slot_table": out_token_slot_table,
    }
