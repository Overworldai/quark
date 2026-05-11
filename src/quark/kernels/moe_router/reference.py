"""moe_router numpy reference — capacity-bounded top-K with full-fallback.

Greedy fill in token-id order (token 0 first, then 1, ...). The CUDA
kernel uses atomic counters so the order is *non-deterministic*, but
the per-expert assignment counts and the per-(expert, slot) population
match this reference up to a within-expert permutation. Correctness
validation in fuzz/bench compares the *unordered set* of (expert, token,
weight) tuples emitted, not the within-expert ordering.

For each token:
  * Compute softmax over the top-K logits — those are the slot weights,
    matched by *original rank* (not by which expert ultimately fills
    the slot).
  * For each k in [0, K), try expert pri[k] first; on overflow, try
    every other expert in priority order, with dedup against prior
    claims by this token. Drops only happen if no expert with room
    remains uncliamed — structurally impossible when E*C >= M*top_k
    and K << E (the typical MoE regime).
"""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import to_f32_numpy


def moe_router_reference_numpy(spec, *, logits, token_ids=None, slot_weights=None, counts=None):
    del token_ids, slot_weights, counts
    M = spec.M
    E = spec.E
    K = spec.top_k
    C = spec.capacity

    L = to_f32_numpy(logits, dtype_hint="f32").reshape(M, E)

    # Full-priority sort: every expert is a candidate, in priority order.
    pri_idx = np.argsort(-L, axis=1)  # [M, E], descending by score
    row_idx = np.arange(M)[:, None]
    pri_score = L[row_idx, pri_idx]

    # Softmax over the top-K scores only — those are the per-slot weights
    # stored regardless of which expert actually fills the slot.
    topk_score = pri_score[:, :K]
    topk_shift = topk_score - topk_score.max(axis=1, keepdims=True)
    topk_exp = np.exp(topk_shift)
    pri_weight = topk_exp / topk_exp.sum(axis=1, keepdims=True)  # [M, K]

    out_token_ids = np.zeros((E * C,), dtype=np.int32)
    out_slot_weights = np.zeros((E * C,), dtype=np.float32)
    out_counts = np.zeros((E,), dtype=np.int32)

    for t in range(M):
        claimed: list[int] = []
        for k in range(K):
            # Try pri[k] first, then every other expert in priority order.
            # If none has room AND isn't already claimed, the slot is
            # dropped — only reachable under adversarial K-near-E configs.
            try_order = [k] + [i for i in range(E) if i != k]
            for i in try_order:
                e = int(pri_idx[t, i])
                if e in claimed:
                    continue
                slot = out_counts[e]
                if slot < C:
                    out_token_ids[e * C + slot] = t
                    out_slot_weights[e * C + slot] = pri_weight[t, k]
                    out_counts[e] = slot + 1
                    claimed.append(e)
                    break

    return {
        "token_ids": out_token_ids,
        "slot_weights": out_slot_weights,
        "counts": out_counts,
    }
