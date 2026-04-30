"""moe_router_shared numpy reference — shared-experts routing.

Pick the K experts whose summed-across-tokens softmax probability is
highest (the "cumulative preference" criterion). Every token then uses
those same K experts, with per-token weights = softmax over the K
chosen logits (so each token's K weights sum to 1).
"""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import to_f32_numpy


def moe_router_shared_reference_numpy(
    spec, *, logits, token_ids=None, slot_weights=None, counts=None, work_list=None
):
    del token_ids, slot_weights, counts, work_list
    M = spec.M
    E = spec.E
    K = spec.top_k
    BM = 32

    L = to_f32_numpy(logits, dtype_hint="f32").reshape(M, E)

    # Per-token softmax over E.
    L_shift = L - L.max(axis=-1, keepdims=True)
    probs = np.exp(L_shift)
    probs = probs / probs.sum(axis=-1, keepdims=True)  # [M, E]

    # Cumulative preference across tokens; pick top-K.
    cum_probs = probs.sum(axis=0)  # [E]
    chosen = np.argpartition(-cum_probs, K - 1)[:K]
    # Sort by score descending so chosen[0] is the best — kernel matches this order.
    chosen = chosen[np.argsort(-cum_probs[chosen])]  # [K] in E-space

    # Per-token softmax over the K chosen logits.
    topk_logits = L[:, chosen]  # [M, K]
    topk_shift = topk_logits - topk_logits.max(axis=-1, keepdims=True)
    topk_exp = np.exp(topk_shift)
    weights = topk_exp / topk_exp.sum(axis=-1, keepdims=True)  # [M, K]

    # Build outputs in the (K blocks of M slots) layout.
    out_token_ids = np.tile(np.arange(M, dtype=np.int32), K)  # [K*M]
    out_slot_weights = weights.T.reshape(-1).astype(np.float32)  # [K*M], k-major
    out_counts = np.zeros((E,), dtype=np.int32)
    out_counts[chosen] = M

    # work_list: K*M/BM chunks of BM slots each, labelled with the
    # chosen expert ID in E-space.
    n_chunks = (K * M) // BM
    grp_starts = np.arange(n_chunks, dtype=np.int32) * BM
    chunk_to_k_pos = grp_starts // M  # which of the K positions
    expert_ids = chosen[chunk_to_k_pos].astype(np.int32)
    out_work_list = np.stack([grp_starts, expert_ids], axis=1).reshape(-1)

    return {
        "token_ids": out_token_ids,
        "slot_weights": out_slot_weights,
        "counts": out_counts,
        "work_list": out_work_list,
    }
