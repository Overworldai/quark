"""Attention numpy reference — standard scaled-dot-product attention
with GQA head replication. f32 throughout, cast to ``spec.a_dtype``
on the way out.

Q: [B, Hq, Lq, Dh]    flattened to [B*Hq*Lq, Dh] on disk
K: [B, Hk, Lkv, Dh]   flattened to [B*Hk*Lkv, Dh]
V: [B, Hk, Lkv, Dh]   (stored transposed as V_t [B*Hk*Dh, Lkv])

Output: [B, Hq, Lq, Dh]  flattened to [B*Hq*Lq, Dh]
"""

from __future__ import annotations

import numpy as np

from popcorn.runtime.npconv import astype_numpy, to_f32_numpy


def attn_reference_numpy(spec, *, Q, K, V_t, output=None):
    del output
    hint = spec.a_dtype.value
    B, Hq, Hk = spec.B, spec.n_q_heads, spec.n_kv_heads
    Lq, Lkv, Dh = spec.seq_len, spec.kv_len, spec.Dh
    gqa = Hq // Hk

    q = to_f32_numpy(Q, dtype_hint=hint).reshape(B, Hq, Lq, Dh)
    k = to_f32_numpy(K, dtype_hint=spec.b_dtype.value).reshape(B, Hk, Lkv, Dh)
    # V_t stored as [B, Hk, Dh, Lkv] — transpose last two to recover V.
    v = to_f32_numpy(V_t, dtype_hint=spec.b_dtype.value).reshape(B, Hk, Dh, Lkv)
    v = np.swapaxes(v, -2, -1)  # → [B, Hk, Lkv, Dh]

    # Broadcast KV heads to match Q's head count (GQA).
    if gqa != 1:
        k = np.repeat(k, gqa, axis=1)
        v = np.repeat(v, gqa, axis=1)

    scale = 1.0 / np.sqrt(Dh)
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale  # [B, Hq, Lq, Lkv]
    # f32 softmax with per-row max subtraction for numerical stability.
    m = scores.max(axis=-1, keepdims=True)
    ex = np.exp(scores - m)
    p = ex / ex.sum(axis=-1, keepdims=True)
    out = np.matmul(p, v)  # [B, Hq, Lq, Dh]

    return astype_numpy(out.reshape(B * Hq * Lq, Dh), spec.a_dtype)
