"""owl_attn numpy reference — Q-RoPE + segment-masked GQA attention.

Flow:
  1. Inline RoPE: cos/sin from (H_spatial, W_spatial, frame_t, Dh).
  2. Apply ortho-RoPE to Q in f32 (concat layout: ``cat([y0, y1], -1)``).
  3. Additive mask ``[B, 1, 1, capacity]`` from (segments, n_segments):
     0 for valid positions, -inf outside.
  4. Standard SDPA in f32 with per-head GQA replication.
"""

from __future__ import annotations

import numpy as np

from popcorn.runtime.npconv import astype_numpy, to_f32_numpy


def _apply_rope_fp32(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    x0 = x32[..., 0::2]
    x1 = x32[..., 1::2]
    c = cos.astype(np.float32, copy=False)
    s = sin.astype(np.float32, copy=False)
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s
    return np.concatenate([y0, y1], axis=-1)


def _additive_mask_from_segments(
    segments: np.ndarray, n_segments: np.ndarray, capacity: int
) -> np.ndarray:
    """``segments[B, max_segments, 2] s32`` → ``[B, 1, 1, capacity] f32``
    additive mask (0 for valid, -inf for masked)."""
    B = int(segments.shape[0])
    mask = np.full((B, capacity), -np.inf, dtype=np.float32)
    for b in range(B):
        n = int(n_segments[b])
        for i in range(n):
            st = int(segments[b, i, 0])
            ln = int(segments[b, i, 1])
            end = min(st + ln, capacity)
            if st < end:
                mask[b, st:end] = 0.0
    return mask.reshape(B, 1, 1, capacity)


def owl_attn_reference_numpy(
    spec, *, Q, K_cache, Vt_cache, segments, n_segments, frame_t, output=None
):
    del output
    from popcorn.kernels.kv_cache_update.reference import _make_ortho_rope_freqs

    s = spec
    ft_arr = to_f32_numpy(frame_t, dtype_hint="s32")
    ft = int(ft_arr[0])

    # cos/sin for this frame.
    cos_full, sin_full = _make_ortho_rope_freqs(s.H_spatial, s.W_spatial, ft + 1, s.Dh)
    cos = cos_full[ft * s.tpf : (ft + 1) * s.tpf, :].reshape(s.tpf, s.Dh // 2)
    sin = sin_full[ft * s.tpf : (ft + 1) * s.tpf, :].reshape(s.tpf, s.Dh // 2)

    # Q: packed → per-head slice, else already [B, Hq, tpf, Dh]-shaped.
    Q_np = to_f32_numpy(Q, dtype_hint=s.a_dtype.value)
    if s.packed_qkv:
        q_cols = s.n_q_heads * s.Dh
        Q_raw = Q_np[:, :q_cols]
        Q4 = Q_raw.reshape(s.B, s.tpf, s.n_q_heads, s.Dh).transpose(0, 2, 1, 3)
    else:
        Q4 = Q_np.reshape(s.B, s.n_q_heads, s.tpf, s.Dh)

    K_np = to_f32_numpy(K_cache, dtype_hint=s.kv_dtype.value)
    Vt_np = to_f32_numpy(Vt_cache, dtype_hint=s.kv_dtype.value)
    Kc = K_np.reshape(s.B, s.n_kv_heads, s.capacity, s.Dh)
    Vtc = Vt_np.reshape(s.B, s.n_kv_heads, s.Dh, s.capacity)
    V = np.swapaxes(Vtc, -2, -1)  # [B, Hk, capacity, Dh]

    segs = to_f32_numpy(segments, dtype_hint="s32").astype(np.int32).reshape(s.B, s.max_segments, 2)
    nsegs = to_f32_numpy(n_segments, dtype_hint="s32").astype(np.int32).reshape(s.B)

    # Q-side RoPE.
    Q_rot = _apply_rope_fp32(Q4, cos[None, None, :, :], sin[None, None, :, :])

    # GQA: broadcast K/V heads to match Q.
    gqa = s.gqa_ratio
    if gqa != 1:
        Kc = np.repeat(Kc, gqa, axis=1)
        V = np.repeat(V, gqa, axis=1)

    scale = 1.0 / np.sqrt(s.Dh)
    scores = np.matmul(Q_rot, np.swapaxes(Kc, -2, -1)) * scale  # [B, Hq, tpf, capacity]

    # Additive mask, broadcast over Hq / tpf axes.
    mask = _additive_mask_from_segments(segs, nsegs, s.capacity)
    scores = scores + mask

    m = scores.max(axis=-1, keepdims=True)
    # After masking, rows with every entry = -inf produce NaN in
    # exp(-inf - (-inf)) = exp(NaN). Guard by replacing -inf row maxes
    # with 0 (those rows are all-masked anyway → output 0).
    m = np.where(np.isfinite(m), m, 0.0)
    ex = np.exp(scores - m)
    denom = ex.sum(axis=-1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    p = ex / denom
    out = np.matmul(p, V)  # [B, Hq, tpf, Dh]

    if s.packed_qkv:
        out = out.transpose(0, 2, 1, 3).reshape(s.B * s.tpf, s.n_q_heads * s.Dh)
    else:
        out = out.reshape(s.B * s.n_q_heads * s.tpf, s.Dh)
    return astype_numpy(out, s.out_dtype)
