"""Reference for owl_attn — Q-RoPE + masked SDPA, polymorphic.

Flow (matches kv_cache_update.reference + owl_attn semantics):
  1. Q-side ortho-RoPE in fp32 with concat-output layout
     (``cat([y0, y1], -1)``), cast back to Q dtype.
  2. Build an additive ``[B, 1, 1, capacity]`` mask from ``segments``
     (0 = valid, -inf = masked).
  3. ``PT.attention(Q, K, V, mask=...)`` — dispatches to each
     backend's fast SDPA (flash-attn on MLX, torch SDPA on CUDA).

Inline RoPE: cos/sin computed from (H, W, frame_t, Dh) — no
precomputed table tensors.
"""

from __future__ import annotations

from popcorn.backend import PT


def apply_rope_fp32_concat(x, cos, sin):
    """Ortho-RoPE in fp32 with concat layout."""
    orig_dt = x.dtype
    x32 = PT.astype(x, PT.float32)
    x0 = x32[..., 0::2]
    x1 = x32[..., 1::2]
    c = PT.astype(cos, PT.float32)
    s = PT.astype(sin, PT.float32)
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s
    return PT.astype(PT.cat((y0, y1), dim=-1), orig_dt)


def _additive_mask_from_segments(segments, n_segments, capacity: int):
    B = int(segments.shape[0])
    rows = [[float("-inf")] * capacity for _ in range(B)]
    seg_list = PT.to_cpu_numpy(segments).tolist()
    nseg_list = PT.to_cpu_numpy(n_segments).tolist()
    for b in range(B):
        n = int(nseg_list[b])
        for i in range(n):
            st, ln = int(seg_list[b][i][0]), int(seg_list[b][i][1])
            for j in range(st, min(st + ln, capacity)):
                rows[b][j] = 0.0
    m = PT.tensor(rows, dtype=PT.float32)
    return m.reshape(B, 1, 1, capacity)


def owl_attn_reference(Q, K_cache, Vt_cache, cos, sin, segments, n_segments, *, out_dtype):
    """Returns output ``[B, Hq, tpf, Dh]`` in ``out_dtype``."""
    capacity = K_cache.shape[2]
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    Q_rot = apply_rope_fp32_concat(Q, cos_b, sin_b)
    V_cache = PT.transpose(Vt_cache)
    mask = _additive_mask_from_segments(segments, n_segments, capacity)
    Qf = PT.astype(Q_rot, PT.bfloat16)
    Kf = PT.astype(K_cache, PT.bfloat16)
    Vf = PT.astype(V_cache, PT.bfloat16)
    y = PT.attention(Qf, Kf, Vf, mask=PT.astype(mask, PT.bfloat16))
    return PT.astype(y, out_dtype)


def owl_attn_reference_for_spec(kernel, Q, K_cache, Vt_cache, segments, n_segments, frame_t):
    """Reference matching inline-RoPE TENSORS: Q, K_cache, Vt_cache,
    segments, n_segments, frame_t, output."""
    from popcorn.kernels.kv_cache_update.reference import make_ortho_rope_freqs

    s = kernel.spec
    out_dt = s.out_dtype.backend

    # Compute cos/sin from frame_t (inline RoPE).
    ft = int(frame_t.item() if hasattr(frame_t, "item") else frame_t[0])
    cos_full, sin_full = make_ortho_rope_freqs(s.H_spatial, s.W_spatial, ft + 1, s.Dh)
    cos = cos_full[ft * s.tpf : (ft + 1) * s.tpf, :]
    sin = sin_full[ft * s.tpf : (ft + 1) * s.tpf, :]

    # Extract Q from packed QKV if needed.
    if s.packed_qkv:
        # Q is [B*tpf, qkv_dim] — first n_q_heads*Dh columns are Q.
        q_cols = s.n_q_heads * s.Dh
        Q_raw = Q[:, :q_cols]  # [B*tpf, Hq*Dh]
        Q4 = Q_raw.reshape(s.B, s.tpf, s.n_q_heads, s.Dh)
        Q4 = PT.permute(Q4, (0, 2, 1, 3))  # [B, Hq, tpf, Dh]
    else:
        Q4 = Q.reshape(s.B, s.n_q_heads, s.tpf, s.Dh)

    Kc = K_cache.reshape(s.B, s.n_kv_heads, s.capacity, s.Dh)
    Vtc = Vt_cache.reshape(s.B, s.n_kv_heads, s.Dh, s.capacity)
    cos2 = cos.reshape(s.tpf, s.Dh // 2)
    sin2 = sin.reshape(s.tpf, s.Dh // 2)
    segs3 = segments.reshape(s.B, s.max_segments, 2)
    nseg1 = n_segments.reshape(s.B)
    out = owl_attn_reference(Q4, Kc, Vtc, cos2, sin2, segs3, nseg1, out_dtype=out_dt)

    # Output shape matches kernel: packed → [B*tpf, Hq*Dh], else [B*Hq*tpf, Dh].
    if s.packed_qkv:
        # out is [B, Hq, tpf, Dh] → [B, tpf, Hq, Dh] → [B*tpf, Hq*Dh]
        out = PT.permute(out, (0, 2, 1, 3))
        return out.reshape(s.B * s.tpf, s.n_q_heads * s.Dh)
    return out.reshape(s.B * s.n_q_heads * s.tpf, s.Dh)
