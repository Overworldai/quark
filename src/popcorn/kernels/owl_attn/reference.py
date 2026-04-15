"""Reference for owl_attn — Q-RoPE + masked SDPA, polymorphic.

Flow (matches kv_cache_update.reference + owl_attn semantics):
  1. Q-side ortho-RoPE in fp32 with concat-output layout
     (``cat([y0, y1], -1)``), cast back to Q dtype.
  2. Build an additive ``[B, 1, 1, capacity]`` mask from ``segments``
     (0 = valid, -inf = masked).
  3. ``PT.attention(Q, K, V, mask=...)`` — dispatches to each
     backend's fast SDPA (flash-attn on MLX, torch SDPA on CUDA).

No manual softmax, no backend-type branching — the reference only
talks to ``PT`` and lets it route to the right hardware path.
"""

from __future__ import annotations

from popcorn.backend import PT


def apply_rope_fp32_concat(x, cos, sin):
    """Ortho-RoPE in fp32 with concat layout — mirrors
    ``kv_cache_update.reference.apply_rope_fp32``."""
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
    """Build a ``[B, 1, 1, capacity]`` additive mask: 0 where valid,
    ``-inf`` elsewhere. Segments are tiny (2 per batch in steady
    state) so a Python-side scan is fine. Returns a PT tensor in the
    same backend as ``segments``."""
    B = int(segments.shape[0])
    rows = [[float("-inf")] * capacity for _ in range(B)]
    # Pull the segment data to Python once — segments is [B, max_seg, 2]
    # int32, a handful of entries per batch.
    seg_list = PT.to_cpu_numpy(segments).tolist()
    nseg_list = PT.to_cpu_numpy(n_segments).tolist()
    for b in range(B):
        n = int(nseg_list[b])
        for i in range(n):
            st, ln = int(seg_list[b][i][0]), int(seg_list[b][i][1])
            for j in range(st, min(st + ln, capacity)):
                rows[b][j] = 0.0
    m = PT.tensor(rows, dtype=PT.float32)
    # Broadcast-friendly shape: [B, 1, 1, capacity].
    return m.reshape(B, 1, 1, capacity)


def owl_attn_reference(
    Q,  # [B, Hq, tpf, Dh]     a_dtype
    K_cache,  # [B, Hk, capacity, Dh]  kv_dtype
    Vt_cache,  # [B, Hk, Dh, capacity]  kv_dtype
    cos,  # [tpf, Dh//2]      f32
    sin,  # [tpf, Dh//2]      f32
    segments,  # [B, max_segments, 2] int32
    n_segments,  # [B]          int32
    *,
    out_dtype,
):
    """Returns output ``[B, Hq, tpf, Dh]`` in ``out_dtype``."""
    capacity = K_cache.shape[2]

    # 1) Q-side ortho-RoPE in fp32.
    cos_b = cos[None, None, :, :]  # [1, 1, tpf, Dh//2]
    sin_b = sin[None, None, :, :]
    Q_rot = apply_rope_fp32_concat(Q, cos_b, sin_b)  # [B, Hq, tpf, Dh]

    # 2) Reconstruct V from Vt_cache: swap last two dims.
    V_cache = PT.transpose(Vt_cache)

    # 3) Additive mask over capacity.
    mask = _additive_mask_from_segments(segments, n_segments, capacity)

    # 4) Cast everything to bf16 so the SDPA matmul stays bf16 — matches
    #    the kernel's bf16 MMA precision.
    Qf = PT.astype(Q_rot, PT.bfloat16)
    Kf = PT.astype(K_cache, PT.bfloat16)
    Vf = PT.astype(V_cache, PT.bfloat16)
    y = PT.attention(Qf, Kf, Vf, mask=PT.astype(mask, PT.bfloat16))
    return PT.astype(y, out_dtype)


def owl_attn_reference_for_spec(kernel, Q, K_cache, Vt_cache, cos, sin, segments, n_segments):
    s = kernel.spec
    out_dt = s.out_dtype.backend
    Q4 = Q.reshape(s.B, s.n_q_heads, s.tpf, s.Dh)
    Kc = K_cache.reshape(s.B, s.n_kv_heads, s.capacity, s.Dh)
    Vtc = Vt_cache.reshape(s.B, s.n_kv_heads, s.Dh, s.capacity)
    cos2 = cos.reshape(s.tpf, s.Dh // 2)
    sin2 = sin.reshape(s.tpf, s.Dh // 2)
    segs3 = segments.reshape(s.B, s.max_segments, 2)
    nseg1 = n_segments.reshape(s.B)
    out = owl_attn_reference(Q4, Kc, Vtc, cos2, sin2, segs3, nseg1, out_dtype=out_dt)
    return out.reshape(s.B * s.n_q_heads * s.tpf, s.Dh)
