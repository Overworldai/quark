"""OwlAttnInt reference — Phase 3.3b full-attention ground truth.

Per the current default ``config.phase = "full"`` the kernel emits
the standard attention output of shape ``(out_rows, out_cols)`` in
``spec.out_dtype`` (bf16 by default).

Kernel math (per Q row m, KV col n, output dh-col):

    Q_f32         = Q.astype(f32)
    Q_absmax[m]   = max_k |Q_f32[m, k]|
    Q_scale[m]    = Q_absmax[m] / 127  (1.0 if 0)
    Q_s8[m, k]    = round(Q_f32[m, k] / Q_scale[m])
    s_s32[m, n]   = sum_k Q_s8[m, k] * K_s8[n, k]
    s_f32[m, n]   = s_s32[m, n] * Q_scale[m] * K_scales[n]
    m[m]          = max_n s_f32[m, n]
    e[m, n]       = exp(s_f32[m, n] - m[m])
    l[m]          = sum_n e[m, n]
    P_scaled[m,n] = e[m, n] * V_scales[n]
    P_scale[m]    = max_n |P_scaled[m, :]| / 127  (1.0 if 0)
    P_s8[m, n]    = round(P_scaled[m, n] / P_scale[m])
    o_s32[m, dh]  = sum_n P_s8[m, n] * Vt_s8[dh, n]
    out[m, dh]    = o_s32[m, dh] * P_scale[m] / l[m]   →  cast out_dtype

The reference mirrors this *exact* sequence (including the
add-half-with-sign rounding the kernel uses) so the comparison
isolates kernel correctness from quantization noise. Tolerances
are tight (~1e-3) because the only quant losses are Q-row and
P-row, both also applied by this numpy ref.
"""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import astype_numpy, to_f32_numpy


def _quantize_per_row_s8(x_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8 quant matching the kernel:
       absmax → scale = absmax/127 (guard zero); s8 = round(x/scale)."""
    absmax = np.abs(x_f32).max(axis=-1)
    scale = np.where(absmax == 0, 1.0, absmax / 127.0)
    inv = 1.0 / scale
    scaled = x_f32 * inv[..., None]
    half = np.where(scaled < 0, -0.5, 0.5)
    rounded = (scaled + half).astype(np.int32)
    rounded = np.clip(rounded, -127, 127).astype(np.int8)
    return rounded, scale.astype(np.float32)


def _apply_rope_fp32(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Ortho-RoPE on last axis (matches OwlAttn): input is interleaved
    (pairs at ``2c, 2c+1``); output is concat (``y0`` in first half,
    ``y1`` in second half). Mirrors ``owl_attn/reference._apply_rope_fp32``.
    """
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    y0 = x0 * cos - x1 * sin
    y1 = x1 * cos + x0 * sin
    return np.concatenate([y0, y1], axis=-1)


def _scores_f32(spec, Q, K_s8, K_scales, frame_t):
    """Compute dequantized QK scores ``s_f32[q_idx, kv]`` + Q quant.

    For both packed and unpacked layouts we build a virtual "per-head"
    Q tensor of shape ``(B*n_q_heads*tpf, Dh)`` keyed by the canonical
    ``qr = (b*n_q_heads + q_head)*tpf + t`` so the quantization runs
    over exactly the kernel's per-row footprint (one Dh-vec at a time).
    Packed Q's other channels (the K/V slices in QKV) are correctly
    ignored — the kernel only reads the q_head*Dh : (q_head+1)*Dh
    range of each packed row.

    Phase 3.5b: applies ortho-RoPE to each per-head Q row in fp32
    BEFORE quantization, matching the kernel's on-load RoPE pass.
    """
    from quark.kernels.kv_cache_update.reference import _make_ortho_rope_freqs

    s = spec
    q_f_in = to_f32_numpy(Q, dtype_hint=s.a_dtype.value).astype(np.float32)
    k_s8_arr = to_f32_numpy(K_s8, dtype_hint="s8").astype(np.int32)
    k_scales = to_f32_numpy(K_scales).astype(np.float32)
    ft_arr = to_f32_numpy(frame_t, dtype_hint="s32").astype(np.int32)
    ft = int(ft_arr[0])

    # Per-frame cos/sin tables (same logic as owl_attn reference).
    cos_full, sin_full = _make_ortho_rope_freqs(s.H_spatial, s.W_spatial, ft + 1, s.Dh)
    cos = cos_full[ft * s.tpf : (ft + 1) * s.tpf, :]   # (tpf, Dh//2)
    sin = sin_full[ft * s.tpf : (ft + 1) * s.tpf, :]

    Dh = s.Dh
    if s.packed_qkv:
        # q_f_in: (B*tpf, n_q_heads*Dh + 2*n_kv_heads*Dh)
        q_f = np.zeros((s.B * s.n_q_heads * s.tpf, Dh), dtype=np.float32)
        for b in range(s.B):
            for q_head in range(s.n_q_heads):
                for t in range(s.tpf):
                    qr = (b * s.n_q_heads + q_head) * s.tpf + t
                    src_row = b * s.tpf + t
                    q_f[qr] = q_f_in[src_row, q_head * Dh : (q_head + 1) * Dh]
    else:
        q_f = q_f_in

    # Apply RoPE per-row: each row's t index = qr % tpf gives the
    # spatial position whose (cos, sin) row to use.
    q_f_4d = q_f.reshape(s.B, s.n_q_heads, s.tpf, Dh)
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    q_f_rot = _apply_rope_fp32(q_f_4d, cos_b, sin_b)
    q_f = q_f_rot.reshape(s.B * s.n_q_heads * s.tpf, Dh).astype(np.float32)

    q_s8, q_scale = _quantize_per_row_s8(q_f)
    q_s8_i32 = q_s8.astype(np.int32)

    scores = np.zeros((s.B * s.n_q_heads * s.tpf, s.capacity), dtype=np.float32)
    for b in range(s.B):
        for q_head in range(s.n_q_heads):
            kv_h = q_head // s.gqa_ratio
            kv_row_start = (b * s.n_kv_heads + kv_h) * s.capacity
            kv_row_end = kv_row_start + s.capacity
            k_block = k_s8_arr[kv_row_start:kv_row_end]
            k_sc_block = k_scales[kv_row_start:kv_row_end]
            for t in range(s.tpf):
                qr = (b * s.n_q_heads + q_head) * s.tpf + t
                s_s32 = k_block @ q_s8_i32[qr]
                scores[qr] = s_s32.astype(np.float32) * q_scale[qr] * k_sc_block
    return scores, q_scale


def owl_attn_int8_reference_numpy(
    spec, *, Q, K_s8, K_scales, Vt_s8, V_scales, frame_t, output=None,
):
    """Compute the Phase 3.3b full-attention reference output.

    Returns ``{"output": <(out_rows, out_cols) out_dtype>}``.

    Mirrors the kernel's int8 pipeline exactly (Q quant + softmax +
    V_scale fold + P quant + AV MMA + epilogue scale + cast).
    """
    del output

    s = spec
    scores, _q_scale = _scores_f32(spec, Q, K_s8, K_scales, frame_t)
    vt_s8_arr = to_f32_numpy(Vt_s8, dtype_hint="s8").astype(np.int32)
    v_scales = to_f32_numpy(V_scales).astype(np.float32)

    # Softmax: m, e, l per (q_row).
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    l = e.sum(axis=-1, keepdims=True)
    l_safe = np.where(l > 0, l, 1.0)

    # V_scale fold: P_scaled[m, n] = e[m, n] * V_scales[n].
    # V_scales is indexed by the *KV cache slot*; for each (b, kv_h, t):
    #   v_block = v_scales[(b*n_kv_heads + kv_h)*capacity : ... + capacity]
    out_shape = (s.out_rows, s.out_cols)
    out = np.zeros(out_shape, dtype=np.float32)

    Dh = s.Dh
    for b in range(s.B):
        for q_head in range(s.n_q_heads):
            kv_h = q_head // s.gqa_ratio
            kv_row_start = (b * s.n_kv_heads + kv_h) * s.capacity
            kv_row_end = kv_row_start + s.capacity
            v_sc_block = v_scales[kv_row_start:kv_row_end]    # (cap,)
            # Vt_s8 layout: (B*n_kv_heads*Dh, capacity). For (b, kv_h),
            # row = (b*n_kv_heads + kv_h)*Dh + dh ∈ [vt_base, vt_base+Dh).
            vt_base = (b * s.n_kv_heads + kv_h) * Dh
            vt_block = vt_s8_arr[vt_base : vt_base + Dh]      # (Dh, cap)

            for t in range(s.tpf):
                qr = (b * s.n_q_heads + q_head) * s.tpf + t
                p_scaled = e[qr] * v_sc_block                  # (cap,)

                # Per-row P quant (match kernel: add-half-with-sign + clip).
                p_absmax = np.abs(p_scaled).max()
                p_scale = (p_absmax / 127.0) if p_absmax > 0 else 1.0
                p_inv = 1.0 / p_scale
                p_x = p_scaled * p_inv
                p_half = np.where(p_x < 0, -0.5, 0.5)
                p_rounded = (p_x + p_half).astype(np.int32)
                p_s8 = np.clip(p_rounded, -127, 127).astype(np.int32)

                # AV MMA equivalent: o_s32[dh] = sum_n P_s8[n] * Vt_s8[dh, n].
                o_s32 = vt_block @ p_s8                        # (Dh,)

                # Epilogue scale + softmax-l division.
                o_f32 = o_s32.astype(np.float32) * p_scale / l_safe[qr][0]

                # Output position: (q_row, dh_col).
                # Output layout is (out_rows, out_cols) = (q_rows, Dh) here
                # (unpacked), so q_row == qr and dh_col ∈ [0, Dh).
                if s.packed_qkv:
                    out_row = b * s.tpf + t
                    out_col_base = q_head * Dh
                    out[out_row, out_col_base : out_col_base + Dh] = o_f32
                else:
                    out[qr, :] = o_f32

    return {"output": astype_numpy(out.astype(np.float32), s.out_dtype.value)}
