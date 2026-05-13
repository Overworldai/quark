"""Inline ortho-RoPE frequency computation in IR.

Computes cos/sin for one spatial position from (h, w, frame_t, Dh)
directly in the kernel — no precomputed table needed.

The ortho-RoPE frequency vector for position (h, w) in frame t:
  freqs[0 : Dh//8]      = pixel freqs for x (w position)
  freqs[Dh//8 : Dh//4]  = pixel freqs for y (h position)
  freqs[Dh//4 : Dh//2]  = lang freqs for t (frame position)

Each band: position * base_freq, where base_freq is a constant
derived from (H, W, Dh) at kernel compile time.
"""

from __future__ import annotations

import math

from quark.ir import DType


def emit_rope_cos_sin(
    bctx,
    *,
    h_idx,  # Value: spatial row index [0, H)
    w_idx,  # Value: spatial column index [0, W)
    frame_t,  # Value: frame index (u32)
    c_idx,  # Value: frequency pair index [0, Dh//2)
    H: int,
    W: int,
    Dh: int,
):
    """Emit IR to compute (cos, sin) for one RoPE frequency pair.

    Returns ``(cos_val, sin_val)`` as f32 Values.

    ``c_idx`` is the column index into the Dh//2 frequency vector.
    The function determines which band (x, y, t) ``c_idx`` falls in
    and computes the appropriate frequency.

    Uses structured ``if_`` control flow to avoid computing out-of-band
    frequencies (which would cause u32 underflow in band index math).
    """
    import quark.lang as qk

    half_Dh = Dh // 2
    x_dim = Dh // 8  # number of x-freq pairs
    y_dim = Dh // 8  # number of y-freq pairs
    t_dim = Dh // 4  # number of t-freq pairs
    assert x_dim + y_dim + t_dim == half_Dh

    max_freq = min(H, W) * 0.8
    theta = 10000.0

    # Position values: x_pos ∈ [-1+1/W, 1-1/W], y_pos similar, t_pos = frame_t.
    # ts_mult = base_fps // (inference_fps / temporal_compression) = 15 // 15 = 1.
    w_f = qk.convert(w_idx, DType.F32)
    h_f = qk.convert(h_idx, DType.F32)
    t_f = qk.convert(frame_t, DType.F32)

    inv_W = bctx.c(1.0 / W, dtype=DType.F32)
    inv_H = bctx.c(1.0 / H, dtype=DType.F32)
    two = bctx.c(2.0, dtype=DType.F32)
    one = bctx.c(1.0, dtype=DType.F32)
    neg_one = bctx.c(-1.0, dtype=DType.F32)

    x_pos = neg_one + (two * w_f + one) * inv_W  # -1 + (2w+1)/W
    y_pos = neg_one + (two * h_f + one) * inv_H  # -1 + (2h+1)/H

    # Band boundaries.
    c_x_dim = bctx.c(x_dim, dtype=DType.U32)
    c_xy_dim = bctx.c(x_dim + y_dim, dtype=DType.U32)

    is_x = qk.cmp("lt", c_idx, c_x_dim)
    is_y = qk.cmp("lt", c_idx, c_xy_dim)

    # Shared constants for x/y band base computation.
    pi_c = bctx.c(math.pi, dtype=DType.F32)
    xy_scale = bctx.c(math.pi * (max_freq / 2 - 1.0) / max(x_dim // 2 - 1, 1), dtype=DType.F32)
    one_u = bctx.c(1, dtype=DType.U32)
    zero_u = bctx.c(0, dtype=DType.U32)

    # Compute all three band frequencies unconditionally and select. The
    # earlier version used nested ``qk.if_`` to protect ``c_idx -
    # c_x_dim`` / ``c_idx - c_xy_dim`` from u32 underflow when c_idx is
    # in a lower band; the OpSelect version is strictly simpler — fewer
    # OpPhi at merges, no structured-CF blocks, identical numerics on
    # every backend. To avoid the underflow, mask the diff with
    # ``qk.select`` before subtracting so the unselected branch sees a
    # benign operand.
    c_idx_x = qk.select(is_x, c_idx, c_x_dim)   # max(0, ...) for x branch
    c_idx_y = qk.select(is_y, c_idx, c_xy_dim)  # max(c_x_dim, ...) for y branch
    # X band: local_i = c_idx / 2, base = pi * (1 + local_i * scale)
    half_c_x = qk.convert(c_idx_x >> one_u, DType.F32)
    x_base = pi_c + half_c_x * xy_scale
    x_freq = x_pos * x_base
    # Y band: local_i = (c_idx - x_dim) / 2. Operand was clamped so the
    # u32 subtract can't underflow when this branch is the live one.
    y_local = qk.convert(qk.select(is_x, zero_u, c_idx_y - c_x_dim) >> one_u, DType.F32)
    y_base = pi_c + y_local * xy_scale
    y_freq = y_pos * y_base
    # T band: base = 1 / theta^(local_pair_idx * 2 / t_dim)
    t_diff = qk.select(is_y, zero_u, c_idx - c_xy_dim)
    t_local = qk.convert(t_diff >> one_u, DType.F32)
    t_exp = t_local * bctx.c(2.0 / t_dim, dtype=DType.F32)
    log2_theta = bctx.c(math.log2(theta), dtype=DType.F32)
    t_base = qk.ex2_approx(qk.neg(t_exp * log2_theta))
    t_freq = t_f * t_base

    # 3-way select: c_idx ∈ [0,x_dim) → x_freq; [x_dim, xy_dim) → y_freq;
    # else → t_freq.
    yt_freq = qk.select(is_y, y_freq, t_freq)
    freq = qk.select(is_x, x_freq, yt_freq)

    # cos/sin via SFU.
    cos_val = qk.cos(freq)
    sin_val = qk.sin(freq)

    return cos_val, sin_val
