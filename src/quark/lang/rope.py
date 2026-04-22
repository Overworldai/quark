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

    # Shared constants for x/y band base computation.
    pi_c = bctx.c(math.pi, dtype=DType.F32)
    xy_scale = bctx.c(math.pi * (max_freq / 2 - 1.0) / max(x_dim // 2 - 1, 1), dtype=DType.F32)

    # Use a zero placeholder for the carried freq value.
    zero_f = bctx.c(0.0, dtype=DType.F32)

    # Structured if: only compute the band that c_idx actually falls in.
    # This avoids u32 underflow from (c_idx - x_dim) when c_idx < x_dim.
    with qk.if_(is_x, carried=[zero_f]) as (then_in, else_in, arms):
        with arms.then_():
            # X band: local_i = c_idx / 2, base = pi * (1 + local_i * scale)
            half_c = qk.convert(c_idx >> bctx.c(1, dtype=DType.U32), DType.F32)
            x_base = pi_c + half_c * xy_scale
            x_freq = x_pos * x_base
            qk.yield_(x_freq)
        with arms.else_():
            # Not x-band: determine if y or t.
            is_y = qk.cmp("lt", c_idx, c_xy_dim)
            with qk.if_(is_y, carried=[zero_f]) as (y_then, y_else, y_arms):
                with y_arms.then_():
                    # Y band: local_i = (c_idx - x_dim) / 2
                    y_local = qk.convert((c_idx - c_x_dim) >> bctx.c(1, dtype=DType.U32), DType.F32)
                    y_base = pi_c + y_local * xy_scale
                    y_freq = y_pos * y_base
                    qk.yield_(y_freq)
                with y_arms.else_():
                    # T band: base = 1 / theta^(local_pair_idx * 2 / t_dim)
                    t_local = qk.convert(
                        (c_idx - c_xy_dim) >> bctx.c(1, dtype=DType.U32), DType.F32
                    )
                    t_exp = t_local * bctx.c(2.0 / t_dim, dtype=DType.F32)
                    log2_theta = bctx.c(math.log2(theta), dtype=DType.F32)
                    t_base = qk.ex2_approx(qk.neg(t_exp * log2_theta))
                    t_freq = t_f * t_base
                    qk.yield_(t_freq)
            (inner_freq,) = qk.last_results()
            qk.yield_(inner_freq)
    (freq,) = qk.last_results()

    # cos/sin via SFU.
    cos_val = qk.cos(freq)
    sin_val = qk.sin(freq)

    return cos_val, sin_val
