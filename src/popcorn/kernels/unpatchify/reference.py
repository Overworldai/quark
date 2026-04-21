"""Unpatchify reference."""

from __future__ import annotations

from popcorn.backend import PT


def unpatchify_reference_for_spec(kernel, X, W, Bias):
    s = kernel.spec

    X_f = PT.astype(X, PT.float32)
    W_f = PT.astype(W, PT.float32)
    h = PT.matmul(X_f, PT.transpose(W_f))  # [M, C*ph*pw]
    if s.has_bias:
        h = h + PT.astype(Bias, PT.float32)

    # Rearrange to [B, C*H*W].
    B, C = s.B, s.C
    Hp, Wp = s.Hp, s.Wp
    ph, pw = s.ph, s.pw
    h_r = h.reshape(B, Hp, Wp, C, ph, pw)
    if PT._is_mx(h_r):
        import mlx.core as mx

        h_r = mx.transpose(h_r, (0, 3, 1, 4, 2, 5))
    else:
        h_r = h_r.permute(0, 3, 1, 4, 2, 5).contiguous()
    out = h_r.reshape(B, C * s.H * s.W)
    return PT.astype(out, s.dtype.backend)
