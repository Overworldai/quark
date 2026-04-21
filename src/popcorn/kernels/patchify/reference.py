"""Patchify reference — reshape + matmul."""

from __future__ import annotations

from popcorn.backend import PT


def patchify_reference_for_spec(kernel, X, W):
    s = kernel.spec
    B, C = s.B, s.C
    ph, pw = s.ph, s.pw
    Hp, Wp = s.Hp, s.Wp

    X_f = PT.astype(X, PT.float32)
    W_f = PT.astype(W, PT.float32)

    # Reshape + permute on the reference side (CPU/backend, fine for correctness).
    X_tiled = X_f.reshape(B, C, Hp, ph, Wp, pw)
    if PT._is_mx(X_tiled):
        import mlx.core as mx

        X_perm = mx.transpose(X_tiled, (0, 2, 4, 1, 3, 5))
    else:
        X_perm = X_tiled.permute(0, 2, 4, 1, 3, 5).contiguous()
    X_flat = X_perm.reshape(B * Hp * Wp, C * ph * pw)

    Out = PT.matmul(X_flat, PT.transpose(W_f))
    return PT.astype(Out, s.dtype.backend)
