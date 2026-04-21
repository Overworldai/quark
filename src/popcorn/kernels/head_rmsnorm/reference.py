"""HeadRMSNorm reference."""

from __future__ import annotations

from popcorn.backend import PT


def head_rmsnorm_reference_for_spec(kernel, X):
    s = kernel.spec
    Dh, nq, Hk = s.Dh, s.n_q_heads, s.n_kv_heads
    eps = s.eps
    out_dt = s.dtype.backend

    X_f = PT.astype(X, PT.float32)

    if PT._is_mx(X_f):
        import mlx.core as mx

        def _rnorm_heads(t, n_heads, col_start):
            for h in range(n_heads):
                c0 = col_start + h * Dh
                c1 = c0 + Dh
                chunk = t[:, c0:c1]
                inv = mx.rsqrt(mx.mean(chunk * chunk, axis=-1, keepdims=True) + eps)
                t = mx.concatenate([t[:, :c0], chunk * inv, t[:, c1:]], axis=1)
            return t

        Y = _rnorm_heads(X_f, nq, 0)
        Y = _rnorm_heads(Y, Hk, nq * Dh)
    else:
        import torch

        def _rnorm_heads(t, n_heads, col_start):
            for h in range(n_heads):
                c0 = col_start + h * Dh
                c1 = c0 + Dh
                t[:, c0:c1] = torch.nn.functional.rms_norm(t[:, c0:c1], (Dh,), eps=eps)
            return t

        Y = _rnorm_heads(X_f.clone(), nq, 0)
        Y = _rnorm_heads(Y, Hk, nq * Dh)

    return PT.astype(Y, out_dt)
