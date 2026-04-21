"""Backend-agnostic RMSNorm reference (gainless, last-dim reduction)."""

from __future__ import annotations

from popcorn.backend import PT
from popcorn.ir import DType


def rmsnorm_reference(X, *, eps: float = 1.1920929e-07, dtype: DType | str = DType.BF16):
    """``y = x * rsqrt(mean(x², axis=-1, keepdim=True) + eps)``.

    Accumulates in f32, casts back to ``dtype`` at the end. No
    learnable gain — matches ``F.rms_norm(x)`` with weight=None.
    """
    out_dt = DType(dtype) if isinstance(dtype, str) else dtype
    X_f32 = PT.astype(X, PT.float32)
    X_sq = X_f32 * X_f32
    if PT._is_mx(X_sq):
        import mlx.core as mx

        mean_sq = mx.mean(X_sq, axis=-1, keepdims=True)
        inv = mx.rsqrt(mean_sq + eps)
    else:
        import torch

        mean_sq = torch.mean(X_sq, dim=-1, keepdim=True)
        inv = torch.rsqrt(mean_sq + eps)
    Y = X_f32 * inv
    return PT.astype(Y, out_dt.backend)


def rmsnorm_reference_for_spec(kernel, X):
    s = kernel.spec
    return rmsnorm_reference(X, eps=s.eps, dtype=s.dtype)
