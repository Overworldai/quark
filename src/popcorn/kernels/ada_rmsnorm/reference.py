"""AdaRMSNorm reference — ``y = rmsnorm(x) * (1 + scale) + bias``."""

from __future__ import annotations

from popcorn.backend import PT
from popcorn.ir import DType


def ada_rmsnorm_reference(
    X, scale, bias, *, eps: float = 1.1920929e-07, dtype: DType | str = DType.BF16
):
    """X: [G*M, D], scale/bias: [G, D]. Broadcast by repeating each
    scale/bias row M times along axis 0 before applying the epilogue.
    """
    out_dt = DType(dtype) if isinstance(dtype, str) else dtype

    B = int(X.shape[0])
    G = int(scale.shape[0])
    M = B // G
    assert G * M == B, f"X rows ({B}) not divisible by scale groups ({G})"

    X_f = PT.astype(X, PT.float32)
    if PT._is_mx(X_f):
        import mlx.core as mx

        mean_sq = mx.mean(X_f * X_f, axis=-1, keepdims=True)
        inv = mx.rsqrt(mean_sq + eps)
    else:
        import torch

        mean_sq = torch.mean(X_f * X_f, dim=-1, keepdim=True)
        inv = torch.rsqrt(mean_sq + eps)
    Y = X_f * inv

    # Broadcast scale/bias [G, D] → [G*M, D] by repeating along axis 0.
    scale_f = PT.astype(scale, PT.float32)
    bias_f = PT.astype(bias, PT.float32)
    if PT._is_mx(scale_f):
        import mlx.core as mx

        scale_bm = mx.repeat(scale_f, M, axis=0)
        bias_bm = mx.repeat(bias_f, M, axis=0)
    else:
        scale_bm = scale_f.repeat_interleave(M, dim=0)
        bias_bm = bias_f.repeat_interleave(M, dim=0)

    Y = Y * (1.0 + scale_bm) + bias_bm
    return PT.astype(Y, out_dt.backend)


def ada_rmsnorm_reference_for_spec(kernel, X, scale, bias):
    s = kernel.spec
    return ada_rmsnorm_reference(X, scale, bias, eps=s.eps, dtype=s.dtype)
