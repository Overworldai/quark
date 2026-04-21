"""AdaRMSNorm baselines — plain torch/MLX reference timing."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def ada_rmsnorm_baselines(kernel, tensors: dict) -> list[Baseline]:
    from popcorn.backend import IS_METAL

    X = tensors["X"]
    scale = tensors["scale"]
    bias = tensors["bias"]
    D = int(X.shape[-1])
    G = int(scale.shape[0])
    M = int(X.shape[0]) // G
    eps = kernel.spec.eps

    if IS_METAL:
        import mlx.core as mx

        def _mx():
            X_f = X.astype(mx.float32)
            inv = mx.rsqrt(mx.mean(X_f * X_f, axis=-1, keepdims=True) + eps)
            Y = X_f * inv
            s_bm = mx.repeat(scale, M, axis=0)
            b_bm = mx.repeat(bias, M, axis=0)
            Y = Y * (1.0 + s_bm) + b_bm
            mx.eval(Y.astype(mx.bfloat16))

        return [Baseline("mx.ada_rmsnorm", _mx)]

    import torch

    def _torch():
        y = torch.nn.functional.rms_norm(X, (D,), eps=eps)
        s_bm = scale.repeat_interleave(M, dim=0)
        b_bm = bias.repeat_interleave(M, dim=0)
        _ = y * (1 + s_bm) + b_bm

    return [Baseline("F.rms_norm+affine", _torch)]
