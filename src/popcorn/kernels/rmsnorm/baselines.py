"""Backend-dispatched RMSNorm baselines."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def rmsnorm_baselines(kernel, tensors: dict) -> list[Baseline]:
    from popcorn.backend import IS_METAL

    X = tensors["X"]
    D = int(X.shape[-1])
    eps = kernel.spec.eps

    if IS_METAL:
        import mlx.core as mx

        def _mx():
            sq = X * X
            inv = mx.rsqrt(mx.mean(sq, axis=-1, keepdims=True) + eps)
            mx.eval(X * inv)

        return [Baseline("mx.rms_norm", _mx)]

    import torch

    def _torch():
        torch.nn.functional.rms_norm(X, (D,), eps=eps)

    return [Baseline("F.rms_norm", _torch)]
