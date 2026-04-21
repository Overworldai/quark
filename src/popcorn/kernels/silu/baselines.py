"""SiLU baselines."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def silu_baselines(kernel, tensors: dict) -> list[Baseline]:
    from popcorn.backend import IS_METAL

    X = tensors["X"]
    if IS_METAL:
        import mlx.core as mx

        def _mx():
            mx.eval(mx.sigmoid(X.astype(mx.float32)).astype(X.dtype) * X)

        return [Baseline("mx.silu", _mx)]

    import torch

    def _torch():
        torch.nn.functional.silu(X)

    return [Baseline("F.silu", _torch)]
