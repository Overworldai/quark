"""Unpatchify baselines."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def unpatchify_baselines(kernel, tensors: dict) -> list[Baseline]:
    from popcorn.backend import IS_METAL

    X = tensors["X"]
    W = tensors["W"]
    if IS_METAL:
        import mlx.core as mx

        def _mx():
            mx.eval(X @ W.T)

        return [Baseline("mx.matmul", _mx)]
    return [Baseline("noop", lambda: None)]
