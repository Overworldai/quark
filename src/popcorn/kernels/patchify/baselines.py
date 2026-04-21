"""Patchify baselines."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def patchify_baselines(kernel, tensors: dict) -> list[Baseline]:
    from popcorn.backend import IS_METAL

    X = tensors["X"]
    W = tensors["W"]
    if IS_METAL:
        import mlx.core as mx

        def _mx():
            mx.eval(X.reshape(-1) @ W.reshape(W.shape[0], -1).T)

        return [Baseline("mx.matmul", _mx)]
    return [Baseline("noop", lambda: None)]
