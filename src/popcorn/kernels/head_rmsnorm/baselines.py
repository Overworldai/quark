"""HeadRMSNorm baselines."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def head_rmsnorm_baselines(kernel, tensors: dict) -> list[Baseline]:
    from popcorn.backend import IS_METAL

    X = tensors["X"]
    if IS_METAL:
        import mlx.core as mx

        def _mx():
            mx.eval(X)

        return [Baseline("mx.noop", _mx)]
    return [Baseline("noop", lambda: None)]
