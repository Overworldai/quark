"""ValueResidualPacked baselines."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def value_residual_packed_baselines(kernel, tensors: dict) -> list[Baseline]:
    from popcorn.backend import IS_METAL

    if IS_METAL:
        import mlx.core as mx

        def _mx():
            mx.eval(tensors["QKV_curr"])

        return [Baseline("mx.noop", _mx)]
    return [Baseline("noop", lambda: None)]
