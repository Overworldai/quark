"""ValueResidual baselines."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def value_residual_baselines(kernel, tensors: dict) -> list[Baseline]:
    from popcorn.backend import IS_METAL

    V = tensors["V"]
    V1 = tensors["V1"]
    lamb = tensors["lamb"]

    if IS_METAL:
        import mlx.core as mx

        def _mx():
            out = V + lamb * (V1 - V)
            mx.eval(out)

        return [Baseline("mx.value_residual", _mx)]

    import torch

    def _torch():
        _ = torch.lerp(V, V1, lamb)

    return [Baseline("torch.lerp", _torch)]
