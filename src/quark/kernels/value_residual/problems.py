"""Bench/fuzz problems for ValueResidual."""

from __future__ import annotations

from quark.kernels.base import Problem


def value_residual_problems() -> list[Problem]:
    bf16 = {"dtype": "bf16"}
    # wp1.5: B=1, Hk=16, tpf=512, Dh=128 → 1,048,576 elements per layer per frame.
    return [
        Problem("wp15_layer", {"N": 1_048_576, **bf16}, tags={"smoke", "production"}),
        Problem("small", {"N": 4096, **bf16}, tags={"small"}),
    ]
