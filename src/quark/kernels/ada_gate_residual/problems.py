"""Bench/fuzz problems for AdaGateResidual."""

from __future__ import annotations

from quark.kernels.base import Problem


def ada_gate_residual_problems() -> list[Problem]:
    bf16 = {"dtype": "bf16"}
    return [
        Problem("wp15_frame", {"G": 1, "M": 512, "D": 2048, **bf16}, tags={"smoke", "production"}),
        Problem("wp15_prefill8", {"G": 8, "M": 512, "D": 2048, **bf16}, tags={"production"}),
        Problem("small", {"G": 2, "M": 4, "D": 128, **bf16}, tags={"small"}),
    ]
