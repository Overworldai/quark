"""SiLU problems."""

from __future__ import annotations

from popcorn.kernels.base import Problem


def silu_problems() -> list[Problem]:
    return [
        Problem("wp15_cond", {"N": 2048}, tags={"smoke", "production"}),
        Problem("wp15_tokens", {"N": 512 * 2048}, tags={"production"}),
    ]
