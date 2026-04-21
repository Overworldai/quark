"""Unpatchify problems."""

from __future__ import annotations

from popcorn.kernels.base import Problem


def unpatchify_problems() -> list[Problem]:
    return [
        Problem(
            "wp15_360p",
            {"B": 1, "C": 32, "H": 16, "W": 32, "d_model": 2048},
            tags={"smoke", "production"},
        ),
    ]
