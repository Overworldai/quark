"""Patchify problems."""

from __future__ import annotations

from quark.kernels.base import Problem


def patchify_problems() -> list[Problem]:
    return [
        Problem(
            "wp15_360p",
            {"B": 1, "C": 32, "H": 16, "W": 32, "d_model": 2048},
            tags={"smoke", "production"},
        ),
        Problem(
            "wp15_512p",
            {"B": 1, "C": 32, "H": 32, "W": 64, "d_model": 2048},
            tags={"production"},
        ),
    ]
