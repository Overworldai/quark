"""ValueResidualPacked problems."""

from __future__ import annotations

from popcorn.kernels.base import Problem


def value_residual_packed_problems() -> list[Problem]:
    return [
        Problem(
            "wp15_360p",
            {"M": 128, "D_full": 4096, "v_col_offset": 3072, "v_width": 1024},
            tags={"smoke", "production"},
        ),
    ]
