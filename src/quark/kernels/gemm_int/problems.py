"""GemmInt problems — minimal set for correctness validation."""

from __future__ import annotations

from quark.kernels.base import Problem


def gemm_int_problems() -> list[Problem]:
    return [
        # Single MMA-tile smoke: exercises the bare s8/s8/s32 path
        # without any block tiling. Useful for correctness-first
        # validation; performance config will come once correctness
        # is locked.
        Problem("smoke_8x16x32", {"M": 8, "N": 16, "K": 32}, tags={"smoke"}),
    ]
