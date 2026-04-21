"""Test problems for the elementwise kernel."""

from __future__ import annotations

from popcorn.kernels.base import Problem
from popcorn.kernels.elementwise.spec import ALL_OPS, BINARY_OPS


def elementwise_problems() -> list[Problem]:
    problems = []
    bf16 = {"dtype": "bf16"}

    for op in sorted(ALL_OPS):
        n = 1024
        problems.append(Problem(f"{op}_1k", {"N": n, "op": op, **bf16}, tags={"smoke"}))

    # Larger sizes for binary ops.
    for op in sorted(BINARY_OPS):
        problems.append(Problem(f"{op}_64k", {"N": 65536, "op": op, **bf16}, tags={"production"}))

    # Cast bf16→f32.
    problems.append(
        Problem(
            "cast_bf16_f32",
            {"N": 1024, "op": "cast", "dtype": "bf16", "out_dtype": "f32"},
            tags={"smoke"},
        )
    )

    return problems
