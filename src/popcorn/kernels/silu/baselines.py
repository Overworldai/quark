"""silu baselines — stubbed empty in the numpy-refs era.

The prior implementation drove backend-fast references (torch / mlx)
for the ``make bench`` speed column. The numpy-refs migration
dropped torch as a runtime dep, so baselines return empty by default.
Wire a conditional torch import here to re-enable perf comparisons
against backend-fast libs when needed."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def silu_baselines(kernel, tensors: dict) -> list[Baseline]:
    del kernel, tensors
    return []
