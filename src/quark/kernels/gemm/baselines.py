"""GEMM baselines — stubbed empty in the numpy-refs era.

The prior implementation drove backend-fast references (cuBLAS /
``torch._scaled_mm`` / MLX) for the speed column in ``make bench``.
Those required torch / mlx to be importable; the numpy-refs migration
dropped torch as a runtime dep, so we've neutered this to an empty
list. If you need perf comparisons back, wire a conditional torch
import here and reconstruct the baseline calls — but keep the rest
of the codebase torch-free."""

from __future__ import annotations

from quark.kernels.base import Baseline


def gemm_baselines(tensors: dict) -> list[Baseline]:
    return []
