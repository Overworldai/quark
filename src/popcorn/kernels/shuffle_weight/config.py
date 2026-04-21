"""ShuffleWeightConfig — n_warps only; elems_per_block derived from tile size."""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.kernels.base import KernelConfig


@dataclass(frozen=True)
class ShuffleWeightConfig(KernelConfig):
    n_warps: int = 4

    @classmethod
    def default_for(cls, spec) -> ShuffleWeightConfig:
        return cls(n_warps=4)
