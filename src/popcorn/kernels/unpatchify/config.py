"""UnpatchifyConfig."""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.kernels.base import KernelConfig


@dataclass(frozen=True)
class UnpatchifyConfig(KernelConfig):
    BM: int = 64
    BN: int = 64
    BK: int = 16
    n_warps: int = 4
    n_stages: int = 2
    a_pad: int = 0
    b_pad: int = 0
    main_shape: str = ""

    @classmethod
    def default_for(cls, spec) -> UnpatchifyConfig:
        n_stages = 2 if (spec.K % (2 * 32) == 0) else 1
        return cls(BM=64, BN=64, BK=32, n_warps=4, n_stages=n_stages, a_pad=8)
