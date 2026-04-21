"""ValueResidualPackedConfig — n_warps + chunk_D."""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.kernels.base import KernelConfig


@dataclass(frozen=True)
class ValueResidualPackedConfig(KernelConfig):
    n_warps: int = 4
    chunk_D: int = 256

    @classmethod
    def default_for(cls, spec) -> ValueResidualPackedConfig:
        for nw in (4, 8, 2, 1):
            if spec.D_full % (nw * 32) == 0 and spec.M % nw == 0:
                return cls(n_warps=nw, chunk_D=256)
        return cls(n_warps=4, chunk_D=256)
