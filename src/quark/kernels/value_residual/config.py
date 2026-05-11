"""ValueResidualConfig — n_warps + elements-per-block."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class ValueResidualConfig(KernelConfig):
    n_warps: int = 4
    # Elements per block — must divide N. Default gives one vec_load +
    # vec_store per thread for 128-thread blocks (8 bf16 per op).
    elems_per_block: int = 1024

    @classmethod
    def default_for(cls, spec) -> ValueResidualConfig:
        # Walk a couple of candidate configs; pick the first that divides N.
        for epb in (1024, 512, 256, 128):
            if spec.N % epb == 0 and epb % (4 * 32) == 0:
                return cls(n_warps=4, elems_per_block=epb)
        # Fallback: simpler but less efficient.
        for epb in (64, 32, 16):
            if spec.N % epb == 0 and epb % (1 * 32) == 0:
                return cls(n_warps=1, elems_per_block=epb)
        return cls(n_warps=4, elems_per_block=1024)
