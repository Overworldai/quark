"""ElementwiseConfig — n_warps + elements per block."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class ElementwiseConfig(KernelConfig):
    n_warps: int = 4
    elems_per_block: int = 1024

    @classmethod
    def default_for(cls, spec) -> ElementwiseConfig:
        for epb in (1024, 512, 256, 128):
            if spec.N % epb == 0 and epb % (4 * 32) == 0:
                return cls(n_warps=4, elems_per_block=epb)
        for epb in (64, 32):
            if spec.N % epb == 0 and epb % 32 == 0:
                return cls(n_warps=1, elems_per_block=epb)
        return cls(n_warps=4, elems_per_block=1024)
