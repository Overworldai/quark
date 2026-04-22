"""QuantizeE4M3Config."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class QuantizeE4M3Config(KernelConfig):
    n_warps: int = 4
    elems_per_block: int = 1024

    @classmethod
    def default_for(cls, spec) -> QuantizeE4M3Config:
        pairs = spec.N // 2
        for epb in (1024, 512, 256, 128):
            if pairs % epb == 0 and epb % (4 * 32) == 0:
                return cls(n_warps=4, elems_per_block=epb)
        for epb in (64, 32):
            if pairs % epb == 0 and epb % 32 == 0:
                return cls(n_warps=1, elems_per_block=epb)
        return cls(n_warps=4, elems_per_block=128)
