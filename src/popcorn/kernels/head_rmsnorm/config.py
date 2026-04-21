"""HeadRMSNormConfig — n_warps sized for Dh reduction."""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.kernels.base import KernelConfig


@dataclass(frozen=True)
class HeadRMSNormConfig(KernelConfig):
    n_warps: int = 2  # Dh=64 → 2 warps × 32 = 64 threads = 1 elem/thread

    @classmethod
    def default_for(cls, spec) -> HeadRMSNormConfig:
        for nw in (2, 4, 1):
            if spec.Dh % (nw * 32) == 0:
                return cls(n_warps=nw)
        return cls(n_warps=2)
