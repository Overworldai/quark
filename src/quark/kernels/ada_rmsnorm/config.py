"""AdaRMSNormConfig — n_warps + chunk_D for the chunked pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class AdaRMSNormConfig(KernelConfig):
    n_warps: int = 4
    chunk_D: int = 256  # 0 = use full D (no chunking)

    @classmethod
    def default_for(cls, spec) -> AdaRMSNormConfig:
        for nw in (4, 8, 2, 1):
            if spec.D % (nw * 32) == 0:
                return cls(n_warps=nw, chunk_D=256)
        return cls(n_warps=4, chunk_D=256)
