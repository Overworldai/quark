"""AdaGateResidualConfig — n_warps + chunk_D + direct."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class AdaGateResidualConfig(KernelConfig):
    n_warps: int = 4
    chunk_D: int = 256
    # direct=True: read gate/X/Y directly from gmem (no smem staging).
    # Eliminates the gate-smem barrier and the gmem→smem→regs double-copy,
    # at the cost of exposed L2 latency. Wins on small G where the gate is
    # L2-resident and smem pipeline overhead dominates.
    direct: bool = True

    @classmethod
    def default_for(cls, spec) -> AdaGateResidualConfig:
        # chunk_D must divide D and be at least one warp's worth of elements
        # (32 lanes × 8 bf16/vec = 256 elements minimum).
        chunk_D = 256
        while chunk_D > spec.D or spec.D % chunk_D != 0:
            chunk_D //= 2
            if chunk_D < 32:
                chunk_D = spec.D  # fallback: one chunk covering all of D
                break
        for nw in (4, 8, 2, 1):
            if spec.D % (nw * 32) == 0 and spec.B % nw == 0:
                return cls(n_warps=nw, chunk_D=chunk_D, direct=True)
        return cls(n_warps=1, chunk_D=chunk_D, direct=True)
