"""MoeRouterCorrectConfig — tunable parameters for moe_router_correct."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class MoeRouterCorrectConfig(KernelConfig):
    n_warps: int = 16

    @classmethod
    def default_for(cls, spec) -> MoeRouterCorrectConfig:
        for w in (4, 8, 16, 32):
            n_threads = w * 32
            if spec.M % n_threads == 0 and n_threads >= spec.M:
                return cls(n_warps=w)
        return cls(n_warps=32)
