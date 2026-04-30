"""MoeRouterSharedConfig — tunable parameters for moe_router_shared."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class MoeRouterSharedConfig(KernelConfig):
    # Single-block kernel; n_warps controls thread count for cooperative
    # phases (per-token softmax accumulation, slot writes, work_list build).
    n_warps: int = 16

    @classmethod
    def default_for(cls, spec) -> MoeRouterSharedConfig:
        for w in (4, 8, 16, 32):
            n_threads = w * 32
            if spec.M % n_threads == 0 and n_threads >= spec.M:
                return cls(n_warps=w)
        return cls(n_warps=32)
