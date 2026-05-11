"""MoeRouterConfig — tunable parameters for the moe_router kernel."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class MoeRouterConfig(KernelConfig):
    # Single-block kernel; n_warps controls the thread count for the
    # cooperative zero-init pass and the per-token routing pass.
    # Each thread handles ``M / n_threads`` tokens (must divide cleanly).
    n_warps: int = 16

    @classmethod
    def default_for(cls, spec) -> MoeRouterConfig:
        # Pick the smallest n_warps that gives one thread per token (so
        # the common W1.5 case with M=512 lands at exactly n_warps=16
        # and avoids any per-thread token loop).
        for w in (4, 8, 16, 32):
            n_threads = w * 32
            if spec.M % n_threads == 0 and n_threads >= spec.M:
                return cls(n_warps=w)
        return cls(n_warps=32)
