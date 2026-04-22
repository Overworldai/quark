"""RMSNormConfig — n_warps is the only knob.

One block per row; each block's ``n_warps * 32`` threads cooperate
on a single D-element reduction + rescale. ``D`` must be divisible
by ``n_warps * 32`` (the kernel's ``is_valid`` enforces this).
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class RMSNormConfig(KernelConfig):
    n_warps: int = 4

    @classmethod
    def default_for(cls, spec) -> RMSNormConfig:
        # Pick the smallest n_warps that cleanly partitions D so the
        # kernel is valid out of the box. Autotune will pick the best.
        for nw in (4, 8, 2, 1):
            if spec.D % (nw * 32) == 0:
                return cls(n_warps=nw)
        return cls(n_warps=4)
