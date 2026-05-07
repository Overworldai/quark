"""MoeReduceConfig — only knob is ``n_warps``.

One warp per output row; D split into ``vec`` chunks per lane. The
top_k axis is python-unrolled inside the loop body so each lane
issues ``vecs_per_lane × top_k`` vec_loads against the partials buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class MoeReduceConfig(KernelConfig):
    n_warps: int = 4

    @classmethod
    def default_for(cls, spec) -> MoeReduceConfig:
        # Pick the largest n_warps that cleanly partitions M.
        for nw in (8, 4, 2, 1):
            if spec.M % nw == 0:
                return cls(n_warps=nw)
        return cls(n_warps=1)
