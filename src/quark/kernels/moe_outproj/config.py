"""MoeOutprojConfig — tunable parameters for the MoE out-projection."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class MoeOutprojConfig(KernelConfig):
    BM: int = 32
    BN: int = 64
    BK: int = 32  # K chunk — ≥2 MMAs/iter amortizes cp.async commit/wait cost
    n_warps: int = 4
    n_stages: int = 2  # double-buffered cp.async K pipeline by default
    a_pad: int = 0
    b_pad: int = 0
    b_shuffle: bool = False  # preshuffled B weight layout
    # Per-site shape (MMA_SHAPES M3). Autotune fills from device caps;
    # "" falls back to the compute-dtype default (k=16).
    main_shape: str = ""

    @classmethod
    def default_for(cls, spec) -> MoeOutprojConfig:
        # n_stages=2 needs K // BK to split into at least 4 chunks so
        # the software-pipelined KLoop has something to drain in its
        # epilogue; fall back to 1 on degenerate shapes.
        n_stages = 2 if (spec.H // 32) >= 4 and (spec.H // 32) % 2 == 0 else 1
        return cls(BM=32, BN=64, BK=32, n_warps=4, n_stages=n_stages)
