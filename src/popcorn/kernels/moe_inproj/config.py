"""MoeInprojConfig — tunable parameters for the MoE in-projection."""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.kernels.base import KernelConfig


@dataclass(frozen=True)
class MoeInprojConfig(KernelConfig):
    BM: int = 32  # block rows (slots per work item)
    BN: int = 64  # block cols along H
    BK: int = 32  # K chunk — ≥2 MMAs/iter amortizes cp.async commit/wait cost
    n_warps: int = 4
    n_stages: int = 2  # double-buffered cp.async K pipeline by default
    a_pad: int = 0  # smem row padding for A (bank conflicts)
    b_pad: int = 0  # smem row padding for B
    b_shuffle: bool = False  # preshuffled B weight layout
    vec_epilogue: bool = True  # staged smem → vectorized gmem epilogue
    # Per-site shape (MMA_SHAPES M3). Autotune fills from the device's
    # legal shape set; "" falls back to compute-dtype default (k=16).
    main_shape: str = ""

    @classmethod
    def default_for(cls, spec) -> MoeInprojConfig:
        # Scalar epilogue (`vec_epilogue=False`) is the universal path;
        # autotune picks `True` when the backend supports it.
        return cls(BM=32, BN=64, BK=32, n_warps=4, n_stages=2, vec_epilogue=False)
