"""GemmIntConfig — tunable knobs for the int8 GEMM."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class GemmIntConfig(KernelConfig):
    """int8 GEMM tile + warp + stage knobs.

    Defaults target a single-warp, single-MMA correctness-first
    config. Production tuning will widen BM/BN/n_warps once the
    end-to-end pipeline is validated.

    ``main_shape`` is FIXED to ``m8n16k32_intel_s8_s32`` for the
    int8 path — the only KHR coopmat shape Battlemage exposes for
    s8/s8/s32 MMA.
    """

    BM: int = 8       # = mma.shape.m (single m-tile, m8 native)
    BN: int = 16      # = mma.shape.n (single n-tile, n16 native)
    BK: int = 32      # = mma.shape.k (single k-tile, k32 native)
    n_warps: int = 1  # single warp first; expand once core path works
    n_stages: int = 1 # no double-buffer initially
    main_shape: str = "m8n16k32_intel_s8_s32"
