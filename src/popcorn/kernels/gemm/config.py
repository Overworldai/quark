"""GemmConfig — tunable parameters for the universal GEMM.

The config controls the block tiling, warp count, pipeline depth,
and which mma shape the kernel uses. The autotune search walks the
cartesian product of `tune_space()` and picks the fastest valid
config per problem × device.
"""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.kernels.base import KernelConfig


@dataclass(frozen=True)
class GemmConfig(KernelConfig):
    """Tunable GEMM knobs.

    BM / BN / BK are the tile dimensions the proposal calls
    MTile / NTile / KChunk. n_stages is NStages (cp.async pipeline
    depth: 1 = synchronous, 2 = double buffer, 3 = triple).
    """

    BM: int = 64
    BN: int = 64
    BK: int = 16  # must be a multiple of the mma inner-K dimension
    n_warps: int = 4
    n_stages: int = 2
    a_pad: int = 0  # smem row padding for A (bank-conflict avoidance)
    b_pad: int = 0  # smem row padding for B
    # Per-MMA-site shape (MMA_SHAPES M3). Autotune fills this from
    # device.caps.matmul_shapes filtered by mma_sites(). Empty string
    # falls back to the kernel's compute-dtype default (k=16).
    main_shape: str = ""

    @classmethod
    def default_for(cls, spec) -> GemmConfig:
        # Pad granule follows compute dtype (16 for 1-byte fp8, 8 for
        # 2-byte bf16/fp16). n_stages=2 needs an even K-iter count.
        # Shape selection is now the autotuner's job via ``main_shape``
        # (MMA_SHAPES M3) — default_for leaves it empty; ``_mma_cfg``
        # resolves empty to the compute-dtype default.
        fp8 = spec.compute_dtype_resolved.bytes == 1
        pad = 16 if fp8 else 8
        n_stages = 2 if (spec.K % (2 * 32) == 0) else 1
        return cls(BM=64, BN=64, BK=32, n_warps=4, n_stages=n_stages, a_pad=pad)
