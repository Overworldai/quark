"""AttnConfig — tunable parameters for flash attention.

Architecture matches the old owl_attn kernel:
  - GQA: gqa_ratio Q heads per KV head, handled within one block
  - NCW: consumer warps per GQA group (total warps = gqa_ratio * NCW)
  - MTiles: M-tiles per warp (each = 16 Q rows)
  - BlockQRows = NCW * MTiles * 16
  - KvTile: KV columns per pipeline chunk
  - KvPad: smem padding for bank conflict avoidance
  - Double-buffered cp.async KV pipeline (2 stages)
"""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.kernels.base import KernelConfig


@dataclass(frozen=True)
class AttnConfig(KernelConfig):
    KvTile: int = 64  # KV chunk size (32/64/128)
    MTiles: int = 2  # M-tiles per warp (1/2/4, each = 16 Q rows)
    NCW: int = 2  # consumer warps per GQA group
    KvPad: int = 8  # smem padding for K and V (bank conflict avoidance)
    # Per-site shape (MMA_SHAPES M3). Autotune fills from device caps;
    # "" falls back to the compute-dtype default (k=16).
    main_shape: str = ""

    @property
    def n_warps(self) -> int:
        """Total warps per block. Set by kernel from gqa_ratio * NCW."""
        # This is computed by the kernel, not stored here.
        # Kept as a property for compatibility with base.Kernel.block()
        raise AttributeError("Use gqa_ratio * NCW instead")

    @classmethod
    def default_for(cls, spec) -> AttnConfig:
        from popcorn.ir import DType

        is_fp8_compute = spec.compute_dtype_resolved in (DType.E4M3, DType.E5M2)
        return cls(
            KvTile=32,
            MTiles=1,
            NCW=1,
            KvPad=16 if is_fp8_compute else 8,
        )
