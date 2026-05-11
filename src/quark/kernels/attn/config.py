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

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class AttnConfig(KernelConfig):
    KvTile: int = 64  # KV chunk size; 8 is the SIMD-friendly minimum
    MTiles: int = 2  # M-tiles per warp (1/2/4, each = 16 Q rows)
    NCW: int = 2  # consumer warps per GQA group
    KvPad: int = 8  # smem padding for K and V (bank conflict avoidance)
    # Pipeline depth across the KV loop. 1 = synchronous (load, wait,
    # compute); 2 = software double buffer (prefetch chunk N+1 while
    # computing chunk N). Small KvTile autotune winners typically pair
    # with n_stages=2 — more iterations give double-buffering more room
    # to hide latency.
    n_stages: int = 1
    # Per-site shape knob (MMA_SHAPES M3). Autotune fills this from the
    # device's legal shapes filtered by mma_sites. "" falls back to
    # lookup_mma(a, b) — the legacy fixed-shape path.
    main_shape: str = ""

    @property
    def n_warps(self) -> int:
        """Total warps per block. Set by kernel from gqa_ratio * NCW."""
        # This is computed by the kernel, not stored here.
        # Kept as a property for compatibility with base.Kernel.block()
        raise AttributeError("Use gqa_ratio * NCW instead")

    @classmethod
    def default_for(cls, spec) -> AttnConfig:
        # Small default so the fallback fits Metal's 32KB smem and stays
        # under the MSL JIT's per-function ceilings.
        return cls(KvTile=32, MTiles=1, NCW=1, KvPad=8)
