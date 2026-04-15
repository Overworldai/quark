"""KVCacheUpdateConfig — tuning knobs for the KV cache write kernel."""

from __future__ import annotations

from dataclasses import dataclass

from popcorn.kernels.base import KernelConfig


@dataclass(frozen=True)
class KVCacheUpdateConfig(KernelConfig):
    # Tokens-per-frame slice handled per threadblock. Must divide tpf.
    tile_T: int = 64
    # Warps per threadblock. cp.async + RoPE + vectorized store is bandwidth-bound;
    # 4 warps (128 threads) is usually plenty for the elementwise traffic.
    n_warps: int = 4
    # Smem column padding to dodge bank conflicts on (Dh+pad) strides.
    smem_pad: int = 0

    @classmethod
    def default_for(cls, spec) -> KVCacheUpdateConfig:
        # tile_T must divide tpf AND satisfy the vec_store alignment:
        #   (tile_T * Dh // vec_elems) % n_threads == 0
        # vec_elems = 16 // kv_dtype.bytes  (8 for bf16, 16 for fp8).
        # For fp8 with n_warps=4 (128 threads) and Dh=64 this requires
        # tile_T >= 32 — tile_T=16 gives only 64 vecs for 128 threads.
        n_warps = 4
        n_threads = n_warps * 32
        kv_b = spec.kv_dtype.bytes
        vec_elems = 16 // kv_b
        for tile_T in [16, 32, 64, 128]:
            if spec.tpf % tile_T != 0:
                continue
            if (tile_T * spec.Dh // vec_elems) % n_threads == 0:
                return cls(tile_T=tile_T, n_warps=n_warps, smem_pad=0)
        # Unreachable for standard Dh/tpf values (power-of-2 geometry).
        return cls(tile_T=128, n_warps=n_warps, smem_pad=0)
