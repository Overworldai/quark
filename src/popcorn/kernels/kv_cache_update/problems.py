"""Canonical bench/fuzz problems for the kv_cache_update kernel.

Runs before owl_attn on every layer, so every production model path
(metal-dense / metal-moe / cuda-dense / cuda-moe) includes it. Metal
keeps the KV cache in bf16; CUDA stores it in fp8 for bandwidth.
"""

from __future__ import annotations

from popcorn.kernels.base import Problem

# kv_cache_update runs every layer in both model paths.
_METAL = {"metal", "metal-dense", "metal-moe", "production"}
_CUDA = {"cuda", "cuda-dense", "cuda-moe", "production"}


def kv_cache_update_problems() -> list[Problem]:
    common = {"B": 1, "n_kv_heads": 16, "Dh": 64}
    return [
        Problem(
            "smoke_dense",
            {
                **common,
                "n_kv_heads": 2,
                "H_spatial": 8,
                "W_spatial": 8,
                "num_buckets": 4,
                "pinned_dilation": 1,
            },
            tags={"smoke", "small", "dense"},
        ),
        # bf16 KV cache — Metal production path.
        Problem(
            "owl_360p_dense",
            {**common, "H_spatial": 16, "W_spatial": 8, "num_buckets": 16, "pinned_dilation": 1},
            tags={"owl", "dense", "360p", "bf16"} | _METAL,
        ),
        Problem(
            "owl_360p_dilated",
            {**common, "H_spatial": 16, "W_spatial": 8, "num_buckets": 16, "pinned_dilation": 8},
            tags={"owl", "dilated", "360p", "bf16"} | _METAL,
        ),
        Problem(
            "owl_720p_dense",
            {**common, "H_spatial": 32, "W_spatial": 16, "num_buckets": 16, "pinned_dilation": 1},
            tags={"owl", "dense", "720p", "bf16"} | _METAL,
        ),
        Problem(
            "owl_720p_dilated",
            {**common, "H_spatial": 32, "W_spatial": 16, "num_buckets": 16, "pinned_dilation": 8},
            tags={"owl", "dilated", "720p", "bf16"} | _METAL,
        ),
        # fp8 KV cache — CUDA production path.
        Problem(
            "owl_360p_dense_fp8",
            {
                **common,
                "H_spatial": 16,
                "W_spatial": 8,
                "num_buckets": 16,
                "pinned_dilation": 1,
                "kv_dtype": "e4m3",
            },
            tags={"owl", "dense", "360p", "fp8"} | _CUDA,
        ),
        Problem(
            "owl_360p_dilated_fp8",
            {
                **common,
                "H_spatial": 16,
                "W_spatial": 8,
                "num_buckets": 16,
                "pinned_dilation": 8,
                "kv_dtype": "e4m3",
            },
            tags={"owl", "dilated", "360p", "fp8"} | _CUDA,
        ),
        Problem(
            "owl_720p_dense_fp8",
            {
                **common,
                "H_spatial": 32,
                "W_spatial": 16,
                "num_buckets": 16,
                "pinned_dilation": 1,
                "kv_dtype": "e4m3",
            },
            tags={"owl", "dense", "720p", "fp8"} | _CUDA,
        ),
        Problem(
            "owl_720p_dilated_fp8",
            {
                **common,
                "H_spatial": 32,
                "W_spatial": 16,
                "num_buckets": 16,
                "pinned_dilation": 8,
                "kv_dtype": "e4m3",
            },
            tags={"owl", "dilated", "720p", "fp8"} | _CUDA,
        ),
    ]
