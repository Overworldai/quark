"""Canonical bench/fuzz problems for the owl_attn kernel.

Production model: 32 Q heads × 16 KV heads, Dh=64. Attention compute
stays bf16 on both backends; KV cache is bf16 on both as well (the
earlier fp8 KV cache CUDA variants were removed — owl_attn CUDA uses
the same bf16 KV path as Metal). owl_attn runs on every transformer
layer so problems tag into both ``*-dense`` and ``*-moe`` model paths.
"""

from __future__ import annotations

from quark.kernels.base import Problem

# owl_attn runs every layer in both model paths on both backends.
_PROD = {
    "metal",
    "metal-dense",
    "metal-moe",
    "cuda",
    "cuda-dense",
    "cuda-moe",
    "production",
}


def owl_attn_problems() -> list[Problem]:
    common = {"B": 1, "n_kv_heads": 16, "gqa_ratio": 2, "Dh": 64}
    return [
        Problem(
            "owl_360p_dense",
            {**common, "H_spatial": 16, "W_spatial": 8, "num_buckets": 16, "pinned_dilation": 1},
            tags={"owl", "dense", "360p", "bf16"} | _PROD,
        ),
        Problem(
            "owl_360p_dilated",
            {**common, "H_spatial": 16, "W_spatial": 8, "num_buckets": 16, "pinned_dilation": 8},
            tags={"owl", "dilated", "360p", "bf16"} | _PROD,
        ),
        Problem(
            "owl_720p_dense",
            {**common, "H_spatial": 32, "W_spatial": 16, "num_buckets": 16, "pinned_dilation": 1},
            tags={"owl", "dense", "720p", "bf16"} | _PROD,
        ),
        Problem(
            "owl_720p_dilated",
            {**common, "H_spatial": 32, "W_spatial": 16, "num_buckets": 16, "pinned_dilation": 8},
            tags={"owl", "dilated", "720p", "bf16"} | _PROD,
        ),
    ]
