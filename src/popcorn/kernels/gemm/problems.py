"""Canonical bench/fuzz problems for the GEMM kernel.

Production workload = 32 Q heads × 16 KV heads × Dh 64 model.
Per layer the gemms are:

  QKV proj       M × 2048 → 4096   (shared by dense + MoE)
  attn out proj  M × 2048 → 2048   (shared by dense + MoE)
  FFN up/gate    M × 2048 → 8192   (dense only)
  FFN down       M × 8192 → 2048   (dense only)

Metal production stack is bf16 everywhere; CUDA production stack is
A=bf16, B=e4m3 preshuffled, compute=e4m3, out=bf16. Tags let bench
pick the exact model path::

    make bench TAG=metal-dense      # metal, dense FFN
    make bench TAG=cuda-moe         # cuda, MoE FFN (no FFN gemms here)
    make bench TAG=metal            # everything Metal-relevant
"""

from __future__ import annotations

from popcorn.kernels.base import Problem

# ── Dtype presets ────────────────────────────────────────────────────
_METAL = {"a_dtype": "bf16", "b_dtype": "bf16", "out_dtype": "bf16"}
_CUDA = {
    "a_dtype": "bf16",
    "b_dtype": "e4m3",
    "compute_dtype": "e4m3",
    "out_dtype": "bf16",
    "b_shuffle": True,
}

# ── Tag presets ──────────────────────────────────────────────────────
# Shared gemms (QKV, attn-out) appear in both dense and moe paths.
_METAL_SHARED = {"metal", "metal-dense", "metal-moe", "production"}
_CUDA_SHARED = {"cuda", "cuda-dense", "cuda-moe", "production"}
_METAL_DENSE = {"metal", "metal-dense", "production"}
_CUDA_DENSE = {"cuda", "cuda-dense", "production"}


def _prod(name: str, *, M: int, N: int, K: int, dtype: dict, tags: set) -> Problem:
    return Problem(name, {"M": M, "N": N, "K": K, **dtype}, tags=tags)


def gemm_problems() -> list[Problem]:
    mixed = {  # bf16 × e4m3 → e4m3 compute → bf16 out
        "a_dtype": "bf16",
        "b_dtype": "e4m3",
        "compute_dtype": "e4m3",
        "out_dtype": "bf16",
    }
    owl = {"N": 2048, "K": 2048, **mixed}
    owl_tags = {"owl", "production", "mixed"}
    shuf_tags = owl_tags | {"shuffle"}
    bf16_2k = {"a_dtype": "bf16", "b_dtype": "bf16", "out_dtype": "bf16"}
    return [
        # ── Legacy / regression problems ─────────────────────────────
        Problem("owl_360p", {"M": 128, **owl}, tags=owl_tags),
        Problem("owl_720p", {"M": 512, **owl}, tags=owl_tags),
        Problem("owl_360p_shuf", {"M": 128, **owl, "b_shuffle": True}, tags=shuf_tags),
        Problem("owl_720p_shuf", {"M": 512, **owl, "b_shuffle": True}, tags=shuf_tags),
        Problem(
            "owl_360p_bf16",
            {"M": 128, "N": 2048, "K": 2048, **bf16_2k},
            tags={"owl", "production", "bf16"},
        ),
        Problem(
            "owl_720p_bf16",
            {"M": 512, "N": 2048, "K": 2048, **bf16_2k},
            tags={"owl", "production", "bf16"},
        ),
        Problem(
            "2k_bf16",
            {"M": 2048, "N": 2048, "K": 2048, **bf16_2k},
            tags={"production", "bf16", "large"},
        ),
        Problem(
            "4k_bf16",
            {"M": 4096, "N": 4096, "K": 4096, **bf16_2k},
            tags={"production", "bf16", "large"},
        ),
        # SiLU-fused regression (MLP fc1 path for waypoint-1.5)
        Problem(
            "ffn_up_silu_360p",
            {"M": 128, "N": 8192, "K": 2048, **bf16_2k, "activation": "silu"},
            tags={"production", "bf16", "silu", "smoke"},
        ),
        Problem(
            "ffn_up_silu_720p",
            {"M": 512, "N": 8192, "K": 2048, **bf16_2k, "activation": "silu"},
            tags={"production", "bf16", "silu"},
        ),
        # ── Production model workload ────────────────────────────────
        # QKV fused projection (2048 → 4096). Shared across dense + moe.
        _prod("qkv_proj_360p_metal", M=128, N=4096, K=2048, dtype=_METAL, tags=_METAL_SHARED),
        _prod("qkv_proj_720p_metal", M=512, N=4096, K=2048, dtype=_METAL, tags=_METAL_SHARED),
        _prod("qkv_proj_360p_cuda", M=128, N=4096, K=2048, dtype=_CUDA, tags=_CUDA_SHARED),
        _prod("qkv_proj_720p_cuda", M=512, N=4096, K=2048, dtype=_CUDA, tags=_CUDA_SHARED),
        # Attn output projection (2048 → 2048). Shared across dense + moe.
        _prod("attn_out_360p_metal", M=128, N=2048, K=2048, dtype=_METAL, tags=_METAL_SHARED),
        _prod("attn_out_720p_metal", M=512, N=2048, K=2048, dtype=_METAL, tags=_METAL_SHARED),
        _prod("attn_out_360p_cuda", M=128, N=2048, K=2048, dtype=_CUDA, tags=_CUDA_SHARED),
        _prod("attn_out_720p_cuda", M=512, N=2048, K=2048, dtype=_CUDA, tags=_CUDA_SHARED),
        # Dense FFN up / gate (2048 → 8192). One shape; fire twice per
        # layer in production.
        _prod("ffn_up_360p_metal", M=128, N=8192, K=2048, dtype=_METAL, tags=_METAL_DENSE),
        _prod("ffn_up_720p_metal", M=512, N=8192, K=2048, dtype=_METAL, tags=_METAL_DENSE),
        _prod("ffn_up_360p_cuda", M=128, N=8192, K=2048, dtype=_CUDA, tags=_CUDA_DENSE),
        _prod("ffn_up_720p_cuda", M=512, N=8192, K=2048, dtype=_CUDA, tags=_CUDA_DENSE),
        # Dense FFN down (8192 → 2048).
        _prod("ffn_down_360p_metal", M=128, N=2048, K=8192, dtype=_METAL, tags=_METAL_DENSE),
        _prod("ffn_down_720p_metal", M=512, N=2048, K=8192, dtype=_METAL, tags=_METAL_DENSE),
        _prod("ffn_down_360p_cuda", M=128, N=2048, K=8192, dtype=_CUDA, tags=_CUDA_DENSE),
        _prod("ffn_down_720p_cuda", M=512, N=2048, K=8192, dtype=_CUDA, tags=_CUDA_DENSE),
    ]
