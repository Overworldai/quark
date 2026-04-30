"""Canonical bench/fuzz problems for the moe_outproj kernel.

Shapes match the production model (D=2048, H=2048, top_k=4,
n_experts=16). Only the MoE model path exercises this kernel, so the
tags are ``metal-moe`` / ``cuda-moe`` — never the ``*-dense`` variants.
"""

from __future__ import annotations

from quark.kernels.base import Problem

_METAL = {"metal", "metal-moe", "production"}
_CUDA = {"cuda", "cuda-moe", "production"}


def moe_outproj_problems() -> list[Problem]:
    c_e4m3 = {
        "a_dtype": "bf16",
        "b_dtype": "e4m3",
        "compute_dtype": "e4m3",
        "out_dtype": "bf16",
    }
    base = {"D": 2048, "H": 2048, "n_experts": 16, "top_k": 4}
    otg = {"owl", "moe"}
    shuf = {"b_shuffle": True}
    # W1.5 MoE shapes: E=8, top_k=2, H = mlp_ratio*D/top_k = 4096.
    w15 = {"D": 2048, "H": 4096, "n_experts": 8, "top_k": 2}
    w15tg = {"w15", "moe"}
    return [
        # bf16 — Metal production path.
        Problem("owl_360p", {"M": 128, **base}, tags=otg | {"bf16"} | _METAL),
        Problem("owl_720p", {"M": 512, **base}, tags=otg | {"bf16"} | _METAL),
        Problem(
            "owl_360p_shuf",
            {"M": 128, **base},
            tags=otg | {"bf16"},
            config_overrides=shuf,
        ),
        Problem(
            "owl_720p_shuf",
            {"M": 512, **base},
            tags=otg | {"bf16"},
            config_overrides=shuf,
        ),
        # fp8 compute, shuffle off — regression gate for the scalar path.
        Problem("owl_360p_c_e4m3", {"M": 128, **base, **c_e4m3}, tags=otg | {"mixed"}),
        Problem("owl_720p_c_e4m3", {"M": 512, **base, **c_e4m3}, tags=otg | {"mixed"}),
        # fp8 compute + preshuffled B — CUDA production path.
        Problem(
            "owl_360p_prod_cuda",
            {"M": 128, **base, **c_e4m3},
            tags=otg | {"mixed"} | _CUDA,
            config_overrides=shuf,
        ),
        Problem(
            "owl_720p_prod_cuda",
            {"M": 512, **base, **c_e4m3},
            tags=otg | {"mixed"} | _CUDA,
            config_overrides=shuf,
        ),
        # W1.5 MoE — bf16-only path (moe_inproj fused-silu epilogue
        # can't store fp8 yet, so the whole MoE block stays bf16
        # regardless of cfg.use_fp8).
        Problem("w15_360p", {"M": 128, **w15}, tags=w15tg | {"bf16"} | _CUDA),
        Problem("w15_720p", {"M": 512, **w15}, tags=w15tg | {"bf16"} | _CUDA),
    ]
