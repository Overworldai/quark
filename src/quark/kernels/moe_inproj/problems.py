"""Canonical bench/fuzz problems for the moe_inproj kernel.

Shapes already match the production model (D=2048, H=2048, top_k=4,
n_experts=16). Only the MoE model path exercises this kernel, so the
tags are ``metal-moe`` / ``cuda-moe`` — never the ``*-dense`` variants.
"""

from __future__ import annotations

from quark.kernels.base import Problem

# MoE path only.
_METAL = {"metal", "metal-moe", "production"}
_CUDA = {"cuda", "cuda-moe", "production"}


def moe_inproj_problems() -> list[Problem]:
    bf16 = {"a_dtype": "bf16", "b_dtype": "bf16", "out_dtype": "bf16"}
    c_e4m3 = {
        "a_dtype": "bf16",
        "b_dtype": "e4m3",
        "compute_dtype": "e4m3",
        "out_dtype": "bf16",
    }
    owl = {"D": 2048, "H": 2048, "n_experts": 16, "top_k": 4, **bf16}
    owl_e4m3 = {"D": 2048, "H": 2048, "n_experts": 16, "top_k": 4, **c_e4m3}
    shuf = {"b_shuffle": True}
    otg = {"owl", "moe"}
    return [
        # bf16 — Metal production path (and a CUDA bf16 regression gate).
        Problem("owl_360p", {"M": 128, **owl}, tags=otg | {"bf16"} | _METAL),
        Problem("owl_720p", {"M": 512, **owl}, tags=otg | {"bf16"} | _METAL),
        Problem("owl_360p_shuf", {"M": 128, **owl}, tags=otg | {"bf16"}, config_overrides=shuf),
        Problem("owl_720p_shuf", {"M": 512, **owl}, tags=otg | {"bf16"}, config_overrides=shuf),
        # fp8 compute — shuffle off (regression gate for the scalar path).
        Problem("owl_360p_c_e4m3", {"M": 128, **owl_e4m3}, tags=otg | {"mixed"}),
        Problem("owl_720p_c_e4m3", {"M": 512, **owl_e4m3}, tags=otg | {"mixed"}),
        # fp8 compute + preshuffled B — CUDA production path.
        Problem(
            "owl_360p_prod_cuda",
            {"M": 128, **owl_e4m3},
            tags=otg | {"mixed"} | _CUDA,
            config_overrides=shuf,
        ),
        Problem(
            "owl_720p_prod_cuda",
            {"M": 512, **owl_e4m3},
            tags=otg | {"mixed"} | _CUDA,
            config_overrides=shuf,
        ),
    ]
