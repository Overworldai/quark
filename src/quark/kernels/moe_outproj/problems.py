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
    ]
