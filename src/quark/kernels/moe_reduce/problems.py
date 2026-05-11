"""Canonical bench/fuzz problems for moe_reduce.

Shapes match the production model paths that exercise this kernel —
the ones the modified ``moe_outproj`` writes per-slot partials for.
"""

from __future__ import annotations

from quark.kernels.base import Problem

_METAL = {"metal", "metal-moe", "production"}
_CUDA = {"cuda", "cuda-moe", "production"}


def moe_reduce_problems() -> list[Problem]:
    # owl_360p / owl_720p: M=128/512, D=2048, E=16, top_k=4
    owl = {"D": 2048, "n_experts": 16, "top_k": 4}
    otg = {"owl", "moe"}
    # w15: M=128/512, D=2048, E=8, top_k=2
    w15 = {"D": 2048, "n_experts": 8, "top_k": 2}
    w15tg = {"w15", "moe"}

    # capacity sized to the worst-case BM-padded layout used by
    # ``moe_router_correct``: (M*top_k + E*31) rounded up to BM=32 per
    # expert. Equivalent to the value ``nn.MoE`` computes for routing="correct".
    def cap(M, E, K):
        worst_case = M * K + E * (32 - 1)
        cap_per_expert = (worst_case + E - 1) // E
        return ((cap_per_expert + 31) // 32) * 32

    return [
        Problem(
            "owl_360p",
            {"M": 128, **owl, "capacity": cap(128, 16, 4)},
            tags=otg | {"bf16"} | _METAL | _CUDA,
        ),
        Problem(
            "owl_720p",
            {"M": 512, **owl, "capacity": cap(512, 16, 4)},
            tags=otg | {"bf16"} | _METAL | _CUDA,
        ),
        Problem(
            "w15_360p",
            {"M": 128, **w15, "capacity": cap(128, 8, 2)},
            tags=w15tg | {"bf16"} | _METAL | _CUDA,
        ),
        Problem(
            "w15_720p",
            {"M": 512, **w15, "capacity": cap(512, 8, 2)},
            tags=w15tg | {"bf16"} | _METAL | _CUDA,
        ),
    ]
