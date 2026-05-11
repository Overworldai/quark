"""Canonical bench/fuzz problems for moe_router.

Shapes match the W1.5 production MoE setup
(D=2048, E=8, top_k=2 → C=128). The router itself is dtype-agnostic
on its outputs (always emits int32 token_ids + f32 slot_weights), so
unlike moe_inproj/outproj there's no bf16/e4m3 axis here.
"""

from __future__ import annotations

from quark.kernels.base import Problem


def moe_router_problems() -> list[Problem]:
    moe = {"production", "moe", "cuda-moe"}
    # Capacity = ceil(M * top_k / E) rounded up to BM (32). For these
    # shapes M*top_k is already a clean multiple of E*BM.
    return [
        # 360p frame, top_k=2: M*top_k = 256, E*C = 8*32 = 256.
        Problem("w15_360p", {"M": 128, "E": 8, "top_k": 2, "capacity": 32}, tags=moe),
        # 720p frame, top_k=2: M*top_k = 1024, E*C = 8*128 = 1024.
        Problem("w15_720p", {"M": 512, "E": 8, "top_k": 2, "capacity": 128}, tags=moe),
        # Larger E, top_k=4 — exercises wider top-K selection.
        Problem("w15_720p_e16k4", {"M": 512, "E": 16, "top_k": 4, "capacity": 128}, tags=moe),
    ]
