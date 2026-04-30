"""Canonical bench/fuzz problems for moe_router_correct."""

from __future__ import annotations

from quark.kernels.base import Problem


def moe_router_correct_problems() -> list[Problem]:
    moe = {"production", "moe", "cuda-moe"}
    # capacity is the per-expert slot count; total_slots = E*capacity must
    # cover the worst case M*K + E*(BM-1).
    return [
        # Waypoint15 360p, K=4, E=16: M*K=512, worst-case 992, capacity=64.
        Problem(
            "w15_360p_e16k4",
            {"M": 128, "E": 16, "top_k": 4, "capacity": 64},
            tags=moe,
        ),
        # 720p, K=4, E=16: M*K=2048, worst-case 2528, capacity=160 (round to 32).
        Problem(
            "w15_720p_e16k4",
            {"M": 512, "E": 16, "top_k": 4, "capacity": 160},
            tags=moe,
        ),
        # Smaller top_k=2, E=8: M*K=256, worst-case 504, capacity=64.
        Problem(
            "w15_360p_e8k2",
            {"M": 128, "E": 8, "top_k": 2, "capacity": 64},
            tags=moe,
        ),
    ]
