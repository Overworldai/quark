"""Canonical bench/fuzz problems for moe_router_shared.

Same shapes as moe_router so the two routers can be compared on
matched configs.
"""

from __future__ import annotations

from quark.kernels.base import Problem


def moe_router_shared_problems() -> list[Problem]:
    moe = {"production", "moe", "cuda-moe"}
    return [
        Problem("w15_360p", {"M": 128, "E": 8, "top_k": 2, "capacity": 32}, tags=moe),
        Problem("w15_720p", {"M": 512, "E": 8, "top_k": 2, "capacity": 128}, tags=moe),
        Problem("w15_720p_e16k4", {"M": 512, "E": 16, "top_k": 4, "capacity": 128}, tags=moe),
    ]
