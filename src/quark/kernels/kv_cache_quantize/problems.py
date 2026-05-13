"""KVQuantize bench/test problems."""

from __future__ import annotations

from quark.kernels.base import Problem


def kv_quantize_problems() -> list[Problem]:
    return [
        # Production Waypoint-1.5-1B 360p shape: K cache fully populated
        # = B * n_kv_heads * capacity tokens at Dh=64.
        # B=1, n_kv_heads=16, capacity=8704 → 139,264 tokens.
        Problem("waypoint_15_360p_K",
                {"num_tokens": 1 * 16 * 8704, "Dh": 64},
                tags={"production", "owl"}),
        Problem("smoke", {"num_tokens": 128, "Dh": 64}, tags={"smoke"}),
    ]
