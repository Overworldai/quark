"""Bench/fuzz problems for AdaRMSNorm.

Waypoint-1.5 uses M=tokens_per_frame=512, G=1 per frame, D=2048.
"""

from __future__ import annotations

from popcorn.kernels.base import Problem


def ada_rmsnorm_problems() -> list[Problem]:
    bf16 = {"dtype": "bf16"}
    return [
        Problem("wp15_frame", {"G": 1, "M": 512, "D": 2048, **bf16}, tags={"smoke", "production"}),
        # Multi-frame prefill: 8 frames, each with its own (scale, bias).
        Problem("wp15_prefill8", {"G": 8, "M": 512, "D": 2048, **bf16}, tags={"production"}),
        # Tiny smoke.
        Problem("small", {"G": 2, "M": 4, "D": 128, **bf16}, tags={"small"}),
    ]
