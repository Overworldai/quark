"""Bench/fuzz problems for the RMSNorm kernel.

Waypoint-1.5 production sizes:
- d_model = 2048
- tokens_per_frame = 512 (one frame at a time)
- prefill flows up to ~8k rows during warm-start

RMSNorm is called ~5× per transformer block (pre-attn, pre-MLP, on
Q, on K, and inside ada_rmsnorm) × 24 layers × per-denoise-step.
"""

from __future__ import annotations

from quark.kernels.base import Problem


def rmsnorm_problems() -> list[Problem]:
    bf16 = {"dtype": "bf16"}
    return [
        # Per-frame decode: one transformer forward on 512 tokens.
        Problem("wp15_frame", {"B": 512, "D": 2048, **bf16}, tags={"smoke", "production"}),
        # Single-token decode (rare in this engine but the lower bound).
        Problem("wp15_single", {"B": 1, "D": 2048, **bf16}, tags={"production"}),
        # Prefill-sized: 16 frames × 512 tpf.
        Problem("wp15_prefill", {"B": 8192, "D": 2048, **bf16}, tags={"production"}),
        # Head-wise Q/K pre-RoPE: 32 heads × 64 Dh as a smaller D sweep.
        Problem("head_64", {"B": 512 * 32, "D": 64, **bf16}, tags={"small"}),
    ]
