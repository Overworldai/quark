"""Canonical bench/fuzz problems for the attn kernel."""

from __future__ import annotations

from popcorn.kernels.base import Problem


def attn_problems() -> list[Problem]:
    owl_common = {"B": 1, "n_kv_heads": 16, "gqa_ratio": 2, "Dh": 64}
    return [
        Problem(
            "small_4h",
            {"B": 1, "n_kv_heads": 2, "gqa_ratio": 2, "seq_len": 64, "kv_len": 64, "Dh": 64},
            tags={"smoke", "small", "bf16"},
        ),
        Problem(
            "owl_360p",
            {**owl_common, "seq_len": 128, "kv_len": 2176},
            tags={"owl", "production", "bf16"},
        ),
        Problem(
            "owl_720p",
            {**owl_common, "seq_len": 512, "kv_len": 8704},
            tags={"owl", "production", "bf16"},
        ),
    ]
