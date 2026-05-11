"""HeadRMSNorm problems."""

from __future__ import annotations

from quark.kernels.base import Problem


def head_rmsnorm_problems() -> list[Problem]:
    return [
        Problem(
            "wp15_360p",
            {"M": 128, "D_full": 4096, "n_q_heads": 32, "n_kv_heads": 16, "Dh": 64},
            tags={"smoke", "production"},
        ),
    ]
