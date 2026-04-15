"""SDPA baselines for the attn kernel."""

from __future__ import annotations

from popcorn.backend import PT
from popcorn.kernels.base import Baseline


def attn_baselines(kernel, tensors: dict) -> list[Baseline]:
    s = kernel.spec
    Q = PT.astype(tensors["Q"], PT.bfloat16).reshape(s.B, s.n_q_heads, s.seq_len, s.Dh)
    K = PT.astype(tensors["K"], PT.bfloat16).reshape(s.B, s.n_kv_heads, s.kv_len, s.Dh)
    V = PT.transpose(
        PT.astype(tensors["V_t"], PT.bfloat16).reshape(s.B, s.n_kv_heads, s.Dh, s.kv_len)
    )

    def run():
        y = PT.attention(Q, K, V)
        # Force the computation on mlx — torch SDPA is eager.
        if PT._is_mx(y):
            import mlx.core as mx

            mx.eval(y)

    tag = "mx.fast.sdpa[bf16]" if PT._is_mx(Q) else "F.sdpa[bf16]"
    return [Baseline(tag, run)]
