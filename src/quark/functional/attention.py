"""``quark.functional.attention`` — torch/MLX-native flash attention.

Signature:

    out = pcf.attention(Q, K, V_t, *, B, n_kv_heads, gqa_ratio,
                        seq_len, kv_len)

Q: ``[B*n_q_heads*seq_len, Dh]``. K: ``[B*n_kv_heads*kv_len, Dh]``.
V_t: ``[B*n_kv_heads*Dh, kv_len]`` (V transposed for fast-axis kv_len).
Returns out: same shape/dtype as Q.
"""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

_AttnCls = None


def _cls():
    global _AttnCls
    if _AttnCls is None:
        _AttnCls = get("attn")
    return _AttnCls


def _attn_impl(Q, K, V_t, *, B, n_kv_heads, gqa_ratio, seq_len, kv_len):
    cls = _cls()
    spec = cls.spec_from_tensors(
        Q,
        K,
        V_t,
        B=B,
        n_kv_heads=n_kv_heads,
        gqa_ratio=gqa_ratio,
        seq_len=seq_len,
        kv_len=kv_len,
    )
    result = call_with_bindings(
        cls,
        spec,
        provided={"Q": Q, "K": K, "V_t": V_t},
        auto_alloc=("output",),
        like=Q,
    )
    return result["output"]


def attention(Q, K, V_t, B, n_kv_heads, gqa_ratio, seq_len, kv_len):
    return _attn_impl(
        Q,
        K,
        V_t,
        B=B,
        n_kv_heads=n_kv_heads,
        gqa_ratio=gqa_ratio,
        seq_len=seq_len,
        kv_len=kv_len,
    )


attention.autotune = make_autotune(_attn_impl, _cls)  # ty: ignore[unresolved-attribute]
