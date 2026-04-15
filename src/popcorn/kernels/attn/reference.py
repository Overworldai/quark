"""Backend-agnostic reference for the attn kernel."""

from __future__ import annotations

from popcorn.backend import PT


def attn_reference(kernel, Q, K, V_t):
    s = kernel.spec
    # Reshape to [B, H, L, Dh] layout PT.attention expects. V_t is
    # stored [B, Hkv, Dh, kv_len] — transpose the last two so V ends up
    # [B, Hkv, kv_len, Dh].
    Q_ = PT.astype(Q, PT.bfloat16).reshape(s.B, s.n_q_heads, s.seq_len, s.Dh)
    K_ = PT.astype(K, PT.bfloat16).reshape(s.B, s.n_kv_heads, s.kv_len, s.Dh)
    V_ = PT.transpose(PT.astype(V_t, PT.bfloat16).reshape(s.B, s.n_kv_heads, s.Dh, s.kv_len))
    y = PT.attention(Q_, K_, V_)
    return y.reshape(s.B * s.n_q_heads * s.seq_len, s.Dh)
