"""``popcorn.functional.attention`` — torch/MLX-native flash attention.

Signature:

    out = pcf.attention(Q, K, V_t, *, B, n_kv_heads, gqa_ratio,
                        seq_len, kv_len)

Q: ``[B*n_q_heads*seq_len, Dh]``. K: ``[B*n_kv_heads*kv_len, Dh]``.
V_t: ``[B*n_kv_heads*Dh, kv_len]`` (V transposed for fast-axis kv_len).
Returns out: same shape/dtype as Q.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from popcorn.backend import PT
from popcorn.functional._dispatch import call_with_bindings, make_autotune, torch_op
from popcorn.kernels import get

if TYPE_CHECKING:
    import torch

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


@torch_op("popcorn::attention", mutates_args=())
def attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V_t: torch.Tensor,
    B: int,
    n_kv_heads: int,
    gqa_ratio: int,
    seq_len: int,
    kv_len: int,
) -> torch.Tensor:
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


@attention.register_fake
def _(Q, K, V_t, B, n_kv_heads, gqa_ratio, seq_len, kv_len):
    import torch

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
    cfg = cls.CONFIG_CLS.default_for(spec)
    out_decl = next(d for d in cls.TENSORS if d.name == "output")
    shape = out_decl.shape(spec, cfg)
    dtype = PT.ir_dtype_to_backend(out_decl.dtype(spec, cfg))
    return torch.empty(tuple(shape), dtype=dtype, device=Q.device)


@attention.register_autograd
def _(ctx, grad_out):
    raise NotImplementedError("popcorn.functional.attention: backward not implemented.")


attention.autotune = make_autotune(_attn_impl, _cls)
