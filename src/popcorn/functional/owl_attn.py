"""``popcorn.functional.owl_attn`` — segment-sparse flash attention.

Signature:

    out = pcf.owl_attn(Q, K_cache, Vt_cache, cos, sin, segments, n_segments,
                       *, B, n_kv_heads, gqa_ratio,
                       H_spatial, W_spatial, num_buckets, pinned_dilation,
                       out_dtype=None, compute_dtype=None, max_segments=3)

The seven tensor inputs match the kernel's ``TENSORS`` declaration order;
the allocated output is shape ``[B * n_q_heads * tpf, Dh]`` in ``out_dtype``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from popcorn.backend import PT
from popcorn.functional._dispatch import call_with_bindings, make_autotune, torch_op
from popcorn.kernels import get

if TYPE_CHECKING:
    import torch

_OwlAttnCls = None


def _cls():
    global _OwlAttnCls
    if _OwlAttnCls is None:
        _OwlAttnCls = get("owl_attn")
    return _OwlAttnCls


def _owl_attn_impl(
    Q,
    K_cache,
    Vt_cache,
    cos,
    sin,
    segments,
    n_segments,
    *,
    B,
    n_kv_heads,
    gqa_ratio,
    H_spatial,
    W_spatial,
    num_buckets,
    pinned_dilation,
    out_dtype=None,
    compute_dtype=None,
    max_segments=3,
):
    cls = _cls()
    spec = cls.spec_from_tensors(
        Q,
        K_cache,
        Vt_cache,
        cos,
        sin,
        segments,
        n_segments,
        B=B,
        n_kv_heads=n_kv_heads,
        gqa_ratio=gqa_ratio,
        H_spatial=H_spatial,
        W_spatial=W_spatial,
        num_buckets=num_buckets,
        pinned_dilation=pinned_dilation,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        max_segments=max_segments,
    )
    result = call_with_bindings(
        cls,
        spec,
        provided={
            "Q": Q,
            "K_cache": K_cache,
            "Vt_cache": Vt_cache,
            "cos": cos,
            "sin": sin,
            "segments": segments,
            "n_segments": n_segments,
        },
        auto_alloc=("output",),
        like=Q,
    )
    return result["output"]


@torch_op("popcorn::owl_attn", mutates_args=())
def owl_attn(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    Vt_cache: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    segments: torch.Tensor,
    n_segments: torch.Tensor,
    B: int,
    n_kv_heads: int,
    gqa_ratio: int,
    H_spatial: int,
    W_spatial: int,
    num_buckets: int,
    pinned_dilation: int,
    out_dtype: Optional[str] = None,
    compute_dtype: Optional[str] = None,
    max_segments: int = 3,
) -> torch.Tensor:
    return _owl_attn_impl(
        Q,
        K_cache,
        Vt_cache,
        cos,
        sin,
        segments,
        n_segments,
        B=B,
        n_kv_heads=n_kv_heads,
        gqa_ratio=gqa_ratio,
        H_spatial=H_spatial,
        W_spatial=W_spatial,
        num_buckets=num_buckets,
        pinned_dilation=pinned_dilation,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        max_segments=max_segments,
    )


@owl_attn.register_fake
def _(
    Q,
    K_cache,
    Vt_cache,
    cos,
    sin,
    segments,
    n_segments,
    B,
    n_kv_heads,
    gqa_ratio,
    H_spatial,
    W_spatial,
    num_buckets,
    pinned_dilation,
    out_dtype=None,
    compute_dtype=None,
    max_segments=3,
):
    import torch

    cls = _cls()
    spec = cls.spec_from_tensors(
        Q,
        K_cache,
        Vt_cache,
        cos,
        sin,
        segments,
        n_segments,
        B=B,
        n_kv_heads=n_kv_heads,
        gqa_ratio=gqa_ratio,
        H_spatial=H_spatial,
        W_spatial=W_spatial,
        num_buckets=num_buckets,
        pinned_dilation=pinned_dilation,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        max_segments=max_segments,
    )
    cfg = cls.CONFIG_CLS.default_for(spec)
    out_decl = next(d for d in cls.TENSORS if d.name == "output")
    shape = out_decl.shape(spec, cfg)
    dtype = PT.ir_dtype_to_backend(out_decl.dtype(spec, cfg))
    return torch.empty(tuple(shape), dtype=dtype, device=Q.device)


@owl_attn.register_autograd
def _(ctx, grad_out):
    raise NotImplementedError("popcorn.functional.owl_attn: backward not implemented.")


owl_attn.autotune = make_autotune(_owl_attn_impl, _cls)
