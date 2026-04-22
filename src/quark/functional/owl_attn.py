"""``quark.functional.owl_attn`` — segment-sparse flash attention.

Signature:

    out = pcf.owl_attn(Q, K_cache, Vt_cache, cos, sin, segments, n_segments,
                       *, B, n_kv_heads, gqa_ratio,
                       H_spatial, W_spatial, num_buckets, pinned_dilation,
                       out_dtype=None, compute_dtype=None, max_segments=3)

The seven tensor inputs match the kernel's ``TENSORS`` declaration order;
the allocated output is shape ``[B * n_q_heads * tpf, Dh]`` in ``out_dtype``.
"""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

_OwlAttnCls = None


def _cls():
    global _OwlAttnCls
    if _OwlAttnCls is None:
        _OwlAttnCls = get("owl_attn")
    return _OwlAttnCls


_DUMMY_FRAME_T = None


def _owl_attn_impl(
    Q,
    K_cache,
    Vt_cache,
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
    packed_qkv=False,
    frame_t=None,
    out=None,
):
    cls = _cls()
    spec = cls.spec_from_tensors(
        Q,
        K_cache,
        Vt_cache,
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
        packed_qkv=packed_qkv,
    )
    if frame_t is None:
        # Cache a process-wide zero sentinel so callers that don't pass
        # an explicit frame_t don't pay a ``cuMemAllocAsync`` per call.
        global _DUMMY_FRAME_T
        if _DUMMY_FRAME_T is None:
            from quark.nn.module import _tensor

            _DUMMY_FRAME_T = _tensor([0], dtype="s32")
        frame_t = _DUMMY_FRAME_T
    provided = {
        "Q": Q,
        "K_cache": K_cache,
        "Vt_cache": Vt_cache,
        "segments": segments,
        "n_segments": n_segments,
        "frame_t": frame_t,
    }
    auto_alloc: tuple[str, ...] = ("output",)
    if out is not None:
        provided["output"] = out
        auto_alloc = ()
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=Q)
    return result["output"]


def owl_attn(
    Q,
    K_cache,
    Vt_cache,
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
    packed_qkv=False,
    frame_t=None,
    *,
    out=None,
):
    return _owl_attn_impl(
        Q,
        K_cache,
        Vt_cache,
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
        packed_qkv=packed_qkv,
        frame_t=frame_t,
        out=out,
    )


owl_attn.autotune = make_autotune(_owl_attn_impl, _cls)  # ty: ignore[unresolved-attribute]
