"""``popcorn.functional.kv_cache_update`` — RoPE + ring KV-cache write."""

from __future__ import annotations

from popcorn.functional._dispatch import call_with_bindings, make_autotune
from popcorn.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("kv_cache_update")
    return _Cls


def _impl(
    K,
    V,
    frame_t,
    frozen,
    Vt_cache,
    segments,
    n_segments,
    K_cache,
    *,
    B,
    n_kv_heads,
    H_spatial,
    W_spatial,
    num_buckets,
    pinned_dilation,
    kv_dtype=None,
    max_segments=3,
    packed_qkv=False,
    n_q_heads=0,
    rope_n_frames=1,
):
    cls = _cls()
    spec = cls.spec_from_tensors(
        K,
        V,
        frame_t,
        frozen,
        Vt_cache,
        segments,
        n_segments,
        K_cache,
        B=B,
        n_kv_heads=n_kv_heads,
        H_spatial=H_spatial,
        W_spatial=W_spatial,
        num_buckets=num_buckets,
        pinned_dilation=pinned_dilation,
        kv_dtype=kv_dtype,
        max_segments=max_segments,
        packed_qkv=packed_qkv,
        n_q_heads=n_q_heads,
        rope_n_frames=rope_n_frames,
    )
    result = call_with_bindings(
        cls,
        spec,
        provided={
            "K": K,
            "V": V,
            "frame_t": frame_t,
            "frozen": frozen,
            "Vt_cache": Vt_cache,
            "segments": segments,
            "n_segments": n_segments,
            "K_cache": K_cache,
        },
        auto_alloc=(),
        like=K,
    )
    return result["K_cache"], result["Vt_cache"], result["segments"], result["n_segments"]


def kv_cache_update(
    K,
    V,
    frame_t,
    frozen,
    Vt_cache,
    segments,
    n_segments,
    K_cache,
    *,
    B,
    n_kv_heads,
    H_spatial,
    W_spatial,
    num_buckets,
    pinned_dilation,
    kv_dtype=None,
    max_segments=3,
    packed_qkv=False,
    n_q_heads=0,
    rope_n_frames=1,
):
    """In-place RoPE + KV-cache update. frozen=1 skips ring writes."""
    return _impl(
        K,
        V,
        frame_t,
        frozen,
        Vt_cache,
        segments,
        n_segments,
        K_cache,
        B=B,
        n_kv_heads=n_kv_heads,
        H_spatial=H_spatial,
        W_spatial=W_spatial,
        num_buckets=num_buckets,
        pinned_dilation=pinned_dilation,
        kv_dtype=kv_dtype,
        max_segments=max_segments,
        packed_qkv=packed_qkv,
        n_q_heads=n_q_heads,
        rope_n_frames=rope_n_frames,
    )


kv_cache_update.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
