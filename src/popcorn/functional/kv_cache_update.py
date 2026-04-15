"""``popcorn.functional.kv_cache_update`` — RoPE + ring KV-cache write.

Signature:

    K_cache, Vt_cache, segments, n_segments = pcf.kv_cache_update(
        K, V, cos, sin, frame_t, Vt_cache, segments, n_segments, K_cache,
        *, B, n_kv_heads, H_spatial, W_spatial, num_buckets, pinned_dilation,
        kv_dtype=None, max_segments=3,
    )

All four role="out" tensors (``Vt_cache``, ``segments``, ``n_segments``,
``K_cache``) are in-out: caller provides the pre-existing state, the
kernel partially writes it. On torch the returned tuple elements ARE
the same tensors the caller passed (mutated in place); on MLX they are
fresh ``mx.array`` instances (arrays are immutable).

Mutating API — registered as a ``torch.library.custom_op`` with
``mutates_args`` covering the four in-out tensors. The torch custom
op must return ``None`` (the dispatcher forbids returning aliased
inputs as outputs); the public ``kv_cache_update(...)`` wrapper
calls it for the side effect and returns the original tensor refs.
On Metal, MLX arrays are immutable, so the wrapper instead returns
``_impl``'s freshly-constructed result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from popcorn.backend import IS_METAL
from popcorn.functional._dispatch import call_with_bindings, make_autotune, torch_op
from popcorn.kernels import get

if TYPE_CHECKING:
    import torch

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("kv_cache_update")
    return _Cls


def _impl(
    K,
    V,
    cos,
    sin,
    frame_t,
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
):
    cls = _cls()
    spec = cls.spec_from_tensors(
        K,
        V,
        cos,
        sin,
        frame_t,
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
    )
    result = call_with_bindings(
        cls,
        spec,
        provided={
            "K": K,
            "V": V,
            "cos": cos,
            "sin": sin,
            "frame_t": frame_t,
            "Vt_cache": Vt_cache,
            "segments": segments,
            "n_segments": n_segments,
            "K_cache": K_cache,
        },
        auto_alloc=(),
        like=K,
    )
    return (
        result["K_cache"],
        result["Vt_cache"],
        result["segments"],
        result["n_segments"],
    )


@torch_op(
    "popcorn::kv_cache_update",
    mutates_args=("Vt_cache", "segments", "n_segments", "K_cache"),
)
def _kv_cache_update_op(
    K: torch.Tensor,
    V: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    frame_t: torch.Tensor,
    Vt_cache: torch.Tensor,
    segments: torch.Tensor,
    n_segments: torch.Tensor,
    K_cache: torch.Tensor,
    B: int,
    n_kv_heads: int,
    H_spatial: int,
    W_spatial: int,
    num_buckets: int,
    pinned_dilation: int,
    kv_dtype: Optional[str] = None,
    max_segments: int = 3,
) -> None:
    # mutates_args ops MUST return None — torch.library forbids
    # returning aliased inputs as outputs. The wrapper below returns
    # the original tensor refs after the in-place update.
    _impl(
        K,
        V,
        cos,
        sin,
        frame_t,
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
    )


@_kv_cache_update_op.register_fake
def _(
    K,
    V,
    cos,
    sin,
    frame_t,
    Vt_cache,
    segments,
    n_segments,
    K_cache,
    B,
    n_kv_heads,
    H_spatial,
    W_spatial,
    num_buckets,
    pinned_dilation,
    kv_dtype=None,
    max_segments=3,
):
    return None


def kv_cache_update(
    K,
    V,
    cos,
    sin,
    frame_t,
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
):
    """In-place RoPE + KV-cache update. Returns the four mutated
    tensors (same refs as input on torch; fresh mx.arrays on MLX).
    """
    if IS_METAL:
        # MLX arrays are immutable; ``_impl`` returns freshly built
        # arrays instead of mutating in place. Surface those.
        return _impl(
            K,
            V,
            cos,
            sin,
            frame_t,
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
        )
    _kv_cache_update_op(
        K,
        V,
        cos,
        sin,
        frame_t,
        Vt_cache,
        segments,
        n_segments,
        K_cache,
        B,
        n_kv_heads,
        H_spatial,
        W_spatial,
        num_buckets,
        pinned_dilation,
        kv_dtype,
        max_segments,
    )
    return K_cache, Vt_cache, segments, n_segments


kv_cache_update.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
