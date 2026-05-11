"""``quark.functional.kv_cache_update`` — RoPE + ring KV-cache write."""

from __future__ import annotations

import sys

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

_Cls = None
# Metal fast path cache: spec key → (compiled_module, threads_grid, block).
_FASTPATH_CACHE: dict = {}


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
    quilt_factor=1,
    quilt_offset=0,
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
        quilt_factor=quilt_factor,
        quilt_offset=quilt_offset,
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


def _try_fastpath(
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
    kv_dtype,
    max_segments,
    packed_qkv,
    n_q_heads,
    rope_n_frames,
):
    """Drive ``_md.launch`` directly, skipping ``call_with_bindings``.

    The kv_cache_update kernel has 4 ``role="out"`` tensors that are all
    caller-pinned QuarkTensors (Vt_cache, segments, n_segments, K_cache).
    ``call_with_bindings`` walks param-spec → buffer-list → bytes/handles
    on every call (~190 µs), which dominates the 24-layer × 5-NFE inner
    loop. Here we cache the (spec, compiled module, grid, block) tuple
    keyed on hot-path-stable kwargs, then hit ``launcher()._launch_metal``
    once per call with prepared in/out arrays.

    Returns True on hit (kernel dispatched), False to fall through.
    """
    if sys.platform != "darwin":
        return False
    # Need every state buffer to be a pinned QuarkTensor for the
    # output_handles route. K and V are inputs, the rest are outputs.
    needed = (Vt_cache, segments, n_segments, K_cache)
    for t in needed:
        if getattr(t, "metal_handle", None) is None:
            return False

    cache_key = (
        B,
        n_kv_heads,
        H_spatial,
        W_spatial,
        num_buckets,
        pinned_dilation,
        kv_dtype,
        max_segments,
        packed_qkv,
        n_q_heads,
        rope_n_frames,
        getattr(K, "dtype", None),
    )
    cached = _FASTPATH_CACHE.get(cache_key)
    if cached is None:
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
        from quark.functional._dispatch import launcher

        lc = launcher()
        config = lc._autotune.lookup_or_search(cls, spec)
        compiled = lc.compile(cls, spec, config)
        _FASTPATH_CACHE[cache_key] = (compiled, spec)
        cached = (compiled, spec)

    compiled, spec = cached
    # Ordered list mirroring TENSORS in kernel.py: K, V, frame_t,
    # frozen, Vt_cache, segments, n_segments, K_cache.
    buffers = [K, V, frame_t, frozen, Vt_cache, segments, n_segments, K_cache]
    persistent_outs = {"Vt_cache", "segments", "n_segments", "K_cache"}
    compiled.launch(buffers=buffers, persistent_outs=persistent_outs)
    return True


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
    quilt_factor=1,
    quilt_offset=0,
):
    """In-place RoPE + KV-cache update. frozen=1 skips ring writes.

    Quilt attention: when ``quilt_factor>1`` (power of 2), the cache
    only stores pixels with ``pixel_idx % quilt_factor == quilt_offset``.
    Cache capacity shrinks by ``quilt_factor``.
    """
    # Fast path: skip the param-spec / buffer-list dance for the common
    # quilt_factor=1 case. ``_try_fastpath`` doesn't yet wire the quilt
    # offsets through; quilt_factor>1 falls through to ``_impl`` which
    # carries them via ``call_with_bindings``.
    if quilt_factor == 1 and _try_fastpath(
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
    ):
        return None
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
        quilt_factor=quilt_factor,
        quilt_offset=quilt_offset,
    )


kv_cache_update.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
