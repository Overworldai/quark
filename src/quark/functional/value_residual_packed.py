"""``quark.functional.value_residual_packed`` — lerp V cols of packed QKV."""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings
from quark.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("value_residual_packed")
    return _Cls


_VRP_FASTPATH_CACHE: dict = {}


def value_residual_packed(qkv_curr, qkv_first, lamb, *, v_col_offset: int, v_width: int, out=None):
    cls = _cls()
    # Match dtypes to qkv_curr (the kernel converts to f32 internally).
    curr_dt = qkv_curr.dtype if isinstance(qkv_curr.dtype, str) else str(qkv_curr.dtype)
    first_dt = qkv_first.dtype if isinstance(qkv_first.dtype, str) else str(qkv_first.dtype)
    if first_dt != curr_dt:
        qkv_first = qkv_first.astype(curr_dt)

    import sys

    if sys.platform == "darwin":
        from quark.functional._dispatch import launcher, queue_launch_ir

        M = int(qkv_curr.shape[0])
        D_full = int(qkv_curr.shape[1])
        cache_key = (M, D_full, curr_dt, v_col_offset, v_width)
        cached = _VRP_FASTPATH_CACHE.get(cache_key)
        if cached is None:
            spec = cls.spec_from_tensors(
                qkv_curr,
                qkv_first,
                lamb,
                v_col_offset=v_col_offset,
                v_width=v_width,
            )
            config = launcher()._autotune.lookup_or_search(cls, spec)
            _VRP_FASTPATH_CACHE[cache_key] = (spec, config)
            cached = (spec, config)
        spec, config = cached
        out_h = -1
        if out is not None:
            h = getattr(out, "metal_handle", None)
            if h is not None:
                out_h = int(h)
        return queue_launch_ir(
            cls,
            spec,
            config,
            inputs=[qkv_curr, qkv_first, lamb],
            out_shape=(M, D_full),
            out_dtype=curr_dt,
            out_handle=out_h,
        )

    spec = cls.spec_from_tensors(
        qkv_curr, qkv_first, lamb, v_col_offset=v_col_offset, v_width=v_width
    )
    provided = {"QKV_curr": qkv_curr, "QKV_first": qkv_first, "lamb": lamb}
    auto_alloc: tuple[str, ...] = ("Out",)
    if out is not None:
        provided["Out"] = out
        auto_alloc = ()
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=qkv_curr)
    return result["Out"]
