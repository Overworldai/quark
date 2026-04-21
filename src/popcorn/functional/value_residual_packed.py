"""``popcorn.functional.value_residual_packed`` — lerp V cols of packed QKV."""

from __future__ import annotations

from popcorn.functional._dispatch import call_with_bindings
from popcorn.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("value_residual_packed")
    return _Cls


def value_residual_packed(qkv_curr, qkv_first, lamb, *, v_col_offset: int, v_width: int, out=None):
    cls = _cls()
    # Match dtypes to qkv_curr (the kernel converts to f32 internally).
    curr_dt = qkv_curr.dtype if isinstance(qkv_curr.dtype, str) else str(qkv_curr.dtype)
    first_dt = qkv_first.dtype if isinstance(qkv_first.dtype, str) else str(qkv_first.dtype)
    if first_dt != curr_dt:
        qkv_first = qkv_first.astype(curr_dt)
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
