"""``popcorn.functional.head_rmsnorm`` — per-head RMSNorm on packed QKV."""

from __future__ import annotations

from popcorn.functional._dispatch import call_with_bindings, make_autotune
from popcorn.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("head_rmsnorm")
    return _Cls


def _impl(QKV, *, n_q_heads: int, n_kv_heads: int, Dh: int, eps: float = 1e-6, out=None):
    cls = _cls()
    spec = cls.spec_from_tensors(QKV, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads, Dh=Dh, eps=eps)
    provided = {"X": QKV}
    auto_alloc: tuple[str, ...] = ("Out",)
    if out is not None:
        provided["Out"] = out
        auto_alloc = ()
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=QKV)
    return result["Out"]


def head_rmsnorm(QKV, n_q_heads: int, n_kv_heads: int, Dh: int, eps: float = 1e-6, *, out=None):
    return _impl(QKV, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads, Dh=Dh, eps=eps, out=out)


head_rmsnorm.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
