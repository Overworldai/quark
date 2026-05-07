"""``quark.functional.head_rmsnorm`` — per-head RMSNorm on packed QKV."""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

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


_HRN_FASTPATH_CACHE: dict = {}


def head_rmsnorm(QKV, n_q_heads: int, n_kv_heads: int, Dh: int, eps: float = 1e-6, *, out=None):
    import sys

    if sys.platform == "darwin":
        from quark.functional._dispatch import queue_launch_ir

        cls = _cls()
        # Spec construction reads tensor shape; cache the resulting
        # spec/config keyed on (M, n_q, n_kv, Dh, dtype).
        M = int(QKV.shape[0])
        D_full = int(QKV.shape[1])
        dtype_str = getattr(QKV, "quark_dtype", None) or (
            QKV.dtype if hasattr(QKV, "dtype") and isinstance(QKV.dtype, str) else "f32"
        )
        cache_key = (M, n_q_heads, n_kv_heads, Dh, dtype_str, eps)
        cached = _HRN_FASTPATH_CACHE.get(cache_key)
        if cached is None:
            spec = cls.spec_from_tensors(
                QKV,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                Dh=Dh,
                eps=eps,
            )
            from quark.functional._dispatch import launcher

            config = launcher()._autotune.lookup_or_search(cls, spec)
            _HRN_FASTPATH_CACHE[cache_key] = (spec, config)
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
            inputs=[QKV],
            out_shape=(M, D_full),
            out_dtype=dtype_str,
            out_handle=out_h,
        )
    return _impl(QKV, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads, Dh=Dh, eps=eps, out=out)


head_rmsnorm.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
