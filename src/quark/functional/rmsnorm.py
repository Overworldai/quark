"""``quark.functional.rmsnorm`` — plain RMSNorm over QuarkTensor.

Signature:

    Y = pcf.rmsnorm(X, *, eps=1e-6)

``X``: any shape ``[*, D]``. Returns ``Y`` same shape, same dtype,
with ``y = x * rsqrt(mean(x², last-dim) + eps)``. No learnable gain —
matches ``F.rms_norm(X, (D,), weight=None, eps=eps)``.
"""

from __future__ import annotations

from quark.functional._dispatch import IS_METAL, call_with_bindings, make_autotune
from quark.kernels import get

_Cls = None
# Metal fast-path cache: (B, D, eps) → (compiled, out_shapes, out_dtypes)
_FAST_CACHE: dict[tuple, tuple] = {}


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("rmsnorm")
    return _Cls


def _rmsnorm_impl(X, *, eps: float = 1e-6, out=None):
    cls = _cls()
    # Flatten leading dims — kernel is 2D [B, D].
    orig_shape = tuple(X.shape)
    X2 = X.reshape(-1, orig_shape[-1])
    spec = cls.spec_from_tensors(X2, eps=eps)
    provided = {"X": X2}
    auto_alloc: tuple[str, ...] = ("Out",)
    if out is not None:
        provided["Out"] = out.reshape(-1, orig_shape[-1]) if len(orig_shape) != 2 else out
        auto_alloc = ()
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=X2)
    Y = result["Out"]
    if len(orig_shape) != 2:
        Y = Y.reshape(*orig_shape)
    return Y


_RMSN_FASTPATH_CACHE: dict = {}


def rmsnorm(X, eps: float = 1e-6, *, out=None):
    # Metal fast path: drive the IR-compiled kernel via queue_launch.
    # Saves ~35 µs/call on the eager call_with_bindings path. Caller-
    # pinned ``out`` is supported via queue_launch_ir's out_handle
    # override — the kernel writes straight into the caller's cached
    # buffer (the typical RMSNorm module path with _maybe_cached_out).
    if IS_METAL:
        from quark.functional._dispatch import queue_launch_ir
        from quark.ir import DType
        from quark.kernels.rmsnorm.config import RMSNormConfig
        from quark.kernels.rmsnorm.spec import RMSNormSpec

        orig_shape = tuple(X.shape)
        D = orig_shape[-1]
        B = 1
        for s in orig_shape[:-1]:
            B *= int(s)
        dtype_str = getattr(X, "quark_dtype", None) or (
            X.dtype if hasattr(X, "dtype") and isinstance(X.dtype, str) else "f32"
        )
        cache_key = (B, D, dtype_str, eps)
        cached = _RMSN_FASTPATH_CACHE.get(cache_key)
        if cached is None:
            spec = RMSNormSpec(B=B, D=D, dtype=DType(dtype_str), eps=eps)
            config = None
            for nw in (4, 2, 1, 8):
                cand = RMSNormConfig(n_warps=nw)
                if _cls()(spec=spec, config=cand).is_valid():
                    config = cand
                    break
            if config is not None:
                _RMSN_FASTPATH_CACHE[cache_key] = (spec, config)
                cached = (spec, config)
        if cached is not None:
            spec, config = cached
            out_h = -1
            if out is not None:
                h = getattr(out, "metal_handle", None)
                if h is not None:
                    out_h = int(h)
            result = queue_launch_ir(
                _cls(),
                spec,
                config,
                inputs=[X],
                out_shape=(B, D),
                out_dtype=dtype_str,
                out_handle=out_h,
            )
            return result.reshape(*orig_shape) if len(orig_shape) != 2 else result

    return _rmsnorm_impl(X, eps=eps, out=out)


rmsnorm.autotune = make_autotune(_rmsnorm_impl, _cls)  # ty: ignore[unresolved-attribute]
