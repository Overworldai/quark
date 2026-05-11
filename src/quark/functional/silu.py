"""``quark.functional.silu`` — standalone element-wise SiLU kernel."""

from __future__ import annotations

import sys

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

_IS_METAL = sys.platform == "darwin"

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("silu")
    return _Cls


def _impl(X, *, out=None):
    cls = _cls()
    orig_shape = tuple(X.shape)
    X_flat = X.reshape(-1)
    spec = cls.spec_from_tensors(X_flat)
    provided = {"X": X_flat}
    auto_alloc: tuple[str, ...] = ("Out",)
    if out is not None:
        provided["Out"] = out.reshape(-1) if len(orig_shape) != 1 else out
        auto_alloc = ()
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=X_flat)
    Y = result["Out"]
    if len(orig_shape) != 1:
        Y = Y.reshape(*orig_shape)
    return Y


_SILU_FASTPATH_CACHE: dict = {}


def silu(X, *, out=None):
    if _IS_METAL:
        from quark.functional._dispatch import queue_launch_ir
        from quark.ir import DType
        from quark.kernels.silu.config import SiLUConfig
        from quark.kernels.silu.spec import SiLUSpec

        orig_shape = tuple(X.shape)
        N = 1
        for s in orig_shape:
            N *= int(s)
        dtype_str = getattr(X, "quark_dtype", None) or (
            X.dtype if hasattr(X, "dtype") and isinstance(X.dtype, str) else "f32"
        )
        # Per-(N, dtype) config cache: skip the spec/config search after
        # the first call. Each transformer block hits the same shape
        # (128, 8192 → 1048576) for its mlp.silu, so 24 calls/forward
        # all share one cache entry.
        cache_key = (N, dtype_str)
        cached = _SILU_FASTPATH_CACHE.get(cache_key)
        if cached is None:
            spec = SiLUSpec(N=N, dtype=DType(dtype_str))
            config = None
            # 256-thread blocks (n_warps=8) processing 1024 elems each via
            # vec4. Falls through to the launcher path if the spec can't
            # pick a legal IR config (rare — only weird N's).
            for nw, epb in [(8, 1024), (4, 1024), (4, 512), (2, 512), (1, 256)]:
                cand = SiLUConfig(n_warps=nw, elems_per_block=epb)
                if _cls()(spec=spec, config=cand).is_valid():
                    config = cand
                    break
            if config is not None:
                _SILU_FASTPATH_CACHE[cache_key] = (spec, config)
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
                out_shape=(N,),
                out_dtype=dtype_str,
                out_handle=out_h,
            )
            return result.reshape(*orig_shape) if len(orig_shape) != 1 else result
    return _impl(X, out=out)


silu.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
