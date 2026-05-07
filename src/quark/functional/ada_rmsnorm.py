"""``quark.functional.ada_rmsnorm`` — fused RMSNorm + (1+scale)·y + bias."""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("ada_rmsnorm")
    return _Cls


def _impl(X, scale, bias, *, eps: float = 1e-6, activation: str | None = None, out=None):
    cls = _cls()
    orig_shape = tuple(X.shape)
    X2 = X.reshape(-1, orig_shape[-1])
    # Match scale/bias dtype to X so the kernel sees uniform dtype.
    # The kernel converts to f32 internally anyway.
    x_quark_dt = getattr(X2, "quark_dtype", None) or (
        X2.dtype if isinstance(X2.dtype, str) else None
    )
    s_quark_dt = getattr(scale, "quark_dtype", None) or (
        scale.dtype if isinstance(scale.dtype, str) else None
    )
    if x_quark_dt and s_quark_dt and s_quark_dt != x_quark_dt:
        # Cast through the runtime path that handles bf16-as-uint16 carriers.
        from quark.runtime.npconv import astype_numpy, to_f32_numpy

        s_f32 = to_f32_numpy(scale, dtype_hint=s_quark_dt)
        b_f32 = to_f32_numpy(bias, dtype_hint=s_quark_dt)
        scale = astype_numpy(s_f32, x_quark_dt)
        bias = astype_numpy(b_f32, x_quark_dt)
    S2 = scale.reshape(-1, scale.shape[-1])
    B2 = bias.reshape(-1, bias.shape[-1])
    spec = cls.spec_from_tensors(X2, S2, B2, eps=eps)
    # Patch activation into the frozen spec if requested.
    if activation is not None:
        import dataclasses

        spec = dataclasses.replace(spec, activation=activation)
    provided = {"X": X2, "scale": S2, "bias": B2}
    auto_alloc: tuple[str, ...] = ("Out",)
    if out is not None:
        provided["Out"] = out.reshape(-1, orig_shape[-1]) if len(orig_shape) != 2 else out
        auto_alloc = ()
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=X2)
    Y = result["Out"]
    if len(orig_shape) != 2:
        Y = Y.reshape(*orig_shape)
    return Y


_ARN_FASTPATH_CACHE: dict = {}


def ada_rmsnorm(X, scale, bias, eps: float = 1e-6, activation: str | None = None, *, out=None):
    import sys

    # Fast path: caller-cached out= buffer (the typical AdaRMSNorm
    # module path) routes through the queue_launch_ir handle override
    # so writes land in the cache the way call_with_bindings does, but
    # with ~10 µs less Python overhead per call. Now also handles the
    # ``activation="silu"`` case via dataclasses.replace on the spec —
    # the kernel's silu path is already wired in build, the fast path
    # just needs to patch the spec like ``_impl`` does.
    if sys.platform == "darwin":
        from quark.functional._dispatch import queue_launch_ir
        from quark.ir import DType
        from quark.kernels.ada_rmsnorm.config import AdaRMSNormConfig
        from quark.kernels.ada_rmsnorm.spec import AdaRMSNormSpec

        orig_shape = tuple(X.shape)
        D = orig_shape[-1]
        B = 1
        for s in orig_shape[:-1]:
            B *= int(s)
        dtype_str = getattr(X, "quark_dtype", None) or (
            X.dtype if hasattr(X, "dtype") and isinstance(X.dtype, str) else "f32"
        )
        G = int(scale.shape[0])
        if B % G != 0:
            return _impl(X, scale, bias, eps=eps, activation=activation, out=out)
        M = B // G
        cache_key = (G, M, D, dtype_str, eps, activation)
        cached = _ARN_FASTPATH_CACHE.get(cache_key)
        if cached is None:
            spec = AdaRMSNormSpec(G=G, M=M, D=D, dtype=DType(dtype_str), eps=eps)
            if activation is not None:
                import dataclasses

                spec = dataclasses.replace(spec, activation=activation)
            config = None
            for nw, cd in [(4, 512), (4, 1024), (2, 512), (1, 512)]:
                cand = AdaRMSNormConfig(n_warps=nw, chunk_D=cd)
                if _cls()(spec=spec, config=cand).is_valid():
                    config = cand
                    break
            if config is not None:
                _ARN_FASTPATH_CACHE[cache_key] = (spec, config)
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
                inputs=[X, scale, bias],
                out_shape=(B, D),
                out_dtype=dtype_str,
                out_handle=out_h,
            )
            return result.reshape(*orig_shape) if len(orig_shape) != 2 else result

    return _impl(X, scale, bias, eps=eps, activation=activation, out=out)


ada_rmsnorm.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
