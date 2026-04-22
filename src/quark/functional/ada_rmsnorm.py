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
    x_dtype = X2.dtype if isinstance(X2.dtype, str) else str(X2.dtype)
    s_dtype = scale.dtype if isinstance(scale.dtype, str) else str(scale.dtype)
    if s_dtype != x_dtype:
        scale = scale.astype(x_dtype)
        bias = bias.astype(x_dtype)
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


def ada_rmsnorm(X, scale, bias, eps: float = 1e-6, activation: str | None = None, *, out=None):
    return _impl(X, scale, bias, eps=eps, activation=activation, out=out)


ada_rmsnorm.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
