"""``quark.functional.silu`` — standalone element-wise SiLU kernel."""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

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


def silu(X, *, out=None):
    return _impl(X, out=out)


silu.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
