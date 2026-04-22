"""``quark.functional.euler_step`` — fused ``x + dsig * v`` kernel."""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("euler_step")
    return _Cls


def _impl(X, V, Dsig, *, out=None):
    cls = _cls()
    orig_shape = tuple(X.shape)
    X_flat = X.reshape(-1)
    V_flat = V.reshape(-1)
    spec = cls.spec_from_tensors(X_flat, V_flat, Dsig)
    provided = {"X": X_flat, "V": V_flat, "Dsig": Dsig}
    auto_alloc: tuple[str, ...] = ("Out",)
    if out is not None:
        provided["Out"] = out.reshape(-1) if len(orig_shape) != 1 else out
        auto_alloc = ()
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=X_flat)
    Y = result["Out"]
    if len(orig_shape) != 1:
        Y = Y.reshape(*orig_shape)
    return Y


def euler_step(X, V, Dsig, *, out=None):
    """Compute ``Out = cast(f32(X) + f32(Dsig[0]) * f32(V), X.dtype)``.

    ``Dsig``: single-element f32 tensor on the same device.
    """
    return _impl(X, V, Dsig, out=out)


euler_step.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
