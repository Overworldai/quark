"""``popcorn.functional.ada_gate_residual`` — ``out = x + sigmoid(gate) * y``."""

from __future__ import annotations

from popcorn.functional._dispatch import call_with_bindings, make_autotune
from popcorn.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("ada_gate_residual")
    return _Cls


def _impl(X, Y, gate, *, out=None):
    cls = _cls()
    orig_shape = tuple(X.shape)
    # Match all dtypes to X (the kernel converts to f32 internally).
    x_dtype = X.dtype if isinstance(X.dtype, str) else str(X.dtype)
    if (Y.dtype if isinstance(Y.dtype, str) else str(Y.dtype)) != x_dtype:
        Y = Y.astype(x_dtype)
    if (gate.dtype if isinstance(gate.dtype, str) else str(gate.dtype)) != x_dtype:
        gate = gate.astype(x_dtype)
    X2 = X.reshape(-1, orig_shape[-1])
    Y2 = Y.reshape(-1, orig_shape[-1])
    G2 = gate.reshape(-1, gate.shape[-1])
    spec = cls.spec_from_tensors(X2, Y2, G2)
    provided = {"X": X2, "Y": Y2, "gate": G2}
    auto_alloc: tuple[str, ...] = ("Out",)
    if out is not None:
        provided["Out"] = out.reshape(-1, orig_shape[-1]) if len(orig_shape) != 2 else out
        auto_alloc = ()
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=X2)
    Out = result["Out"]
    if len(orig_shape) != 2:
        Out = Out.reshape(*orig_shape)
    return Out


def ada_gate_residual(X, Y, gate, *, out=None):
    return _impl(X, Y, gate, out=out)


ada_gate_residual.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
