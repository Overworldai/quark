"""``quark.functional.ada_gate_residual`` — ``out = x + sigmoid(gate) * y``."""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.ir import DType
from quark.kernels import get

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
    x_dtype = X.dtype if isinstance(X.dtype, str) else DType.from_backend(X.dtype)
    if (Y.dtype if isinstance(Y.dtype, str) else DType.from_backend(Y.dtype)) != x_dtype:
        Y = Y.astype(x_dtype)
    if (gate.dtype if isinstance(gate.dtype, str) else DType.from_backend(gate.dtype)) != x_dtype:
        gate = gate.astype(x_dtype)
    X2 = X.reshape(-1, orig_shape[-1])
    Y2 = Y.reshape(-1, orig_shape[-1])
    G2 = gate.reshape(-1, gate.shape[-1])
    spec = cls.spec_from_tensors(X2, Y2, G2)
    provided: dict = {"X": X2, "Y": Y2, "gate": G2}
    if out is not None:
        provided["Out"] = out
    result = call_with_bindings(
        cls, spec, provided=provided, auto_alloc=() if out is not None else ("Out",), like=X2
    )
    Out = result["Out"]
    if len(orig_shape) != 2:
        Out = Out.reshape(*orig_shape)
    return Out


def ada_gate_residual(X, Y, gate, *, out=None):
    """Compute ``out = x + gate * y``.

    ``out``: optional pre-allocated output buffer. When provided, skips
    auto_alloc — lets callers (e.g. AdaGateResidual layers) reuse a
    cached buffer and avoid per-call allocations.
    """
    return _impl(X, Y, gate, out=out)


ada_gate_residual.autotune = make_autotune(_impl, _cls)
