"""``popcorn.functional.value_residual`` — ``out = v + lamb * (v1 - v)``."""

from __future__ import annotations

from popcorn.functional._dispatch import call_with_bindings, make_autotune
from popcorn.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("value_residual")
    return _Cls


def _impl(V, V1, lamb):
    cls = _cls()
    orig_shape = tuple(V.shape)
    V_flat = V.reshape(-1)
    V1_flat = V1.reshape(-1)
    spec = cls.spec_from_tensors(V_flat, V1_flat, lamb)
    result = call_with_bindings(
        cls,
        spec,
        provided={"V": V_flat, "V1": V1_flat, "lamb": lamb},
        auto_alloc=("Out",),
        like=V_flat,
    )
    Out = result["Out"]
    if len(orig_shape) != 1:
        Out = Out.reshape(*orig_shape)
    return Out


def value_residual(V, V1, lamb):
    return _impl(V, V1, lamb)


value_residual.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
