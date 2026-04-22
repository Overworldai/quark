"""``quark.functional.rmsnorm`` — torch/MLX-native plain RMSNorm.

Signature:

    Y = pcf.rmsnorm(X, *, eps=1e-6)

``X``: any shape ``[*, D]``. Returns ``Y`` same shape, same dtype,
with ``y = x * rsqrt(mean(x², last-dim) + eps)``. No learnable gain —
matches ``F.rms_norm(X, (D,), weight=None, eps=eps)``.
"""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

_Cls = None


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


def rmsnorm(X, eps: float = 1e-6, *, out=None):
    return _rmsnorm_impl(X, eps=eps, out=out)


rmsnorm.autotune = make_autotune(_rmsnorm_impl, _cls)  # ty: ignore[unresolved-attribute]
