"""``quark.functional.value_residual`` — ``out = v + lamb * (v1 - v)``."""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

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
    import os
    import sys

    if sys.platform == "darwin" and os.environ.get("QUARK_FORCE_FASTPATH") == "1":
        from quark.functional._dispatch import queue_launch_ir
        from quark.ir import DType
        from quark.kernels.value_residual.config import ValueResidualConfig
        from quark.kernels.value_residual.spec import ValueResidualSpec

        orig_shape = tuple(V.shape)
        N = 1
        for s in orig_shape:
            N *= int(s)
        dtype_str = getattr(V, "quark_dtype", None) or (
            V.dtype if hasattr(V, "dtype") and isinstance(V.dtype, str) else "f32"
        )
        spec = ValueResidualSpec(N=N, dtype=DType(dtype_str))
        config = None
        for epb in (1024, 512, 256, 128):
            cand = ValueResidualConfig(n_warps=4, elems_per_block=epb)
            if _cls()(spec=spec, config=cand).is_valid():
                config = cand
                break
        if config is not None:
            V_flat = V.reshape(-1) if len(orig_shape) != 1 else V
            V1_flat = V1.reshape(-1) if len(orig_shape) != 1 else V1
            result = queue_launch_ir(
                _cls(),
                spec,
                config,
                inputs=[V_flat, V1_flat, lamb],
                out_shape=(N,),
                out_dtype=dtype_str,
            )
            return result.reshape(*orig_shape) if len(orig_shape) != 1 else result

    return _impl(V, V1, lamb)


value_residual.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
