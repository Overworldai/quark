"""ValueResidual numpy reference — ``out = v + lamb * (v1 - v)``."""

from __future__ import annotations

from popcorn.runtime.npconv import astype_numpy, to_f32_numpy


def value_residual_reference_numpy(spec, *, V, V1, lamb, Out=None):
    del Out
    hint = spec.dtype.value
    v = to_f32_numpy(V, dtype_hint=hint)
    v1 = to_f32_numpy(V1, dtype_hint=hint)
    lamb_f = to_f32_numpy(lamb, dtype_hint="f32")
    out = v + lamb_f[0] * (v1 - v)
    return astype_numpy(out, spec.dtype)
