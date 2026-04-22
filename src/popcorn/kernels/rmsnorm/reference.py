"""RMSNorm numpy reference — gainless, last-dim reduction."""

from __future__ import annotations

import numpy as np

from popcorn.runtime.npconv import astype_numpy, to_f32_numpy


def rmsnorm_reference_numpy(spec, *, X, Out=None):
    """``y = x * rsqrt(mean(x², axis=-1, keepdim=True) + eps)``.
    Accumulates in f32, casts back to ``spec.dtype`` at the end."""
    del Out
    hint = spec.dtype.value
    x = to_f32_numpy(X, dtype_hint=hint)
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(mean_sq + spec.eps)
    y = x * inv
    return astype_numpy(y, spec.dtype)
