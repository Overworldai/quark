"""AdaRMSNorm numpy reference — ``y = rmsnorm(x) * (1 + scale) + bias``."""

from __future__ import annotations

import numpy as np

from popcorn.runtime.npconv import astype_numpy, to_f32_numpy


def ada_rmsnorm_reference_numpy(spec, *, X, scale, bias, Out=None):
    """X: [G*M, D], scale/bias: [G, D]. Broadcast each scale/bias row M
    times along axis 0 before applying the epilogue."""
    del Out
    hint = spec.dtype.value

    x = to_f32_numpy(X, dtype_hint=hint)
    s = to_f32_numpy(scale, dtype_hint=hint)
    b = to_f32_numpy(bias, dtype_hint=hint)

    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(mean_sq + spec.eps)
    y = x * inv

    B, G = spec.B, spec.G
    M = B // G
    scale_bm = np.repeat(s, M, axis=0)
    bias_bm = np.repeat(b, M, axis=0)
    y = y * (1.0 + scale_bm) + bias_bm
    return astype_numpy(y, spec.dtype)
