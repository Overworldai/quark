"""HeadRMSNorm numpy reference — per-head RMS over Dh slices."""

from __future__ import annotations

import numpy as np

from popcorn.runtime.npconv import astype_numpy, to_f32_numpy


def _rnorm_heads(t: np.ndarray, n_heads: int, col_start: int, Dh: int, eps: float) -> np.ndarray:
    for h in range(n_heads):
        c0 = col_start + h * Dh
        c1 = c0 + Dh
        chunk = t[:, c0:c1]
        inv = 1.0 / np.sqrt(np.mean(chunk * chunk, axis=-1, keepdims=True) + eps)
        t[:, c0:c1] = chunk * inv
    return t


def head_rmsnorm_reference_numpy(spec, *, X, Out=None):
    del Out
    hint = spec.dtype.value
    x = to_f32_numpy(X, dtype_hint=hint).copy()
    y = _rnorm_heads(x, spec.n_q_heads, 0, spec.Dh, spec.eps)
    y = _rnorm_heads(y, spec.n_kv_heads, spec.n_q_heads * spec.Dh, spec.Dh, spec.eps)
    return astype_numpy(y, spec.dtype)
