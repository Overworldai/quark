"""SiLU numpy reference."""

from __future__ import annotations

import numpy as np

from popcorn.runtime.npconv import astype_numpy, to_f32_numpy


def silu_reference_numpy(spec, *, X, Out=None):
    """``y = x * sigmoid(x)`` in f32, cast back to ``spec.dtype`` on the
    way out. ``Out`` is accepted as a keyword so the caller can pass
    the full ``make_tensors_numpy`` dict through."""
    del Out  # reference writes to a fresh buffer
    x = to_f32_numpy(X, dtype_hint=spec.dtype.value)
    sig = 1.0 / (1.0 + np.exp(-x))
    y = x * sig
    return astype_numpy(y, spec.dtype)
