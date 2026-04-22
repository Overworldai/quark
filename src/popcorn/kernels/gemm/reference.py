"""GEMM numpy reference — ``C = A @ B^T`` with f32 accumulation.

When ``compute_dtype`` differs from ``a_dtype`` / ``b_dtype`` we
round-trip A / B through the compute carrier before the matmul so
the cos-sim gate sees the same on-load precision loss the kernel
does.
"""

from __future__ import annotations

import numpy as np

from popcorn.ir import DType
from popcorn.runtime.npconv import astype_numpy, to_f32_numpy


def _roundtrip_through(arr_f32: np.ndarray, target: DType) -> np.ndarray:
    """f32 → carrier → f32. Matches the kernel's on-load cast."""
    carrier = astype_numpy(arr_f32, target)
    return to_f32_numpy(carrier, dtype_hint=target.value)


def gemm_reference_numpy(spec, *, A, B, Bias=None, Out=None):
    del Out
    a_hint = spec.a_dtype.value
    b_hint = spec.b_dtype.value

    a = to_f32_numpy(A, dtype_hint=a_hint)
    b = to_f32_numpy(B, dtype_hint=b_hint)

    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])

    # Simulate the compute-dtype cast when it differs from the source.
    compute_dt = spec.compute_dtype_resolved
    if compute_dt is not spec.a_dtype:
        a = _roundtrip_through(a, compute_dt)
    if compute_dt is not spec.b_dtype:
        b = _roundtrip_through(b, compute_dt)

    c = a @ b.T

    if spec.has_bias and Bias is not None:
        c = c + to_f32_numpy(Bias, dtype_hint=spec.out_dtype.value)

    if spec.activation == "silu":
        c = c * (1.0 / (1.0 + np.exp(-c)))
    elif spec.activation is not None:
        raise ValueError(f"gemm_reference: unsupported activation {spec.activation!r}")

    return astype_numpy(c, spec.out_dtype)
