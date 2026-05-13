"""Numpy reference for the int8 GEMM with per-row scales."""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import astype_numpy, to_f32_numpy


def gemm_int_reference_numpy(spec, *, A, B, A_scales, B_scales, Out=None):
    """C[M, N] = (A_s8 @ B_s8) * A_scales[M, None] * B_scales[None, N].

    Computed in float64 then cast to ``spec.out_dtype`` to match
    what the GPU epilogue produces. ``A`` and ``B`` are int8;
    ``A_scales`` and ``B_scales`` are f32."""
    del Out
    a = to_f32_numpy(A, dtype_hint="s8").astype(np.int32)
    b = to_f32_numpy(B, dtype_hint="s8").astype(np.int32)
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if b.ndim > 2:
        b = b.reshape(-1, b.shape[-1])

    as_f32 = to_f32_numpy(A_scales).ravel().astype(np.float64)
    bs_f32 = to_f32_numpy(B_scales).ravel().astype(np.float64)

    # B layout is [K, N] (matching the shader's B-row layout).
    int_dot = a.astype(np.int64) @ b.astype(np.int64)  # [M, N] s64
    out_f64 = int_dot.astype(np.float64) * as_f32[:, None] * bs_f32[None, :]
    out = out_f64.astype(np.float32)
    return {"Out": astype_numpy(out, spec.out_dtype.value)}
