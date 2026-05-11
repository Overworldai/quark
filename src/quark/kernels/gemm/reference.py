"""GEMM numpy reference — ``C = A @ B^T`` with f32 accumulation.

The reference stays in f32 end-to-end. It used to round-trip A / B
through the compute carrier (e.g. f32 → e4m3 → f32) to mirror the
kernel's on-load precision loss, but Python-side fp8 encoding is
per-element and dominated autotune cold-start wall time on 2048-wide
GEMMs. The correctness gate compensates with a more generous cos-sim
threshold on fp8 outputs (see ``KernelCls.correctness_threshold``).
"""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import astype_numpy, to_f32_numpy


def gemm_reference_numpy(spec, *, A, B, Bias=None, Gate=None, Residual=None, Out=None):
    del Out
    a_hint = spec.a_dtype.value
    b_hint = spec.b_dtype.value

    a = to_f32_numpy(A, dtype_hint=a_hint)
    b = to_f32_numpy(B, dtype_hint=b_hint)

    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])

    c = a @ b.T

    if spec.has_bias and Bias is not None:
        c = c + to_f32_numpy(Bias, dtype_hint=spec.out_dtype.value)

    if spec.activation == "silu":
        c = c * (1.0 / (1.0 + np.exp(-c)))
    elif spec.activation is not None:
        raise ValueError(f"gemm_reference: unsupported activation {spec.activation!r}")

    if spec.has_gate_residual and Gate is not None and Residual is not None:
        out_hint = spec.out_dtype.value
        gate_f32 = to_f32_numpy(Gate, dtype_hint=out_hint)  # [G, N]
        residual_f32 = to_f32_numpy(Residual, dtype_hint=out_hint)  # [M, N]
        m_per_group = spec.M // spec.G
        # Broadcast gate over each group's M//G rows: gate[m // (M/G)].
        gate_bcast = np.repeat(gate_f32, m_per_group, axis=0)  # [M, N]
        c = residual_f32 + gate_bcast * c

    return astype_numpy(c, spec.out_dtype)
