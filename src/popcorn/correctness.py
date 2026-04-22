"""Correctness metric for popcorn kernels — cosine similarity, numpy-only.

Per the numpy-refs migration: zero torch / mlx dependency at the
autotune correctness gate. Inputs are normalized to f32 numpy via
``popcorn.runtime.npconv.to_f32_numpy`` regardless of whether they
arrive as ``PopcornTensor``, ``mx.array``, or raw numpy arrays.

Why cosine similarity:

- bf16 / fp16 / fp8 outputs accumulate elementwise rounding error
  that grows with reduction depth (large K). Direction of the
  output vector stays correct even when individual elements drift,
  so a directional metric is right.
- ``max_abs`` and ``mean_abs`` exist as **diagnostic** fields on the
  result struct, but are NEVER consulted for the pass/fail decision.
- NaN / Inf in the output is a hard failure regardless of cosine
  similarity. NaN in the *reference* is rejected at the helper's
  entry — fix the reference instead of comparing to NaN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from popcorn.ir import DType
from popcorn.runtime.npconv import to_f32_numpy

# ---------------------------------------------------------------------------
# Threshold table
# ---------------------------------------------------------------------------


_THRESHOLD_TABLE: dict[tuple[DType, DType], float] = {
    (DType.F32, DType.F32): 1 - 1e-5,
    (DType.F32, DType.F16): 1 - 1e-3,
    (DType.BF16, DType.F32): 1 - 1e-3,
    (DType.BF16, DType.F16): 1 - 3e-3,
    (DType.F16, DType.F32): 1 - 1e-3,
    (DType.F16, DType.F16): 1 - 5e-3,
    (DType.E4M3, DType.F32): 1 - 5e-3,
    (DType.S32, DType.S32): 1 - 1e-9,
    (DType.U32, DType.U32): 1 - 1e-9,
}


def _threshold_for(out: DType, accum: DType) -> float:
    key = (out, accum)
    return _THRESHOLD_TABLE.get(key, 1 - 5e-3)


# ---------------------------------------------------------------------------
# Result struct
# ---------------------------------------------------------------------------


@dataclass
class CorrectnessResult:
    passed: bool
    cos_sim: float
    threshold: float = 0.0
    reason: str | None = None
    mean_abs: float | None = None
    max_abs: float | None = None

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dtype_hint(x: Any, fallback: DType | None) -> str | None:
    """Best-effort dtype hint for ``to_f32_numpy``. PopcornTensor and
    mx.array self-describe their dtype; raw numpy u16/u8 arrays can't,
    so we fall back to the kernel's declared ``out_dtype`` when the
    caller passed a raw numpy carrier."""
    if isinstance(x, np.ndarray) and x.dtype in (np.uint16, np.uint8):
        if fallback is None:
            return None
        return fallback.value  # DType enum values are the short strings
    return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """f32 cosine similarity on flat views. Uses ``float64`` accum so
    the norms + dot don't underflow to zero on long low-magnitude
    reductions."""
    af = a.ravel().astype(np.float64, copy=False)
    bf = b.ravel().astype(np.float64, copy=False)
    dot = float(np.dot(af, bf))
    na = float(np.linalg.norm(af))
    nb = float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        # Two zero vectors are equal by definition; one-zero is drift.
        if na == 0.0 and nb == 0.0:
            return 1.0
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def check_correctness(
    out: Any,
    ref: Any,
    *,
    out_dtype: DType,
    accum_dtype: DType = DType.F32,
    threshold: float | None = None,
) -> CorrectnessResult:
    """Compare ``out`` against ``ref`` via cosine similarity.

    Both arguments are normalized to f32 numpy internally — accepts
    ``PopcornTensor`` (CUDA runtime), ``mx.array`` (Metal), or raw
    numpy arrays. ``out_dtype`` serves double duty: it picks the
    default threshold *and* disambiguates numpy carrier dtypes (e.g.
    a raw ``uint16`` array could be bf16 or a genuine u16 index
    buffer — the declared kernel ``out_dtype`` decides).

    Raises ``ValueError`` if ``ref`` contains NaN — the reference is
    the source of truth, so a NaN there is a reference bug and we
    refuse to silently compare against it.
    """
    thresh = threshold if threshold is not None else _threshold_for(out_dtype, accum_dtype)

    out_np = to_f32_numpy(out, dtype_hint=_dtype_hint(out, out_dtype))
    ref_np = to_f32_numpy(ref, dtype_hint=_dtype_hint(ref, out_dtype))

    if out_np.size != ref_np.size:
        return CorrectnessResult(
            passed=False,
            cos_sim=float("nan"),
            threshold=thresh,
            reason=f"output / reference numel mismatch: out={out_np.size} ref={ref_np.size}",
        )

    if np.isnan(ref_np).any():
        raise ValueError(
            "check_correctness: reference contains NaN — fix the reference, don't compare to NaN"
        )

    finite_mask = np.isfinite(out_np)
    if not finite_mask.all():
        bad_flat = np.argmax(~finite_mask)
        return CorrectnessResult(
            passed=False,
            cos_sim=float("nan"),
            threshold=thresh,
            reason=f"output contains NaN/Inf at flat index {int(bad_flat)}",
            mean_abs=float("nan"),
            max_abs=float("nan"),
        )

    cos_sim = _cosine_sim(out_np, ref_np)
    passed = cos_sim >= thresh
    if passed:
        return CorrectnessResult(passed=True, cos_sim=cos_sim, threshold=thresh)

    diff = np.abs(out_np - ref_np)
    return CorrectnessResult(
        passed=False,
        cos_sim=cos_sim,
        threshold=thresh,
        reason=f"cos_sim={cos_sim:.6f} < {thresh:.6f}",
        mean_abs=float(diff.mean()),
        max_abs=float(diff.max()),
    )


# ---------------------------------------------------------------------------
# Kernel-side override hook
# ---------------------------------------------------------------------------


def threshold_for_kernel(kernel_cls, out_dtype: DType, accum_dtype: DType = DType.F32) -> float:
    override = getattr(kernel_cls, "CORRECTNESS_THRESHOLD", None)
    if override is not None and not (isinstance(override, float) and math.isnan(override)):
        return float(override)
    return _threshold_for(out_dtype, accum_dtype)
