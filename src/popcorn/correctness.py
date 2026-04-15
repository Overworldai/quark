"""Correctness metric for popcorn kernels — cosine similarity only.

Per popcorn cleanup proposal §7. Replaces today's multi-metric
`Kernel.correctness()` routine. The single gating criterion is
cosine similarity, with thresholds picked from a dtype × accumulator
table tuned against measured rounding behavior on real kernels.

Why cosine similarity:

- bf16 / fp16 / fp8 outputs accumulate elementwise rounding error
  that grows with reduction depth (large K). Direction of the
  output vector stays correct even when individual elements drift,
  so a directional metric is right.
- `max_abs` and `mean_abs` exist as **diagnostic** fields on the
  result struct, but are NEVER consulted for the pass/fail decision.
  Tightening pass/fail to elementwise tolerances re-introduces
  shape-scaled false positives the cosine path is designed to
  avoid.
- NaN / Inf in the output is a hard failure regardless of cosine
  similarity. NaN in the *reference* is rejected at the helper's
  entry — fix the reference instead of comparing to NaN.

The proposal also bans shape-scaled thresholds: the table here is
**fixed** at construction time. If a huge-K test fails on
correctness, the response is to verify the reference computes the
same way (f32 accumulation, same reduction order) or override the
threshold on that ONE kernel via `Kernel.CORRECTNESS_THRESHOLD`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from popcorn.ir import DType

# ---------------------------------------------------------------------------
# Threshold table
# ---------------------------------------------------------------------------


# Default cosine similarity thresholds, keyed by (out_dtype, accum_dtype).
# Rationale:
#   - wider output dtype  → tighter threshold
#   - wider accumulator   → tighter threshold at a given output dtype
# Tuned against measured rounding behavior on real kernels, not from
# first principles. **Do not loosen these without a documented incident.**
_THRESHOLD_TABLE: dict[tuple[DType, DType], float] = {
    # f32 output
    (DType.F32, DType.F32): 1 - 1e-5,  # exact modulo FMA reordering
    (DType.F32, DType.F16): 1 - 1e-3,  # rare: f16 accum into f32 out
    # bf16 output, various accumulators
    (DType.BF16, DType.F32): 1 - 1e-3,  # the common path — GEMM, rmsnorm, MoE
    (DType.BF16, DType.F16): 1 - 3e-3,
    # f16 output
    (DType.F16, DType.F32): 1 - 1e-3,
    (DType.F16, DType.F16): 1 - 5e-3,  # fp16 accumulated in fp16 — loose
    # e4m3 output (rare — usually only intermediates)
    (DType.E4M3, DType.F32): 1 - 5e-3,
    # int output (scatter/gather kernels, indexing)
    (DType.S32, DType.S32): 1 - 1e-9,  # integer must be bitwise equal
    (DType.U32, DType.U32): 1 - 1e-9,
}


def _threshold_for(out: DType, accum: DType) -> float:
    """Look up the default threshold for a (output, accumulator) pair.

    Falls back to a conservative `1 - 5e-3` when the combination
    isn't in the table — that's loose enough to catch direction
    flips but tight enough that any real bug shows up.
    """
    key = (out, accum)
    if key in _THRESHOLD_TABLE:
        return _THRESHOLD_TABLE[key]
    return 1 - 5e-3


# ---------------------------------------------------------------------------
# Result struct
# ---------------------------------------------------------------------------


@dataclass
class CorrectnessResult:
    """Outcome of one `check_correctness` call.

    `passed` and `cos_sim` are the only fields the caller should
    branch on. `reason` is a human string for failure reporting.
    `mean_abs` and `max_abs` are populated only when the cosine
    check FAILS — they're diagnostic-only and should never be
    consulted for pass/fail logic by upstream code.
    """

    passed: bool
    cos_sim: float
    threshold: float = 0.0
    reason: str | None = None
    mean_abs: float | None = None
    max_abs: float | None = None

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def check_correctness(
    out,
    ref,
    *,
    out_dtype: DType,
    accum_dtype: DType = DType.F32,
    threshold: float | None = None,
) -> CorrectnessResult:
    """Compare `out` against `ref` via cosine similarity.

    Args:
      out: kernel output (torch or mlx tensor; any device).
      ref: reference output from the kernel's `reference()` method
        (also torch or mlx — may differ from `out`'s backend).
      out_dtype: the kernel's declared output dtype, used to look up
        the default threshold from the dtype table.
      accum_dtype: the kernel's declared accumulator dtype. Default
        F32 — most kernels in this repo accumulate in f32 even when
        outputting bf16 / fp16. Pass the actual accumulator if it
        differs (e.g. f16 accum on older GPUs).
      threshold: override the dtype-table default. Use sparingly —
        per-kernel overrides should sit in
        `KernelClass.CORRECTNESS_THRESHOLD` instead.

    Returns:
      `CorrectnessResult` with `passed`, `cos_sim`, and `threshold`
      filled in. On failure, `reason`, `mean_abs`, `max_abs` are
      populated for diagnostic printing.

    Raises:
      ValueError: if `ref` contains NaN. The reference is the source
        of truth — a NaN there is a bug in the reference, not a
        kernel bug, and we don't compare against it.
    """
    from popcorn.backend import PT

    thresh = threshold if threshold is not None else _threshold_for(out_dtype, accum_dtype)

    n_out = PT.numel(out)
    n_ref = PT.numel(ref)
    if n_out != n_ref:
        return CorrectnessResult(
            passed=False,
            cos_sim=float("nan"),
            threshold=thresh,
            reason=f"output / reference numel mismatch: out={n_out} ref={n_ref}",
        )

    # NaN/Inf hard checks. NaN in the reference is a bug we refuse to
    # silently compare against; NaN/Inf in the output is a hard failure
    # that short-circuits the cosine math.
    if PT.has_nan(ref):
        raise ValueError(
            "check_correctness: reference contains NaN — fix the reference, don't compare to NaN"
        )
    if not PT.all_finite(out):
        bad_idx = PT.first_non_finite_index(out)
        return CorrectnessResult(
            passed=False,
            cos_sim=float("nan"),
            threshold=thresh,
            reason=f"output contains NaN/Inf at flat index {bad_idx}",
            mean_abs=float("nan"),
            max_abs=float("nan"),
        )

    cos_sim = PT.cosine_sim(out, ref)
    passed = cos_sim >= thresh

    if passed:
        return CorrectnessResult(passed=True, cos_sim=cos_sim, threshold=thresh)

    mean_abs, max_abs = PT.abs_diff_stats(out, ref)
    return CorrectnessResult(
        passed=False,
        cos_sim=cos_sim,
        threshold=thresh,
        reason=f"cos_sim={cos_sim:.6f} < {thresh:.6f}",
        mean_abs=mean_abs,
        max_abs=max_abs,
    )


# ---------------------------------------------------------------------------
# Kernel-side override hook
# ---------------------------------------------------------------------------


def threshold_for_kernel(kernel_cls, out_dtype: DType, accum_dtype: DType = DType.F32) -> float:
    """Pick the threshold a given kernel class wants.

    Reads `kernel_cls.CORRECTNESS_THRESHOLD` if defined, otherwise
    falls back to the dtype table. Per the proposal: per-kernel
    overrides are a code smell — only set them when the dtype table
    really doesn't capture the kernel's precision profile (e.g.
    online softmax + RoPE branch dispatch).
    """
    override = getattr(kernel_cls, "CORRECTNESS_THRESHOLD", None)
    if override is not None and not (isinstance(override, float) and math.isnan(override)):
        return float(override)
    return _threshold_for(out_dtype, accum_dtype)
