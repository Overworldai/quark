"""Tests for popcorn.correctness — cosine similarity gating helper."""

import pytest

pytest.importorskip(
    "torch",
    reason="torch removed from runtime; numpy-refs migration — test kept for dev-only cross-check when torch is installed",
)

import math

import pytest
import torch

from popcorn.correctness import (
    _THRESHOLD_TABLE,
    CorrectnessResult,
    _threshold_for,
    check_correctness,
    threshold_for_kernel,
)
from popcorn.ir import DType

# ---------------------------------------------------------------------------
# Threshold table + lookup
# ---------------------------------------------------------------------------


class TestThresholdTable:
    def test_bf16_f32_is_primary_path(self):
        """The most-used combo (GEMM/MoE/rmsnorm bf16 outputs with
        f32 accumulator) lands at 1 - 1e-3."""
        assert _threshold_for(DType.BF16, DType.F32) == pytest.approx(1 - 1e-3)

    def test_f32_f32_is_tightest(self):
        assert _threshold_for(DType.F32, DType.F32) == pytest.approx(1 - 1e-5)

    def test_int_combos_require_bitwise_match(self):
        for combo in [(DType.S32, DType.S32), (DType.U32, DType.U32)]:
            assert _threshold_for(*combo) == pytest.approx(1 - 1e-9)

    def test_unknown_combo_falls_back_to_loose(self):
        # F64 / F64 isn't in the table — falls back to 1 - 5e-3.
        assert _threshold_for(DType.F64, DType.F64) == pytest.approx(1 - 5e-3)

    def test_table_entries_are_strict_upper_bounds(self):
        """Every threshold is < 1.0 (cosine similarity strictly less
        than perfect)."""
        for value in _THRESHOLD_TABLE.values():
            assert 0.0 < value < 1.0


# ---------------------------------------------------------------------------
# CorrectnessResult
# ---------------------------------------------------------------------------


class TestCorrectnessResult:
    def test_passed_truthiness(self):
        good = CorrectnessResult(passed=True, cos_sim=0.9999)
        bad = CorrectnessResult(passed=False, cos_sim=0.5)
        assert bool(good) is True
        assert bool(bad) is False

    def test_diagnostic_fields_default_none(self):
        r = CorrectnessResult(passed=True, cos_sim=1.0)
        assert r.mean_abs is None
        assert r.max_abs is None
        assert r.reason is None


# ---------------------------------------------------------------------------
# check_correctness — happy paths
# ---------------------------------------------------------------------------


class TestCheckCorrectnessHappy:
    def test_identical_tensors_pass(self):
        x = torch.arange(64, dtype=torch.float32)
        result = check_correctness(x, x, out_dtype=DType.F32)
        assert result.passed
        assert result.cos_sim == pytest.approx(1.0, abs=1e-6)
        assert result.threshold == pytest.approx(1 - 1e-5)

    def test_bf16_round_trip_within_threshold(self):
        """A bf16-rounded version of an f32 tensor should still pass
        the bf16/f32 threshold."""
        ref = torch.arange(1024, dtype=torch.float32) * 0.001
        out = ref.to(torch.bfloat16).to(torch.float32)
        result = check_correctness(out, ref, out_dtype=DType.BF16)
        assert result.passed, f"unexpected failure: {result}"

    def test_explicit_threshold_override_loose(self):
        """A loose explicit override accepts a 0.9-similar tensor."""
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.95, 0.31, 0.0])  # cos ≈ 0.95
        result = check_correctness(a, b, out_dtype=DType.F32, threshold=0.9)
        assert result.passed

    def test_explicit_threshold_override_tight(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.95, 0.31, 0.0])
        result = check_correctness(a, b, out_dtype=DType.F32, threshold=0.999)
        assert not result.passed
        assert result.reason is not None and "cos_sim=" in result.reason


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestCheckCorrectnessFailures:
    def test_orthogonal_vectors_fail(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        result = check_correctness(a, b, out_dtype=DType.F32)
        assert not result.passed
        assert result.cos_sim == pytest.approx(0.0, abs=1e-6)
        assert result.mean_abs is not None
        assert result.max_abs is not None

    def test_nan_in_output_short_circuits(self):
        ref = torch.ones(8)
        out = torch.ones(8)
        out[3] = float("nan")
        result = check_correctness(out, ref, out_dtype=DType.F32)
        assert not result.passed
        assert math.isnan(result.cos_sim)
        assert "NaN" in (result.reason or "")
        assert "3" in (result.reason or "")  # flat index

    def test_inf_in_output_short_circuits(self):
        ref = torch.ones(8)
        out = torch.ones(8)
        out[5] = float("inf")
        result = check_correctness(out, ref, out_dtype=DType.F32)
        assert not result.passed
        assert "Inf" in (result.reason or "")

    def test_nan_in_reference_raises(self):
        ref = torch.ones(8)
        ref[3] = float("nan")
        out = torch.ones(8)
        with pytest.raises(ValueError, match="reference contains NaN"):
            check_correctness(out, ref, out_dtype=DType.F32)

    def test_size_mismatch_short_circuits(self):
        out = torch.ones(8)
        ref = torch.ones(16)
        result = check_correctness(out, ref, out_dtype=DType.F32)
        assert not result.passed
        assert "numel mismatch" in (result.reason or "")

    def test_zero_vectors_dont_nan(self):
        """Both vectors zero → cos_sim = 0/(0+eps) = 0 instead of NaN.
        The point is to not blow up; pass/fail is whatever it is."""
        a = torch.zeros(8)
        b = torch.zeros(8)
        result = check_correctness(a, b, out_dtype=DType.F32)
        assert not math.isnan(result.cos_sim)


# ---------------------------------------------------------------------------
# threshold_for_kernel — per-kernel override hook
# ---------------------------------------------------------------------------


class TestThresholdForKernel:
    def test_no_override_uses_dtype_table(self):
        class Plain:
            pass

        assert threshold_for_kernel(Plain, DType.BF16, DType.F32) == pytest.approx(1 - 1e-3)

    def test_override_takes_precedence(self):
        class Looser:
            CORRECTNESS_THRESHOLD = 1 - 5e-3

        assert threshold_for_kernel(Looser, DType.BF16, DType.F32) == pytest.approx(1 - 5e-3)

    def test_nan_override_falls_through(self):
        """Setting CORRECTNESS_THRESHOLD = NaN is treated as 'not set'
        so the dtype table wins."""

        class Bad:
            CORRECTNESS_THRESHOLD = float("nan")

        assert threshold_for_kernel(Bad, DType.BF16, DType.F32) == pytest.approx(1 - 1e-3)


# ---------------------------------------------------------------------------
# Device round-trip — runs on CUDA if available, otherwise CPU
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestCheckCorrectnessOnCuda:
    def test_cuda_tensor_input(self):
        x = torch.arange(256, dtype=torch.bfloat16, device="cuda")
        ref = x.to(torch.float32)
        out = ref.to(torch.bfloat16)
        result = check_correctness(out, ref, out_dtype=DType.BF16)
        assert result.passed
