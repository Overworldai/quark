"""Correctness tests for the cublasLt dispatch.

Strategy: for each supported combo, run ``pcf.gemm`` twice — once with
``POPCORN_DISABLE_CUBLAS=1`` to force the custom kernel, once without
to take the cuBLAS path — and compare via cos_sim. The custom kernel
is exercised nightly by the full test suite, so using it as the
reference keeps this file self-contained.

Unsupported combos (mixed bf16×e4m3, pre-shuffled B, fused activation)
have a dedicated fall-through test so the dispatch gate can't
silently start routing them to cuBLAS.
"""
# ruff: noqa: E402  -- imports below pytest.skip(allow_module_level=True)
# are intentional; they would fail at module load on a CUDA-less system,
# so they sit behind the runtime gate above.

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

IS_METAL = sys.platform == "darwin"
if IS_METAL:
    pytest.skip("cuBLAS is CUDA-only", allow_module_level=True)

from popcorn.runtime.cuda import CudaRuntime

if not CudaRuntime.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from popcorn.runtime.cublas import CublasRuntime

if not CublasRuntime.is_available():
    pytest.skip("libcublasLt not available", allow_module_level=True)

import popcorn.functional as pcf
from popcorn.correctness import check_correctness
from popcorn.ir import DType
from popcorn.runtime.tensor import PopcornTensor

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _env(name: str, value: str | None):
    """Context helper via try/finally — avoids pulling in ``contextlib``."""

    class _Ctx:
        def __enter__(self_):
            self_._prev = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
            return None

        def __exit__(self_, *exc):
            if self_._prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = self_._prev

    return _Ctx()


def _rand_f32(*shape, seed=0, scale=0.05):
    """Small-scale random tensor so e4m3's narrow range doesn't saturate."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32) * scale


def _make_ab(M, N, K, a_dtype, b_dtype):
    """Build (A, B) PopcornTensors in the requested dtypes from the same
    random seed pair, routed through e4m3 quantization when needed so
    both dispatch paths see identical inputs.
    """
    a_f32 = PopcornTensor.from_numpy(_rand_f32(M, K, seed=1), dtype="f32")
    b_f32 = PopcornTensor.from_numpy(_rand_f32(N, K, seed=2), dtype="f32")
    A = a_f32.astype(a_dtype)
    B = b_f32.astype(b_dtype)
    return A, B


def _run(A, B, out_dtype, *, disable_cublas: bool):
    """Run ``pcf.gemm`` with cuBLAS toggled via env var."""
    with _env("POPCORN_DISABLE_CUBLAS", "1" if disable_cublas else None):
        return pcf.gemm(A, B, out_dtype=out_dtype)


def _assert_matches(out, ref, out_dtype):
    ir_dt = DType(out_dtype)
    cr = check_correctness(out, ref, out_dtype=ir_dt)
    assert cr.passed, (
        f"cuBLAS vs custom-kernel mismatch for out_dtype={out_dtype}: "
        f"cos_sim={cr.cos_sim:.6f} (threshold={cr.threshold})"
    )


# ---------------------------------------------------------------------------
# CublasRuntime direct API
# ---------------------------------------------------------------------------


class TestCublasRuntime:
    def test_singleton_is_idempotent(self):
        a = CublasRuntime.instance()
        b = CublasRuntime.instance()
        assert a is b

    def test_matmul_bf16_small(self):
        M, N, K = 16, 16, 32
        A, B = _make_ab(M, N, K, "bf16", "bf16")
        out = PopcornTensor.empty(M, N, dtype="bf16")
        CublasRuntime.instance().matmul(
            a_ptr=A.data_ptr(),
            b_ptr=B.data_ptr(),
            c_ptr=out.data_ptr(),
            M=M,
            N=N,
            K=K,
            a_dtype="bf16",
            b_dtype="bf16",
            c_dtype="bf16",
        )
        # Numpy reference from the device bf16 tensors we just wrote.
        a_np = A.astype("f32").to_numpy()
        b_np = B.astype("f32").to_numpy()
        ref_np = a_np @ b_np.T
        cr = check_correctness(out, ref_np, out_dtype=DType.BF16)
        assert cr.passed, f"bf16 matmul cos_sim={cr.cos_sim:.6f}"


# ---------------------------------------------------------------------------
# Dispatch parity — cuBLAS vs custom kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("M", "N", "K", "a_dtype", "b_dtype", "out_dtype"),
    [
        (32, 64, 64, "bf16", "bf16", "bf16"),
        (64, 128, 128, "bf16", "bf16", "bf16"),
        (16, 32, 64, "bf16", "bf16", "f32"),
        (32, 64, 64, "e4m3", "e4m3", "bf16"),
        (128, 128, 256, "e4m3", "e4m3", "bf16"),
    ],
)
def test_cublas_matches_custom(M, N, K, a_dtype, b_dtype, out_dtype):
    A, B = _make_ab(M, N, K, a_dtype, b_dtype)
    ref = _run(A, B, out_dtype, disable_cublas=True)
    out = _run(A, B, out_dtype, disable_cublas=False)
    _assert_matches(out, ref, out_dtype)


# ---------------------------------------------------------------------------
# Gate — unsupported combos must fall through to the custom kernel
# ---------------------------------------------------------------------------


def _patch_record_cublas_calls(monkeypatch):
    """Replace ``CublasRuntime.matmul`` with a recorder. Returns the
    call list, so tests can assert ``len(calls) == 0`` when they
    expect the gate to route to the custom kernel.
    """
    calls: list[dict] = []
    rt = CublasRuntime.instance()
    real = rt.matmul

    def fake(**kw):
        calls.append(kw)
        return real(**kw)

    monkeypatch.setattr(rt, "matmul", fake)
    return calls


class TestDispatchGate:
    def test_env_disables_cublas(self, monkeypatch):
        A, B = _make_ab(16, 16, 32, "bf16", "bf16")
        calls = _patch_record_cublas_calls(monkeypatch)
        with _env("POPCORN_DISABLE_CUBLAS", "1"):
            pcf.gemm(A, B, out_dtype="bf16")
        assert calls == [], "POPCORN_DISABLE_CUBLAS=1 did not suppress cuBLAS dispatch"

    def test_bf16_bf16_bf16_uses_cublas(self, monkeypatch):
        A, B = _make_ab(16, 16, 32, "bf16", "bf16")
        calls = _patch_record_cublas_calls(monkeypatch)
        with _env("POPCORN_DISABLE_CUBLAS", None):
            pcf.gemm(A, B, out_dtype="bf16")
        assert len(calls) == 1

    def test_e4m3_e4m3_bf16_uses_cublas(self, monkeypatch):
        A, B = _make_ab(32, 32, 64, "e4m3", "e4m3")
        calls = _patch_record_cublas_calls(monkeypatch)
        with _env("POPCORN_DISABLE_CUBLAS", None):
            pcf.gemm(A, B, out_dtype="bf16")
        assert len(calls) == 1

    def test_mixed_bf16_e4m3_falls_through(self, monkeypatch):
        A, B = _make_ab(32, 32, 64, "bf16", "e4m3")
        calls = _patch_record_cublas_calls(monkeypatch)
        with _env("POPCORN_DISABLE_CUBLAS", None):
            pcf.gemm(A, B, out_dtype="bf16", compute_dtype="e4m3")
        assert calls == [], "mixed bf16×e4m3 should not route to cuBLAS"

    def test_activation_falls_through(self, monkeypatch):
        A, B = _make_ab(32, 32, 64, "bf16", "bf16")
        calls = _patch_record_cublas_calls(monkeypatch)
        with _env("POPCORN_DISABLE_CUBLAS", None):
            pcf.gemm(A, B, out_dtype="bf16", activation="silu")
        assert calls == [], "fused activation should not route to cuBLAS"

    def test_b_shuffled_falls_through(self, monkeypatch):
        A, B = _make_ab(32, 32, 64, "bf16", "bf16")
        calls = _patch_record_cublas_calls(monkeypatch)
        with _env("POPCORN_DISABLE_CUBLAS", None):
            try:
                pcf.gemm(A, B, out_dtype="bf16", b_shuffled=True)
            except Exception:
                # Custom kernel may reject b_shuffled=True on a
                # non-shuffled B tensor; the gate decision is what
                # this test cares about.
                pass
        assert calls == [], "b_shuffled=True should not route to cuBLAS"

    def test_e4m3_out_falls_through(self, monkeypatch):
        """cuBLAS fp8 path produces bf16 only — requesting e4m3 output
        must stay on the custom kernel."""
        A, B = _make_ab(32, 32, 64, "e4m3", "e4m3")
        calls = _patch_record_cublas_calls(monkeypatch)
        with _env("POPCORN_DISABLE_CUBLAS", None):
            pcf.gemm(A, B, out_dtype="e4m3")
        assert calls == [], "e4m3 output should not route to cuBLAS"
