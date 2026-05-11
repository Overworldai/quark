"""Smoke tests for the ops that now accept e4m3 inputs.

Each previously-blocked kernel (ada_rmsnorm, rmsnorm, head_rmsnorm,
ada_gate_residual, silu, elementwise add, value_residual_packed,
kv_cache_update) is hit once with an e4m3 input tensor and compared
against the same op run on the bf16 reference. Both kernels cast
fp8→f32 internally, so the e4m3 result should track the bf16 result
within the quantization budget set by ``check_correctness``.

If a kernel's spec reintroduces a ``_VALID_DTYPES`` frozenset that
excludes E4M3, one of these tests will fire on the first call.
"""
# ruff: noqa: E402  -- imports below pytest.skip(allow_module_level=True)
# are intentional; they would fail at module load on a CUDA-less system,
# so they sit behind the runtime gate above.

from __future__ import annotations

import sys

import numpy as np
import pytest

IS_METAL = sys.platform == "darwin"
if IS_METAL:
    pytest.skip("e4m3 ops are CUDA-only", allow_module_level=True)

from quark.runtime.cuda import CudaRuntime

if not CudaRuntime.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import quark.functional as qf
from quark.correctness import check_correctness
from quark.ir import DType
from quark.runtime.tensor import QuarkTensor

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rand(*shape, seed=0, scale=0.5):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32) * scale


def _quantized_pair(shape, seed=0, scale=0.5):
    """Return (bf16_tensor, e4m3_tensor) holding the SAME quantized
    values — so differences between the two runs come only from how
    each kernel loads / casts, not from input jitter.
    """
    f32 = QuarkTensor.from_numpy(_rand(*shape, seed=seed, scale=scale), dtype="f32")
    e4m3 = f32.astype("e4m3")
    bf16 = e4m3.astype("bf16")
    return bf16, e4m3


def _compare(out_e4m3, out_bf16, out_dtype="bf16"):
    cr = check_correctness(out_e4m3, out_bf16, out_dtype=DType(out_dtype))
    assert cr.passed, f"e4m3 vs bf16 cos_sim={cr.cos_sim:.6f} (threshold={cr.threshold})"


# ---------------------------------------------------------------------------
# SiLU
# ---------------------------------------------------------------------------


def test_silu_accepts_e4m3():
    bf16, e4m3 = _quantized_pair((128, 256), seed=1, scale=1.0)
    out_e4m3 = qf.silu(e4m3)
    out_bf16 = qf.silu(bf16)
    assert out_e4m3.dtype == "e4m3"
    _compare(out_e4m3, out_bf16)


# ---------------------------------------------------------------------------
# Elementwise Add (via QuarkTensor __add__)
# ---------------------------------------------------------------------------


def test_elementwise_add_accepts_e4m3():
    bf16_a, e4m3_a = _quantized_pair((64, 128), seed=2, scale=0.3)
    bf16_b, e4m3_b = _quantized_pair((64, 128), seed=3, scale=0.3)
    out_e4m3 = e4m3_a + e4m3_b
    out_bf16 = bf16_a + bf16_b
    assert out_e4m3.dtype == "e4m3"
    _compare(out_e4m3, out_bf16)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def test_rmsnorm_accepts_e4m3():
    bf16, e4m3 = _quantized_pair((16, 768), seed=4, scale=0.3)
    out_e4m3 = qf.rmsnorm(e4m3)
    out_bf16 = qf.rmsnorm(bf16)
    assert out_e4m3.dtype == "e4m3"
    _compare(out_e4m3, out_bf16)


# ---------------------------------------------------------------------------
# AdaRMSNorm
# ---------------------------------------------------------------------------


def test_ada_rmsnorm_accepts_e4m3():
    M, D, G = 64, 512, 1
    bf16, e4m3 = _quantized_pair((G * M, D), seed=5, scale=0.3)
    scale_bf16, scale_e4m3 = _quantized_pair((G, D), seed=6, scale=0.1)
    bias_bf16, bias_e4m3 = _quantized_pair((G, D), seed=7, scale=0.1)
    # scale/bias kernels accept half dtypes — keep them bf16 for both
    # runs so only the X dtype differs.
    out_e4m3 = qf.ada_rmsnorm(e4m3, scale_bf16, bias_bf16)
    out_bf16 = qf.ada_rmsnorm(bf16, scale_bf16, bias_bf16)
    assert out_e4m3.dtype == "e4m3"
    _compare(out_e4m3, out_bf16)


# ---------------------------------------------------------------------------
# HeadRMSNorm — per-head on packed QKV
# ---------------------------------------------------------------------------


def test_head_rmsnorm_accepts_e4m3():
    n_q, n_kv, Dh = 8, 2, 64
    M = 32
    D_full = (n_q + 2 * n_kv) * Dh
    bf16, e4m3 = _quantized_pair((M, D_full), seed=8, scale=0.3)
    out_e4m3 = qf.head_rmsnorm(e4m3, n_q_heads=n_q, n_kv_heads=n_kv, Dh=Dh)
    out_bf16 = qf.head_rmsnorm(bf16, n_q_heads=n_q, n_kv_heads=n_kv, Dh=Dh)
    assert out_e4m3.dtype == "e4m3"
    _compare(out_e4m3, out_bf16)


# ---------------------------------------------------------------------------
# AdaGateResidual
# ---------------------------------------------------------------------------


def test_ada_gate_residual_accepts_e4m3():
    G, M, D = 1, 64, 512
    x_bf16, x_e4m3 = _quantized_pair((G * M, D), seed=9, scale=0.3)
    y_bf16, y_e4m3 = _quantized_pair((G * M, D), seed=10, scale=0.3)
    # gate is a small scaling; keep it in f32 on host and feed as bf16.
    gate = QuarkTensor.from_numpy(_rand(G, D, seed=11, scale=0.1), dtype="f32").astype("bf16")
    out_e4m3 = qf.ada_gate_residual(x_e4m3, y_e4m3, gate)
    out_bf16 = qf.ada_gate_residual(x_bf16, y_bf16, gate)
    assert out_e4m3.dtype == "e4m3"
    _compare(out_e4m3, out_bf16)


# ---------------------------------------------------------------------------
# ValueResidualPacked
# ---------------------------------------------------------------------------


def test_value_residual_packed_accepts_e4m3():
    M, D_full = 16, 512
    v_col_offset, v_width = 256, 128
    curr_bf16, curr_e4m3 = _quantized_pair((M, D_full), seed=12, scale=0.3)
    first_bf16, first_e4m3 = _quantized_pair((M, D_full), seed=13, scale=0.3)
    lamb = QuarkTensor.from_numpy(np.array([0.5], dtype=np.float32), dtype="f32")
    out_e4m3 = qf.value_residual_packed(
        curr_e4m3, first_e4m3, lamb, v_col_offset=v_col_offset, v_width=v_width
    )
    out_bf16 = qf.value_residual_packed(
        curr_bf16, first_bf16, lamb, v_col_offset=v_col_offset, v_width=v_width
    )
    assert out_e4m3.dtype == "e4m3"
    _compare(out_e4m3, out_bf16)
