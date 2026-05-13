"""GEMM bf16 OCL smoke — Phase 2 (compute-cohort coverage).

Runs ``pcf.gemm`` end-to-end (Launcher.compile + OclDriver.launch on
Intel) and compares against ``gemm_reference_numpy``. Gate:
cos_sim ≥ 0.9999.
"""

from __future__ import annotations

import numpy as np
import pytest

import quark.functional as pcf
from quark.kernels.gemm.reference import gemm_reference_numpy
from quark.kernels.gemm.spec import GemmSpec
from quark.runtime.tensor import QuarkTensor


def _ocl_available() -> bool:
    try:
        from quark.runtime.sync import _IS_OCL
    except Exception:
        return False
    return bool(_IS_OCL)


pytestmark = pytest.mark.skipif(
    not _ocl_available(),
    reason="OCL backend not available — skipping on non-Intel hosts",
)


def _f32_to_bf16(arr): return (np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
def _bf16_to_f32(arr): return (arr.astype(np.uint32) << 16).view(np.float32)


def _cos_sim(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0: return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


@pytest.mark.parametrize("M,N,K", [
    (64, 128, 128),
    (128, 256, 256),
])
def test_gemm_bf16_ocl_smoke_vs_numpy(M, N, K):
    """Simple bf16 GEMM (no bias / no gate / no activation)."""
    from quark.ir import DType
    rng = np.random.default_rng(0xA17E + M)
    A_f32 = rng.standard_normal((M, K)).astype(np.float32) * 0.25
    B_f32 = rng.standard_normal((N, K)).astype(np.float32) * 0.25  # B^T storage
    A_bf16 = _f32_to_bf16(A_f32)
    B_bf16 = _f32_to_bf16(B_f32)

    A_qt = QuarkTensor.from_numpy(A_bf16, dtype="bf16")
    B_qt = QuarkTensor.from_numpy(B_bf16, dtype="bf16")
    out_qt = pcf.gemm(A_qt, B_qt)

    from quark.runtime.sync import synchronize as _sync
    _sync()
    out_f32 = _bf16_to_f32(out_qt.to_numpy())

    spec = GemmSpec(M=M, N=N, K=K, a_dtype=DType.BF16, b_dtype=DType.BF16,
                    out_dtype=DType.BF16)
    ref = _bf16_to_f32(gemm_reference_numpy(spec, A=A_bf16, B=B_bf16))

    cs = _cos_sim(out_f32, ref)
    print(f"\n  M={M} N={N} K={K}: cos_sim = {cs:.6f}")
    assert cs >= 0.9999, (
        f"OCL gemm bf16 (M={M} N={N} K={K}) cos_sim {cs:.6f} < 0.9999"
    )
