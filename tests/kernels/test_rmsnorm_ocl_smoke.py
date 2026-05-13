"""RMSNorm bf16 numerics smoke on OCL — Phase 2B.2 of
``docs/OCL_E2E_PLAN.md``.

Runs ``pcf.rmsnorm`` end-to-end (Launcher.compile + OclDriver.launch
on Intel) and compares against ``rmsnorm_reference_numpy``. Gate:
cos_sim ≥ 0.9999.

RMSNorm is a pure compute-cohort kernel (elementwise + reduce, no
MMA), so unlike the OwlAttn smoke this one doesn't depend on
FragConvertOp K-widening or the cooperative_matrix path.
"""

from __future__ import annotations

import numpy as np
import pytest

import quark.functional as pcf
from quark.kernels.rmsnorm.reference import rmsnorm_reference_numpy
from quark.kernels.rmsnorm.spec import RMSNormSpec
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


def _f32_to_bf16_carrier(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return (arr.view(np.uint32) >> 16).astype(np.uint16)


def _bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


@pytest.mark.parametrize("D", [128, 512, 2048])
def test_rmsnorm_ocl_smoke_vs_numpy(D):
    """RMSNorm at three D widths — small (one warp), production
    (Waypoint d_model=2048), and an in-between to catch off-by-one
    tile bugs."""
    from quark.ir import DType

    rng = np.random.default_rng(0xA17E + D)
    M = 64  # rows
    x_f32 = (rng.standard_normal((M, D)) * 0.5).astype(np.float32)
    x_bf16 = _f32_to_bf16_carrier(x_f32)

    x_qt = QuarkTensor.from_numpy(x_bf16, dtype="bf16")
    out_qt = pcf.rmsnorm(x_qt, eps=1e-6)

    from quark.runtime.sync import synchronize as _sync
    _sync()
    out_bf16 = out_qt.to_numpy()
    out_f32 = _bf16_to_f32(out_bf16)

    spec = RMSNormSpec(B=M, D=D, dtype=DType.BF16, eps=1e-6)
    ref_bf16 = rmsnorm_reference_numpy(spec, X=x_bf16)
    ref_f32 = _bf16_to_f32(ref_bf16)

    cs = _cos_sim(out_f32, ref_f32)
    print(f"\n  D={D}: cos_sim = {cs:.6f}")
    assert cs >= 0.9999, (
        f"OCL RMSNorm D={D} cos_sim {cs:.6f} < 0.9999 vs numpy reference"
    )
