"""gemm_int s8/s32 numerics smoke on OCL — Phase 2C.2 of
``docs/OCL_E2E_PLAN.md``.

Exercises the Intel cl_intel_subgroup_matrix_multiply_accumulate
s8/s32 path (``m8n16k32_intel_s8_s32`` MMA shape) end-to-end:
s8 inputs + per-row f32 scales → bf16 output. Compared against
``gemm_int_reference_numpy``. Gate: cos_sim ≥ 0.9999.

There's no ``pcf.gemm_int`` dispatcher so we drive the kernel
directly via ``call_with_bindings``.
"""

from __future__ import annotations

import numpy as np
import pytest

from quark.functional._dispatch import call_with_bindings
from quark.kernels.gemm_int import GemmIntKernel, GemmIntSpec
from quark.kernels.gemm_int.reference import gemm_int_reference_numpy
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


@pytest.mark.parametrize("M,N,K", [
    (32, 32, 64),   # tiny: one block per dim
    (64, 64, 128),  # multi-block
])
def test_gemm_int_ocl_smoke_vs_numpy(M, N, K):
    """s8 × s8 → s32 (with per-row scale) → bf16. Production
    Waypoint shapes are (512, 4096, 2048)-ish; smoke uses smaller
    shapes that still exercise multi-block dispatch."""
    from quark.ir import DType

    rng = np.random.default_rng(0xA17E + M * 100 + N)
    A_s8 = rng.integers(-8, 8, size=(M, K), dtype=np.int8)
    B_s8 = rng.integers(-8, 8, size=(K, N), dtype=np.int8)
    A_scales = (rng.standard_normal(M).astype(np.float32) * 0.1 + 1.0)
    B_scales = (rng.standard_normal(N).astype(np.float32) * 0.1 + 1.0)

    A_qt = QuarkTensor.from_numpy(A_s8, dtype="s8")
    B_qt = QuarkTensor.from_numpy(B_s8, dtype="s8")
    A_scales_qt = QuarkTensor.from_numpy(A_scales, dtype="f32")
    B_scales_qt = QuarkTensor.from_numpy(B_scales, dtype="f32")

    spec = GemmIntSpec(M=M, N=N, K=K, out_dtype=DType.BF16)
    result = call_with_bindings(
        GemmIntKernel, spec,
        provided={
            "A": A_qt, "B": B_qt,
            "A_scales": A_scales_qt, "B_scales": B_scales_qt,
        },
        auto_alloc=("Out",),
        like=A_qt,
    )
    out_qt = result["Out"]

    from quark.runtime.sync import synchronize as _sync
    _sync()
    out_bf16 = out_qt.to_numpy()  # uint16 bf16 carrier
    out_f32 = _bf16_to_f32(out_bf16)

    ref = gemm_int_reference_numpy(
        spec, A=A_s8, B=B_s8, A_scales=A_scales, B_scales=B_scales,
    )
    ref_bf16 = ref["Out"]
    ref_f32 = _bf16_to_f32(ref_bf16)

    cs = _cos_sim(out_f32, ref_f32)
    print(f"\n  M={M} N={N} K={K}: cos_sim = {cs:.6f}")
    assert cs >= 0.9999, (
        f"OCL gemm_int (M={M} N={N} K={K}) cos_sim {cs:.6f} < 0.9999 "
        f"vs numpy reference"
    )
