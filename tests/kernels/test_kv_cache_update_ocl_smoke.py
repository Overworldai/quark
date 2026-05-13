"""KV cache update OCL numerics smoke — Phase 2D.2 of
``docs/OCL_E2E_PLAN.md``.

Runs ``pcf.kv_cache_update`` end-to-end (Launcher.compile +
OclDriver.launch on Intel) and compares against
``kv_cache_update_reference_numpy``. Gate: cos_sim ≥ 0.9999 on
each output tensor (K_cache, Vt_cache).

KV cache update is a pure compute-cohort kernel (RoPE + ring write,
no MMA, no FragConvert). Same shape as the RMSNorm smoke that
passed in 2B.2.
"""

from __future__ import annotations

import numpy as np
import pytest

import quark.functional as pcf
from quark.kernels.kv_cache_update.reference import (
    kv_cache_update_reference_numpy,
)
from quark.kernels.kv_cache_update.spec import KVCacheUpdateSpec
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


def test_kv_cache_update_ocl_smoke_vs_numpy():
    """Single-frame ring write. Small shape exercises RoPE + segment
    write + ring buffer indexing."""
    from quark.ir import DType

    B = 1
    n_kv_heads = 2
    Dh = 64
    H_spatial = 8
    W_spatial = 8
    num_buckets = 4
    pinned_dilation = 1
    max_segments = 3
    tpf = H_spatial * W_spatial
    capacity = num_buckets * tpf + tpf

    rng = np.random.default_rng(0xCAFE)
    K_f32 = rng.standard_normal((B * n_kv_heads * tpf, Dh)).astype(np.float32) * 0.25
    V_f32 = rng.standard_normal((B * n_kv_heads * tpf, Dh)).astype(np.float32) * 0.25
    K_bf16 = _f32_to_bf16_carrier(K_f32)
    V_bf16 = _f32_to_bf16_carrier(V_f32)

    # frame_t=0: first frame. segments + n_segments are out-only, init
    # them to zeros; the kernel writes the new segment entry.
    frame_t_np = np.array([0], dtype=np.int32)
    frozen_np = np.array([0], dtype=np.int32)

    K_qt = QuarkTensor.from_numpy(K_bf16, dtype="bf16")
    V_qt = QuarkTensor.from_numpy(V_bf16, dtype="bf16")
    frame_t_qt = QuarkTensor.from_numpy(frame_t_np, dtype="s32")
    frozen_qt = QuarkTensor.from_numpy(frozen_np, dtype="s32")

    # Pre-allocate output buffers — the functional dispatcher expects
    # K_cache / Vt_cache / segments / n_segments to exist and writes
    # into them. Initialize to zero so unwritten positions stay 0 in
    # both the device + reference.
    K_cache_qt = QuarkTensor.zeros(
        B * n_kv_heads * capacity, Dh, dtype="bf16",
    )
    Vt_cache_qt = QuarkTensor.zeros(
        B * n_kv_heads * Dh, capacity, dtype="bf16",
    )
    segments_qt = QuarkTensor.zeros(B * max_segments * 2, dtype="s32")
    n_segments_qt = QuarkTensor.zeros(B, dtype="s32")

    pcf.kv_cache_update(
        K_qt, V_qt, frame_t_qt, frozen_qt,
        Vt_cache_qt, segments_qt, n_segments_qt, K_cache_qt,
        B=B, n_kv_heads=n_kv_heads,
        H_spatial=H_spatial, W_spatial=W_spatial,
        num_buckets=num_buckets, pinned_dilation=pinned_dilation,
        max_segments=max_segments,
    )

    from quark.runtime.sync import synchronize as _sync
    _sync()
    K_out_f32 = _bf16_to_f32(K_cache_qt.to_numpy())
    Vt_out_f32 = _bf16_to_f32(Vt_cache_qt.to_numpy())

    # Reference.
    spec = KVCacheUpdateSpec(
        B=B, n_kv_heads=n_kv_heads, Dh=Dh,
        H_spatial=H_spatial, W_spatial=W_spatial,
        num_buckets=num_buckets, pinned_dilation=pinned_dilation,
        in_dtype=DType.BF16, kv_dtype=DType.BF16,
        max_segments=max_segments,
    )
    ref = kv_cache_update_reference_numpy(
        spec, K=K_bf16, V=V_bf16,
        frame_t=frame_t_np, frozen=frozen_np,
    )
    K_ref_f32 = _bf16_to_f32(ref["K_cache"])
    Vt_ref_f32 = _bf16_to_f32(ref["Vt_cache"])

    cs_k = _cos_sim(K_out_f32, K_ref_f32)
    cs_vt = _cos_sim(Vt_out_f32, Vt_ref_f32)
    print(f"\n  K_cache  cos_sim = {cs_k:.6f}")
    print(f"  Vt_cache cos_sim = {cs_vt:.6f}")
    assert cs_k >= 0.9999, f"K_cache cos_sim {cs_k:.6f} < 0.9999"
    assert cs_vt >= 0.9999, f"Vt_cache cos_sim {cs_vt:.6f} < 0.9999"
