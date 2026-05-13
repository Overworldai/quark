"""OwlAttn bf16 numerics smoke on OCL — closes Phase 2A.2 of
``docs/OCL_E2E_PLAN.md``.

One shape per ``mma_cfg`` (bf16/bf16/f32 and bf16/bf16/bf16),
end-to-end through ``OclDriver`` on Intel hardware, compared
against ``owl_attn_reference_numpy``. Gate: cos_sim ≥ 0.9999.

Skipped on non-OCL hosts (no Intel iGPU/Arc).
"""

from __future__ import annotations

import numpy as np
import pytest

import quark.functional as pcf
from quark.kernels.owl_attn.reference import owl_attn_reference_numpy
from quark.kernels.owl_attn.spec import OwlAttnSpec
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
    """``[*] f32`` → ``[*] uint16`` bf16 carrier (high 16 bits of f32)."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return (arr.view(np.uint32) >> 16).astype(np.uint16)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def _bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    """Inverse of ``_f32_to_bf16_carrier``."""
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


def _make_inputs(
    *,
    seed: int,
    B: int = 1,
    n_kv_heads: int = 2,
    gqa_ratio: int = 2,
    H_spatial: int = 8,
    W_spatial: int = 8,
    num_buckets: int = 4,
    pinned_dilation: int = 1,
    Dh: int = 64,
    max_segments: int = 3,
):
    """Smoke-sized owl_attn inputs. Shapes are smaller than production
    (Waypoint runs 16×32 spatial, num_buckets=16) but exercise the
    full kernel — quilt_factor=1, packed_qkv=False, 1 segment of
    length=tpf so masking trivially fires."""
    rng = np.random.default_rng(seed)
    tpf = H_spatial * W_spatial
    n_q_heads = n_kv_heads * gqa_ratio
    capacity = num_buckets * tpf + tpf

    # head_first Q layout: [n_q_heads * tpf, Dh] bf16
    Q_f32 = rng.standard_normal((n_q_heads * tpf, Dh)).astype(np.float32) * 0.125
    K_f32 = rng.standard_normal((n_kv_heads * capacity, Dh)).astype(np.float32) * 0.125
    Vt_f32 = rng.standard_normal((n_kv_heads * Dh, capacity)).astype(np.float32) * 0.125

    Q_bf16 = _f32_to_bf16_carrier(Q_f32)
    K_bf16 = _f32_to_bf16_carrier(K_f32)
    Vt_bf16 = _f32_to_bf16_carrier(Vt_f32)

    # One full-length segment, rest zero — exercises the mask path
    # without making the test reduce to a trivial all-valid case.
    segments = np.zeros((B, max_segments, 2), dtype=np.int32)
    segments[0, 0, 0] = 0
    segments[0, 0, 1] = tpf  # length covers the first tpf tokens
    n_segments = np.array([1] * B, dtype=np.int32)
    frame_t = np.array([0], dtype=np.int32)

    return {
        "Q_qt": QuarkTensor.from_numpy(Q_bf16, dtype="bf16"),
        "K_cache_qt": QuarkTensor.from_numpy(K_bf16, dtype="bf16"),
        "Vt_cache_qt": QuarkTensor.from_numpy(Vt_bf16, dtype="bf16"),
        "segments_qt": QuarkTensor.from_numpy(
            segments.reshape(-1), dtype="s32",
        ),
        "n_segments_qt": QuarkTensor.from_numpy(n_segments, dtype="s32"),
        "frame_t_qt": QuarkTensor.from_numpy(frame_t, dtype="s32"),
        # Raw numpy too, for the reference.
        "Q_np": Q_bf16,
        "K_np": K_bf16,
        "Vt_np": Vt_bf16,
        "segments_np": segments,
        "n_segments_np": n_segments,
        "frame_t_np": frame_t,
        # Geometry.
        "B": B,
        "n_kv_heads": n_kv_heads,
        "gqa_ratio": gqa_ratio,
        "H_spatial": H_spatial,
        "W_spatial": W_spatial,
        "num_buckets": num_buckets,
        "pinned_dilation": pinned_dilation,
        "max_segments": max_segments,
    }


@pytest.mark.parametrize("compute_dtype", ["f32", "bf16"])
def test_owl_attn_ocl_smoke_vs_numpy(compute_dtype):
    """OwlAttn bf16 inputs, compute in f32 (acc_dtype=f32) and bf16
    (acc_dtype=bf16). Compares OCL output against numpy ref."""
    from quark.ir import DType

    inputs = _make_inputs(seed=0xA17E)

    # Run through pcf.owl_attn (Launcher.compile + OclDriver.launch
    # on Intel). Returns a QuarkTensor.
    out_qt = pcf.owl_attn(
        inputs["Q_qt"],
        inputs["K_cache_qt"],
        inputs["Vt_cache_qt"],
        inputs["segments_qt"],
        inputs["n_segments_qt"],
        B=inputs["B"],
        n_kv_heads=inputs["n_kv_heads"],
        gqa_ratio=inputs["gqa_ratio"],
        H_spatial=inputs["H_spatial"],
        W_spatial=inputs["W_spatial"],
        num_buckets=inputs["num_buckets"],
        pinned_dilation=inputs["pinned_dilation"],
        compute_dtype=DType.BF16 if compute_dtype == "bf16" else DType.F32,
        max_segments=inputs["max_segments"],
        frame_t=inputs["frame_t_qt"],
    )

    # Bring OCL output to numpy.
    from quark.runtime.sync import synchronize as _sync
    _sync()
    out_bf16 = out_qt.to_numpy()  # uint16 bf16 carrier
    out_f32 = _bf16_to_f32(out_bf16)

    # Build a matching spec for the reference.
    spec = OwlAttnSpec(
        B=inputs["B"],
        n_kv_heads=inputs["n_kv_heads"],
        gqa_ratio=inputs["gqa_ratio"],
        H_spatial=inputs["H_spatial"],
        W_spatial=inputs["W_spatial"],
        num_buckets=inputs["num_buckets"],
        pinned_dilation=inputs["pinned_dilation"],
        Dh=64,
        a_dtype=DType.BF16,
        kv_dtype=DType.BF16,
        out_dtype=DType.BF16,
        compute_dtype=DType.BF16 if compute_dtype == "bf16" else DType.F32,
        max_segments=inputs["max_segments"],
        packed_qkv=False,
    )

    ref_bf16 = owl_attn_reference_numpy(
        spec,
        Q=inputs["Q_np"],
        K_cache=inputs["K_np"],
        Vt_cache=inputs["Vt_np"],
        segments=inputs["segments_np"],
        n_segments=inputs["n_segments_np"],
        frame_t=inputs["frame_t_np"],
    )
    ref_f32 = _bf16_to_f32(ref_bf16)

    cs = _cos_sim(out_f32, ref_f32)
    print(f"\n  compute_dtype={compute_dtype}: cos_sim = {cs:.6f}")
    assert cs >= 0.9999, (
        f"OCL owl_attn (compute={compute_dtype}) cos_sim {cs:.6f} < 0.9999 "
        f"vs numpy reference"
    )
