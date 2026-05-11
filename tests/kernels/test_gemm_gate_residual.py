"""Parity test for the fused GEMM + AdaGate-residual epilogue.

The kernel emits ``Out = Residual + Gate_bcast * (A @ Bᵀ)`` in one
dispatch, replacing the standalone ``qf.gemm`` followed by
``qf.ada_gate_residual``. This test verifies the fused output
matches the reference numpy computation within the GEMM kernel's
correctness threshold.

Metal-only PoC: NAX path falls through to the simdgroup_matrix
build() when ``has_gate_residual=True`` (see
``GemmKernel.is_valid``). CUDA backend is wired through the same
spec/store_acc plumbing but not exercised here — the smoke
``tests/kernels/test_smoke.py`` covers the CUDA path once a GEMM
gate-residual problem is added to the registry.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

IS_METAL = sys.platform == "darwin"

if not IS_METAL:
    pytest.skip("PoC is Metal-only (CUDA path TBD)", allow_module_level=True)


def _bf16_carrier(arr_f32: np.ndarray) -> np.ndarray:
    """numpy f32 → bf16-as-uint16 carrier (the Metal model's storage)."""
    raw = (np.ascontiguousarray(arr_f32, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return raw


def _bf16_to_f32(carrier_u16: np.ndarray) -> np.ndarray:
    return (carrier_u16.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize(
    "M,N,K,G",
    [
        # Waypoint attn_gate / mlp_gate shapes at 360p (tpf=128, d=2048):
        (128, 2048, 2048, 1),  # out_proj → attn_gate
        (128, 2048, 8192, 1),  # mlp.fc2 → mlp_gate
        # Smaller for fast iteration:
        (64, 128, 256, 1),
        # Multi-group case:
        (32, 64, 128, 4),
    ],
)
def test_fused_gate_residual_matches_reference(M, N, K, G):
    """Fused kernel output matches ``residual + gate_bcast * (A @ B.T)``."""
    from quark.functional._dispatch import IS_METAL as _ENV_METAL

    if not _ENV_METAL:
        pytest.skip("Metal env required")

    import quark.functional as qf

    rng = np.random.default_rng(0xA17E)
    a_f32 = rng.standard_normal((M, K)).astype(np.float32) * 0.1
    b_f32 = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    gate_f32 = rng.standard_normal((G, N)).astype(np.float32) * 0.5
    residual_f32 = rng.standard_normal((M, N)).astype(np.float32) * 0.1

    # Reference in f32: residual + repeat(gate, M/G, axis=0) * (A @ B.T).
    m_per_group = M // G
    gate_bcast_f32 = np.repeat(gate_f32, m_per_group, axis=0)
    ref_f32 = residual_f32 + gate_bcast_f32 * (a_f32 @ b_f32.T)

    # Convert to the Metal numpy carriers (bf16-as-uint16, tagged).
    from quark.runtime.tensor import QuarkTensor

    A = QuarkTensor.from_numpy(_bf16_carrier(a_f32), dtype="bf16").reshape(M, K)
    B = QuarkTensor.from_numpy(_bf16_carrier(b_f32), dtype="bf16").reshape(N, K)
    Gate = QuarkTensor.from_numpy(_bf16_carrier(gate_f32), dtype="bf16").reshape(G, N)
    Residual = QuarkTensor.from_numpy(_bf16_carrier(residual_f32), dtype="bf16").reshape(M, N)

    out = qf.gemm(A, B, gate=Gate, residual=Residual, gate_groups=G, out_dtype="bf16")
    # Explicit drain: ``qf.gemm`` enqueues the dispatch through
    # ``queue_launch`` (the NAX fast path) and ``np.asarray(out)``'s
    # implicit flush has been observed to race with the kernel completion
    # under pytest's tighter scheduling. The world-engine bench wraps
    # forwards in ``quark.lazy()`` which fences end-of-frame; tests that
    # read a single output should sync explicitly.
    from quark.runtime.sync import synchronize

    synchronize()

    out_np = np.asarray(out).view(np.uint16).reshape(M, N)
    out_f32 = _bf16_to_f32(out_np)

    # bf16 has ~3 decimal digits of mantissa; ref is f32. Use cosine
    # similarity (the GEMM correctness gate's metric) — ada_gate_residual
    # uses the same dtype so this should clear 0.999 easily.
    flat_out = out_f32.reshape(-1)
    flat_ref = ref_f32.reshape(-1)
    cos = float(
        np.dot(flat_out, flat_ref) / (np.linalg.norm(flat_out) * np.linalg.norm(flat_ref) + 1e-30)
    )
    max_abs = float(np.max(np.abs(flat_out - flat_ref)))
    print(f"[M={M} N={N} K={K} G={G}] cos={cos:.6f} max|diff|={max_abs:.4f}")
    assert cos >= 0.99, f"fused gate_residual cos={cos:.6f} below 0.99 threshold"
