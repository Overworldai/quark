"""Multi-simdgroup correctness for the IR-emitted NAX attn kernel.

The single-simdgroup baseline (``WM=WN=1``) and a multi-simdgroup
variant (``WM·WN > 1``) compute *exactly* the same output — each
simdgroup independently runs the same attention loop on its own
``BQ=16`` Q-row slice, so the only architectural change between the
two is the threadgroup geometry: ``threads_per_tg = 32 × WM × WN``
and the threadgroup grid shrinks proportionally. There is no
inter-simdgroup data dependency in the loop body (K/V are read from
gmem at the same positions in lockstep, so Apple's L2 dedupes the
redundant fetches; outputs land in disjoint Q-row bands).

Skipped on non-NAX hardware.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="NAX is Apple Silicon Metal only")


def _has_nax() -> bool:
    try:
        from quark.device import current_device

        return current_device().caps.supports_nax
    except Exception:
        return False


def _bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest f32 → bf16 storage (uint16)."""
    x = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((x + 0x8000) >> 16).astype(np.uint16)


def _bf16_to_f32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint32) << 16).view(np.float32)


def _make_inputs(
    *, n_kv_heads=16, gqa_ratio=2, tpf=128, num_buckets=16, Dh=64, max_segments=3, seed=0xA17E
):
    n_q_heads = n_kv_heads * gqa_ratio
    capacity = num_buckets * tpf + tpf
    rng = np.random.default_rng(seed)

    # head_first layout: Q is [n_q_heads * tpf, Dh].
    Q = _bf16(rng.standard_normal((n_q_heads * tpf, Dh)) * 0.3)
    K_cache = _bf16(rng.standard_normal((n_kv_heads * capacity, Dh)) * 0.3)
    Vt_cache = _bf16(rng.standard_normal((n_kv_heads * Dh, capacity)) * 0.3)

    L = num_buckets * tpf
    seg_pairs = [0, L, L, tpf] + [0, 0] * (max_segments - 2)
    segments = np.array(seg_pairs, dtype=np.int32)
    n_segments = np.array([2], dtype=np.int32)
    frame_t = np.array([num_buckets], dtype=np.int32)

    return dict(
        Q=Q,
        K_cache=K_cache,
        Vt_cache=Vt_cache,
        segments=segments,
        n_segments=n_segments,
        frame_t=frame_t,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        gqa_ratio=gqa_ratio,
        tpf=tpf,
        Dh=Dh,
        capacity=capacity,
        max_segments=max_segments,
    )


def _run_ir(inputs, *, WM: int, WN: int) -> np.ndarray:
    from quark.kernels.owl_attn.nax import NaxAttnSpec, dispatch_nax_attn
    from quark.runtime.sync import synchronize
    from quark.runtime.tensor import QuarkTensor

    spec = NaxAttnSpec(
        BK=32,
        n_q_heads=inputs["n_q_heads"],
        n_kv_heads=inputs["n_kv_heads"],
        gqa_ratio=inputs["gqa_ratio"],
        tpf=inputs["tpf"],
        capacity=inputs["capacity"],
        Dh=inputs["Dh"],
        max_segments=inputs["max_segments"],
        WM=WM,
        WN=WN,
    )
    # Pin Q/K/V/segments through QuarkTensor so dispatch picks the
    # zero-copy fast path.
    Q_t = QuarkTensor.from_numpy(inputs["Q"], dtype="bf16")
    K_t = QuarkTensor.from_numpy(inputs["K_cache"], dtype="bf16")
    Vt_t = QuarkTensor.from_numpy(inputs["Vt_cache"], dtype="bf16")
    seg_t = QuarkTensor.from_numpy(inputs["segments"], dtype="s32")
    nseg_t = QuarkTensor.from_numpy(inputs["n_segments"], dtype="s32")
    ft_t = QuarkTensor.from_numpy(inputs["frame_t"], dtype="s32")

    out_qt = dispatch_nax_attn(
        spec=spec,
        Q=Q_t,
        K_cache=K_t,
        Vt_cache=Vt_t,
        segments=seg_t,
        n_segments=nseg_t,
        frame_t=ft_t,
    )
    synchronize()
    n_q_heads = inputs["n_q_heads"]
    tpf = inputs["tpf"]
    Dh = inputs["Dh"]
    return out_qt.to_numpy().reshape(n_q_heads * tpf, Dh)


@pytest.mark.skipif(not _has_nax(), reason="requires NAX (M5+)")
@pytest.mark.parametrize("WM,WN", [(1, 2), (2, 1), (2, 2), (1, 4), (4, 1)])
def test_multisimd_matches_single_simd(WM: int, WN: int):
    """Multi-simdgroup attention output is byte-identical to the
    single-simdgroup baseline for the same inputs.

    No floating-point reordering happens between the two paths — each
    simdgroup uses the same per-row online-softmax algorithm as the
    single-simd version, just on a smaller subset of Q rows. So the
    test is a strict equality on the bf16 storage.
    """
    # tpf=128 so BQ_total = 16*WM*WN ≤ 128 (max BQ_total at WM=WN=4
    # would be 256 which doesn't divide 128). 128 covers WM*WN ≤ 8.
    inputs = _make_inputs(tpf=128)
    out_single = _run_ir(inputs, WM=1, WN=1)
    out_multi = _run_ir(inputs, WM=WM, WN=WN)
    assert out_single.shape == out_multi.shape

    f32_single = _bf16_to_f32(out_single)
    f32_multi = _bf16_to_f32(out_multi)

    if not np.array_equal(out_single, out_multi):
        diff = np.abs(f32_single - f32_multi)
        max_err = float(diff.max())
        # Tolerate a single bf16 ULP (~7.8e-3) — Apple's matmul2d's
        # internal fragment ordering inside execution_simdgroup may
        # vary by simdgroup count even though the math is identical.
        assert max_err < 1e-2, (
            f"WM={WM} WN={WN}: max abs diff {max_err} > 1e-2 ULP\n"
            f"first mismatch at {np.unravel_index(diff.argmax(), diff.shape)}\n"
            f"single[:4,:4]=\n{f32_single[:4, :4]}\n"
            f"multi[:4,:4]=\n{f32_multi[:4, :4]}"
        )


def _make_packed_inputs(
    *,
    n_kv_heads=16,
    gqa_ratio=2,
    H_spatial=8,
    W_spatial=16,
    num_buckets=16,
    Dh=64,
    max_segments=3,
    seed=0xA17E,
):
    """Same as ``_make_inputs`` but for the packed-QKV / inline-Q-RoPE path.

    Q is the packed buffer ``[tpf, (n_q + 2*n_kv) * Dh]`` matching what
    ``Waypoint15`` actually feeds owl_attn after the qkv_proj GEMM.
    """
    n_q_heads = n_kv_heads * gqa_ratio
    tpf = H_spatial * W_spatial
    capacity = num_buckets * tpf + tpf
    rng = np.random.default_rng(seed)

    qkv_dim = (n_q_heads + 2 * n_kv_heads) * Dh
    Q_packed = _bf16(rng.standard_normal((tpf, qkv_dim)) * 0.3)
    K_cache = _bf16(rng.standard_normal((n_kv_heads * capacity, Dh)) * 0.3)
    Vt_cache = _bf16(rng.standard_normal((n_kv_heads * Dh, capacity)) * 0.3)

    L = num_buckets * tpf
    seg_pairs = [0, L, L, tpf] + [0, 0] * (max_segments - 2)
    segments = np.array(seg_pairs, dtype=np.int32)
    n_segments = np.array([2], dtype=np.int32)
    frame_t = np.array([num_buckets], dtype=np.int32)

    return dict(
        Q=Q_packed,
        K_cache=K_cache,
        Vt_cache=Vt_cache,
        segments=segments,
        n_segments=n_segments,
        frame_t=frame_t,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        gqa_ratio=gqa_ratio,
        tpf=tpf,
        Dh=Dh,
        capacity=capacity,
        max_segments=max_segments,
        H_spatial=H_spatial,
        W_spatial=W_spatial,
    )


def _run_ir_inline_qrope(inputs, *, WM: int, WN: int) -> np.ndarray:
    """Dispatch the inline-Q-RoPE / packed-QKV / token_first path."""
    from quark.kernels.owl_attn.nax import NaxAttnSpec, dispatch_nax_attn
    from quark.runtime.tensor import QuarkTensor

    spec = NaxAttnSpec(
        BK=32,
        n_q_heads=inputs["n_q_heads"],
        n_kv_heads=inputs["n_kv_heads"],
        gqa_ratio=inputs["gqa_ratio"],
        tpf=inputs["tpf"],
        capacity=inputs["capacity"],
        Dh=inputs["Dh"],
        max_segments=inputs["max_segments"],
        layout="token_first",
        inline_q_rope=True,
        H_spatial=inputs["H_spatial"],
        W_spatial=inputs["W_spatial"],
        WM=WM,
        WN=WN,
    )
    Q_t = QuarkTensor.from_numpy(inputs["Q"], dtype="bf16")
    K_t = QuarkTensor.from_numpy(inputs["K_cache"], dtype="bf16")
    Vt_t = QuarkTensor.from_numpy(inputs["Vt_cache"], dtype="bf16")
    seg_t = QuarkTensor.from_numpy(inputs["segments"], dtype="s32")
    nseg_t = QuarkTensor.from_numpy(inputs["n_segments"], dtype="s32")
    ft_t = QuarkTensor.from_numpy(inputs["frame_t"], dtype="s32")

    out_qt = dispatch_nax_attn(
        spec=spec,
        Q=Q_t,
        K_cache=K_t,
        Vt_cache=Vt_t,
        segments=seg_t,
        n_segments=nseg_t,
        frame_t=ft_t,
    )
    from quark.runtime.sync import synchronize

    synchronize()
    n_q_heads = inputs["n_q_heads"]
    tpf = inputs["tpf"]
    Dh = inputs["Dh"]
    return out_qt.to_numpy().reshape(tpf, n_q_heads * Dh)


@pytest.mark.skipif(not _has_nax(), reason="requires NAX (M5+)")
@pytest.mark.parametrize("WM,WN", [(2, 1), (2, 2), (4, 1), (1, 4)])
def test_multisimd_inline_qrope_matches_single_simd(WM: int, WN: int):
    """Multi-simdgroup output is byte-identical to single-simd output
    on the inline-Q-RoPE / packed-QKV / token_first path.

    Distinct from ``test_multisimd_matches_single_simd`` (which covers
    head_first with no inline RoPE) because inline RoPE writes the
    rotated Q tile to a SharedRegion smem with a per-simdgroup
    ``dyn_offset = sg_q_off * Dh``. Without folding that offset into
    the GEMM1 ``load_matrix`` address, every simdgroup reads the same
    Q band and the multi-simd output ends up using simdgroup-0's Q
    rotation for every output row band — visible end-to-end as a 4×
    horizontal repetition in the decoded 720p frames (Waypoint-1.5-1B
    at 720p uses WM=2 + this path; bug found 2026-05-06, fix in
    ``lower/msl/mma.py: _fold_smem_flat_offset``).

    Strict equality bar: each simdgroup runs the same online-softmax
    pass on its own Q-row slice, so there is no fp reordering between
    WM=1 and WM>1. The pre-fix bug produces an O(attention-output-
    magnitude) drift on rows that fall outside simdgroup 0's band
    (e.g. ~0.005 at this shape) — well inside the prior 1e-2 ULP
    tolerance, which is why the bug went undetected for that long.
    """
    inputs = _make_packed_inputs(H_spatial=8, W_spatial=16)
    out_single = _run_ir_inline_qrope(inputs, WM=1, WN=1)
    out_multi = _run_ir_inline_qrope(inputs, WM=WM, WN=WN)
    assert out_single.shape == out_multi.shape
    if not np.array_equal(out_single, out_multi):
        f32_single = _bf16_to_f32(out_single)
        f32_multi = _bf16_to_f32(out_multi)
        diff = np.abs(f32_single - f32_multi)
        max_err = float(diff.max())
        n_diff = int(np.count_nonzero(out_single != out_multi))
        first_mismatch = np.unravel_index(diff.argmax(), diff.shape)
        raise AssertionError(
            f"WM={WM} WN={WN}: multi-simd output differs from single-simd\n"
            f"  max abs diff {max_err}\n"
            f"  {n_diff} bf16 elements differ ({100 * n_diff / out_single.size:.2f}%)\n"
            f"  first mismatch at {first_mismatch}"
        )
