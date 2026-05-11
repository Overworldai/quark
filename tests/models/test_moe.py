"""Smoke + capacity tests for ``nn.MoE``.

Exercises constructor, buffer sizing, and routing-mode validation for
every supported routing. The MoE block allocates its cached buffers
in ``__init__`` (via ``QuarkTensor.empty`` on CUDA / ``mx.zeros`` on
Metal), so the tests need a runnable backend even though they don't
launch any kernels. CUDA-less Linux CI is skipped at module load.
The kernel-level launches are covered by ``tests/kernels/test_smoke``.
"""

# are intentional; constructing nn.MoE allocates QuarkTensor buffers
# which need libcuda on non-Metal platforms.

from __future__ import annotations

import sys

import pytest

if sys.platform != "darwin":
    # On Linux / Windows we need CUDA for the QuarkTensor allocator.
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip(
            "nn.MoE construction needs CUDA on non-Metal platforms",
            allow_module_level=True,
        )

from quark.nn.moe import MoE

# ---------------------------------------------------------------
# Constructor + buffer sizing
# ---------------------------------------------------------------


@pytest.mark.parametrize("routing", ["balanced", "shared_experts", "correct"])
def test_construct_each_routing_mode(routing):
    """Every routing mode constructs cleanly at the canonical W1.5
    config (M=128, E=16, K=4) and exposes the expected buffer shapes."""
    m = MoE(
        M=128,
        d_model=2048,
        d_intermediate=2048,
        n_experts=16,
        top_k=4,
        routing=routing,
    )
    assert m._routing == routing
    assert m._E == 16
    assert m._K == 4
    assert m._M == 128
    # token_ids / slot_weights buffer always sized E*capacity.
    assert m._buf_token_ids.shape == (m._E * m._C,)
    assert m._buf_slot_weights.shape == (m._E * m._C,)
    assert m._buf_counts.shape == (m._E,)
    # Per-mode workspace buffers exist only where needed.
    if routing == "shared_experts":
        assert hasattr(m, "_buf_chosen_experts")
        assert hasattr(m, "_buf_cum_probs")
    if routing == "correct":
        assert hasattr(m, "_buf_offsets")


def test_balanced_capacity_is_M_times_K_over_E_rounded_to_BM():
    m = MoE(M=128, d_model=2048, d_intermediate=2048, n_experts=16, top_k=4)
    # M*K/E = 128*4/16 = 32 → already a multiple of BM=32.
    assert m._C == 32


def test_correct_capacity_is_worst_case_padded():
    m = MoE(
        M=128,
        d_model=2048,
        d_intermediate=2048,
        n_experts=16,
        top_k=4,
        routing="correct",
    )
    # Worst case = M*K + E*(BM-1) = 512 + 480 = 992; per-expert 992/16 = 62
    # rounded up to BM=32 → 64. So total_slots = 16*64 = 1024.
    assert m._C == 64
    assert m._E * m._C == 1024


def test_shared_experts_requires_clean_K_M_E_alignment():
    """Shared-experts routing requires E*capacity == K*M exactly. This
    holds when M*K is a multiple of E*BM=512; it doesn't for arbitrary M."""
    # M=128, K=4, E=16 → M*K=512, E*BM=512: clean.
    MoE(M=128, d_model=2048, d_intermediate=2048, n_experts=16, top_k=4, routing="shared_experts")
    # M=144 (not a multiple of 128) → M*K=576, would round capacity to 64,
    # giving E*C=1024 != M*K=576. Should reject.
    with pytest.raises(ValueError, match="shared_experts requires"):
        MoE(
            M=144,
            d_model=2048,
            d_intermediate=2048,
            n_experts=16,
            top_k=4,
            routing="shared_experts",
        )


def test_unknown_routing_rejected():
    with pytest.raises(ValueError, match="routing must be"):
        MoE(M=128, d_model=2048, d_intermediate=2048, n_experts=16, top_k=4, routing="nonsense")


# ---------------------------------------------------------------
# MoE-specific fp8 opt-out (via ``moe_fp8`` kwarg)
# ---------------------------------------------------------------


def test_moe_fp8_kwarg_keeps_experts_bf16():
    """``MoE.prepare(fp8=True, moe_fp8=False)`` keeps experts in bf16 even
    when the surrounding model passes ``fp8=True``. Mirrors what
    ``Waypoint15.prepare`` does when ``cfg.quant.moe == "bf16"``.

    Scoped to the ``effective_fp8 == False`` branch so the assertion
    runs on both Metal (no e4m3 in the allocator) and CUDA. The
    ``moe_fp8=True`` quantize path is exercised by the CUDA-only
    waypoint smoke tests.
    """
    moe = MoE(M=128, d_model=64, d_intermediate=32, n_experts=4, top_k=2, routing="balanced")
    assert getattr(moe, "_fp8", False) is False
    moe.prepare(fp8=True, moe_fp8=False)
    assert getattr(moe, "_fp8", False) is False, "moe_fp8=False must keep experts bf16"
