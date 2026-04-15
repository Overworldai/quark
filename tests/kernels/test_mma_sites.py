"""Tests for per-MMA-site shape autotune knobs — MMA_SHAPES M3.

Verifies the base-class machinery (``mma_sites`` / ``tune_space_resolved``)
and the GEMM pilot. Bench-time perf deltas between shapes live in
``tools/bench.py``; this module just confirms the knob plumbing.
"""

from __future__ import annotations

from popcorn.device import ChipGeneration, make_test_device
from popcorn.ir import DType
from popcorn.kernels.base import Kernel, MmaSite


class _DummyKernel(Kernel):
    """Minimal kernel for testing the base-class hooks — never emits."""

    NAME = "dummy_mma_sites"

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"BM": [16, 32, 64]}

    @classmethod
    def mma_sites(cls, spec) -> list[MmaSite]:
        return [
            MmaSite(name="main", a_dtype=DType.BF16, b_dtype=DType.BF16),
        ]

    @classmethod
    def problems(cls):
        return []

    def param_spec(self):
        return None


class _TwoSiteKernel(_DummyKernel):
    """A kernel with two sites — the attention pattern."""

    NAME = "dummy_two_sites"

    @classmethod
    def mma_sites(cls, spec) -> list[MmaSite]:
        return [
            MmaSite(name="gemm1_qk", a_dtype=DType.BF16, b_dtype=DType.BF16),
            MmaSite(name="gemm2_pv", a_dtype=DType.BF16, b_dtype=DType.BF16),
        ]


def test_mma_site_shape_matches_by_dtypes():
    site = MmaSite(name="x", a_dtype=DType.BF16, b_dtype=DType.BF16)
    assert site.shape_matches("m16n8k16_bf16")
    assert site.shape_matches("m16n8k8_bf16")
    assert not site.shape_matches("m16n8k16_e4m3")  # wrong dtypes
    assert not site.shape_matches("nonexistent_shape")


def test_tune_space_resolved_injects_shape_knob_on_cuda():
    """CUDA sm_89 has both m16n8k16_bf16 and m16n8k8_bf16 — both appear
    as values of the ``main_shape`` knob."""
    d = make_test_device(chip_gen=ChipGeneration.SM_89)
    space = _DummyKernel.tune_space_resolved(None, d)
    assert space["BM"] == [16, 32, 64]
    assert set(space["main_shape"]) == {"m16n8k16_bf16", "m16n8k8_bf16"}


def test_tune_space_resolved_metal_uses_native_8x8x8():
    """Metal M3's bf16 shape set is just m8n8k8 — the larger tiles were
    dropped from the registry after autotune showed m8n8k8 dominated
    every production problem on M3 Ultra."""
    d = make_test_device(chip_gen=ChipGeneration.METAL_M3)
    space = _DummyKernel.tune_space_resolved(None, d)
    assert set(space["main_shape"]) == {"m8n8k8_bf16"}


def test_tune_space_resolved_empty_when_unknown_chip():
    """UNKNOWN chip → no matmul shapes → no shape knob injected at all
    (knob omitted rather than empty-listed so autotune doesn't try to
    mutate over an empty set)."""
    d = make_test_device(chip_gen=ChipGeneration.UNKNOWN)
    space = _DummyKernel.tune_space_resolved(None, d)
    assert "main_shape" not in space
    assert space["BM"] == [16, 32, 64]


def test_two_site_kernel_gets_two_shape_knobs():
    """A kernel with two MMA sites gets two INDEPENDENT shape knobs —
    the autotuner's GA can mutate them separately."""
    d = make_test_device(chip_gen=ChipGeneration.SM_89)
    space = _TwoSiteKernel.tune_space_resolved(None, d)
    assert "gemm1_qk_shape" in space
    assert "gemm2_pv_shape" in space
    assert space["gemm1_qk_shape"] == space["gemm2_pv_shape"]  # same legal set
    # The GA treats each as an ordinary axis — neighbors differ in one
    # shape independently.


def test_default_mma_sites_empty():
    """Kernels that don't override mma_sites get no shape knobs — pure
    back-compat path for non-MMA kernels (kv_cache_update, epilogues)."""

    class _NoMma(Kernel):
        NAME = "no_mma"

        @classmethod
        def tune_space(cls):
            return {"BM": [32]}

        @classmethod
        def problems(cls):
            return []

        def param_spec(self):
            return None

    d = make_test_device(chip_gen=ChipGeneration.SM_89)
    space = _NoMma.tune_space_resolved(None, d)
    assert space == {"BM": [32]}


# GEMM pilot — end-to-end from tune_space_resolved through emit().
def test_gemm_pilot_explicit_main_shape_emits_correctly():
    """GEMM accepts every bf16 shape in the registry — the tile math
    derives MT / NT / acc_width from mma_cfg.shape.m / n / c_regs
    after the M3 generalization."""
    from popcorn.ir.validator import validate_module
    from popcorn.kernels.gemm.config import GemmConfig
    from popcorn.kernels.gemm.kernel import GemmKernel
    from popcorn.kernels.gemm.spec import GemmSpec

    spec = GemmSpec(M=128, N=128, K=128, a_dtype="bf16", b_dtype="bf16", out_dtype="bf16")
    for shape_id in ("m16n8k16_bf16", "m16n8k8_bf16", "m8n8k8_bf16"):
        cfg = GemmConfig(BM=32, BN=32, BK=16, n_warps=4, n_stages=1, main_shape=shape_id)
        k = GemmKernel(spec, cfg)
        assert k.is_valid(), f"GEMM should accept main_shape={shape_id}"
        assert k._mma_cfg().shape_id == shape_id
        validate_module(k.emit())


def test_gemm_pilot_legacy_mma_k_fallback_still_works():
    """main_shape='' falls back to lookup_mma(compute, compute, mma_k).
    Preserves behaviour for callers (tuned JSONs, tests) that haven't
    migrated to main_shape yet — removed in M4."""
    from popcorn.kernels.gemm.config import GemmConfig
    from popcorn.kernels.gemm.kernel import GemmKernel
    from popcorn.kernels.gemm.spec import GemmSpec

    spec = GemmSpec(M=128, N=128, K=128, a_dtype="bf16", b_dtype="bf16", out_dtype="bf16")
    cfg = GemmConfig(BM=32, BN=32, BK=16, n_warps=4, n_stages=1)  # main_shape default ""
    k = GemmKernel(spec, cfg)
    assert k.is_valid()
    assert k._mma_cfg().shape_id == "m16n8k16_bf16"  # mma_k=16 default
