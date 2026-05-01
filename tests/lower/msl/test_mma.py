"""Tests for MMA (simdgroup_matrix) MSL lowering."""

from quark.device import DeviceFamily
from quark.ir import Builder, DType, MmaShape
from quark.ir.mma_registry import register_backend_payload
from quark.lower.msl import MslLowerer
from tests.lower.msl.conftest import METAL_CAPS_FAKE


def _lower_mma_module(shape: MmaShape, a_offsets, b_offsets, cd_offsets):
    """Build a minimal module with load_matrix → mma → store_matrix."""
    b = Builder("mma_test")
    b.begin_function("gemm_mma")
    b.module.register_shape(shape)

    # Allocate smem tiles for A, B, and output C.
    A_smem = b.smem_alloc("A", shape.a_dtype, (shape.m, shape.k))
    B_smem = b.smem_alloc("B", shape.b_dtype, (shape.k, shape.n))
    C_smem = b.smem_alloc("C", shape.acc_dtype, (shape.m, shape.n))

    row_a = b.const(DType.U32, 0)
    col_a = b.const(DType.U32, 0)
    row_b = b.const(DType.U32, 0)
    col_b = b.const(DType.U32, 0)

    # Load A and B fragments.
    a_frag = b.load_matrix(
        A_smem,
        shape.name,
        which="a",
        row=row_a,
        col=col_a,
        reg_offsets=a_offsets,
    )
    b_frag = b.load_matrix(
        B_smem,
        shape.name,
        which="b",
        row=row_b,
        col=col_b,
        reg_offsets=b_offsets,
    )
    # Load C (accumulator) — zero-init.
    c_frag = b.load_matrix(
        C_smem,
        shape.name,
        which="c",
        row=row_a,
        col=col_b,
        reg_offsets=cd_offsets,
    )
    # MMA.
    d_frag = b.mma(shape.name, a_frag, b_frag, c_frag)
    # Store result.
    b.store_matrix(
        C_smem,
        d_frag,
        shape.name,
        which="d",
        row=row_a,
        col=col_b,
        reg_offsets=cd_offsets,
    )
    b.end_function()
    return MslLowerer(METAL_CAPS_FAKE).lower_module(b.module)


class TestMmaBf16:
    """Test MMA lowering for bf16 × bf16 → f32 (m16n8k16)."""

    def test_simdgroup_load_emitted(self):
        shape = MmaShape(
            name="m16n8k16_bf16",
            m=16,
            n=8,
            k=16,
            a_dtype=DType.BF16,
            b_dtype=DType.BF16,
            acc_dtype=DType.F32,
            a_regs=4,
            b_regs=2,
            c_regs=4,
        )
        register_backend_payload(
            "m16n8k16_bf16", DeviceFamily.CUDA, "m16n8k16.row.col.f32.bf16.bf16.f32"
        )
        register_backend_payload("m16n8k16_bf16", DeviceFamily.METAL, "half:2:1:2")
        a_offsets = ((0, 0), (8, 0), (0, 8), (8, 8))
        b_offsets = ((0, 0), (0, 8))
        cd_offsets = ((0, 0), (0, 1), (8, 0), (8, 1))
        result = _lower_mma_module(shape, a_offsets, b_offsets, cd_offsets)
        src = result.source
        # Should contain simdgroup_load calls for A, B, C fragments.
        assert "simdgroup_load(" in src
        # Should contain simdgroup_multiply_accumulate.
        assert "simdgroup_multiply_accumulate(" in src
        # Should contain simdgroup_store for the result.
        assert "simdgroup_store(" in src

    def test_header_includes_simdgroup(self):
        shape = MmaShape(
            name="m16n8k16_bf16",
            m=16,
            n=8,
            k=16,
            a_dtype=DType.BF16,
            b_dtype=DType.BF16,
            acc_dtype=DType.F32,
            a_regs=4,
            b_regs=2,
            c_regs=4,
        )
        register_backend_payload(
            "m16n8k16_bf16", DeviceFamily.CUDA, "m16n8k16.row.col.f32.bf16.bf16.f32"
        )
        register_backend_payload("m16n8k16_bf16", DeviceFamily.METAL, "half:2:1:2")
        a_offsets = ((0, 0), (8, 0), (0, 8), (8, 8))
        b_offsets = ((0, 0), (0, 8))
        cd_offsets = ((0, 0), (0, 1), (8, 0), (8, 1))
        result = _lower_mma_module(shape, a_offsets, b_offsets, cd_offsets)
        assert "#include <metal_simdgroup_matrix>" in result.header

    def test_correct_fragment_count(self):
        shape = MmaShape(
            name="m16n8k16_bf16",
            m=16,
            n=8,
            k=16,
            a_dtype=DType.BF16,
            b_dtype=DType.BF16,
            acc_dtype=DType.F32,
            a_regs=4,
            b_regs=2,
            c_regs=4,
        )
        register_backend_payload(
            "m16n8k16_bf16", DeviceFamily.CUDA, "m16n8k16.row.col.f32.bf16.bf16.f32"
        )
        register_backend_payload("m16n8k16_bf16", DeviceFamily.METAL, "half:2:1:2")
        a_offsets = ((0, 0), (8, 0), (0, 8), (8, 8))
        b_offsets = ((0, 0), (0, 8))
        cd_offsets = ((0, 0), (0, 1), (8, 0), (8, 1))
        result = _lower_mma_module(shape, a_offsets, b_offsets, cd_offsets)
        src = result.source
        # A: 2 m_frags * 2 k_frags = 4 fragments.
        assert "simdgroup_matrix<half, 8, 8>" in src
        # C/D: 2 m_frags * 1 n_frag = 2 fragments.
        assert "simdgroup_matrix<float, 8, 8>" in src


class TestMmaBf16M8n8k8:
    """Direct Metal-native 8x8x8 path — one simdgroup_matrix multiply
    per MMA call, no internal m/n/k tiling. Registered in MMA_SHAPES M2
    as the hypothesized perf win over the m16n8k16 two-tile path."""

    def _m8n8k8_shape(self) -> MmaShape:
        from quark.ir.mma_registry import _BF16_M8N8K8

        return _BF16_M8N8K8.shape

    def test_emits_one_simdgroup_multiply_accumulate(self):
        from quark.ir.mma_registry import _BF16_M8N8K8

        shape = self._m8n8k8_shape()
        result = _lower_mma_module(
            shape,
            _BF16_M8N8K8.a_offsets,
            _BF16_M8N8K8.b_offsets,
            _BF16_M8N8K8.cd_offsets,
        )
        src = result.source
        # Single m=1, n=1, k=1 tiling → exactly one simdgroup multiply.
        assert src.count("simdgroup_multiply_accumulate(") == 1
        # A/B are bfloat16_t; acc is float. One fragment per axis.
        assert "simdgroup_matrix<bfloat16_t, 8, 8>" in src
        assert "simdgroup_matrix<float, 8, 8>" in src


class TestMmaNoMsl:
    """Shapes without msl field should raise."""

    def test_no_msl_raises(self):
        shape = MmaShape(
            name="test_no_msl",
            m=16,
            n=8,
            k=16,
            a_dtype=DType.BF16,
            b_dtype=DType.BF16,
            acc_dtype=DType.F32,
            a_regs=4,
            b_regs=2,
            c_regs=4,
        )
        register_backend_payload(
            "test_no_msl", DeviceFamily.CUDA, "m16n8k16.row.col.f32.bf16.bf16.f32"
        )
        import pytest

        b = Builder("mma_test")
        b.begin_function("f")
        b.module.register_shape(shape)
        A = b.smem_alloc("A", DType.BF16, (16, 16))
        r = b.const(DType.U32, 0)
        c = b.const(DType.U32, 0)
        b.load_matrix(A, shape.name, which="a", row=r, col=c)
        b.end_function()
        with pytest.raises(NotImplementedError, match="no `msl` field"):
            MslLowerer(METAL_CAPS_FAKE).lower_module(b.module)
