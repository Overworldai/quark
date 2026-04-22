"""Tests for module-level lowering wrapper and metadata."""

from quark.ir import BufferType, DType, ParamAttrs, ScalarType
from tests.lower.msl.conftest import lower_full


class TestKernelName:
    def test_kernel_name_has_prefix(self, fresh_builder):
        b = fresh_builder
        result = lower_full(b)
        assert result.kernel_name == "quark_f"


class TestInputOutputSplit:
    def test_readonly_goes_to_input(self, fresh_builder):
        b = fresh_builder
        b.param("A", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
        b.param("B", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
        b.param("C", BufferType(DType.F32))
        result = lower_full(b)
        assert result.input_names == ["A", "B"]
        assert result.output_names == ["C"]

    def test_scalar_goes_to_input(self, fresh_builder):
        b = fresh_builder
        b.param("A", BufferType(DType.F32), attrs=ParamAttrs(readonly=True))
        b.param("stride", ScalarType(DType.U32))
        result = lower_full(b)
        assert "stride" in result.scalar_names
        assert "stride" in result.input_names


class TestEmptyModule:
    def test_empty_function(self, fresh_builder):
        b = fresh_builder
        result = lower_full(b)
        assert result.source is not None
        assert result.smem_bytes == 0
        assert result.atomic_outputs is False
