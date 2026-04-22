"""Tests for quark.ir.types: DType, MemSpace, ValueShape, Param types."""

import pytest

from quark.ir import BufferType, DType, MemSpace, ScalarType, ValueShape
from quark.ir.types import _VALID_VECTOR_WIDTHS


class TestDType:
    def test_byte_sizes(self):
        assert DType.F32.bytes == 4
        assert DType.F16.bytes == 2
        assert DType.BF16.bytes == 2
        assert DType.F64.bytes == 8
        assert DType.U8.bytes == 1
        assert DType.E4M3.bytes == 1
        assert DType.B32.bytes == 4
        assert DType.PRED.bytes == 1

    def test_bit_width(self):
        assert DType.F32.bits == 32
        assert DType.F16.bits == 16

    def test_category_predicates(self):
        assert DType.F32.is_float and not DType.F32.is_int
        assert DType.BF16.is_float
        assert DType.U32.is_int and not DType.U32.is_float
        assert DType.S32.is_int and DType.S32.is_signed_int
        assert DType.U64.is_int and not DType.U64.is_signed_int
        assert DType.B32.is_bit and not DType.B32.is_float
        # PRED is neither float nor int in our scheme
        assert not DType.PRED.is_float
        assert not DType.PRED.is_int


class TestValueShape:
    def test_scalar_default(self):
        s = ValueShape(DType.F32)
        assert s.is_scalar and not s.is_vector
        assert s.width == 1
        assert s.bytes == 4

    @pytest.mark.parametrize("w", sorted(_VALID_VECTOR_WIDTHS))
    def test_valid_widths(self, w: int):
        s = ValueShape(DType.F32, width=w)
        assert s.width == w
        assert s.bytes == 4 * w

    @pytest.mark.parametrize("bad", [0, 5, 6, 7, 9, 32, -1])
    def test_invalid_widths(self, bad: int):
        with pytest.raises(ValueError, match="width must be one of"):
            ValueShape(DType.F32, width=bad)

    def test_equality_and_hash(self):
        a = ValueShape(DType.F32, 1)
        b = ValueShape(DType.F32, 1)
        c = ValueShape(DType.F32, 2)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c

    def test_repr(self):
        assert repr(ValueShape(DType.F32)) == "f32"
        assert repr(ValueShape(DType.BF16, 4)) == "bf16x4"


class TestMemSpace:
    def test_all_members(self):
        for ms in ("GLOBAL", "SHARED", "PRIVATE", "CONSTANT", "PARAM"):
            assert hasattr(MemSpace, ms)


class TestParamTypes:
    def test_scalar_type(self):
        t = ScalarType(DType.U32)
        assert t.dtype is DType.U32
        assert "u32" in repr(t)

    def test_buffer_type_default_space(self):
        t = BufferType(DType.BF16)
        assert t.dtype is DType.BF16
        assert t.space is MemSpace.GLOBAL

    def test_buffer_type_shared(self):
        t = BufferType(DType.F32, MemSpace.SHARED)
        assert t.space is MemSpace.SHARED
