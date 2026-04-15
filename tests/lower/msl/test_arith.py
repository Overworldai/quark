"""Tests for arithmetic / math / cmp / select / convert / bitcast MSL lowering."""

import pytest

from popcorn.ir import DType
from tests.lower.msl.conftest import lower


class TestConst:
    def test_f32_const_uses_hex_float(self, fresh_builder):
        b = fresh_builder
        b.const(DType.F32, 1.0)
        out = lower(b)
        assert "as_type<float>(0x3f800000u)" in out

    def test_u32_const_is_decimal(self, fresh_builder):
        b = fresh_builder
        b.const(DType.U32, 42)
        out = lower(b)
        assert "uint _pc_u32_0 = 42u;" in out

    def test_s32_const(self, fresh_builder):
        b = fresh_builder
        b.const(DType.S32, -7)
        out = lower(b)
        assert "int _pc_s32_0 = -7;" in out

    def test_pred_const(self, fresh_builder):
        b = fresh_builder
        b.const(DType.PRED, True)
        out = lower(b)
        assert "bool _pc_p_0 = true;" in out

    def test_pred_false(self, fresh_builder):
        b = fresh_builder
        b.const(DType.PRED, False)
        out = lower(b)
        assert "bool _pc_p_0 = false;" in out


class TestArithBinary:
    @pytest.mark.parametrize(
        "method,expect",
        [
            ("add", "+"),
            ("sub", "-"),
            ("mul", "*"),
            ("div", "/"),
        ],
    )
    def test_f32_binary_ops(self, fresh_builder, method, expect):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        getattr(b, method)(a, c)
        out = lower(b)
        assert expect in out
        assert "float _pc_f32_2 =" in out

    def test_fma_uses_metal_fma(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        d = b.const(DType.F32, 3.0)
        b.fma(a, c, d)
        out = lower(b)
        assert "metal::fma(" in out

    def test_min_max(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        b.min(a, c)
        b.max(a, c)
        out = lower(b)
        assert "metal::min(" in out
        assert "metal::max(" in out

    def test_neg_unary(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        b.neg(a)
        out = lower(b)
        assert "= -_pc_f32_0;" in out

    def test_abs_unary(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        b.abs(a)
        out = lower(b)
        assert "metal::abs(" in out

    def test_bitwise_ops(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.U32, 3)
        c = b.const(DType.U32, 4)
        b.and_(a, c)
        b.or_(a, c)
        b.xor(a, c)
        b.shl(a, c)
        out = lower(b)
        assert "&" in out
        assert "|" in out
        assert "^" in out
        assert "<<" in out


class TestMath:
    @pytest.mark.parametrize(
        "method,expect",
        [
            ("ex2_approx", "metal::fast::exp2("),
            ("rsqrt_approx", "metal::fast::rsqrt("),
            ("sqrt", "metal::sqrt("),
            ("exp2", "metal::exp2("),
            ("log2", "metal::log2("),
            ("tanh", "metal::tanh("),
        ],
    )
    def test_math_functions(self, fresh_builder, method, expect):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        getattr(b, method)(v)
        out = lower(b)
        assert expect in out


class TestCmpSelect:
    def test_cmp_lt_f32(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        b.cmp("lt", a, c)
        out = lower(b)
        assert "bool _pc_p_0 = _pc_f32_0 < _pc_f32_1;" in out

    def test_select_f32(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        p = b.cmp("lt", a, c)
        b.select(p, a, c)
        out = lower(b)
        assert "? _pc_f32_0 : _pc_f32_1;" in out


class TestConvert:
    def test_f32_to_bf16(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.convert(v, DType.BF16)
        out = lower(b)
        assert "static_cast<bfloat16_t>" in out

    def test_bf16_to_f32(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.BF16, 0)
        b.convert(v, DType.F32)
        out = lower(b)
        assert "static_cast<float>" in out

    def test_f32_to_half(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.convert(v, DType.F16)
        out = lower(b)
        assert "static_cast<half>" in out


class TestBitcast:
    def test_f32_to_b32(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.bitcast(v, DType.B32)
        out = lower(b)
        assert "as_type<uint>" in out
