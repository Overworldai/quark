"""Tests for arithmetic / math / cmp / select / convert / bitcast lowering."""

import pytest

from popcorn.ir import DType
from tests.lower.ptx.conftest import lower


class TestConst:
    def test_f32_const_uses_hex_float(self, fresh_builder):
        b = fresh_builder
        b.const(DType.F32, 1.0)
        out = lower(b)
        assert "mov.f32 %f0, 0f3f800000;" in out

    def test_u32_const_is_decimal(self, fresh_builder):
        b = fresh_builder
        b.const(DType.U32, 42)
        out = lower(b)
        assert "mov.u32 %r0, 42;" in out

    def test_f64_const_uses_hex_double(self, fresh_builder):
        b = fresh_builder
        b.const(DType.F64, 1.0)
        out = lower(b)
        assert "mov.f64 %fd0, 0d3ff0000000000000;" in out

    def test_pred_const(self, fresh_builder):
        b = fresh_builder
        b.const(DType.PRED, True)
        out = lower(b)
        assert "mov.pred %p0, 1;" in out


class TestArithMnemonics:
    @pytest.mark.parametrize(
        "method,kind,expect",
        [
            ("add", "add.f32", "add.f32"),
            ("sub", "sub.f32", "sub.f32"),
            ("mul", "mul.f32", "mul.f32"),
            ("min", "min.f32", "min.f32"),
            ("max", "max.f32", "max.f32"),
            ("div", "div.f32", "div.f32"),
        ],
    )
    def test_f32_binary(self, fresh_builder, method, kind, expect):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        getattr(b, method)(a, c)
        out = lower(b)
        assert expect in out

    def test_fma_uses_rn_suffix(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        d = b.const(DType.F32, 3.0)
        b.fma(a, c, d)
        out = lower(b)
        assert "fma.rn.f32" in out

    def test_int_mul_is_lo(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.U32, 3)
        c = b.const(DType.U32, 4)
        b.mul(a, c)
        out = lower(b)
        assert "mul.lo.u32" in out

    def test_bitwise_uses_bit_suffix(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.U32, 3)
        c = b.const(DType.U32, 4)
        b.and_(a, c)
        b.or_(a, c)
        b.xor(a, c)
        b.shl(a, c)
        out = lower(b)
        for m in ("and.b32", "or.b32", "xor.b32", "shl.b32"):
            assert m in out


class TestMath:
    @pytest.mark.parametrize(
        "method,expect",
        [
            ("ex2_approx", "ex2.approx.f32"),
            ("rcp_approx", "rcp.approx.f32"),
            ("rsqrt_approx", "rsqrt.approx.f32"),
        ],
    )
    def test_approx_math_mnemonics(self, fresh_builder, method, expect):
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
        assert "setp.lt.f32 %p0" in out

    def test_select_f32(self, fresh_builder):
        b = fresh_builder
        a = b.const(DType.F32, 1.0)
        c = b.const(DType.F32, 2.0)
        p = b.cmp("lt", a, c)
        b.select(p, a, c)
        out = lower(b)
        assert "selp.f32" in out

    def test_select_pred_rejected(self, fresh_builder):
        """`selp.pred` is not a real PTX instruction — combining two
        predicates by selection (the AND/OR pattern: select(p1, p2, FALSE) /
        select(p1, TRUE, p2)) must use `bld.and_(...)` / `bld.or_(...)`.

        Regression: previously this lowered to `selp.pred` and ptxas
        rejected the kernel with "Unexpected instruction types specified
        for 'selp'". Now the lowerer rejects it eagerly with a message
        pointing at the right builder method.
        """
        import pytest

        b = fresh_builder
        x = b.const(DType.F32, 1.0)
        y = b.const(DType.F32, 2.0)
        p1 = b.cmp("lt", x, y)
        p2 = b.cmp("eq", x, y)
        false_ = b.const(DType.PRED, False)
        b.select(p1, p2, false_)  # AND-via-select pattern — forbidden
        with pytest.raises(NotImplementedError, match=r"selp\.pred|and\.pred"):
            lower(b)

    def test_and_pred_lowers_to_and_pred(self, fresh_builder):
        """`bld.and_(p1, p2)` lowers to `and.pred` — the supported PTX
        path for combining predicates."""
        b = fresh_builder
        x = b.const(DType.F32, 1.0)
        y = b.const(DType.F32, 2.0)
        p1 = b.cmp("lt", x, y)
        p2 = b.cmp("eq", x, y)
        b.and_(p1, p2)
        out = lower(b)
        assert "and.pred" in out

    def test_or_pred_lowers_to_or_pred(self, fresh_builder):
        b = fresh_builder
        x = b.const(DType.F32, 1.0)
        y = b.const(DType.F32, 2.0)
        p1 = b.cmp("lt", x, y)
        p2 = b.cmp("eq", x, y)
        b.or_(p1, p2)
        out = lower(b)
        assert "or.pred" in out


class TestConvert:
    def test_f32_to_bf16_has_rn(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.convert(v, DType.BF16)
        out = lower(b)
        assert "cvt.rn.bf16.f32" in out

    def test_bf16_to_f32_no_rounding(self, fresh_builder):
        b = fresh_builder
        # Simulate a bf16 Value — we use a const with BF16 dtype (which
        # the lowerer routes through b16 reg class, but the convert
        # site still uses the bf16 spelling).
        v = b.const(DType.BF16, 0)
        b.convert(v, DType.F32)
        out = lower(b)
        # No .rn when widening — just `cvt.f32.bf16`.
        assert "cvt.f32.bf16" in out
        assert "cvt.rn.f32.bf16" not in out


class TestUnpackedConvert:
    """Coverage for `bld.unpacked_convert` — the symmetric inverse of
    `packed_convert`. Required because PTX has NO scalar `cvt.<wider>.e4m3`
    instruction; the only fp8-source cvt is the packed `cvt.<dst>x2.e4m3x2`
    form (PTX 9.2+ for bf16, PTX 7.8+ for f16).

    Regression: previously the scalar tile loader emitted plain
    `cvt.bf16.b8 %h, %rb` for fp8 src → bf16 cast, which ptxas rejects with
    "Unexpected instruction types specified for 'cvt'".
    """

    def test_e4m3_to_bf16_emits_packed_cvt(self, fresh_builder):
        """e4m3 → bf16 routes through the f16x2 packed cvt (PTX 7.8+,
        sm_89+), widens each half to f32, then narrows to bf16 — because
        `cvt.bf16.f16` is sm_90+ and the project still targets sm_89."""
        b = fresh_builder
        # B16 input "carrying" two packed e4m3 bytes.
        packed = b.const(DType.B16, 0)
        b.unpacked_convert(packed, src_dtype=DType.E4M3, dst_dtype=DType.BF16)
        out = lower(b)
        # Expect the f16x2 packed cvt + per-half f32→bf16 narrowing.
        assert "cvt.rn.f16x2.e4m3x2" in out
        assert "cvt.f32.f16" in out
        assert "cvt.rn.bf16.f32" in out
        assert "mov.b32" in out
        # The PTX 9.2-only direct form must NOT appear.
        assert "cvt.rn.bf16x2.e4m3x2" not in out
        # The sm_90+ direct f16→bf16 cvt must NOT appear.
        assert "cvt.rn.bf16.f16" not in out

    def test_e4m3_to_f16_emits_packed_cvt(self, fresh_builder):
        b = fresh_builder
        packed = b.const(DType.B16, 0)
        b.unpacked_convert(packed, src_dtype=DType.E4M3, dst_dtype=DType.F16)
        out = lower(b)
        assert "cvt.rn.f16x2.e4m3x2" in out

    def test_e5m2_to_bf16_emits_packed_cvt(self, fresh_builder):
        """e5m2 → bf16 follows the same f16x2 → f32 → bf16 routing as
        e4m3 (direct bf16.f16 cvt is sm_90+; we target sm_89)."""
        b = fresh_builder
        packed = b.const(DType.B16, 0)
        b.unpacked_convert(packed, src_dtype=DType.E5M2, dst_dtype=DType.BF16)
        out = lower(b)
        assert "cvt.rn.f16x2.e5m2x2" in out
        assert "cvt.f32.f16" in out
        assert "cvt.rn.bf16.f32" in out
        assert "cvt.rn.bf16.f16" not in out

    def test_non_fp8_src_rejected(self, fresh_builder):
        import pytest

        b = fresh_builder
        v = b.const(DType.B16, 0)
        with pytest.raises(TypeError, match="fp8"):
            b.unpacked_convert(v, src_dtype=DType.BF16, dst_dtype=DType.F32)

    def test_non_b16_input_rejected(self, fresh_builder):
        import pytest

        b = fresh_builder
        v = b.const(DType.F32, 0.0)
        with pytest.raises(TypeError, match="B16"):
            b.unpacked_convert(v, src_dtype=DType.E4M3, dst_dtype=DType.BF16)


class TestBitcast:
    def test_f32_to_b32_via_mov(self, fresh_builder):
        b = fresh_builder
        v = b.const(DType.F32, 1.0)
        b.bitcast(v, DType.B32)
        out = lower(b)
        assert "mov.b32" in out
