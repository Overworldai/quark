"""Tests for Op __post_init__ checks.

These hit the op classes directly (not through Builder) to verify
arity, attribute, and type-mismatch rejection paths. The Builder tests
cover the happy paths end-to-end.
"""

import pytest

from popcorn.ir import (
    ArithOp,
    BitcastOp,
    CmpOp,
    ConvertOp,
    DType,
    MathOp,
    SelectOp,
    Value,
    ValueShape,
    YieldOp,
)
from popcorn.ir.op import MergeB32Op, SplitB32Op, VecBuildOp, VecExtractOp
from popcorn.ir.value import ValueAllocator


def _v(alloc: ValueAllocator, dt: DType, w: int = 1) -> Value:
    return alloc.fresh(ValueShape(dt, w))


class TestArithOp:
    def test_unknown_kind(self):
        alloc = ValueAllocator()
        a = _v(alloc, DType.F32)
        b = _v(alloc, DType.F32)
        out = _v(alloc, DType.F32)
        with pytest.raises(ValueError, match="unknown kind"):
            ArithOp(results=(out,), operands=(a, b), attrs={"kind": "woof"})

    def test_wrong_arity_binary(self):
        alloc = ValueAllocator()
        a = _v(alloc, DType.F32)
        out = _v(alloc, DType.F32)
        with pytest.raises(ValueError, match="expected 2 operands"):
            ArithOp(results=(out,), operands=(a,), attrs={"kind": "add"})

    def test_fma_takes_three(self):
        alloc = ValueAllocator()
        a = _v(alloc, DType.F32)
        b = _v(alloc, DType.F32)
        c = _v(alloc, DType.F32)
        out = _v(alloc, DType.F32)
        ArithOp(results=(out,), operands=(a, b, c), attrs={"kind": "fma"})  # OK

    def test_neg_unary(self):
        alloc = ValueAllocator()
        a = _v(alloc, DType.F32)
        out = _v(alloc, DType.F32)
        ArithOp(results=(out,), operands=(a,), attrs={"kind": "neg"})  # OK
        with pytest.raises(ValueError):
            ArithOp(results=(out,), operands=(a, a), attrs={"kind": "neg"})


class TestCmpAndSelect:
    def test_select_pred_required(self):
        alloc = ValueAllocator()
        not_pred = _v(alloc, DType.U32)
        t = _v(alloc, DType.F32)
        f = _v(alloc, DType.F32)
        out = _v(alloc, DType.F32)
        with pytest.raises(TypeError, match="must be PRED"):
            SelectOp(results=(out,), operands=(not_pred, t, f))

    def test_select_shape_mismatch(self):
        alloc = ValueAllocator()
        p = _v(alloc, DType.PRED)
        t = _v(alloc, DType.F32)
        f = _v(alloc, DType.F16)
        out = _v(alloc, DType.F32)
        with pytest.raises(TypeError, match="matching shape"):
            SelectOp(results=(out,), operands=(p, t, f))

    def test_cmp_bad_kind(self):
        alloc = ValueAllocator()
        a = _v(alloc, DType.F32)
        b = _v(alloc, DType.F32)
        out = _v(alloc, DType.PRED)
        with pytest.raises(ValueError, match="unknown kind"):
            CmpOp(results=(out,), operands=(a, b), attrs={"kind": "==?"})


class TestConvertAndBitcast:
    def test_convert_requires_dtypes_and_rounding(self):
        alloc = ValueAllocator()
        v = _v(alloc, DType.F32)
        out = _v(alloc, DType.BF16)
        with pytest.raises(ValueError, match="src_dtype"):
            ConvertOp(results=(out,), operands=(v,), attrs={"dst_dtype": DType.BF16})
        with pytest.raises(ValueError, match="unknown rounding"):
            ConvertOp(
                results=(out,),
                operands=(v,),
                attrs={
                    "src_dtype": DType.F32,
                    "dst_dtype": DType.BF16,
                    "rounding": "??",
                },
            )
        # Happy
        ConvertOp(
            results=(out,),
            operands=(v,),
            attrs={"src_dtype": DType.F32, "dst_dtype": DType.BF16, "rounding": "rn"},
        )

    def test_bitcast_size_check(self):
        alloc = ValueAllocator()
        v = _v(alloc, DType.F32)  # 4 bytes
        ok_out = _v(alloc, DType.B32)  # 4 bytes
        bad_out = _v(alloc, DType.F16)  # 2 bytes
        BitcastOp(results=(ok_out,), operands=(v,), attrs={"dst_dtype": DType.B32})
        with pytest.raises(TypeError, match="bits"):
            BitcastOp(results=(bad_out,), operands=(v,), attrs={"dst_dtype": DType.F16})


class TestMathOp:
    def test_kind_required_and_unary(self):
        alloc = ValueAllocator()
        a = _v(alloc, DType.F32)
        out = _v(alloc, DType.F32)
        with pytest.raises(ValueError):
            MathOp(results=(out,), operands=(a,), attrs={"kind": "bogus"})
        # arity
        b = _v(alloc, DType.F32)
        with pytest.raises(ValueError):
            MathOp(results=(out,), operands=(a, b), attrs={"kind": "ex2_approx"})
        # happy
        out2 = _v(alloc, DType.F32)
        MathOp(results=(out2,), operands=(a,), attrs={"kind": "ex2_approx"})


class TestVecOps:
    def test_vec_build_width_matches(self):
        alloc = ValueAllocator()
        a = _v(alloc, DType.F32)
        b = _v(alloc, DType.F32)
        out2 = _v(alloc, DType.F32, w=2)
        VecBuildOp(results=(out2,), operands=(a, b))
        with pytest.raises(ValueError):
            # result width=2 but only 1 operand
            VecBuildOp(results=(out2,), operands=(a,))

    def test_vec_extract_index_bounds(self):
        alloc = ValueAllocator()
        v = _v(alloc, DType.F32, w=4)
        out = _v(alloc, DType.F32)
        VecExtractOp(results=(out,), operands=(v,), attrs={"index": 2})
        with pytest.raises(ValueError):
            VecExtractOp(results=(out,), operands=(v,), attrs={"index": 4})
        with pytest.raises(ValueError):
            VecExtractOp(results=(out,), operands=(v,), attrs={"index": -1})

    def test_split_merge_b32(self):
        alloc = ValueAllocator()
        src = _v(alloc, DType.B32)
        lo = _v(alloc, DType.B16)
        hi = _v(alloc, DType.B16)
        SplitB32Op(results=(lo, hi), operands=(src,))
        out = _v(alloc, DType.B32)
        MergeB32Op(results=(out,), operands=(lo, hi))
        # Wrong dtype input
        not_b32 = _v(alloc, DType.F32)
        with pytest.raises(TypeError):
            SplitB32Op(results=(lo, hi), operands=(not_b32,))


class TestYield:
    def test_yield_has_no_results(self):
        alloc = ValueAllocator()
        out = _v(alloc, DType.F32)
        with pytest.raises(ValueError, match="no results"):
            YieldOp(results=(out,))
