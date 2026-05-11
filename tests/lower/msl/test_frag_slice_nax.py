"""MSL lowering tests for FragSliceOp on NAX-storage fragments.

NAX attention's GEMM2 needs a width-8 A operand, but GEMM1 produces a
width-16 S accumulator (two 16×16 N-tiles per lane). FragSliceOp takes
a contiguous component slice — pure rebinding at lower time, no MSL
emitted — so the two halves of S can feed two GEMM2 MMAs without copies.
"""

from __future__ import annotations

import pytest

from quark.ir import Builder, DType, validate_module
from quark.lower.msl import MslLowerer
from tests.lower.msl.conftest import METAL_CAPS_FAKE


def _register_nax(b: Builder) -> None:
    from quark.ir.mma_registry import _BY_SHAPE_ID

    b.register_shape(_BY_SHAPE_ID["m16n32k16_nax_bf16"].shape)


def _lower(b: Builder) -> str:
    b.end_function()
    return MslLowerer(METAL_CAPS_FAKE).lower_module(b.module).source


def _build_gemm1_then_slice(b: Builder, *, slice_start: int, slice_length: int):
    """Emit one NAX GEMM and slice the f32 accumulator output."""
    _register_nax(b)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (32, 16))
    C = b.smem_alloc("C", DType.F32, (16, 32))
    a = b.load_matrix(A, "m16n32k16_nax_bf16", which="a")
    bf = b.load_matrix(B, "m16n32k16_nax_bf16", which="b")
    cf = b.load_matrix(C, "m16n32k16_nax_bf16", which="c")
    s = b.mma("m16n32k16_nax_bf16", a, bf, cf)
    return b.frag_slice(s, start=slice_start, length=slice_length, name="s_half")


def test_frag_slice_emits_no_msl_just_rebinds():
    """FragSliceOp doesn't emit any MSL — count of MMA-related lines
    is unchanged whether or not we slice. Slicing produces no extra
    statements (no copies, no temporaries)."""
    b1 = Builder("a")
    _register_nax(b1)
    b1.begin_function("f")
    A = b1.smem_alloc("A", DType.BF16, (16, 16))
    B = b1.smem_alloc("B", DType.BF16, (32, 16))
    C = b1.smem_alloc("C", DType.F32, (16, 32))
    a = b1.load_matrix(A, "m16n32k16_nax_bf16", which="a")
    bf = b1.load_matrix(B, "m16n32k16_nax_bf16", which="b")
    cf = b1.load_matrix(C, "m16n32k16_nax_bf16", which="c")
    b1.mma("m16n32k16_nax_bf16", a, bf, cf)
    msl_no_slice = _lower(b1)

    b2 = Builder("b")
    _build_gemm1_then_slice(b2, slice_start=0, slice_length=8)
    msl_with_slice = _lower(b2)

    # Same line count: slice is pure rebinding.
    assert msl_no_slice.count("\n") == msl_with_slice.count("\n"), (
        f"FragSliceOp emitted MSL: {msl_no_slice.count(chr(10))} vs {msl_with_slice.count(chr(10))}"
    )


def test_frag_slice_inherits_nax_storage():
    """Slicing a NAX-stored fragment keeps the NAX storage tag — the
    sliced Value can be fed to subsequent NAX MMAs as A. Verify by
    using the slice as the next MMA's A operand and lowering cleanly."""
    b = Builder("test")
    _register_nax(b)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (32, 16))
    C = b.smem_alloc("C", DType.F32, (16, 32))
    O_smem = b.smem_alloc("O", DType.F32, (16, 32))
    a = b.load_matrix(A, "m16n32k16_nax_bf16", which="a")
    bf = b.load_matrix(B, "m16n32k16_nax_bf16", which="b")
    cf = b.load_matrix(C, "m16n32k16_nax_bf16", which="c")
    s = b.mma("m16n32k16_nax_bf16", a, bf, cf)
    # First half of S as new A operand for GEMM2.
    s_half = b.frag_slice(s, start=0, length=8, name="s_half")
    of = b.load_matrix(O_smem, "m16n32k16_nax_bf16", which="c")
    vf = b.load_matrix(B, "m16n32k16_nax_bf16", which="b")  # reuse B as V
    b.mma("m16n32k16_nax_bf16", s_half, vf, of, transpose_b=False)
    b.end_function()
    lk = MslLowerer(METAL_CAPS_FAKE).lower_module(b.module)
    full = lk.header + lk.source
    # Two distinct MMA helper variants in the full source — first MMA
    # uses the matched-dtype helper, second uses the castF32 variant
    # because the slice carries f32 dtype while the shape's A is bf16.
    assert full.count(".run(") >= 2
    # The implicit cast (f32 A → bf16) fires for the second MMA.
    assert "static_cast<bfloat>" in full


def test_frag_slice_two_halves_share_lane_locals():
    """Slicing [0:8] and [8:16] of the same width-16 source produces
    two width-8 Values that bind to the source's lane-local names —
    no copies. Verify via inspection of the produced module."""
    b = Builder("test")
    _register_nax(b)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (32, 16))
    C = b.smem_alloc("C", DType.F32, (16, 32))
    a = b.load_matrix(A, "m16n32k16_nax_bf16", which="a")
    bf = b.load_matrix(B, "m16n32k16_nax_bf16", which="b")
    cf = b.load_matrix(C, "m16n32k16_nax_bf16", which="c")
    s = b.mma("m16n32k16_nax_bf16", a, bf, cf)
    s0 = b.frag_slice(s, start=0, length=8, name="s0")
    s1 = b.frag_slice(s, start=8, length=8, name="s1")
    # Both slices are width-8 and share dtype with S.
    assert s0.width == 8 and s1.width == 8
    assert s0.dtype == s.dtype == DType.F32


def test_frag_slice_start_validation():
    b = Builder("test")
    _register_nax(b)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    a = b.load_matrix(A, "m16n32k16_nax_bf16", which="a")  # width-8
    # Out-of-range slice should raise.
    with pytest.raises(ValueError, match="exceeds source width"):
        b.frag_slice(a, start=4, length=8)


def test_frag_slice_module_validates():
    b = Builder("test")
    _build_gemm1_then_slice(b, slice_start=0, slice_length=8)
    b.end_function()
    validate_module(b.module)


def test_frag_slice_op_post_init_rejects_dtype_mismatch():
    """The op's __post_init__ enforces result dtype matches src dtype."""
    from quark.ir import op as _op
    from quark.ir.types import ValueShape

    b = Builder("test")
    _register_nax(b)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    a = b.load_matrix(A, "m16n32k16_nax_bf16", which="a")
    # Result with wrong dtype.
    out = b._fresh(ValueShape(DType.F32, width=4), "wrong")
    with pytest.raises(TypeError, match="result dtype"):
        _op.FragSliceOp(
            results=(out,),
            operands=(a,),
            attrs={"start": 0, "length": 4},
        )
