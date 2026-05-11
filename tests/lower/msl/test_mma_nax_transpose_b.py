"""MSL lowering tests for MmaOp.transpose_b on NAX shapes.

The NAX MMA dispatches to ``matmul2d_descriptor(M, N, K, false,
transpose_b, true, accumulate)``. ``transpose_b`` defaults to ``True``
(matches the GEMM kernel's existing emission). Attention's GEMM2 sets
it to ``False`` — V is read K-contiguous, no transpose needed.
"""

from __future__ import annotations

from quark.ir import Builder, DType, validate_module
from quark.lower.msl import MslLowerer
from tests.lower.msl.conftest import METAL_CAPS_FAKE


def _build_nax_mma(b: Builder, *, transpose_b: bool):
    from quark.ir.mma_registry import _BY_SHAPE_ID

    b.register_shape(_BY_SHAPE_ID["m16n32k16_nax_bf16"].shape)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (32, 16))
    C = b.smem_alloc("C", DType.F32, (16, 32))
    a = b.load_matrix(A, "m16n32k16_nax_bf16", which="a")
    bf = b.load_matrix(B, "m16n32k16_nax_bf16", which="b")
    cf = b.load_matrix(C, "m16n32k16_nax_bf16", which="c")
    return b.mma("m16n32k16_nax_bf16", a, bf, cf, transpose_b=transpose_b)


def _lower(b: Builder) -> tuple[str, str]:
    """Returns (header, source) — the descriptor lives in the header
    after the Phase 3 MMA-helper refactor (one nax_mma_* helper
    function per (shape, tb, acc, cast_a) tuple, called from each
    MmaOp site). The body shows the call site; the header shows the
    descriptor + cooperative_tensor block once."""
    b.end_function()
    lk = MslLowerer(METAL_CAPS_FAKE).lower_module(b.module)
    return lk.header, lk.source


def test_nax_mma_transpose_b_true_is_default():
    """Default mma() emits transpose_b=true in the matmul2d_descriptor."""
    b = Builder("test")
    _build_nax_mma(b, transpose_b=True)
    header, msl = _lower(b)
    # transpose_b is the 5th argument: (M, N, K, false, true, true, mode).
    full = header + msl
    assert "matmul2d_descriptor(16, 32, 16, false, true, true," in full
    # Body should show the helper call site rather than inline desc.
    assert "nax_mma_m16n32k16" in msl and "_tbT_" in msl


def test_nax_mma_transpose_b_false_emits_false():
    """transpose_b=False propagates to the descriptor's 5th arg."""
    b = Builder("test")
    _build_nax_mma(b, transpose_b=False)
    header, msl = _lower(b)
    full = header + msl
    assert "matmul2d_descriptor(16, 32, 16, false, false, true," in full
    # Not the default — no tbT helper for this kernel.
    assert "_tbT_" not in msl
    assert "_tbF_" in msl


def test_nax_mma_transpose_b_module_validates():
    """The IR module with transpose_b=False validates cleanly."""
    b = Builder("test")
    _build_nax_mma(b, transpose_b=False)
    b.end_function()
    validate_module(b.module)


def test_nax_mma_two_descriptors_in_one_function():
    """Two MMA calls with different transpose_b emit two distinct
    constexpr descriptors. Apple's compiler will dedupe identical
    constexpr values, so the cost is purely textual."""
    from quark.ir.mma_registry import _BY_SHAPE_ID

    b = Builder("test")
    b.register_shape(_BY_SHAPE_ID["m16n32k16_nax_bf16"].shape)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (32, 16))
    C = b.smem_alloc("C", DType.F32, (16, 32))
    a = b.load_matrix(A, "m16n32k16_nax_bf16", which="a")
    bf = b.load_matrix(B, "m16n32k16_nax_bf16", which="b")
    cf = b.load_matrix(C, "m16n32k16_nax_bf16", which="c")
    # GEMM1 style: transpose_b=True
    d1 = b.mma("m16n32k16_nax_bf16", a, bf, cf, transpose_b=True)
    # GEMM2 style: transpose_b=False — emitted-only; the test's
    # assertions read both descriptor variants out of the generated MSL.
    b.mma("m16n32k16_nax_bf16", a, bf, d1, transpose_b=False)
    header, msl = _lower(b)
    full = header + msl
    # Both descriptor variants present (in the header — one helper per
    # (shape, tb, acc) tuple).
    assert "matmul2d_descriptor(16, 32, 16, false, true, true," in full
    assert "matmul2d_descriptor(16, 32, 16, false, false, true," in full
    # Both helpers got called.
    assert "_tbT_" in msl
    assert "_tbF_" in msl


def test_ptx_mma_rejects_transpose_b_false():
    """PTX ``mma.sync``'s layout is fixed by the shape mnemonic.
    Accepting transpose_b=False on the PTX path would silently produce
    wrong code, so the visitor raises NotImplementedError.

    Tests the guard directly on the IR + visitor (no lowering harness)
    to avoid the LoadMatrixOp reg_offsets boilerplate that would
    otherwise fire first.
    """
    import pytest

    from quark.ir import op as _op
    from quark.ir.types import DType, ValueShape
    from quark.lower.ptx.lower import PtxLowerer

    # Construct just an MmaOp + dummy operands, bypass the full lowering
    # pipeline — the guard is at the visitor's entry point.
    b = Builder("test")
    b.begin_function("f")
    # Build dummy width-N b32 fragments without LoadMatrixOp.
    a = b._fresh(ValueShape(DType.B32, width=4), "fake_a")
    bf = b._fresh(ValueShape(DType.B32, width=2), "fake_b")
    cf = b._fresh(ValueShape(DType.F32, width=4), "fake_c")
    out = b._fresh(ValueShape(DType.F32, width=4), "fake_d")
    op = _op.MmaOp(
        results=(out,),
        operands=(a, bf, cf),
        attrs={"shape_id": "m16n8k16_bf16", "transpose_b": False},
    )

    # Direct visitor invocation — sidesteps the full lower_module flow.
    lowerer = PtxLowerer()

    # Fake a minimal _FnCtx — the visitor only reads ctx.module for the
    # shape lookup, which happens AFTER the transpose_b guard.
    class _FakeCtx:
        module = b.module

    with pytest.raises(NotImplementedError, match="transpose_b=False is NAX-only"):
        lowerer._visit_mma(op, _FakeCtx())
