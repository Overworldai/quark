"""MSL lowering tests for the NAX MMA's implicit dtype cast at the
cooperative_tensor input assignment.

Apple's ``matmul2d`` cooperative_tensor storage is fixed by its template
parameters. When an MmaOp's IR operand has a different dtype than the
shape declares (e.g. attention's GEMM2 hands an f32 softmax accumulator
to a shape whose A is bf16), the visitor must narrow at the assignment
site — ``ct_a[i] = static_cast<bfloat>(src_i)`` — otherwise the Apple
backend silently reinterprets bits and produces garbage. (Probed
empirically: cosine 0.022 without the cast, 1.000 with it.)

The cast applies whenever ``operand.dtype != shape.{a,b,acc}_dtype``;
the matched-dtype path stays a direct copy.
"""

from __future__ import annotations

from quark.ir import Builder, DType, validate_module
from quark.lower.msl import MslLowerer
from tests.lower.msl.conftest import METAL_CAPS_FAKE


def _register_nax(b: Builder) -> None:
    from quark.ir.mma_registry import _BY_SHAPE_ID

    b.register_shape(_BY_SHAPE_ID["m16n32k16_nax_bf16"].shape)


def _lower(b: Builder) -> str:
    b.end_function()
    return MslLowerer(METAL_CAPS_FAKE).lower_module(b.module).source


def _build_f32_a_mma(b: Builder, *, transpose_b: bool = False):
    """Build an MMA whose A operand is a width-8 f32 vec_build (mimicking
    a softmax accumulator handed to GEMM2). B/C come from regular
    load_matrix on the bf16 NAX shape, so they have matched dtypes."""
    _register_nax(b)
    b.begin_function("f")
    # f32 width-8 A: 8 fresh f32 consts → vec_build.
    a_scalars = [b.const(DType.F32, float(i)) for i in range(8)]
    a_f32 = b.vec_build(a_scalars, name="a_f32")
    # B and C from regular load_matrix on the NAX shape.
    B = b.smem_alloc("B", DType.BF16, (32, 16))
    C = b.smem_alloc("C", DType.F32, (16, 32))
    bf = b.load_matrix(B, "m16n32k16_nax_bf16", which="b")
    cf = b.load_matrix(C, "m16n32k16_nax_bf16", which="c")
    return b.mma("m16n32k16_nax_bf16", a_f32, bf, cf, transpose_b=transpose_b)


def test_f32_a_emits_static_cast_at_ct_a_assignment():
    """An f32 A operand against the bf16 shape narrows via static_cast
    at the ct_a[i] = ... assignment lines."""
    b = Builder("test")
    _build_f32_a_mma(b)
    msl = _lower(b)
    # All 8 ct_a slots assigned with static_cast<bfloat>.
    assert msl.count("static_cast<bfloat>") >= 8
    # Spot-check a representative slot — the slot index is what matters,
    # the source name is allocator-determined.
    assert "= static_cast<bfloat>(" in msl
    # The line shape is `ct_aN[i] = static_cast<bfloat>(<name>);` — make
    # sure the cast is on the ct_a assignment side, not somewhere else.
    for line in msl.splitlines():
        if "static_cast<bfloat>" in line:
            assert "ct_a" in line, f"expected static_cast on ct_a line, got: {line!r}"


def test_b_and_c_matched_dtypes_skip_cast():
    """When B is bf16 (matches shape.b_dtype) and C is f32 (matches
    shape.acc_dtype), their assignments stay direct copies."""
    b = Builder("test")
    _build_f32_a_mma(b)
    msl = _lower(b)
    # No cast on ct_b or ct_c lines.
    for line in msl.splitlines():
        if line.lstrip().startswith(("ct_b", "ct_c")) and "[" in line and "=" in line:
            assert "static_cast" not in line, f"unexpected cast on B/C assignment: {line!r}"


def test_matched_dtype_a_no_cast():
    """Baseline: bf16 A from regular load_matrix → no cast emitted on
    the ct_a assignment (sanity check that the cast is conditional).

    After Phase 3 the matmul body lives in a per-(shape, tb, acc,
    cast_a) helper function in the kernel header. With matched
    dtypes the cast_a tag is absent from the helper name and the
    helper body has no ``static_cast<bfloat>``.
    """
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
    b.mma("m16n32k16_nax_bf16", a, bf, cf)
    b.end_function()
    lk = MslLowerer(METAL_CAPS_FAKE).lower_module(b.module)
    full = lk.header + lk.source
    # Zero bfloat casts in either header or body when all dtypes match.
    # (``reinterpret_cast`` is fine — the vec4 loads use it.)
    assert "static_cast<bfloat>" not in full
    # Helper name has no ``_castX`` tag for matched dtypes.
    assert "_tbT_accT" in full
    assert "_castF32" not in full


def test_f32_a_module_validates():
    """The IR module with a dtype-mismatched A operand still validates —
    the validator only enforces shape_id registration, not operand
    dtype matching (the cast handles the mismatch at lowering time)."""
    b = Builder("test")
    _build_f32_a_mma(b)
    b.end_function()
    validate_module(b.module)
