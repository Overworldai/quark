"""PTX lowering of FragApplyOp — verifies the per-c_reg re-walk pattern.

FragApplyOp's body region is a scalar subgraph with one input (the
per-slot element). The lowerer re-walks it once per c_reg, rebinding
the input to that reg and allocating fresh names for body-local
Values. The emitted PTX must contain one copy of the body transform
per c_reg, each reading from a distinct input reg and writing to a
distinct output reg.
"""

from __future__ import annotations

import re

from popcorn.ir import (
    Builder,
    DType,
    FragApplyOp,
    MmaShape,
    validate_module,
)
from popcorn.lower.ptx import PtxLowerer

_M16N8K16_BF16 = MmaShape(
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
    ptx="m16n8k16.row.col.f32.bf16.bf16.f32",
)


_A_OFFS = ((0, 0), (8, 0), (0, 8), (8, 8))
_B_OFFS = ((0, 0), (0, 8))
_CD_OFFS = ((0, 0), (0, 1), (8, 0), (8, 1))


def _make_acc_builder() -> tuple[Builder, object]:
    """Build a Builder with a live accumulator fragment (the result of
    a zero-init MMA chain). Returns (builder, c_frag_value)."""
    b = Builder("test")
    b.register_shape(_M16N8K16_BF16)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (8, 16))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    a_frag = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=_A_OFFS)
    b_frag = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
    c_frag = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    d = b.mma("m16n8k16_bf16", a_frag, b_frag, c_frag)
    return b, d


# ---------------------------------------------------------------------------
# IR-level checks (no lowering)
# ---------------------------------------------------------------------------


def test_frag_apply_constructs_with_body_region():
    """Builder.frag_apply builds a FragApplyOp whose body region ends
    with a YieldOp producing one Value of the input dtype."""
    b, d = _make_acc_builder()
    scale = b.const(DType.F32, 2.0)
    out = b.frag_apply("m16n8k16_bf16", d, lambda x: b.mul(x, scale))
    assert out.shape == d.shape
    # Look up the op emitted in the current region.
    ops = b.current_region.ops
    apply_ops = [op for op in ops if isinstance(op, FragApplyOp)]
    assert len(apply_ops) == 1, f"expected 1 FragApplyOp, got {len(apply_ops)}"
    ap = apply_ops[0]
    assert ap.attrs["shape_id"] == "m16n8k16_bf16"
    assert ap.body_input_var is not None
    assert ap.body_input_var.dtype is DType.F32
    term = ap.body.terminator
    assert term is not None
    assert len(term.operands) == 1
    assert term.operands[0].dtype is DType.F32


def test_frag_apply_rejects_non_f32_fragment():
    """frag_apply requires an f32 accumulator — bf16 frags don't have
    a per-c_reg scalar lowering (they're packed b32)."""
    import pytest

    b = Builder("test")
    b.register_shape(_M16N8K16_BF16)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    a_frag = b.load_matrix(A, "m16n8k16_bf16", which="a", row=0, col=0)
    with pytest.raises(TypeError, match="f32 accumulator"):
        b.frag_apply("m16n8k16_bf16", a_frag, lambda x: x)


def test_frag_apply_rejects_fn_with_wrong_dtype():
    """fn must preserve the element dtype — dtype-changing maps go
    through FragConvertOp, not FragApplyOp."""
    import pytest

    b, d = _make_acc_builder()
    with pytest.raises(TypeError, match="produced shape"):
        # Convert to bf16 inside the body — shape mismatch.
        b.frag_apply("m16n8k16_bf16", d, lambda x: b.convert(x, DType.BF16))


def test_frag_apply_validates_module():
    """Module with FragApplyOp passes validate_module."""
    b, d = _make_acc_builder()
    scale = b.const(DType.F32, 3.0)
    b.frag_apply("m16n8k16_bf16", d, lambda x: b.mul(x, scale))
    b.end_function()
    validate_module(b.module)


# ---------------------------------------------------------------------------
# PTX lowering checks
# ---------------------------------------------------------------------------


def test_ptx_frag_apply_emits_four_muls_for_m16n8_acc():
    """m16n8 bf16 accumulator has c_regs=4. A map(x → x * scale) body
    should lower to 4 `mul.f32` + 4 `mov.f32` pairs — one per c_reg.
    """
    b, d = _make_acc_builder()
    scale = b.const(DType.F32, 2.5)
    out = b.frag_apply("m16n8k16_bf16", d, lambda x: b.mul(x, scale))
    # Consume the output so it doesn't get DCE'd.
    dummy = b.smem_alloc("O", DType.F32, (16, 8))
    b.store_matrix(dummy, out, "m16n8k16_bf16", which="d", reg_offsets=_CD_OFFS)
    b.end_function()
    ptx = PtxLowerer().lower_module(b.module).ptx
    # One mul.f32 per c_reg → 4 muls whose operands include the scale.
    muls = re.findall(r"mul\.f32 %f\d+, %f\d+, %f\d+;", ptx)
    # There's only one scalar mul we emit here (the scale × c_reg), so
    # all muls in the PTX are from the FragApplyOp body. Expect 4.
    assert len(muls) == 4, f"expected 4 mul.f32 (one per c_reg), got {len(muls)}\n{ptx}"
    # And 4 mov.f32 from yield-coalesce, one per output c_reg.
    assert ptx.count("mov.f32") >= 4


def test_ptx_frag_apply_reads_distinct_input_regs_writes_distinct_outputs():
    """Each body walk should read a DIFFERENT input reg (one per c_reg)
    and write to a DIFFERENT output reg — guards against the
    "name-binding survives across walks" bug where walk #2 emits
    `mul %f7, %f7, scale` instead of `mul %fN, %fN, scale` with a
    fresh dst."""
    b, d = _make_acc_builder()
    scale = b.const(DType.F32, 2.0)
    out = b.frag_apply("m16n8k16_bf16", d, lambda x: b.mul(x, scale))
    dummy = b.smem_alloc("O", DType.F32, (16, 8))
    b.store_matrix(dummy, out, "m16n8k16_bf16", which="d", reg_offsets=_CD_OFFS)
    b.end_function()
    ptx = PtxLowerer().lower_module(b.module).ptx
    # Extract the 4 muls. Each must have a unique dst AND a unique lhs
    # (the c_reg being read). The scale reg can repeat (it's a const).
    muls = re.findall(r"mul\.f32 (%f\d+), (%f\d+), (%f\d+);", ptx)
    assert len(muls) == 4, f"expected 4 muls, got {muls}"
    dsts = [m[0] for m in muls]
    srcs = [m[1] for m in muls]
    assert len(set(dsts)) == 4, f"dst regs must be unique: {dsts}"
    assert len(set(srcs)) == 4, f"src c_regs must be unique: {srcs}"


def test_ptx_frag_apply_chained_maps():
    """Two chained maps produce 8 muls total, each chain's body walked
    4 times (once per c_reg). Verifies the PTX re-walk pattern composes."""
    b, d = _make_acc_builder()
    s1 = b.const(DType.F32, 2.0)
    s2 = b.const(DType.F32, 3.0)
    out1 = b.frag_apply("m16n8k16_bf16", d, lambda x: b.mul(x, s1))
    out2 = b.frag_apply("m16n8k16_bf16", out1, lambda x: b.mul(x, s2))
    dummy2 = b.smem_alloc("O", DType.F32, (16, 8))
    b.store_matrix(dummy2, out2, "m16n8k16_bf16", which="d", reg_offsets=_CD_OFFS)
    b.end_function()
    ptx = PtxLowerer().lower_module(b.module).ptx
    muls = re.findall(r"mul\.f32 %f\d+, %f\d+, %f\d+;", ptx)
    assert len(muls) == 8, f"expected 8 muls (2 maps × 4 c_regs), got {len(muls)}\n{ptx}"


def test_ptx_frag_apply_composite_body_ex2():
    """Body with a compound transform (sub + mul + ex2_approx) — still
    one copy per c_reg. Tests that body-local Value name collection
    handles multi-op bodies correctly."""
    b, d = _make_acc_builder()
    m_new = b.const(DType.F32, 0.5)
    log2e = b.const(DType.F32, 1.4426950408889634)
    out = b.frag_apply(
        "m16n8k16_bf16",
        d,
        lambda x: b.ex2_approx(b.mul(b.sub(x, m_new), log2e)),
    )
    dummy = b.smem_alloc("O", DType.F32, (16, 8))
    b.store_matrix(dummy, out, "m16n8k16_bf16", which="d", reg_offsets=_CD_OFFS)
    b.end_function()
    ptx = PtxLowerer().lower_module(b.module).ptx
    # 4 each of sub, mul, ex2 — one per c_reg.
    assert ptx.count("sub.f32") == 4, f"expected 4 subs\n{ptx}"
    assert ptx.count("mul.f32") == 4
    assert ptx.count("ex2.approx.f32") == 4
