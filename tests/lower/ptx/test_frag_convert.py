"""PTX lowering tests for FragConvertOp.

Currently scoped to ACC f32 → A_FRAG bf16. Arbitrary register-tile
conversions raise NotImplementedError and will be added when a caller
needs them.
"""

from __future__ import annotations

import re

import pytest

from quark.ir import (
    Builder,
    DType,
    FragConvertOp,
    MmaShape,
    validate_module,
)
from quark.lower.ptx import PtxLowerer

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


def _live_acc(b: Builder, *, suffix: str = ""):
    if "m16n8k16_bf16" not in b.module.kernel_shapes:
        b.register_shape(_M16N8K16_BF16)
        b.begin_function("f")
    A = b.smem_alloc(f"A{suffix}", DType.BF16, (16, 16))
    B = b.smem_alloc(f"B{suffix}", DType.BF16, (8, 16))
    C = b.smem_alloc(f"C{suffix}", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=_A_OFFS)
    bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
    cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    return b.mma("m16n8k16_bf16", a, bf, cf)


def test_frag_convert_construct():
    b = Builder("t")
    d = _live_acc(b, suffix="1")
    d2 = _live_acc(b, suffix="2")
    out = b.frag_convert(
        "m16n8k16_bf16",
        src_frags=(d, d2),
        src_layout="acc",
        dst_layout="a_frag",
        src_dtype=DType.F32,
        dst_dtype=DType.BF16,
        cd_offsets=_CD_OFFS,
    )
    assert out.dtype is DType.B32
    assert out.width == 4  # shape.a_regs
    ops = [op for op in b.current_region.ops if isinstance(op, FragConvertOp)]
    assert len(ops) == 1
    assert ops[0].attrs["num_src_frags"] == 2


def test_frag_convert_rejects_non_acc_to_a():
    b = Builder("t")
    d = _live_acc(b)
    with pytest.raises(NotImplementedError, match="acc.+a_frag"):
        b.frag_convert(
            "m16n8k16_bf16",
            src_frags=(d,),
            src_layout="a_frag",
            dst_layout="acc",
            src_dtype=DType.F32,
            dst_dtype=DType.BF16,
            cd_offsets=_CD_OFFS,
        )


def test_frag_convert_rejects_non_f32_to_bf16():
    b = Builder("t")
    d = _live_acc(b)
    with pytest.raises(NotImplementedError, match="f32.+bf16"):
        b.frag_convert(
            "m16n8k16_bf16",
            src_frags=(d,),
            src_layout="acc",
            dst_layout="a_frag",
            src_dtype=DType.F32,
            dst_dtype=DType.F16,
            cd_offsets=_CD_OFFS,
        )


def test_ptx_frag_convert_emits_cvt_and_pack():
    """For 2 src frags × 4 c_regs each = 8 elements, lowering should
    emit 8 ``cvt.rn.bf16.f32`` instructions and 4 ``mov.b32 %o, {%lo, %hi}``
    pack idioms."""
    b = Builder("t")
    d1 = _live_acc(b)
    d2 = _live_acc(b)
    out = b.frag_convert(
        "m16n8k16_bf16",
        src_frags=(d1, d2),
        src_layout="acc",
        dst_layout="a_frag",
        src_dtype=DType.F32,
        dst_dtype=DType.BF16,
        cd_offsets=_CD_OFFS,
    )
    # Make out used.
    b.smem_alloc("Aout", DType.BF16, (16, 16))
    B = b.smem_alloc("Bout", DType.BF16, (8, 16))
    C = b.smem_alloc("Cout", DType.F32, (16, 8))
    bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
    cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    b.mma("m16n8k16_bf16", out, bf, cf)
    b.end_function()
    ptx = PtxLowerer().lower_module(b.module).ptx
    assert ptx.count("cvt.rn.bf16.f32") == 8, (
        f"expected 8 cvt.rn.bf16.f32, got {ptx.count('cvt.rn.bf16.f32')}"
    )
    # 4 packs: mov.b32 %b?, {%h?, %h?};
    packs = re.findall(r"mov\.b32 %b\d+, \{%h\d+, %h\d+\};", ptx)
    assert len(packs) == 4, f"expected 4 b32 packs, got {len(packs)}\n{ptx}"


def test_ptx_frag_convert_with_body_transform():
    """Body applies exp2((x - m_rc) * log2e), with per-row-class m_rc.
    Emits per-c_reg sub/mul/ex2 before the cvt."""
    b = Builder("t")
    d = _live_acc(b)
    m_rc0 = b.const(DType.F32, 0.5)
    m_rc1 = b.const(DType.F32, 1.5)
    log2e = b.const(DType.F32, 1.4426950408889634)

    def fn(elem, m_rc):
        return b.ex2_approx(b.mul(b.sub(elem, m_rc), log2e))

    out = b.frag_convert(
        "m16n8k16_bf16",
        src_frags=(d,),
        src_layout="acc",
        dst_layout="a_frag",
        src_dtype=DType.F32,
        dst_dtype=DType.BF16,
        cd_offsets=_CD_OFFS,
        fn=fn,
        selectors=(m_rc0, m_rc1),
        slot_to_selector_idx=(0, 0, 1, 1),
    )
    b.smem_alloc("Aout", DType.BF16, (16, 16))
    B = b.smem_alloc("Bout", DType.BF16, (8, 16))
    C = b.smem_alloc("Cout", DType.F32, (16, 8))
    bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
    cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    b.mma("m16n8k16_bf16", out, bf, cf)
    b.end_function()
    ptx = PtxLowerer().lower_module(b.module).ptx
    # 4 subs, 4 muls, 4 ex2, 4 cvt for the body × 1 src frag.
    assert ptx.count("sub.f32") >= 4
    assert ptx.count("ex2.approx.f32") >= 4
    assert ptx.count("cvt.rn.bf16.f32") == 4


def test_frag_convert_validates():
    b = Builder("t")
    d = _live_acc(b, suffix="1")
    d2 = _live_acc(b, suffix="2")
    b.frag_convert(
        "m16n8k16_bf16",
        src_frags=(d, d2),
        src_layout="acc",
        dst_layout="a_frag",
        src_dtype=DType.F32,
        dst_dtype=DType.BF16,
        cd_offsets=_CD_OFFS,
    )
    b.end_function()
    validate_module(b.module)
