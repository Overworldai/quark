"""MSL lowering tests for FragConvertOp.

Currently scoped to ACC f32 → A_FRAG bf16 using the thread_elements()
direct-write pattern (no smem round-trip, no simd_shuffle). Other
layouts raise NotImplementedError.
"""

from __future__ import annotations

import pytest

from popcorn.ir import Builder, DType, MmaShape, validate_module
from popcorn.lower.msl import MslLowerer
from tests.lower.msl.conftest import METAL_CAPS_FAKE

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
    msl="bfloat16_t:2:1:2",
)

_A_OFFS = ((0, 0), (8, 0), (0, 8), (8, 8))
_B_OFFS = ((0, 0), (0, 8))
_CD_OFFS = ((0, 0), (0, 1), (8, 0), (8, 1))


def _live_acc(b: Builder, *, suffix=""):
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


def _lower(b: Builder) -> str:
    b.end_function()
    return MslLowerer(METAL_CAPS_FAKE).lower_module(b.module).source


def test_msl_frag_convert_emits_thread_elements_no_simd_shuffle():
    """FragConvert on MSL uses thread_elements() directly — no
    simd_shuffle (the Apple layout is dtype-agnostic so positions match
    between f32 and bf16 tiles)."""
    b = Builder("t")
    d1 = _live_acc(b)
    out = b.frag_convert(
        "m16n8k16_bf16",
        src_frags=(d1,),
        src_layout="acc",
        dst_layout="a_frag",
        src_dtype=DType.F32,
        dst_dtype=DType.BF16,
        cd_offsets=_CD_OFFS,
    )
    # Consume out so it isn't DCE'd — feed into a downstream MMA.
    B = b.smem_alloc("Bo", DType.BF16, (8, 16))
    C = b.smem_alloc("Co", DType.F32, (16, 8))
    bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
    cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    b.mma("m16n8k16_bf16", out, bf, cf)
    msl = _lower(b)
    assert "thread_elements()" in msl
    # The FragConvert body must NOT route through simd_shuffle.
    assert "simd_shuffle" not in msl, f"FragConvertOp MSL body shouldn't use simd_shuffle\n{msl}"
    # Cast pattern appears in the FragConvert body.
    assert "(bfloat16_t)" in msl, f"FragConvertOp MSL should emit (bfloat16_t) cast\n{msl}"


def test_msl_frag_convert_registers_frag_values():
    """Output of FragConvertOp on MSL is registered as a simdgroup_matrix
    array. Downstream MMA consuming this Value reads from frag_values
    and skips the b32→simdgroup_matrix unpack path."""
    b = Builder("t")
    d1 = _live_acc(b, suffix="1")
    d2 = _live_acc(b, suffix="2")
    out = b.frag_convert(
        "m16n8k16_bf16",
        src_frags=(d1, d2),
        src_layout="acc",
        dst_layout="a_frag",
        src_dtype=DType.F32,
        dst_dtype=DType.BF16,
        cd_offsets=_CD_OFFS,
    )
    B = b.smem_alloc("Bout", DType.BF16, (8, 16))
    C = b.smem_alloc("Cout", DType.F32, (16, 8))
    bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
    cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    b.mma("m16n8k16_bf16", out, bf, cf)
    msl = _lower(b)
    # The simdgroup_multiply_accumulate should reference the FragConvert's
    # output frag array directly — no intermediate simd_shuffle_xor /
    # pack_b32 scratch. The old pipeline would emit "ushort2" bitcast
    # patterns; the new one uses thread_elements() writes.
    assert "ushort2" not in msl.split("simdgroup_multiply_accumulate")[0][-500:], (
        "FragConvert output fed MMA through b32 unpack path (unexpected)"
    )


def test_msl_frag_convert_rejects_non_supported():
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


def test_msl_frag_convert_with_body_and_selectors():
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
    B = b.smem_alloc("Bout", DType.BF16, (8, 16))
    C = b.smem_alloc("Cout", DType.F32, (16, 8))
    bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
    cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    b.mma("m16n8k16_bf16", out, bf, cf)
    msl = _lower(b)
    # Body: 4 exp2 calls (one per slot: 2 tiles × 2 slots each).
    assert msl.count("exp2") == 4


def test_msl_frag_convert_validates():
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
