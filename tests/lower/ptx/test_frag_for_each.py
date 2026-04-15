"""PTX lowering tests for FragForEachOp.

Per-slot side-effect body: body sees (elem, row, col) where row and
col are U32 Values bound at lower time to ``gid + dr`` and
``tig*2 + dc`` respectively (per cd_offsets).
"""

from __future__ import annotations

import re

from popcorn.ir import (
    BufferType,
    Builder,
    DType,
    FragForEachOp,
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


def _make_builder_with_gmem():
    b = Builder("test")
    b.register_shape(_M16N8K16_BF16)
    b.begin_function("f")
    # Global output buffer for the epilogue stores.
    g_out = b.param("G", BufferType(DType.F32))
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (8, 16))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=_A_OFFS)
    bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
    cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    d = b.mma("m16n8k16_bf16", a, bf, cf)
    return b, d, g_out


def test_frag_for_each_constructs():
    b, d, g = _make_builder_with_gmem()
    # Use a 2D global tensor proxy via raw store to a buffer param.
    # Simplest: just do an arithmetic side effect.
    sink = b.smem_alloc("sink", DType.F32, (16, 8))

    def fn(elem, row, col):
        b.store(sink, elem, row, col)

    b.frag_for_each("m16n8k16_bf16", d, fn, _CD_OFFS)
    ops = b.current_region.ops
    fe_ops = [op for op in ops if isinstance(op, FragForEachOp)]
    assert len(fe_ops) == 1
    op = fe_ops[0]
    assert op.body_input_var is not None
    assert op.body_row_var is not None
    assert op.body_col_var is not None
    assert len(op.results) == 0


def test_ptx_frag_for_each_emits_per_slot_body():
    """4 c_regs → 4 body walks → 4 st.shared.f32 emissions."""
    b, d, g = _make_builder_with_gmem()
    sink = b.smem_alloc("sink", DType.F32, (16, 8))

    def fn(elem, row, col):
        b.store(sink, elem, row, col)

    b.frag_for_each("m16n8k16_bf16", d, fn, _CD_OFFS)
    b.end_function()
    ptx = PtxLowerer().lower_module(b.module).ptx
    # 4 store emits (one per c_reg, into the sink smem).
    st_count = len(re.findall(r"st\.shared\.(?:f32|b32) ", ptx))
    assert st_count == 4, f"expected 4 stores from for_each, got {st_count}\n{ptx}"


def test_ptx_frag_for_each_row_col_formula():
    """Body should reference gid + dr and tig*2 + dc per slot."""
    b, d, g = _make_builder_with_gmem()
    sink = b.smem_alloc("sink", DType.F32, (16, 8))

    def fn(elem, row, col):
        b.store(sink, elem, row, col)

    b.frag_for_each("m16n8k16_bf16", d, fn, _CD_OFFS)
    b.end_function()
    ptx = PtxLowerer().lower_module(b.module).ptx
    # Look for %laneid shift — the gid computation.
    assert "%laneid" in ptx
    assert "shr.u32" in ptx  # gid = laneid >> 2
    assert "and.b32" in ptx  # tig = laneid & 3


def test_ptx_frag_for_each_validates():
    b, d, g = _make_builder_with_gmem()
    sink = b.smem_alloc("sink", DType.F32, (16, 8))

    def fn(elem, row, col):
        b.store(sink, elem, row, col)

    b.frag_for_each("m16n8k16_bf16", d, fn, _CD_OFFS)
    b.end_function()
    validate_module(b.module)


def test_frag_for_each_rejects_non_void_fn():
    import pytest

    b, d, g = _make_builder_with_gmem()

    def bad_fn(elem, row, col):
        return elem  # should be None

    with pytest.raises(TypeError, match="side-effect body"):
        b.frag_for_each("m16n8k16_bf16", d, bad_fn, _CD_OFFS)
