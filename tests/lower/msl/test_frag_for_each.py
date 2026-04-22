"""MSL lowering tests for FragForEachOp.

Body is walked per (tile fi, thread_elements slot) with body_input_var
bound to ``e_fi[slot]`` and body_row_var / body_col_var bound to
Apple's lane-dependent row/col formulas. No smem round-trip.
"""

from __future__ import annotations

from quark.ir import BufferType, Builder, DType, MmaShape, validate_module
from quark.lower.msl import MslLowerer
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


def _make_builder():
    b = Builder("test")
    b.register_shape(_M16N8K16_BF16)
    b.begin_function("f")
    g_out = b.param("G", BufferType(DType.F32))
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (8, 16))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=_A_OFFS)
    bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
    cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    return b, b.mma("m16n8k16_bf16", a, bf, cf), g_out


def _lower(b: Builder) -> str:
    b.end_function()
    return MslLowerer(METAL_CAPS_FAKE).lower_module(b.module).source


def test_msl_frag_for_each_emits_thread_elements():
    b, d, _ = _make_builder()
    sink = b.smem_alloc("sink", DType.F32, (16, 8))

    def fn(elem, row, col):
        b.store(sink, elem, row, col)

    b.frag_for_each("m16n8k16_bf16", d, fn, _CD_OFFS)
    msl = _lower(b)
    assert "thread_elements" in msl
    assert "thread_index_in_simdgroup" in msl


def test_msl_frag_for_each_no_simdgroup_store():
    """The for-each body accesses frags via thread_elements() — there
    must be no simdgroup_store emissions from this op."""
    b, d, _ = _make_builder()
    sink = b.smem_alloc("sink", DType.F32, (16, 8))

    def fn(elem, row, col):
        b.store(sink, elem, row, col)

    b.frag_for_each("m16n8k16_bf16", d, fn, _CD_OFFS)
    msl = _lower(b)
    assert "simdgroup_store" not in msl


def test_msl_frag_for_each_emits_apple_row_col_bases():
    """The op should emit Apple's 2x2x2 row/col base computations once."""
    b, d, _ = _make_builder()
    sink = b.smem_alloc("sink", DType.F32, (16, 8))

    def fn(elem, row, col):
        b.store(sink, elem, row, col)

    b.frag_for_each("m16n8k16_bf16", d, fn, _CD_OFFS)
    msl = _lower(b)
    # Apple row:  ((L >> 4) & 1) * 4 + ((L >> 1) & 3)
    assert ">> 4u) & 1u) * 4u" in msl
    # Apple col0: ((L >> 3) & 1) * 4 + (L & 1) * 2
    assert ">> 3u) & 1u) * 4u" in msl


def test_msl_frag_for_each_emits_four_bodies():
    """2 tiles × 2 slots = 4 body walks → 4 store ops into the sink."""
    b, d, _ = _make_builder()
    sink = b.smem_alloc("sink", DType.F32, (16, 8))

    def fn(elem, row, col):
        b.store(sink, elem, row, col)

    b.frag_for_each("m16n8k16_bf16", d, fn, _CD_OFFS)
    msl = _lower(b)
    # Count sink writes — pattern: `<sinkname>[...] = ...;`
    # Use the buf name "smem_" prefix + some id. Simpler: look for " = "
    # lines that match the store pattern. Just check we have ≥4
    # assignments into the sink by searching for the sink's buffer.
    # The sink name can vary; instead count "foreach_elem" locals (one
    # per slot walk).
    # Each slot emits one `<acc_ty> foreach_elemN = <e_ref>[<slot>];`
    # decl. There should be 4 such decls (2 tiles × 2 slots).
    import re as _re

    decls = _re.findall(r"float foreach_elem\d+ =", msl)
    assert len(decls) == 4, f"expected 4 slot-elem decls, got {len(decls)}"


def test_msl_frag_for_each_validates():
    b, d, _ = _make_builder()
    sink = b.smem_alloc("sink", DType.F32, (16, 8))

    def fn(elem, row, col):
        b.store(sink, elem, row, col)

    b.frag_for_each("m16n8k16_bf16", d, fn, _CD_OFFS)
    b.end_function()
    validate_module(b.module)
