"""Tests for MmaOp / LoadMatrixOp / StoreMatrixOp lowering.

The PTX lowerer's default `LoadMatrix` / `StoreMatrix` path is the
**manual** per-register scalar-load pattern that matches the hoisted
frag loaders in `mma/frag.py`. `ldmatrix` is reachable as an explicit
opt-in via `layout_hint="ldmatrix"` but is not the default, because the
codebase's real kernels rely on manual assembly for preshuffled layouts
and dtype casts around the load.

## Authoritative per-register offset tables (row.col layouts)

All entries below are `(row_elem, col_elem)` tuples, one per fragment
register, sourced directly from the PTX ISA §9.7.14.5 formulas and
derived for A in row-major smem and B stored as B^T in smem (so the K
dimension is contiguous — what the user calls the "row.col" layout).

The per-lane base — `groupID * row_stride + tidIG * elems_per_lane`
where `groupID = laneid >> 2` and `tidIG = laneid & 3` — is NOT encoded
here. It's the caller's job to fold it into a SharedRegion view via
`dyn_offset=` before calling `load_matrix`. Tests that exercise
per-lane math build their own view.
"""

import re

import pytest

from popcorn.ir import (
    BufferType,
    Builder,
    DType,
    GlobalTensor,
    MmaShape,
    validate_module,
)
from popcorn.lower.ptx import PtxLowerer

# ---------------------------------------------------------------------------
# Shape registry fixtures
# ---------------------------------------------------------------------------

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

_M16N8K16_E4M3 = MmaShape(
    name="m16n8k16_e4m3",
    m=16,
    n=8,
    k=16,
    a_dtype=DType.E4M3,
    b_dtype=DType.E4M3,
    acc_dtype=DType.F32,
    a_regs=2,
    b_regs=1,
    c_regs=4,
    ptx="m16n8k16.row.col.f32.e4m3.e4m3.f32",
)

_M16N8K32_E4M3 = MmaShape(
    name="m16n8k32_e4m3",
    m=16,
    n=8,
    k=32,
    a_dtype=DType.E4M3,
    b_dtype=DType.E4M3,
    acc_dtype=DType.F32,
    a_regs=4,
    b_regs=2,
    c_regs=4,
    ptx="m16n8k32.row.col.f32.e4m3.e4m3.f32",
)


# ---------------------------------------------------------------------------
# Per-shape register-offset tables (row.col layout)
#
# Derived from PTX ISA §9.7.14.5 formulas. Each entry is (dr, dc) in
# element units, relative to the per-lane base at (groupID, tidIG*S)
# where S is the lane col stride (2 for bf16/fp16, 4 for 8-bit types).
#
# A matrix: row-major smem (rows=M, cols=K).
# B matrix: B^T in smem (rows=N, cols=K), so that K is contiguous on the
#           fast axis — the "colmajor transposed" layout the codebase
#           uses throughout.
# C/D accumulator: row-major smem (rows=M, cols=N).
# ---------------------------------------------------------------------------

# m16n8k16 bf16 A — 4 regs, each holding 2 bf16 (1 b32). Packing:
#   reg 0 = (a0, a1) at (groupID+0, tidIG*2+0)
#   reg 1 = (a2, a3) at (groupID+8, tidIG*2+0)
#   reg 2 = (a4, a5) at (groupID+0, tidIG*2+8)
#   reg 3 = (a6, a7) at (groupID+8, tidIG*2+8)
A_BF16_K16_ROW = ((0, 0), (8, 0), (0, 8), (8, 8))

# m16n8k16 bf16 B (B^T in smem) — 2 regs, each holding 2 bf16.
#   reg 0 = (b0, b1) at (groupID, tidIG*2+0)
#   reg 1 = (b2, b3) at (groupID, tidIG*2+8)
B_BF16_K16_COL_T = ((0, 0), (0, 8))

# m16n8k16 f32 C/D — 4 regs, each holding 1 f32. Per the ISA:
#   reg 0 at (groupID+0, tidIG*2+0)
#   reg 1 at (groupID+0, tidIG*2+1)
#   reg 2 at (groupID+8, tidIG*2+0)
#   reg 3 at (groupID+8, tidIG*2+1)
CD_F32_ROW = ((0, 0), (0, 1), (8, 0), (8, 1))

# m16n8k16 e4m3 A — 2 regs, each holding 4 e4m3. Per-lane base uses
# `tidIG*4` (each lane covers 4 e4m3 elements on K).
#   reg 0 = (a0..a3) at (groupID+0, tidIG*4+0)
#   reg 1 = (a4..a7) at (groupID+8, tidIG*4+0)
A_E4M3_K16_ROW = ((0, 0), (8, 0))

# m16n8k16 e4m3 B (B^T in smem) — 1 reg, 4 e4m3.
#   reg 0 = (b0..b3) at (groupID, tidIG*4+0)
B_E4M3_K16_COL_T = ((0, 0),)

# m16n8k32 e4m3 A — 4 regs, each holding 4 e4m3 (k doubles, so there's
# a second pair of regs at col += 16).
#   reg 0 = (a0..a3)   at (groupID+0, tidIG*4+0)
#   reg 1 = (a4..a7)   at (groupID+8, tidIG*4+0)
#   reg 2 = (a8..a11)  at (groupID+0, tidIG*4+16)
#   reg 3 = (a12..a15) at (groupID+8, tidIG*4+16)
A_E4M3_K32_ROW = ((0, 0), (8, 0), (0, 16), (8, 16))

# m16n8k32 e4m3 B (B^T in smem) — 2 regs, each holding 4 e4m3.
#   reg 0 = (b0..b3) at (groupID, tidIG*4+0)
#   reg 1 = (b4..b7) at (groupID, tidIG*4+16)
B_E4M3_K32_COL_T = ((0, 0), (0, 16))


def _builder_with(shape: MmaShape) -> Builder:
    b = Builder("m")
    b.register_shape(shape)
    b.begin_function("f")
    return b


def _lower(b: Builder) -> str:
    """Close `b`'s open function and return the PTX text string."""
    b.end_function()
    return PtxLowerer().lower_module(b.module).ptx


# ---------------------------------------------------------------------------
# LoadMatrix: default manual path
# ---------------------------------------------------------------------------


class TestLoadMatrixManualBf16:
    """bf16 m16n8k16 — 4 regs A, 2 regs B, 4 regs C/D."""

    def test_a_frag_emits_four_scalar_loads_at_iso_offsets(self):
        b = _builder_with(_M16N8K16_BF16)
        # Tile in bf16 with row_stride = 16 (elements), elem_bytes = 2.
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        b.load_matrix(
            A,
            "m16n8k16_bf16",
            which="a",
            row=0,
            col=0,
            reg_offsets=A_BF16_K16_ROW,
        )
        out = _lower(b)
        # row_stride_bytes = 16 * 2 = 32
        # col_stride_bytes = 2
        # Byte offsets per reg:
        #   (0, 0): 0
        #   (8, 0): 8 * 32 = 256
        #   (0, 8): 8 * 2 = 16
        #   (8, 8): 8 * 32 + 8 * 2 = 272
        assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+\];", out)  # (0,0)
        assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+ \+ 256\];", out)
        assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+ \+ 16\];", out)
        assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+ \+ 272\];", out)
        # No ldmatrix emitted on the default path.
        assert "ldmatrix" not in out

    def test_b_frag_emits_two_scalar_loads(self):
        b = _builder_with(_M16N8K16_BF16)
        # B^T smem: (N, K) = (8, 16)
        B = b.smem_alloc("B", DType.BF16, (8, 16))
        b.load_matrix(
            B,
            "m16n8k16_bf16",
            which="b",
            row=0,
            col=0,
            reg_offsets=B_BF16_K16_COL_T,
        )
        out = _lower(b)
        # row_stride_bytes = 16 * 2 = 32
        # (0, 0): 0
        # (0, 8): 8 * 2 = 16
        assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+\];", out)
        assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+ \+ 16\];", out)
        assert out.count("ld.shared.b32") == 2
        assert "ldmatrix" not in out

    def test_a_frag_with_tile_base_offsets_fold(self):
        """Passing a non-zero (row, col) should fold into each reg's
        static byte offset."""
        b = _builder_with(_M16N8K16_BF16)
        A = b.smem_alloc("A", DType.BF16, (64, 16))  # row_stride_bytes = 32
        # m_tile=1 means row_base = 16 elements; kk=0
        b.load_matrix(
            A,
            "m16n8k16_bf16",
            which="a",
            row=16,
            col=0,
            reg_offsets=A_BF16_K16_ROW,
        )
        out = _lower(b)
        # Base bytes = 16 * 32 = 512
        # reg0: 512 + 0    = 512
        # reg1: 512 + 256  = 768
        # reg2: 512 + 16   = 528
        # reg3: 512 + 272  = 784
        for expected in (512, 768, 528, 784):
            assert re.search(rf"ld\.shared\.b32 %b\d+, \[%r\d+ \+ {expected}\];", out), (
                f"missing load at +{expected} bytes:\n{out}"
            )


class TestLoadMatrixManualE4m3:
    """e4m3 m16n8k16 and m16n8k32 — 1-byte element paths."""

    def test_a_k16_emits_two_loads(self):
        b = _builder_with(_M16N8K16_E4M3)
        A = b.smem_alloc("A", DType.E4M3, (32, 16))  # row_stride_bytes = 16
        b.load_matrix(
            A,
            "m16n8k16_e4m3",
            which="a",
            row=0,
            col=0,
            reg_offsets=A_E4M3_K16_ROW,
        )
        out = _lower(b)
        # reg0: 0; reg1: 8 * 16 = 128
        assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+\];", out)
        assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+ \+ 128\];", out)
        assert out.count("ld.shared.b32") == 2

    def test_b_k16_single_load(self):
        b = _builder_with(_M16N8K16_E4M3)
        B = b.smem_alloc("B", DType.E4M3, (8, 16))
        b.load_matrix(
            B,
            "m16n8k16_e4m3",
            which="b",
            row=0,
            col=0,
            reg_offsets=B_E4M3_K16_COL_T,
        )
        out = _lower(b)
        assert out.count("ld.shared.b32") == 1

    def test_a_k32_emits_four_loads_with_col_spread(self):
        b = _builder_with(_M16N8K32_E4M3)
        A = b.smem_alloc("A", DType.E4M3, (32, 32))  # row_stride_bytes = 32
        b.load_matrix(
            A,
            "m16n8k32_e4m3",
            which="a",
            row=0,
            col=0,
            reg_offsets=A_E4M3_K32_ROW,
        )
        out = _lower(b)
        # row_stride_bytes = 32, col_stride_bytes = 1
        # reg0 (0,0):   0
        # reg1 (8,0):   256
        # reg2 (0,16):  16
        # reg3 (8,16):  272
        for off in (0, 256, 16, 272):
            if off == 0:
                assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+\];", out)
            else:
                assert re.search(rf"ld\.shared\.b32 %b\d+, \[%r\d+ \+ {off}\];", out), (
                    f"missing load at +{off}"
                )

    def test_b_k32_two_loads_16_byte_step(self):
        b = _builder_with(_M16N8K32_E4M3)
        B = b.smem_alloc("B", DType.E4M3, (8, 32))  # row_stride_bytes = 32
        b.load_matrix(
            B,
            "m16n8k32_e4m3",
            which="b",
            row=0,
            col=0,
            reg_offsets=B_E4M3_K32_COL_T,
        )
        out = _lower(b)
        # (0, 0): 0 ; (0, 16): 16
        assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+\];", out)
        assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+ \+ 16\];", out)


# ---------------------------------------------------------------------------
# LoadMatrix: error paths
# ---------------------------------------------------------------------------


class TestLoadMatrixErrors:
    def test_missing_reg_offsets_raises_with_hint(self):
        b = _builder_with(_M16N8K16_BF16)
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        # No reg_offsets, no ldmatrix hint.
        b.load_matrix(A, "m16n8k16_bf16", which="a", row=0, col=0)
        with pytest.raises(ValueError, match="reg_offsets"):
            _lower(b)

    def test_reg_offsets_wrong_count_rejected_at_build(self):
        b = _builder_with(_M16N8K16_BF16)
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        with pytest.raises(ValueError, match="reg_offsets"):
            b.load_matrix(
                A,
                "m16n8k16_bf16",
                which="a",
                reg_offsets=((0, 0), (8, 0)),  # should be 4
            )

    def test_gmem_source_rejected(self):
        b = _builder_with(_M16N8K16_BF16)
        b.param("X", BufferType(DType.BF16))
        g = GlobalTensor(
            dtype=DType.BF16,
            shape=(32, 16),
            stride=(16, 1),
            name="X",
            param=b.function.params[-1],
        )
        b.load_matrix(
            g,
            "m16n8k16_bf16",
            which="a",
            reg_offsets=A_BF16_K16_ROW,
        )
        with pytest.raises(NotImplementedError, match="SharedRegion"):
            _lower(b)


# ---------------------------------------------------------------------------
# LoadMatrix: ldmatrix opt-in fast path
# ---------------------------------------------------------------------------


class TestLoadMatrixLdmatrixOptIn:
    def test_ldmatrix_hint_x4_for_a_frag(self):
        b = _builder_with(_M16N8K16_BF16)
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        b.load_matrix(
            A,
            "m16n8k16_bf16",
            which="a",
            row=0,
            col=0,
            layout_hint="ldmatrix",
        )
        out = _lower(b)
        assert re.search(
            r"ldmatrix\.sync\.aligned\.x4\.m8n8\.shared\.b16 "
            r"\{%b\d+, %b\d+, %b\d+, %b\d+\}, \[%r\d+\];",
            out,
        )
        # And no manual loads on the ldmatrix path.
        assert out.count("ld.shared.b32") == 0

    def test_ldmatrix_hint_x2_for_b_frag(self):
        b = _builder_with(_M16N8K16_BF16)
        B = b.smem_alloc("B", DType.BF16, (8, 16))
        b.load_matrix(
            B,
            "m16n8k16_bf16",
            which="b",
            layout_hint="ldmatrix",
        )
        out = _lower(b)
        assert "ldmatrix.sync.aligned.x2.m8n8.shared.b16" in out


# ---------------------------------------------------------------------------
# LoadMatrix: preshuffled (custom offsets) path
# ---------------------------------------------------------------------------


class TestLoadMatrixPreshuffled:
    def test_custom_offsets_work_unchanged(self):
        """A preshuffled smem layout passes arbitrary (dr, dc) offsets;
        the lowerer uses them verbatim. This is the whole reason we
        default to the manual path."""
        b = _builder_with(_M16N8K16_BF16)
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        # Hypothetical shuffled layout: all four regs at row 0, spread
        # through col by 4-element strides. Nothing to do with the real
        # ISA formulas — just proves the offsets flow through.
        shuffled = ((0, 0), (0, 4), (0, 8), (0, 12))
        b.load_matrix(
            A,
            "m16n8k16_bf16",
            which="a",
            reg_offsets=shuffled,
        )
        out = _lower(b)
        # Byte offsets for bf16 (col_stride=2): 0, 8, 16, 24
        for off in (0, 8, 16, 24):
            if off == 0:
                assert re.search(r"ld\.shared\.b32 %b\d+, \[%r\d+\];", out)
            else:
                assert re.search(rf"ld\.shared\.b32 %b\d+, \[%r\d+ \+ {off}\];", out), (
                    f"missing load at +{off}"
                )


# ---------------------------------------------------------------------------
# LoadMatrix: dyn per-lane base via SharedRegion.view
# ---------------------------------------------------------------------------


class TestLoadMatrixPerLaneBase:
    def test_dyn_offset_on_tensor_view_threads_through(self):
        """The per-lane base (groupID*stride + tidIG*elems_per_lane) is
        supplied by the caller as a SharedRegion.view(dyn_offset=...).
        The lowerer must fold it into the address arithmetic so the
        resulting loads are per-lane."""
        b = _builder_with(_M16N8K16_BF16)
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        # Stand-in per-lane base: some arithmetic Value.
        lane = b.lane_id()
        group = b.shr(lane, b.const(DType.U32, 2))  # laneid >> 2
        tidIG = b.and_(lane, b.const(DType.U32, 3))  # laneid & 3
        stride_elems = b.const(DType.U32, 16)
        tidIG_step = b.const(DType.U32, 2)  # 2 elems per lane for bf16
        group_off = b.mul(group, stride_elems)
        tidIG_off = b.mul(tidIG, tidIG_step)
        per_lane_elem_off = b.add(group_off, tidIG_off)
        A_lane = A.view(dyn_offset=per_lane_elem_off)
        b.load_matrix(
            A_lane,
            "m16n8k16_bf16",
            which="a",
            row=0,
            col=0,
            reg_offsets=A_BF16_K16_ROW,
        )
        out = _lower(b)
        # The address math should include a mul.lo.u32 with factor 2
        # (bf16 elem_bytes) turning the element offset into bytes.
        assert re.search(r"mul\.lo\.u32 %r\d+, %r\d+, 2;", out)
        # And an add.u32 combining that with the base.
        assert re.search(r"add\.u32 %r\d+, %r\d+, %r\d+;", out)
        # Four ld.shared.b32 still emitted (one per reg).
        assert out.count("ld.shared.b32") == 4


# ---------------------------------------------------------------------------
# Mma
# ---------------------------------------------------------------------------


class TestAccumulatorCarrierDtype:
    """C / D fragment carrier dtype must match MmaShape.acc_dtype so
    the PTX mma's typed accumulator slot gets the right register class.
    A / B fragments stay b32 regardless (always packed subword)."""

    def test_f32_acc_produces_f32_cd_regs(self):
        b = _builder_with(_M16N8K16_BF16)  # acc_dtype=F32
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        C = b.smem_alloc("C", DType.F32, (16, 8))
        a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=A_BF16_K16_ROW)
        c = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=CD_F32_ROW)
        assert a.dtype is DType.B32
        assert c.dtype is DType.F32
        out = _lower(b)
        # A loads use .b32 regs; C loads use .f32 regs.
        assert re.search(r"ld\.shared\.b32 %b\d+", out)
        assert re.search(r"ld\.shared\.f32 %f\d+", out)

    def test_s32_acc_produces_s32_cd_regs(self):
        s32_shape = MmaShape(
            name="m16n8k16_s8_s32",
            m=16,
            n=8,
            k=16,
            a_dtype=DType.S8,
            b_dtype=DType.S8,
            acc_dtype=DType.S32,
            a_regs=2,
            b_regs=1,
            c_regs=4,
            ptx="m16n8k16.row.col.s32.s8.s8.s32",
        )
        b = _builder_with(s32_shape)
        b.smem_alloc("A", DType.S8, (32, 16))
        C = b.smem_alloc("C", DType.S32, (16, 8))
        c = b.load_matrix(C, "m16n8k16_s8_s32", which="c", reg_offsets=CD_F32_ROW)
        assert c.dtype is DType.S32
        out = _lower(b)
        assert re.search(r"ld\.shared\.s32 %rs\d+", out)

    def test_f16_acc_falls_back_to_b32(self):
        f16_shape = MmaShape(
            name="m16n8k16_f16_f16",
            m=16,
            n=8,
            k=16,
            a_dtype=DType.F16,
            b_dtype=DType.F16,
            acc_dtype=DType.F16,
            a_regs=4,
            b_regs=2,
            c_regs=2,  # f16 accumulator: 2 f16x2 regs
            ptx="m16n8k16.row.col.f16.f16.f16.f16",
        )
        b = _builder_with(f16_shape)
        C = b.smem_alloc("C", DType.F16, (16, 8))
        c = b.load_matrix(C, "m16n8k16_f16_f16", which="c", reg_offsets=((0, 0), (8, 0)))
        # F16 acc uses packed b32 regs (f16x2 carriers).
        assert c.dtype is DType.B32


class TestMma:
    def test_mma_emits_four_braced_lists_in_order(self):
        b = _builder_with(_M16N8K16_BF16)
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        B = b.smem_alloc("B", DType.BF16, (8, 16))
        a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=A_BF16_K16_ROW)
        bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=B_BF16_K16_COL_T)
        c = b.load_matrix(A, "m16n8k16_bf16", which="c", reg_offsets=CD_F32_ROW)
        b.mma("m16n8k16_bf16", a, bf, c)
        out = _lower(b)
        # D / C slots are f32 regs because acc_dtype=F32; A / B stay b32.
        assert re.search(
            r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.bf16\.bf16\.f32 "
            r"\{%f\d+, %f\d+, %f\d+, %f\d+\}, "  # d (4, f32)
            r"\{%b\d+, %b\d+, %b\d+, %b\d+\}, "  # a (4, b32 bf16x2)
            r"\{%b\d+, %b\d+\}, "  # b (2, b32 bf16x2)
            r"\{%f\d+, %f\d+, %f\d+, %f\d+\};",  # c (4, f32)
            out,
        )

    def test_mma_e4m3_k32(self):
        b = _builder_with(_M16N8K32_E4M3)
        A = b.smem_alloc("A", DType.E4M3, (32, 32))
        B = b.smem_alloc("B", DType.E4M3, (8, 32))
        a = b.load_matrix(A, "m16n8k32_e4m3", which="a", reg_offsets=A_E4M3_K32_ROW)
        bf = b.load_matrix(B, "m16n8k32_e4m3", which="b", reg_offsets=B_E4M3_K32_COL_T)
        c = b.load_matrix(A, "m16n8k32_e4m3", which="c", reg_offsets=CD_F32_ROW)
        b.mma("m16n8k32_e4m3", a, bf, c)
        out = _lower(b)
        assert "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32" in out

    def test_mma_bf16_k8(self):
        """m16n8k8 bf16 — the smaller-K bf16 MMA added in MMA_SHAPES M2.
        Same A-reg layout as e4m3 k=16 (2 regs) but with bf16's 2-elem-
        per-b32 packing."""
        from popcorn.ir.mma_registry import _BF16_M16N8K8

        b = _builder_with(_BF16_M16N8K8.shape)
        A = b.smem_alloc("A", DType.BF16, (16, 8))
        B = b.smem_alloc("B", DType.BF16, (8, 8))
        a = b.load_matrix(A, "m16n8k8_bf16", which="a", reg_offsets=_BF16_M16N8K8.a_offsets)
        bf = b.load_matrix(B, "m16n8k8_bf16", which="b", reg_offsets=_BF16_M16N8K8.b_offsets)
        c = b.load_matrix(A, "m16n8k8_bf16", which="c", reg_offsets=_BF16_M16N8K8.cd_offsets)
        b.mma("m16n8k8_bf16", a, bf, c)
        out = _lower(b)
        assert "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32" in out

    def test_mma_without_ptx_suffix_raises_at_lower(self):
        shapeless = MmaShape(
            name="shapeless",
            m=16,
            n=8,
            k=16,
            a_dtype=DType.BF16,
            b_dtype=DType.BF16,
            acc_dtype=DType.F32,
            a_regs=4,
            b_regs=2,
            c_regs=4,
        )
        b = _builder_with(shapeless)
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        a = b.load_matrix(A, "shapeless", which="a", reg_offsets=A_BF16_K16_ROW)
        bf = b.load_matrix(A, "shapeless", which="b", reg_offsets=B_BF16_K16_COL_T)
        c = b.load_matrix(A, "shapeless", which="c", reg_offsets=CD_F32_ROW)
        b.mma("shapeless", a, bf, c)
        with pytest.raises(NotImplementedError, match="no `ptx`"):
            _lower(b)


# ---------------------------------------------------------------------------
# StoreMatrix
# ---------------------------------------------------------------------------


class TestStoreMatrix:
    def test_f32_acc_store_uses_cd_row_offsets(self):
        """Storing the D accumulator to an f32 row-major smem tile uses
        the CD row offsets: (0,0), (0,1), (8,0), (8,1)."""
        b = _builder_with(_M16N8K16_BF16)
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        B = b.smem_alloc("B", DType.BF16, (8, 16))
        D_smem = b.smem_alloc("D", DType.F32, (16, 8))  # row_stride_bytes = 32
        a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=A_BF16_K16_ROW)
        bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=B_BF16_K16_COL_T)
        c = b.load_matrix(A, "m16n8k16_bf16", which="c", reg_offsets=CD_F32_ROW)
        d = b.mma("m16n8k16_bf16", a, bf, c)
        b.store_matrix(D_smem, d, "m16n8k16_bf16", which="d", reg_offsets=CD_F32_ROW)
        out = _lower(b)
        # row_stride_bytes = 8 * 4 = 32, col_stride_bytes = 4
        # reg0 (0,0): 0
        # reg1 (0,1): 4
        # reg2 (8,0): 256
        # reg3 (8,1): 260
        # D fragment is f32 (acc_dtype), so `st.shared.f32` with f32 regs.
        for off in (0, 4, 256, 260):
            if off == 0:
                assert re.search(r"st\.shared\.f32 \[%r\d+\], %f\d+;", out)
            else:
                assert re.search(rf"st\.shared\.f32 \[%r\d+ \+ {off}\], %f\d+;", out), (
                    f"missing store at +{off}"
                )
        assert out.count("st.shared.f32") == 4

    def test_store_to_gmem_also_works(self):
        """D epilogues often store directly to gmem with its own lane
        mapping. The default lowerer handles GlobalTensor destinations
        the same way it handles SharedRegion."""
        b = _builder_with(_M16N8K16_BF16)
        b.param("Y", BufferType(DType.F32))
        Y = GlobalTensor(
            dtype=DType.F32,
            shape=(128, 128),
            stride=(128, 1),
            name="Y",
            param=b.function.params[-1],
        )
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        B = b.smem_alloc("B", DType.BF16, (8, 16))
        a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=A_BF16_K16_ROW)
        bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=B_BF16_K16_COL_T)
        c = b.load_matrix(A, "m16n8k16_bf16", which="c", reg_offsets=CD_F32_ROW)
        d = b.mma("m16n8k16_bf16", a, bf, c)
        b.store_matrix(Y, d, "m16n8k16_bf16", which="d", reg_offsets=CD_F32_ROW)
        out = _lower(b)
        # row_stride_bytes = 128 * 4 = 512
        # 4 global stores at byte offsets:
        #   reg0 (0, 0): 0
        #   reg1 (0, 1): 4
        #   reg2 (8, 0): 8 * 512 = 4096
        #   reg3 (8, 1): 4096 + 4 = 4100
        # D fragment is f32 → `st.global.f32` with f32 regs.
        assert out.count("st.global.f32") == 4
        for off in (0, 4, 4096, 4100):
            if off == 0:
                assert re.search(r"st\.global\.f32 \[%rd\d+\], %f\d+;", out)
            else:
                assert re.search(rf"st\.global\.f32 \[%rd\d+ \+ {off}\], %f\d+;", out)

    def test_missing_reg_offsets_raises(self):
        b = _builder_with(_M16N8K16_BF16)
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        D_smem = b.smem_alloc("D", DType.F32, (16, 8))
        a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=A_BF16_K16_ROW)
        bf = b.load_matrix(A, "m16n8k16_bf16", which="b", reg_offsets=B_BF16_K16_COL_T)
        c = b.load_matrix(A, "m16n8k16_bf16", which="c", reg_offsets=CD_F32_ROW)
        d = b.mma("m16n8k16_bf16", a, bf, c)
        b.store_matrix(D_smem, d, "m16n8k16_bf16", which="d")  # no reg_offsets
        with pytest.raises(ValueError, match="reg_offsets"):
            _lower(b)


# ---------------------------------------------------------------------------
# Full gemm-tile smoke (bf16)
# ---------------------------------------------------------------------------


class TestFullGemmTile:
    def test_full_tile_builds_lowers_validates(self):
        b = _builder_with(_M16N8K16_BF16)
        A = b.smem_alloc("A", DType.BF16, (32, 16))
        B = b.smem_alloc("B", DType.BF16, (8, 16))
        # C / D accumulator lives in its own f32 smem tile — separate
        # from the bf16 multiplicand tiles.
        C_smem = b.smem_alloc("C", DType.F32, (16, 8))
        a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=A_BF16_K16_ROW)
        bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=B_BF16_K16_COL_T)
        c = b.load_matrix(C_smem, "m16n8k16_bf16", which="c", reg_offsets=CD_F32_ROW)
        d = b.mma("m16n8k16_bf16", a, bf, c)
        b.store_matrix(C_smem, d, "m16n8k16_bf16", which="d", reg_offsets=CD_F32_ROW)
        validate_module(b.module)
        out = _lower(b)
        # 4 A loads (b32) + 2 B loads (b32) = 6 ld.shared.b32
        assert out.count("ld.shared.b32") == 6
        # 4 C loads (f32) = 4 ld.shared.f32
        assert out.count("ld.shared.f32") == 4
        assert out.count("mma.sync.aligned") == 1
        # 4 D stores to f32 smem
        assert out.count("st.shared.f32") == 4
