"""Regression tests for PTX mma.sync fragment per-lane layouts.

The PTX backend's MMA lowering relies on the ``a_offsets`` /
``b_offsets`` / ``cd_offsets`` tables in ``quark.kernels.gemm.mma_shapes``
— those tables encode the per-register (row_offset, col_offset)
positions each lane holds for a given ``mma.sync`` shape. PTX ISA
§9.7.14.5 specifies these mappings as part of the instruction
contract; changing them would silently corrupt every GEMM/attention
kernel's fragment math.

Unlike the Apple layout (which we verify empirically via a
``simdgroup_load`` probe), the PTX layout is static — it comes from
the ISA spec and cannot change without NVIDIA's architects shipping
a new ISA revision. So these are static asserts against the spec,
not runtime probes. They lock in the table values and the closed-form
formulas the emit helpers use.

If a future PTX SM introduces a new layout, we'd register a NEW
``MmaShape`` entry with its own table — these existing entries stay
frozen to match the PTX ISA revisions they target (sm_80+).
"""

from __future__ import annotations

from quark.kernels.gemm.mma_shapes import (
    _BF16_K16,
    _E4M3_K16,
    _E4M3_K32,
    _F16_K16,
)

# ---------------------------------------------------------------------------
# m16n8k16 bf16 / fp16 — PTX ISA §9.7.14.5 "Matrix Fragments for mma.m16n8k16"
# ---------------------------------------------------------------------------

# Per-lane A fragment (16×16 bf16/fp16). Lane L = (gid=L/4, tig=L%4):
#   a[0] at (row=gid+0, col=tig*2+0..1)   — top-left  8×8
#   a[1] at (row=gid+8, col=tig*2+0..1)   — bot-left  8×8
#   a[2] at (row=gid+0, col=tig*2+8..9)   — top-right 8×8
#   a[3] at (row=gid+8, col=tig*2+8..9)   — bot-right 8×8
EXPECTED_M16N8K16_A_OFFSETS = ((0, 0), (8, 0), (0, 8), (8, 8))

# Per-lane B fragment (16×8 bf16/fp16). Lane L = (gid=L/4, tig=L%4):
#   b[0] at (row=tig*2+0..1, col=gid+0) — top half, col transpose
#   b[1] at (row=tig*2+8..9, col=gid+0) — bot half
# NOTE: the offsets here name the (row, col) of the BASE of each reg
# for col-layout B, which in our per-reg-offset convention shows up as
# ``(0, 0)`` and ``(0, 8)`` — the same convention load_matrix uses.
EXPECTED_M16N8K16_B_OFFSETS = ((0, 0), (0, 8))

# Per-lane C/D fragment (16×8 f32). Lane L = (gid=L/4, tig=L%4):
#   c[0] at (row=gid+0, col=tig*2+0)
#   c[1] at (row=gid+0, col=tig*2+1)
#   c[2] at (row=gid+8, col=tig*2+0)
#   c[3] at (row=gid+8, col=tig*2+1)
EXPECTED_M16N8K16_CD_OFFSETS = ((0, 0), (0, 1), (8, 0), (8, 1))


def test_bf16_m16n8k16_a_offsets():
    """Table matches the PTX ISA §9.7.14.5 per-register A-layout."""
    assert _BF16_K16.a_offsets == EXPECTED_M16N8K16_A_OFFSETS


def test_bf16_m16n8k16_b_offsets():
    assert _BF16_K16.b_offsets == EXPECTED_M16N8K16_B_OFFSETS


def test_bf16_m16n8k16_cd_offsets():
    assert _BF16_K16.cd_offsets == EXPECTED_M16N8K16_CD_OFFSETS


def test_fp16_m16n8k16_matches_bf16_layout():
    """m16n8k16 A/B/C fragment layouts are dtype-independent across
    bf16 and fp16 (same instruction family, different element dtype)."""
    assert _F16_K16.a_offsets == _BF16_K16.a_offsets
    assert _F16_K16.b_offsets == _BF16_K16.b_offsets
    assert _F16_K16.cd_offsets == _BF16_K16.cd_offsets


def test_bf16_m16n8k16_reg_counts():
    """a_regs=4, b_regs=2, c_regs=4 per PTX ISA for m16n8k16 bf16."""
    assert _BF16_K16.shape.a_regs == 4
    assert _BF16_K16.shape.b_regs == 2
    assert _BF16_K16.shape.c_regs == 4


def test_m16n8k16_closed_form_per_lane_positions():
    """The ``cd_offsets`` + lane decomposition (gid=L/4, tig=L%4)
    covers all 16*8 = 128 C/D matrix cells exactly once across 32
    lanes × 4 regs."""
    cd_offsets = _BF16_K16.cd_offsets
    positions = set()
    for lane in range(32):
        gid = lane // 4
        tig = lane % 4
        for dr, dc in cd_offsets:
            row = gid + dr
            col = tig * 2 + dc
            assert 0 <= row < 16 and 0 <= col < 8, (
                f"cd_offsets iteration produced out-of-range (row, col) "
                f"= ({row}, {col}) for lane {lane}, dr={dr}, dc={dc}"
            )
            positions.add((row, col))
    assert len(positions) == 16 * 8, (
        f"Expected all 128 (row, col) cells covered exactly once; "
        f"got {len(positions)} unique positions"
    )


def test_m16n8k16_closed_form_a_positions():
    """a_offsets + lane decomposition covers all 16*16 = 256 A matrix
    cells exactly once (2 cells per reg × 4 regs × 32 lanes = 256)."""
    a_offsets = _BF16_K16.a_offsets
    positions = set()
    for lane in range(32):
        gid = lane // 4
        tig = lane % 4
        for dr, dc in a_offsets:
            for col_add in range(2):  # bf16x2: 2 elements per b32 reg
                row = gid + dr
                col = tig * 2 + dc + col_add
                assert 0 <= row < 16 and 0 <= col < 16, (
                    f"a_offsets iteration produced OOB ({row}, {col}) "
                    f"for lane {lane}, dr={dr}, dc={dc}, col_add={col_add}"
                )
                positions.add((row, col))
    assert len(positions) == 16 * 16, (
        f"Expected all 256 A cells covered exactly once; got {len(positions)} unique positions"
    )


# ---------------------------------------------------------------------------
# m16n8k16 e4m3 fp8 — PTX ISA §9.7.14.5.d "Matrix Fragments for fp8 mma"
# ---------------------------------------------------------------------------

EXPECTED_M16N8K16_FP8_A_OFFSETS = ((0, 0), (8, 0))
EXPECTED_M16N8K16_FP8_B_OFFSETS = ((0, 0),)
EXPECTED_M16N8K16_FP8_CD_OFFSETS = ((0, 0), (0, 1), (8, 0), (8, 1))


def test_e4m3_m16n8k16_layouts():
    assert _E4M3_K16.a_offsets == EXPECTED_M16N8K16_FP8_A_OFFSETS
    assert _E4M3_K16.b_offsets == EXPECTED_M16N8K16_FP8_B_OFFSETS
    assert _E4M3_K16.cd_offsets == EXPECTED_M16N8K16_FP8_CD_OFFSETS


def test_e4m3_m16n8k16_reg_counts():
    """a_regs=2, b_regs=1, c_regs=4 for m16n8k16 e4m3 — half the A/B
    regs of bf16 because each b32 reg holds 4 fp8 elements vs 2 bf16."""
    assert _E4M3_K16.shape.a_regs == 2
    assert _E4M3_K16.shape.b_regs == 1
    assert _E4M3_K16.shape.c_regs == 4


# ---------------------------------------------------------------------------
# m16n8k32 e4m3 — PTX ISA for k=32 fp8 MMA. Doubled K dimension.
# ---------------------------------------------------------------------------

EXPECTED_M16N8K32_FP8_A_OFFSETS = ((0, 0), (8, 0), (0, 16), (8, 16))
EXPECTED_M16N8K32_FP8_B_OFFSETS = ((0, 0), (0, 16))
EXPECTED_M16N8K32_FP8_CD_OFFSETS = ((0, 0), (0, 1), (8, 0), (8, 1))


def test_e4m3_m16n8k32_layouts():
    assert _E4M3_K32.a_offsets == EXPECTED_M16N8K32_FP8_A_OFFSETS
    assert _E4M3_K32.b_offsets == EXPECTED_M16N8K32_FP8_B_OFFSETS
    assert _E4M3_K32.cd_offsets == EXPECTED_M16N8K32_FP8_CD_OFFSETS


def test_e4m3_m16n8k32_cd_same_as_k16():
    """C/D fragment layout only depends on (m, n), not k — so
    m16n8k16 and m16n8k32 share the same cd_offsets."""
    assert _E4M3_K32.cd_offsets == _E4M3_K16.cd_offsets
    assert _BF16_K16.cd_offsets == _E4M3_K16.cd_offsets
