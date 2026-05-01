"""Unit tests for the RegisterTile / FragLayout machinery in
``quark.ir.frag_tile``. Verifies:
  * Forward lane→position + inverse position→lane_slot formulas agree.
  * Full coverage: iterating all 32 lanes × 2 elements visits every
    (row, col) in the 8×8 tile exactly once.
  * PTX C/D + A-frag per-lane formulas round-trip against the existing
    offset tables in ``quark.kernels.gemm.mma_shapes``.

These tests don't require a device — they're pure closed-form / table
checks. Runtime layout probes (which DO need a device) live in
``tests/lower/msl/test_frag_layout.py``.
"""

from __future__ import annotations

from quark.device import DeviceFamily
from quark.ir.frag_tile import (
    FragLayout,
    RegisterTile,
    apple_lane_to_tile_position,
    apple_position_to_lane_slot,
    ptx_lane_to_a_frag_positions,
    ptx_lane_to_acc_position,
)
from quark.ir.mma_registry import register_backend_payload
from quark.kernels.gemm.mma_shapes import _BF16_K16

# ---------------------------------------------------------------------------
# Apple simdgroup_matrix<T, 8, 8> lane map
# ---------------------------------------------------------------------------


def test_apple_lane_to_position_round_trip():
    """For every Apple lane L, the two positions its thread_elements()
    cover invert back to (L, 0) and (L, 1) via the position→lane_slot
    formula."""
    for lane in range(32):
        r0, c0, r1, c1 = apple_lane_to_tile_position(lane)
        back_l0, back_e0 = apple_position_to_lane_slot(r0, c0)
        back_l1, back_e1 = apple_position_to_lane_slot(r1, c1)
        assert (back_l0, back_e0) == (lane, 0), (
            f"lane {lane}: forward ({r0}, {c0}) then inverse gave "
            f"({back_l0}, {back_e0}), expected ({lane}, 0)"
        )
        assert (back_l1, back_e1) == (lane, 1), (
            f"lane {lane}: forward ({r1}, {c1}) then inverse gave "
            f"({back_l1}, {back_e1}), expected ({lane}, 1)"
        )


def test_apple_full_tile_coverage():
    """All 32 lanes × 2 elements cover every (row, col) in 8×8
    exactly once. If the formulas are slightly wrong (e.g., off-by-one
    in the bit-shift), this catches double-booking or gaps."""
    covered: set[tuple[int, int]] = set()
    for lane in range(32):
        r0, c0, r1, c1 = apple_lane_to_tile_position(lane)
        assert 0 <= r0 < 8 and 0 <= c0 < 8
        assert 0 <= r1 < 8 and 0 <= c1 < 8
        covered.add((r0, c0))
        covered.add((r1, c1))
    assert len(covered) == 64, (
        f"Apple tile coverage has gaps/overlaps: {len(covered)} unique positions, expected 64"
    )


def test_apple_inverse_full_coverage():
    """Every (row, col) in 8×8 maps to some (lane, elem) such that
    the forward formula reproduces (row, col)."""
    for row in range(8):
        for col in range(8):
            lane, elem = apple_position_to_lane_slot(row, col)
            r0, c0, r1, c1 = apple_lane_to_tile_position(lane)
            if elem == 0:
                assert (r0, c0) == (row, col)
            else:
                assert (r1, c1) == (row, col)


# ---------------------------------------------------------------------------
# PTX C/D accumulator lane map
# ---------------------------------------------------------------------------


def test_ptx_acc_full_coverage_bf16_m16n8():
    """All 32 lanes × 4 c_regs cover every (row, col) in 16×8
    exactly once for m16n8k16 bf16's cd_offsets."""
    cd_offsets = _BF16_K16.cd_offsets
    covered: set[tuple[int, int]] = set()
    for lane in range(32):
        for reg_idx in range(len(cd_offsets)):
            row, col = ptx_lane_to_acc_position(lane, reg_idx, cd_offsets)
            assert 0 <= row < 16 and 0 <= col < 8
            covered.add((row, col))
    assert len(covered) == 16 * 8, f"PTX C/D coverage: {len(covered)} unique, expected 128"


# ---------------------------------------------------------------------------
# PTX A-fragment lane map
# ---------------------------------------------------------------------------


def test_ptx_a_frag_full_coverage_bf16_m16n8k16():
    """All 32 lanes × 4 a_regs × 2 elements/reg cover every
    (row, col) in 16×16 exactly once."""
    a_offsets = _BF16_K16.a_offsets
    covered: set[tuple[int, int]] = set()
    for lane in range(32):
        for reg_idx in range(len(a_offsets)):
            for row, col in ptx_lane_to_a_frag_positions(
                lane, reg_idx, a_offsets, elements_per_reg=2
            ):
                assert 0 <= row < 16 and 0 <= col < 16
                covered.add((row, col))
    assert len(covered) == 16 * 16, f"PTX A-frag coverage: {len(covered)} unique, expected 256"


# ---------------------------------------------------------------------------
# RegisterTile dataclass basics
# ---------------------------------------------------------------------------


class _FakeValue:
    """Lightweight Value stand-in for tests that don't need a real
    Builder/Module."""

    width = 4

    class _Dt:
        name = "F32"

    dtype = _Dt


def test_register_tile_single_tile_accessors():
    """Single-tile RegisterTile (tile_grid=(1, 1)) exposes
    rows/cols/dtype and the ``.value`` convenience accessor."""
    v = _FakeValue()
    tile = RegisterTile(
        values=(v,),  # type: ignore[arg-type]
        layout=FragLayout.ACCUMULATOR,
        shape_id="m16n8k16_bf16",
        tile_grid=(1, 1),
        logical_shape=(16, 8),
    )
    assert tile.rows == 16
    assert tile.cols == 8
    assert tile.layout is FragLayout.ACCUMULATOR
    assert tile.shape_id == "m16n8k16_bf16"
    assert tile.value is v


def test_register_tile_multi_tile_grid():
    """Multi-tile (2×2 grid of m16n8 ACCUMULATORs — logical_shape
    = (32, 16)). ``.tile_at()`` indexes into the tile grid; ``.value``
    raises (ambiguous which tile)."""
    import pytest

    vs = tuple(_FakeValue() for _ in range(4))
    tile = RegisterTile(
        values=vs,  # type: ignore[arg-type]
        layout=FragLayout.ACCUMULATOR,
        shape_id="m16n8k16_bf16",
        tile_grid=(2, 2),
        logical_shape=(32, 16),
    )
    assert tile.tile_at(0, 0) is vs[0]
    assert tile.tile_at(0, 1) is vs[1]
    assert tile.tile_at(1, 0) is vs[2]
    assert tile.tile_at(1, 1) is vs[3]
    with pytest.raises(ValueError, match="multi-tile"):
        _ = tile.value
    with pytest.raises(IndexError):
        tile.tile_at(2, 0)


def test_register_tile_grid_shape_mismatch_raises():
    """tile_grid (M, N) must match len(values) == M*N — catch
    inconsistent construction at build time."""
    import pytest

    with pytest.raises(ValueError, match="implies"):
        RegisterTile(
            values=(_FakeValue(),),  # type: ignore[arg-type]
            layout=FragLayout.ACCUMULATOR,
            shape_id="m16n8k16_bf16",
            tile_grid=(2, 2),  # would need 4 values, we gave 1
            logical_shape=(32, 16),
        )


def test_frag_layout_enum_distinct_values():
    """Guard against accidental merging of layout roles — three
    layouts, three distinct enum values."""
    values = {FragLayout.ACCUMULATOR.value, FragLayout.A_FRAG.value, FragLayout.B_FRAG.value}
    assert len(values) == 3


# ---------------------------------------------------------------------------
# RegisterTile.map — fluent API hooking into Builder.frag_apply
# ---------------------------------------------------------------------------

from quark.ir import Builder, DType, FragApplyOp, MmaShape  # noqa: E402

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
)
register_backend_payload("m16n8k16_bf16", DeviceFamily.CUDA, "m16n8k16.row.col.f32.bf16.bf16.f32")
register_backend_payload("m16n8k16_bf16", DeviceFamily.METAL, "bfloat16_t:2:1:2")

_A_OFFS = ((0, 0), (8, 0), (0, 8), (8, 8))
_B_OFFS = ((0, 0), (0, 8))
_CD_OFFS = ((0, 0), (0, 1), (8, 0), (8, 1))


def _live_acc_tile(b: Builder) -> tuple[RegisterTile, object]:
    """Return a RegisterTile wrapping a live accumulator Value."""
    b.register_shape(_M16N8K16_BF16)
    b.begin_function("f")
    A = b.smem_alloc("A", DType.BF16, (16, 16))
    B = b.smem_alloc("B", DType.BF16, (8, 16))
    C = b.smem_alloc("C", DType.F32, (16, 8))
    a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=_A_OFFS)
    bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
    cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
    d = b.mma("m16n8k16_bf16", a, bf, cf)
    tile = RegisterTile(
        values=(d,),
        layout=FragLayout.ACCUMULATOR,
        shape_id="m16n8k16_bf16",
        tile_grid=(1, 1),
        logical_shape=(16, 8),
        _builder=b,
    )
    return tile, d


def test_register_tile_map_emits_frag_apply_op():
    """``tile.map(fn)`` on an accumulator tile emits a FragApplyOp
    and returns a new tile wrapping the op's result."""
    b = Builder("test")
    tile, _d = _live_acc_tile(b)
    scale = b.const(DType.F32, 2.0)
    new_tile = tile.map(lambda x: b.mul(x, scale))
    assert isinstance(new_tile, RegisterTile)
    assert new_tile.layout is FragLayout.ACCUMULATOR
    assert new_tile.shape_id == "m16n8k16_bf16"
    assert new_tile.tile_grid == (1, 1)
    assert new_tile.logical_shape == (16, 8)
    assert new_tile.values[0] is not tile.values[0]
    # The emitted op is a FragApplyOp in the current region.
    apply_ops = [op for op in b.current_region.ops if isinstance(op, FragApplyOp)]
    assert len(apply_ops) == 1
    assert apply_ops[0].results[0] is new_tile.values[0]


def test_register_tile_map_requires_builder():
    """``.map`` without a builder raises a clear error — can't emit
    IR without one."""
    import pytest

    v = _FakeValue()
    tile = RegisterTile(
        values=(v,),  # type: ignore[arg-type]
        layout=FragLayout.ACCUMULATOR,
        shape_id="m16n8k16_bf16",
        tile_grid=(1, 1),
        logical_shape=(16, 8),
    )
    with pytest.raises(RuntimeError, match="no Builder reference"):
        tile.map(lambda x: x)


def test_register_tile_map_rejects_non_accumulator():
    """``.map`` on A_FRAG / B_FRAG tiles raises — packed layouts need
    FragConvertOp (Step 3 in the migration plan)."""
    import pytest

    v = _FakeValue()
    tile = RegisterTile(
        values=(v,),  # type: ignore[arg-type]
        layout=FragLayout.A_FRAG,
        shape_id="m16n8k16_bf16",
        tile_grid=(1, 1),
        logical_shape=(16, 16),
    )
    with pytest.raises(NotImplementedError, match="ACCUMULATOR"):
        tile.map(lambda x: x, builder=object())  # builder arg used only past guard


def test_register_tile_with_builder_returns_new_instance():
    """``.with_builder(b)`` returns a new tile with the builder
    attached — doesn't mutate the frozen original."""
    v = _FakeValue()
    tile = RegisterTile(
        values=(v,),  # type: ignore[arg-type]
        layout=FragLayout.ACCUMULATOR,
        shape_id="m16n8k16_bf16",
        tile_grid=(1, 1),
        logical_shape=(16, 8),
    )
    assert tile._builder is None
    bound = tile.with_builder(Builder("t"))
    assert bound is not tile
    assert tile._builder is None
    assert bound._builder is not None


def test_register_tile_map_chains():
    """Chained ``.map(...).map(...)`` emits two FragApplyOps."""
    b = Builder("test")
    tile, _d = _live_acc_tile(b)
    s1 = b.const(DType.F32, 2.0)
    s2 = b.const(DType.F32, 3.0)
    final = tile.map(lambda x: b.mul(x, s1)).map(lambda x: b.mul(x, s2))
    assert final.layout is FragLayout.ACCUMULATOR
    apply_ops = [op for op in b.current_region.ops if isinstance(op, FragApplyOp)]
    assert len(apply_ops) == 2


def _live_multi_n_tile(b: Builder, nt: int) -> RegisterTile:
    """Multi-n-tile accumulator: one m-tile row with `nt` n-tiles.

    Each n-tile is a separate live MMA accumulator Value. Real kernels
    build these via ``MmaBody`` across an (MT, NT) grid; here we stack
    ``nt`` independent mma ops to get the same Value shape without
    pulling in the full block machinery.
    """
    b.register_shape(_M16N8K16_BF16)
    b.begin_function("f")
    vals = []
    for i in range(nt):
        A = b.smem_alloc(f"A_{i}", DType.BF16, (16, 16))
        B = b.smem_alloc(f"B_{i}", DType.BF16, (8, 16))
        C = b.smem_alloc(f"C_{i}", DType.F32, (16, 8))
        a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=_A_OFFS)
        bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
        cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
        vals.append(b.mma("m16n8k16_bf16", a, bf, cf))
    return RegisterTile(
        values=tuple(vals),
        layout=FragLayout.ACCUMULATOR,
        shape_id="m16n8k16_bf16",
        tile_grid=(1, nt),
        logical_shape=(16, 8 * nt),
        _builder=b,
    )


def test_reduce_along_cols_nt_greater_than_one_folds_first():
    """For ``tile_grid=(1, 3)`` (three n-tiles sharing the row), reduce
    should element-wise fold the three tiles into one before emitting
    the frag_reduce — not raise NotImplementedError as it used to."""
    from quark.ir.op import FragReduceOp, VecBuildOp

    b = Builder("test")
    tile = _live_multi_n_tile(b, nt=3)
    out = tile.reduce_along_cols("max", _CD_OFFS)

    # 2 row classes in m16n8 acc × 1 m-tile = 2 scalars returned.
    assert len(out) == 2
    # Exactly one FragReduceOp (we reduce the folded tile, not each).
    reduce_ops = [op for op in b.current_region.ops if isinstance(op, FragReduceOp)]
    assert len(reduce_ops) == 1, f"expected 1 FragReduceOp, got {len(reduce_ops)}"
    # Fold emitted `nt - 1 = 2` vec_builds (one per pairwise fold).
    vec_builds = [op for op in b.current_region.ops if isinstance(op, VecBuildOp)]
    assert len(vec_builds) >= 2


def test_reduce_along_cols_nt_one_unchanged():
    """Single-n-tile path is the common case — no fold, one FragReduceOp
    per m-tile, matching the pre-fix behavior."""
    from quark.ir.op import FragReduceOp

    b = Builder("test")
    tile, _ = _live_acc_tile(b)
    out = tile.reduce_along_cols("max", _CD_OFFS)

    assert len(out) == 2  # 2 row classes
    reduce_ops = [op for op in b.current_region.ops if isinstance(op, FragReduceOp)]
    assert len(reduce_ops) == 1


def test_reduce_along_cols_mt_and_nt_both_greater_than_one():
    """``tile_grid=(2, 2)`` produces one fold + one reduce per m-tile
    row; returned flat tuple is (mt0_rc0, mt0_rc1, mt1_rc0, mt1_rc1)."""
    from quark.ir.op import FragReduceOp

    b = Builder("test")
    b.register_shape(_M16N8K16_BF16)
    b.begin_function("f")

    def _mma(i: int):
        A = b.smem_alloc(f"A_{i}", DType.BF16, (16, 16))
        B = b.smem_alloc(f"B_{i}", DType.BF16, (8, 16))
        C = b.smem_alloc(f"C_{i}", DType.F32, (16, 8))
        a = b.load_matrix(A, "m16n8k16_bf16", which="a", reg_offsets=_A_OFFS)
        bf = b.load_matrix(B, "m16n8k16_bf16", which="b", reg_offsets=_B_OFFS)
        cf = b.load_matrix(C, "m16n8k16_bf16", which="c", reg_offsets=_CD_OFFS)
        return b.mma("m16n8k16_bf16", a, bf, cf)

    tile = RegisterTile(
        values=(_mma(0), _mma(1), _mma(2), _mma(3)),
        layout=FragLayout.ACCUMULATOR,
        shape_id="m16n8k16_bf16",
        tile_grid=(2, 2),
        logical_shape=(32, 16),
        _builder=b,
    )
    out = tile.reduce_along_cols("max", _CD_OFFS)

    assert len(out) == 4  # 2 m-tiles × 2 row classes
    reduce_ops = [op for op in b.current_region.ops if isinstance(op, FragReduceOp)]
    assert len(reduce_ops) == 2  # one per m-tile row
