"""Regression tests for Apple simdgroup_matrix per-lane layouts.

These are the ground-truth mappings the ``RegisterTile`` abstraction
consults when lowering fragment operations on Metal. They were
discovered empirically by loading ``arange(0, 64)`` into a
``simdgroup_matrix<float, 8, 8>`` via the cooperative ``simdgroup_load``
path, then dumping each lane's ``thread_elements()``. Any change in
Apple's compiler / driver / hardware that alters the mapping will
break these tests — cheap to catch here, expensive (and hard to
diagnose) to catch via owl_attn cos-sim regressions.

If this file ever starts failing, DO NOT adjust the expected values
— run the probe manually, confirm the new Apple mapping makes sense,
update both the test and ``popcorn.ir.frag_tile._MSL_ACC_LANE_MAP``
together.
"""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    HAS_METAL = mx.metal.is_available()
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="no Metal device")


def _probe_layout(acc_dtype_metal: str, acc_dtype_mx) -> list[tuple[int, int, int, int]]:
    """Load arange(0, 64) into a simdgroup_matrix of the given dtype
    and return the per-lane (row0, col0, row1, col1) the elements land at.

    ``acc_dtype_metal`` is the MSL type name (e.g. "float", "bfloat16_t").
    ``acc_dtype_mx`` is the matching mlx dtype.
    """

    source = f"""
        threadgroup {acc_dtype_metal} smem[64];
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        for (uint i = tid; i < 64; i += 32u) smem[i] = ({acc_dtype_metal})(i);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        metal::simdgroup_matrix<{acc_dtype_metal}, 8, 8> M;
        metal::simdgroup_load(M, smem, 8);
        auto e = M.thread_elements();
        Out[lane * 2 + 0] = float(e[0]);
        Out[lane * 2 + 1] = float(e[1]);
    """
    kernel = mx.fast.metal_kernel(
        name=f"probe_layout_{acc_dtype_metal.replace('_t', '')}",
        input_names=[],
        output_names=["Out"],
        source=source,
        header=(
            "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;"
        ),
        ensure_row_contiguous=True,
    )
    outs = kernel(  # ty: ignore[call-non-callable]
        inputs=[],
        template=[],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(64,)],
        output_dtypes=[mx.float32],
    )
    mx.eval(outs[0])
    got = [int(v) for v in __import__("numpy").array(outs[0]).astype(int)]
    out = []
    for lane in range(32):
        v0, v1 = got[lane * 2], got[lane * 2 + 1]
        out.append((v0 // 8, v0 % 8, v1 // 8, v1 % 8))
    return out


# ---------------------------------------------------------------------------
# The canonical 32-entry lane map for simdgroup_matrix<T, 8, 8> on Apple
# silicon (verified on M3 Ultra; re-verify on each new chip). Format:
# lane L → (row0, col0, row1, col1) — thread_elements()[0] at (row0, col0),
# thread_elements()[1] at (row1, col1).
# ---------------------------------------------------------------------------

EXPECTED_APPLE_8x8_LAYOUT: tuple[tuple[int, int, int, int], ...] = (
    # lane 0..3 — row block 0 (rows 0..1), col block 0 (cols 0..3)
    (0, 0, 0, 1),  # 0
    (0, 2, 0, 3),  # 1
    (1, 0, 1, 1),  # 2
    (1, 2, 1, 3),  # 3
    # lane 4..7 — row block 1 (rows 2..3), col block 0 (cols 0..3)
    (2, 0, 2, 1),  # 4
    (2, 2, 2, 3),  # 5
    (3, 0, 3, 1),  # 6
    (3, 2, 3, 3),  # 7
    # lane 8..11 — row block 0 (rows 0..1), col block 1 (cols 4..7)
    (0, 4, 0, 5),  # 8
    (0, 6, 0, 7),  # 9
    (1, 4, 1, 5),  # 10
    (1, 6, 1, 7),  # 11
    # lane 12..15 — row block 1 (rows 2..3), col block 1 (cols 4..7)
    (2, 4, 2, 5),  # 12
    (2, 6, 2, 7),  # 13
    (3, 4, 3, 5),  # 14
    (3, 6, 3, 7),  # 15
    # lane 16..19 — row block 2 (rows 4..5), col block 0 (cols 0..3)
    (4, 0, 4, 1),  # 16
    (4, 2, 4, 3),  # 17
    (5, 0, 5, 1),  # 18
    (5, 2, 5, 3),  # 19
    # lane 20..23 — row block 3 (rows 6..7), col block 0 (cols 0..3)
    (6, 0, 6, 1),  # 20
    (6, 2, 6, 3),  # 21
    (7, 0, 7, 1),  # 22
    (7, 2, 7, 3),  # 23
    # lane 24..27 — row block 2 (rows 4..5), col block 1 (cols 4..7)
    (4, 4, 4, 5),  # 24
    (4, 6, 4, 7),  # 25
    (5, 4, 5, 5),  # 26
    (5, 6, 5, 7),  # 27
    # lane 28..31 — row block 3 (rows 6..7), col block 1 (cols 4..7)
    (6, 4, 6, 5),  # 28
    (6, 6, 6, 7),  # 29
    (7, 4, 7, 5),  # 30
    (7, 6, 7, 7),  # 31
)


def test_apple_8x8_acc_layout_f32():
    """Load arange(0, 64) into simdgroup_matrix<float, 8, 8> and verify
    the per-lane layout matches EXPECTED_APPLE_8x8_LAYOUT. Running this
    on a fresh Apple chip / new MLX release is the cheapest way to
    catch a silent layout break."""
    got = _probe_layout("float", mx.float32)
    for lane, (expected, observed) in enumerate(zip(EXPECTED_APPLE_8x8_LAYOUT, got)):
        assert expected == observed, (
            f"lane {lane}: apple layout changed!\n"
            f"  expected thread_elements() at {expected} "
            f"(row0, col0, row1, col1)\n"
            f"  got {observed}\n"
            f"If Apple's mapping has genuinely changed, update both the\n"
            f"EXPECTED_APPLE_8x8_LAYOUT here AND the LANE_MAP table in\n"
            f"popcorn.ir.frag_tile — together."
        )


def test_apple_8x8_acc_layout_bf16():
    """Same probe at bf16 — verifies the layout is dtype-agnostic
    (which it should be; Apple spec says ``simdgroup_matrix<T, 8, 8>``
    uses the same per-lane layout regardless of T)."""
    got = _probe_layout("bfloat16_t", mx.bfloat16)
    for lane, (expected, observed) in enumerate(zip(EXPECTED_APPLE_8x8_LAYOUT, got)):
        assert expected == observed, (
            f"lane {lane}: bf16 layout ≠ f32 layout. Apple mapping is "
            f"dtype-dependent? expected {expected}, got {observed}."
        )


def test_lane_map_formula_matches_probe():
    """The closed-form used in ``popcorn.ir.frag_tile`` must match the
    measured table. Any drift means the formula is wrong, not the
    table — fix the formula."""
    for lane in range(32):
        row0 = ((lane >> 4) & 1) * 4 + ((lane >> 1) & 3)
        col0 = ((lane >> 3) & 1) * 4 + (lane & 1) * 2
        expected = EXPECTED_APPLE_8x8_LAYOUT[lane]
        actual = (row0, col0, row0, col0 + 1)
        assert actual == expected, (
            f"lane {lane}: closed-form formula gives {actual}, probe table expects {expected}"
        )


def test_write_via_thread_elements_round_trip():
    """Round-trip: each Apple lane writes KNOWN values at ITS
    thread_elements() positions via the closed-form, then we
    simdgroup_store the matrix and verify each element landed at the
    (row, col) the closed-form predicts.

    This is the exact pattern the b32-pack optimization and the
    future RegisterTile.convert() rely on: writing elements into a
    simdgroup_matrix via ``thread_elements()`` at Apple's per-lane
    positions. If this test passes, writes via thread_elements() are
    safe to use throughout the lowerer."""
    import numpy as np

    # Each lane writes (L, L, ..., L) — different value per lane.
    # If the layout we expect is correct, the OUTPUT 8x8 matrix has
    # each cell equal to the LANE whose thread_elements cover that cell.
    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        metal::simdgroup_matrix<float, 8, 8> M = metal::simdgroup_matrix<float, 8, 8>(0);
        thread auto& e = M.thread_elements();
        e[0] = float(lane) + 0.25f;  // tag element 0
        e[1] = float(lane) + 0.75f;  // tag element 1 (distinct)
        threadgroup float smem[64];
        metal::simdgroup_store(M, smem, 8);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        for (uint i = tid; i < 64; i += 32u) Out[i] = smem[i];
    """
    kernel = mx.fast.metal_kernel(
        name="probe_write",
        input_names=[],
        output_names=["Out"],
        source=source,
        header=(
            "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;"
        ),
        ensure_row_contiguous=True,
    )
    outs = kernel(  # ty: ignore[call-non-callable]
        inputs=[],
        template=[],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(8, 8)],
        output_dtypes=[mx.float32],
    )
    mx.eval(outs[0])
    m = np.array(outs[0])
    for lane, (r0, c0, r1, c1) in enumerate(EXPECTED_APPLE_8x8_LAYOUT):
        assert abs(m[r0, c0] - (lane + 0.25)) < 1e-5, (
            f"lane {lane}'s e[0] at ({r0}, {c0}) has {m[r0, c0]}, expected {lane + 0.25}"
        )
        assert abs(m[r1, c1] - (lane + 0.75)) < 1e-5, (
            f"lane {lane}'s e[1] at ({r1}, {c1}) has {m[r1, c1]}, expected {lane + 0.75}"
        )
