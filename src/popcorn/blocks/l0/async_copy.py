"""L0: Cooperative cp.async 16-byte tile load."""

from __future__ import annotations

from popcorn.ir import Builder, DType, GlobalTensor, SharedRegion, Value


def emit_cp_async_tile(
    b: Builder,
    *,
    dst_smem: SharedRegion,
    src_gmem: GlobalTensor,
    rows: int,
    cols: int,
    elem_bytes: int,
    gmem_row_base: Value,
    gmem_col_base: Value,
    tid: Value,
    n_threads: int,
) -> None:
    """Cooperatively cp.async a (rows x cols) element tile."""
    row_stride_bytes = cols * elem_bytes
    total_bytes = rows * row_stride_bytes
    total_lines = total_bytes // 16
    assert row_stride_bytes % 16 == 0
    lines_per_row = row_stride_bytes // 16

    n_passes = (total_lines + n_threads - 1) // n_threads

    for p in range(n_passes):
        line_id = tid if p == 0 else b.add(tid, b.const(DType.U32, p * n_threads))
        pred = b.cmp("lt", line_id, b.const(DType.U32, total_lines))

        lines_per_row_c = b.const(DType.U32, lines_per_row)
        row_in_tile = b.div(line_id, lines_per_row_c)
        line_in_row = b.rem(line_id, lines_per_row_c)

        elems_per_line = 16 // elem_bytes
        smem_col = b.mul(line_in_row, b.const(DType.U32, elems_per_line))

        gmem_row = b.add(gmem_row_base, row_in_tile)
        gmem_col = b.add(gmem_col_base, smem_col)

        b.async_copy(
            dst_smem,
            src_gmem,
            dst_idx=(row_in_tile, smem_col),
            src_idx=(gmem_row, gmem_col),
            count=16,
            pred=pred,
        )
