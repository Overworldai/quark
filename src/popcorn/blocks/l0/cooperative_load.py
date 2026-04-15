"""L0: Cooperative scalar gmem → smem tile load."""

from __future__ import annotations

from popcorn.ir import Builder, DType, GlobalTensor, SharedRegion, Value


def emit_cooperative_gmem_to_smem(
    b: Builder,
    *,
    gmem: GlobalTensor,
    smem: SharedRegion,
    rows: int,
    cols: int,
    gmem_row_base: Value,
    gmem_col_base: Value,
    tid: Value,
    n_threads: int,
) -> None:
    """Scalar fallback: each thread loads rows*cols/n_threads elements."""
    total = rows * cols
    per_thread = total // n_threads
    if total % n_threads != 0:
        raise ValueError(
            f"emit_cooperative_gmem_to_smem: {rows}*{cols}={total} "
            f"not divisible by n_threads={n_threads}"
        )
    for i in range(per_thread):
        flat_idx = b.add(
            b.mul(tid, b.const(DType.U32, per_thread)),
            b.const(DType.U32, i),
        )
        cols_c = b.const(DType.U32, cols)
        row_in_tile = b.div(flat_idx, cols_c)
        col_in_tile = b.rem(flat_idx, cols_c)
        gmem_row = b.add(gmem_row_base, row_in_tile)
        gmem_col = b.add(gmem_col_base, col_in_tile)
        val = b.load(gmem, gmem_row, gmem_col)
        b.store(smem, val, row_in_tile, col_in_tile)
