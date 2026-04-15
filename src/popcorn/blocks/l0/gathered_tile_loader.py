"""GatheredTileLoader — tile load where each row's gmem row index is
looked up from an smem index cache, but the K dimension is contiguous.

Each row issues cp.async 16-byte lines along the contiguous K columns.
The per-row gmem address is token_ids[row] * row_stride + k_col.
"""

from __future__ import annotations

from popcorn.ir import Builder, DType, GlobalTensor, SharedRegion, Value


def emit_gathered_tile_load(
    b: Builder,
    *,
    dst_smem: SharedRegion,
    src_gmem: GlobalTensor,
    index_smem: SharedRegion,
    rows: int,
    cols: int,
    gmem_col_base: Value,
    tid: Value,
    n_threads: int,
    cast: DType | None = None,
    use_async: bool = True,
) -> None:
    """Cooperatively load a gathered tile using cp.async per row.

    Each row's gmem row index comes from index_smem[row]. The K-dimension
    columns [gmem_col_base..+cols] are contiguous in gmem, so each row
    is loaded as one or more cp.async 16-byte lines.

    Falls back to scalar loads if ``use_async=False``, if the row byte
    width isn't 16-aligned, or if ``cast`` is set — cp.async is a byte
    copy and can't perform element conversion, so casts must flow
    through register scalars.
    """
    elem_bytes = src_gmem.dtype.bytes
    row_bytes = cols * elem_bytes

    if use_async and cast is None and row_bytes % 16 == 0:
        _emit_gathered_async(
            b,
            dst_smem=dst_smem,
            src_gmem=src_gmem,
            index_smem=index_smem,
            rows=rows,
            cols=cols,
            gmem_col_base=gmem_col_base,
            tid=tid,
            n_threads=n_threads,
            elem_bytes=elem_bytes,
        )
    else:
        _emit_gathered_scalar(
            b,
            dst_smem=dst_smem,
            src_gmem=src_gmem,
            index_smem=index_smem,
            rows=rows,
            cols=cols,
            gmem_col_base=gmem_col_base,
            tid=tid,
            n_threads=n_threads,
            cast=cast,
        )


def _emit_gathered_async(
    b: Builder,
    *,
    dst_smem: SharedRegion,
    src_gmem: GlobalTensor,
    index_smem: SharedRegion,
    rows: int,
    cols: int,
    gmem_col_base: Value,
    tid: Value,
    n_threads: int,
    elem_bytes: int,
) -> None:
    """cp.async per row: each row has cols*elem_bytes contiguous bytes."""
    row_bytes = cols * elem_bytes
    lines_per_row = row_bytes // 16
    total_lines = rows * lines_per_row
    elems_per_line = 16 // elem_bytes

    n_passes = (total_lines + n_threads - 1) // n_threads

    for p in range(n_passes):
        line_id = tid if p == 0 else b.add(tid, b.const(DType.U32, p * n_threads))

        # Decompose line_id → (row_in_tile, line_in_row). No clamp: for
        # excess threads (line_id >= total_lines), row_in_tile is OOB
        # of index_smem, so the token_id smem load MUST be
        # predicate-guarded to avoid an OOB smem read. The async_copy
        # itself is also predicate-guarded — OOB threads end up with
        # OOB smem write addresses that the Metal JIT's alias tracker
        # correctly recognizes as non-overlapping with the tile, so
        # the subsequent simdgroup_load reads clean data.
        #
        # (We tried clamping row_in_tile into range so OOB threads did
        # a duplicate write to the last valid line. That combined with
        # the divergent `if (pred)` guard around the write made Metal's
        # alias tracker see in-range writes from divergent lanes and
        # miscompile the following simdgroup_load — cos collapsed to
        # ~0.26. Without the clamp the addresses fall outside the tile
        # and that interaction doesn't trigger.)
        pred = b.cmp("lt", line_id, b.const(DType.U32, total_lines))
        row_in_tile = b.div(line_id, b.const(DType.U32, lines_per_row))
        line_in_row = b.rem(line_id, b.const(DType.U32, lines_per_row))
        smem_col = b.mul(line_in_row, b.const(DType.U32, elems_per_line))

        # Gather: look up gmem row from index cache (predicated to
        # avoid OOB smem reads on excess threads when total_lines
        # doesn't fill n_threads).
        token_id = b.load(index_smem, row_in_tile, pred=pred)
        token_id_u32 = b.bitcast(token_id, DType.U32) if index_smem.dtype == DType.S32 else token_id

        gmem_col = b.add(gmem_col_base, smem_col)

        b.async_copy(
            dst_smem,
            src_gmem,
            dst_idx=(row_in_tile, smem_col),
            src_idx=(token_id_u32, gmem_col),
            count=16,
            pred=pred,
        )


_FP8_DTYPES = (DType.E4M3, DType.E5M2)


def _emit_gathered_scalar(
    b: Builder,
    *,
    dst_smem: SharedRegion,
    src_gmem: GlobalTensor,
    index_smem: SharedRegion,
    rows: int,
    cols: int,
    gmem_col_base: Value,
    tid: Value,
    n_threads: int,
    cast: DType | None = None,
) -> None:
    """Scalar fallback for non-16B-aligned rows or cast paths.

    When `cast` is set, each loaded gmem element is converted to `cast`
    before the smem store. fp8 casts go through `packed_convert` (two
    source elements per iteration → one b16 packed store), matching
    the tile_loader scalar path so the smem layout stays identical.
    """
    total = rows * cols
    per_thread = total // n_threads
    if total % n_threads != 0:
        raise ValueError(
            f"emit_gathered_tile_load: {rows}*{cols}={total} not divisible by n_threads={n_threads}"
        )
    cols_c = b.const(DType.U32, cols)
    pack2 = cast is not None and cast in _FP8_DTYPES and dst_smem.dtype is cast
    step = 2 if pack2 else 1
    if pack2 and per_thread % 2 != 0:
        raise ValueError(
            f"_emit_gathered_scalar: fp8 cast path needs per_thread even, "
            f"got {per_thread} (tile={rows}x{cols}, n_threads={n_threads})"
        )

    for i in range(0, per_thread, step):
        base_idx = b.add(
            b.mul(tid, b.const(DType.U32, per_thread)),
            b.const(DType.U32, i),
        )
        row_in_tile = b.div(base_idx, cols_c)
        col_in_tile = b.rem(base_idx, cols_c)

        token_id = b.load(index_smem, row_in_tile)
        token_id_u32 = b.bitcast(token_id, DType.U32) if index_smem.dtype == DType.S32 else token_id
        gmem_col = b.add(gmem_col_base, col_in_tile)
        v0 = b.load(src_gmem, token_id_u32, gmem_col)

        if pack2:
            assert cast is not None  # narrowing: pack2 ⇒ cast is fp8
            # Second element in the flat tile index. Since the gather
            # looks up the row from the index cache separately for
            # each element, the pair can span rows safely — index_smem
            # remaps them row-independently. The b16 store uses a
            # single `(row_in_tile, col_in_tile)` though, so we require
            # the pair to land in the same row (which holds whenever
            # per_thread slots stay row-aligned — enforced for all
            # MoE tile sizes in practice).
            next_idx = b.add(base_idx, b.const(DType.U32, 1))
            row_n = b.div(next_idx, cols_c)
            col_n = b.rem(next_idx, cols_c)
            token_id_n = b.load(index_smem, row_n)
            token_id_n_u32 = (
                b.bitcast(token_id_n, DType.U32) if index_smem.dtype == DType.S32 else token_id_n
            )
            gmem_col_n = b.add(gmem_col_base, col_n)
            v1 = b.load(src_gmem, token_id_n_u32, gmem_col_n)
            packed = b.packed_convert(v0, v1, cast)
            b.store(dst_smem, packed, row_in_tile, col_in_tile)
            continue

        if cast is not None and v0.dtype is not cast:
            v0 = b.convert(v0, cast)
        b.store(dst_smem, v0, row_in_tile, col_in_tile)
