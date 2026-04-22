"""TileLoader — cooperative tile load from gmem into an smem stage.

Two paths:
  - `emit_tile_load` with `use_async=False` (default): scalar loads.
    Correct and simple. Each thread loads rows*cols/n_threads elements.
  - `emit_tile_load` with `use_async=True`: cp.async 16-byte lines.
    Faster on Ampere+ but requires 16-byte-aligned rows.

Both paths are stateless L1 helpers. The KPipeline L2 block calls
them at the right point in the pipeline (prologue, prefetch, etc.).
"""

from __future__ import annotations

from quark.ir import Builder, DType, GlobalTensor, SharedRegion, Value


def emit_tile_load(
    b: Builder,
    *,
    dst_smem: SharedRegion,
    src_gmem: GlobalTensor,
    rows: int,
    cols: int,
    gmem_row_base: Value,
    gmem_col_base: Value,
    tid: Value,
    n_threads: int,
    use_async: bool = False,
    pred: Value | None = None,
    cast: DType | None = None,
) -> None:
    """Cooperatively load a (rows × cols) element tile into smem.

    `use_async=True` uses cp.async 16-byte lines (requires cols *
    elem_bytes to be 16-aligned). `use_async=False` uses scalar
    loads (always works).

    `cast` converts each element from the src dtype to `cast` before
    the smem store. cp.async can't do element conversion, so passing
    `cast` forces the scalar path.
    """
    if cast is not None:
        _emit_scalar_tile(
            b,
            gmem=src_gmem,
            smem=dst_smem,
            rows=rows,
            cols=cols,
            gmem_row_base=gmem_row_base,
            gmem_col_base=gmem_col_base,
            tid=tid,
            n_threads=n_threads,
            cast=cast,
        )
    elif use_async:
        _emit_async_tile(
            b,
            dst_smem=dst_smem,
            src_gmem=src_gmem,
            rows=rows,
            cols=cols,
            gmem_row_base=gmem_row_base,
            gmem_col_base=gmem_col_base,
            tid=tid,
            n_threads=n_threads,
            outer_pred=pred,
        )
    else:
        _emit_scalar_tile(
            b,
            gmem=src_gmem,
            smem=dst_smem,
            rows=rows,
            cols=cols,
            gmem_row_base=gmem_row_base,
            gmem_col_base=gmem_col_base,
            tid=tid,
            n_threads=n_threads,
        )


_FP8_DTYPES = (DType.E4M3, DType.E5M2)


def _emit_scalar_tile(
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
    cast: DType | None = None,
) -> None:
    total = rows * cols
    per_thread = total // n_threads
    if total % n_threads != 0:
        raise ValueError(
            f"_emit_scalar_tile: {rows}*{cols}={total} not divisible by n_threads={n_threads}"
        )
    cols_c = b.const(DType.U32, cols)

    # Two paired-pack paths handle the two PTX cvt-for-fp8 quirks:
    #   pack2_dst — fp8 DEST: PTX has no scalar `cvt.<fp8>.<src>`; only
    #     packed `cvt.<fp8>x2.<src>x2`. Pair source loads, packed_convert,
    #     write 2 fp8 bytes via one b16 store.
    #   pack2_src — fp8 SOURCE → wider DEST: PTX has no scalar
    #     `cvt.<wider>.<fp8>`; only packed `cvt.<dst>x2.<fp8>x2`. Read 2
    #     fp8 bytes via one b16 load, unpacked_convert, store 2 wider
    #     scalars.
    src_is_fp8 = gmem.dtype in _FP8_DTYPES
    pack2_dst = cast is not None and cast in _FP8_DTYPES and smem.dtype is cast
    pack2_src = cast is not None and src_is_fp8 and cast not in _FP8_DTYPES
    step = 2 if (pack2_dst or pack2_src) else 1
    if (pack2_dst or pack2_src) and per_thread % 2 != 0:
        raise ValueError(
            f"_emit_scalar_tile: paired fp8 cast path needs per_thread even, got "
            f"{per_thread} (tile={rows}x{cols}, n_threads={n_threads})"
        )

    for i in range(0, per_thread, step):
        base_idx = b.add(
            b.mul(tid, b.const(DType.U32, per_thread)),
            b.const(DType.U32, i),
        )
        row_in_tile = b.div(base_idx, cols_c)
        col_in_tile = b.rem(base_idx, cols_c)
        gmem_row = b.add(gmem_row_base, row_in_tile)
        gmem_col = b.add(gmem_col_base, col_in_tile)

        if pack2_src:
            # Load 2 contiguous fp8 bytes as one B16 ("bytes are bytes" — the
            # `dtype=B16` override on `b.load` makes the lowerer emit a
            # plain `ld.global.b16` at this address, even though the gmem
            # tensor's element dtype is fp8). Then unpacked_convert splits
            # the b16 into 2 wider scalars and we store both to smem.
            assert cast is not None
            packed_b16 = b.load(gmem, gmem_row, gmem_col, dtype=DType.B16)
            unpacked = b.unpacked_convert(packed_b16, src_dtype=gmem.dtype, dst_dtype=cast)
            lo = b.vec_extract(unpacked, 0)
            hi = b.vec_extract(unpacked, 1)
            next_idx = b.add(base_idx, b.const(DType.U32, 1))
            row_n = b.div(next_idx, cols_c)
            col_n = b.rem(next_idx, cols_c)
            b.store(smem, lo, row_in_tile, col_in_tile)
            b.store(smem, hi, row_n, col_n)
            continue

        v0 = b.load(gmem, gmem_row, gmem_col)
        if pack2_dst:
            # Load the next element in the same row — `per_thread` is
            # chosen so each thread's `per_thread` slots are contiguous
            # in the flat (row-major) tile index, so `base_idx+1`
            # lands at the next column. (When `base_idx+1 == cols`,
            # the address arithmetic wraps to the next row, which is
            # the same behavior the scalar path would have produced
            # for element `i+1`.)
            assert cast is not None  # pack2_dst ⇒ cast is an fp8 DType
            next_idx = b.add(base_idx, b.const(DType.U32, 1))
            row_n = b.div(next_idx, cols_c)
            col_n = b.rem(next_idx, cols_c)
            v1 = b.load(gmem, b.add(gmem_row_base, row_n), b.add(gmem_col_base, col_n))
            packed = b.packed_convert(v0, v1, cast)
            # Packed b16 store lays down two contiguous fp8 bytes at
            # (row_in_tile, col_in_tile) in the E4M3/E5M2 smem tile —
            # StoreOp treats it as a 2-byte packed write.
            b.store(smem, packed, row_in_tile, col_in_tile)
            continue

        if cast is not None and v0.dtype is not cast:
            v0 = b.convert(v0, cast)
        b.store(smem, v0, row_in_tile, col_in_tile)


def _emit_async_tile(
    b: Builder,
    *,
    dst_smem: SharedRegion,
    src_gmem: GlobalTensor,
    rows: int,
    cols: int,
    gmem_row_base: Value,
    gmem_col_base: Value,
    tid: Value,
    n_threads: int,
    outer_pred: Value | None = None,
) -> None:
    elem_bytes = src_gmem.dtype.bytes
    row_stride_bytes = cols * elem_bytes
    total_bytes = rows * row_stride_bytes
    total_lines = total_bytes // 16
    assert row_stride_bytes % 16 == 0
    lines_per_row = row_stride_bytes // 16
    elems_per_line = 16 // elem_bytes

    n_passes = (total_lines + n_threads - 1) // n_threads

    for p in range(n_passes):
        if p == 0:
            line_id = tid
        else:
            line_id = b.add(tid, b.const(DType.U32, p * n_threads))

        pred = b.cmp("lt", line_id, b.const(DType.U32, total_lines))
        if outer_pred is not None:
            pred = b.select(pred, outer_pred, b.const(DType.PRED, False))

        lines_c = b.const(DType.U32, lines_per_row)
        row_in_tile = b.div(line_id, lines_c)
        line_in_row = b.rem(line_id, lines_c)
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
