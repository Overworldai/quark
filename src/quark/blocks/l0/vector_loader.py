"""VectorLoader — cooperative 1D gmem→smem load.

Companion to :func:`emit_tile_load`, at rank-1.

Two paths:
  - ``use_async=True`` (default): cp.async 16-byte lines, one or more
    per thread. Requires ``length * elem_bytes`` to be a multiple of 16.
  - ``use_async=False`` *or* ``cast is not None``: scalar loads.
    Needed when the gmem element type differs from the smem type.

The source may be rank-1 or rank-2. When rank-2, pass ``gmem_row``
to pick a single row (``src_idx=[gmem_row, col]``); this matches the
``async_copy(dst=S_smem, src=g.scale, src_idx=[group, elem])`` pattern
currently used by the norm kernels to load scale/bias vectors.
"""

from __future__ import annotations

from quark.ir import Builder, DType, GlobalTensor, SharedRegion, Value


def emit_vector_load(
    b: Builder,
    *,
    dst_smem: SharedRegion,
    src_gmem: GlobalTensor,
    length: int,
    tid: Value,
    n_threads: int,
    gmem_row: Value | None = None,
    use_async: bool = True,
    cast: DType | None = None,
    pred: Value | None = None,
) -> None:
    """Cooperatively load a length-N vector from gmem into a rank-1
    SharedRegion.

    ``gmem_row`` is required when ``src_gmem.rank == 2`` — the vector
    lives at row ``gmem_row`` of the 2D tensor.
    """
    if dst_smem.rank != 1:
        raise ValueError(f"emit_vector_load: dst must be rank-1 (got shape {dst_smem.shape})")
    if src_gmem.rank == 2 and gmem_row is None:
        raise ValueError("emit_vector_load: src_gmem is rank-2; pass gmem_row= to select a row")
    if src_gmem.rank not in (1, 2):
        raise ValueError(
            f"emit_vector_load: src must be rank-1 or rank-2 (got shape {src_gmem.shape})"
        )

    if cast is not None or not use_async:
        _emit_scalar_vector(
            b,
            dst_smem=dst_smem,
            src_gmem=src_gmem,
            length=length,
            gmem_row=gmem_row,
            tid=tid,
            n_threads=n_threads,
            cast=cast,
        )
    else:
        _emit_async_vector(
            b,
            dst_smem=dst_smem,
            src_gmem=src_gmem,
            length=length,
            gmem_row=gmem_row,
            tid=tid,
            n_threads=n_threads,
            outer_pred=pred,
        )


def _emit_async_vector(
    b: Builder,
    *,
    dst_smem: SharedRegion,
    src_gmem: GlobalTensor,
    length: int,
    gmem_row: Value | None,
    tid: Value,
    n_threads: int,
    outer_pred: Value | None = None,
) -> None:
    elem_bytes = src_gmem.dtype.bytes
    total_bytes = length * elem_bytes
    if total_bytes % 16 != 0:
        raise ValueError(
            f"_emit_async_vector: length*elem_bytes ({total_bytes}) must be 16-aligned "
            f"for cp.async; pass use_async=False for scalar fallback"
        )
    total_lines = total_bytes // 16
    elems_per_line = 16 // elem_bytes

    n_passes = (total_lines + n_threads - 1) // n_threads
    has_tail = total_lines % n_threads != 0

    for p in range(n_passes):
        if p == 0:
            line_id = tid
        else:
            line_id = b.add(tid, b.const(DType.U32, p * n_threads))

        # Per-thread predicate only needed when the last pass doesn't
        # fill all lanes. Preserves the invariant that the common-case
        # `total_lines % n_threads == 0` path emits no comparison.
        if has_tail:
            pred = b.cmp("lt", line_id, b.const(DType.U32, total_lines))
            if outer_pred is not None:
                pred = b.select(pred, outer_pred, b.const(DType.PRED, False))
        else:
            pred = outer_pred

        elem_off = b.mul(line_id, b.const(DType.U32, elems_per_line))

        if src_gmem.rank == 2:
            assert gmem_row is not None
            src_idx = (gmem_row, elem_off)
        else:
            src_idx = (elem_off,)

        b.async_copy(
            dst_smem,
            src_gmem,
            dst_idx=(elem_off,),
            src_idx=src_idx,
            count=16,
            pred=pred,
        )


def _emit_scalar_vector(
    b: Builder,
    *,
    dst_smem: SharedRegion,
    src_gmem: GlobalTensor,
    length: int,
    gmem_row: Value | None,
    tid: Value,
    n_threads: int,
    cast: DType | None,
) -> None:
    per_thread = length // n_threads
    has_tail = length % n_threads != 0
    if per_thread == 0 and not has_tail:
        return

    for i in range(per_thread):
        base_idx = b.add(
            b.mul(tid, b.const(DType.U32, per_thread)),
            b.const(DType.U32, i),
        )
        if src_gmem.rank == 2:
            assert gmem_row is not None
            v = b.load(src_gmem, gmem_row, base_idx)
        else:
            v = b.load(src_gmem, base_idx)
        if cast is not None and v.dtype is not cast:
            v = b.convert(v, cast)
        b.store(dst_smem, v, base_idx)

    if has_tail:
        tail_base = per_thread * n_threads
        tail_idx = b.add(tid, b.const(DType.U32, tail_base))
        tail_pred = b.cmp("lt", tail_idx, b.const(DType.U32, length))
        if src_gmem.rank == 2:
            assert gmem_row is not None
            v = b.load(src_gmem, gmem_row, tail_idx, pred=tail_pred)
        else:
            v = b.load(src_gmem, tail_idx, pred=tail_pred)
        if cast is not None and v.dtype is not cast:
            v = b.convert(v, cast)
        b.store(dst_smem, v, tail_idx, pred=tail_pred)
