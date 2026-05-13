"""L0: Cooperative cp.async 16-byte tile load."""

from __future__ import annotations

from quark.ir import Builder, DType, GlobalTensor, SharedRegion, Value


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
    """Cooperatively cp.async a (rows x cols) element tile.

    The pass loop is emitted as ``for_loop(unroll=True)`` with the
    bound derived from ``block_dim_x()`` rather than the Python
    ``n_threads`` arg. This makes the emit SG-agnostic — when the
    OCL/IGC backend forces SG=16 for MMA kernels (rescaling
    local_size), ``block_dim_x()`` reports the actual thread count
    at runtime and each thread covers the right number of passes.
    On CUDA, PTX const-folds ``block_dim_x()`` via
    ``fn.attrs.max_threads_per_block`` → fully unrolled, same PTX
    as the prior Python-side emit. See ``quark.ir.op.ForLoopOp``
    docstring and memory/project_openvino_taehv.md "Phase 3 step (17)".
    """
    row_stride_bytes = cols * elem_bytes
    total_bytes = rows * row_stride_bytes
    total_lines = total_bytes // 16
    assert row_stride_bytes % 16 == 0
    lines_per_row = row_stride_bytes // 16
    elems_per_line = 16 // elem_bytes

    n_threads_ir = b.block_dim("x")
    total_lines_c = b.const(DType.U32, total_lines)
    lines_per_row_c = b.const(DType.U32, lines_per_row)
    elems_per_line_c = b.const(DType.U32, elems_per_line)

    # n_passes = ceil(total_lines / n_threads) = (total_lines +
    # n_threads - 1) / n_threads. Computed as IR; on CUDA the PTX
    # const-fold path resolves it to a Python int via the
    # block_dim_x → fn.attrs.max_threads_per_block route.
    n_minus_one = b.add(n_threads_ir, b.const(DType.U32, total_lines - 1))
    n_passes_ir = b.div(n_minus_one, n_threads_ir)

    zero = b.const(DType.U32, 0)
    one = b.const(DType.U32, 1)
    with b.for_loop(zero, n_passes_ir, one, iv_name="ap", unroll=True) as (p, _):
        # line_id = tid + p * n_threads
        line_id = b.add(tid, b.mul(p, n_threads_ir))
        # Predicate on line_id < total_lines so the last pass handles
        # the tail correctly (when total_lines isn't a multiple of
        # n_threads).
        pred = b.cmp("lt", line_id, total_lines_c)
        row_in_tile = b.div(line_id, lines_per_row_c)
        line_in_row = b.rem(line_id, lines_per_row_c)
        smem_col = b.mul(line_in_row, elems_per_line_c)
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
        b.yield_()
