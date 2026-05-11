"""L0: Per-lane smem view for MMA fragment loads."""

from __future__ import annotations

from quark.ir import Builder, DType, SharedRegion, Value


def emit_smem_base(
    b: Builder,
    smem: SharedRegion,
    lane_col_step: int,
    *,
    gid: Value | None = None,
    tig: Value | None = None,
) -> SharedRegion:
    """Compute the per-lane smem view for fragment loads.

    Returns `smem.view(dyn_offset=groupID * row_stride + tidIG * step)`
    in element units.

    `gid` / `tig` may be passed in by the caller to reuse hoisted
    values (e.g. `BlockContext.gid` / `BlockContext.tig`). When not
    provided, fresh `group_id()` / `thread_id_in_group()` ops are
    emitted — every extra call duplicates those at the IR level.
    """
    if gid is None:
        gid = b.group_id()
    if tig is None:
        tig = b.thread_id_in_group()
    stride_elems = b.const(DType.U32, smem.stride[0])
    step = b.const(DType.U32, lane_col_step)
    per_lane = b.add(b.mul(gid, stride_elems), b.mul(tig, step))
    return smem.view(dyn_offset=per_lane)
