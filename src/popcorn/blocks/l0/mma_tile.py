"""L0: Single mma.sync — load A/B fragments from smem, accumulate."""

from __future__ import annotations

from popcorn.ir import Builder, SharedRegion, Value


def emit_mma_tile(
    b: Builder,
    *,
    a_smem_lane: SharedRegion,
    b_smem_lane: SharedRegion,
    shape_id: str,
    a_offsets: tuple[tuple[int, int], ...],
    b_offsets: tuple[tuple[int, int], ...],
    cd_offsets: tuple[tuple[int, int], ...],
    m_tile: int,
    n_tile: int,
    kk: int,
    acc_in: Value,
    m_stride: int = 16,
    n_stride: int = 8,
) -> Value:
    """One mma.sync: load A/B frags from smem, accumulate.

    ``m_stride`` / ``n_stride`` are the per-tile row offsets — ``shape.m``
    and ``shape.n`` of the MMA descriptor. Defaults match the legacy
    m16n8 shapes; callers iterating a grid of m8n8 tiles pass 8/8.
    """
    a_frag = b.load_matrix(
        a_smem_lane,
        shape_id,
        which="a",
        row=m_tile * m_stride,
        col=kk,
        reg_offsets=a_offsets,
    )
    b_frag = b.load_matrix(
        b_smem_lane,
        shape_id,
        which="b",
        row=n_tile * n_stride,
        col=kk,
        reg_offsets=b_offsets,
    )
    return b.mma(shape_id, a_frag, b_frag, acc_in)
