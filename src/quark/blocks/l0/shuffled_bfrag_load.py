"""L0: Shuffled B-fragment load — one vectorized ld.shared.v{N}.b32 per frag.

The B weight matrix has been offline-permuted so each warp lane's entire
B fragment for one (n_tile, k_step) is ``FRAG_BYTES`` contiguous bytes in
smem, at flat byte offset ``ks * 32 * FRAG_BYTES + lane_id * FRAG_BYTES``
within the 8-row n-tile block. This lets us replace the scalar
``ld.shared.b32`` per register with one vectorized ``ld.shared.v{N}.b32``.

The B smem view passed in must have its ``dyn_offset`` set to encode:
  ``warp_n_base_elems + lane_id * (FRAG_BYTES / elem_bytes)``

The (row, col) static offsets passed to ``vec_load`` encode the
(n_tile, k_step) tile coordinates:
  row = ``n_tile * 8`` (in elements — rows of the [BN, bstride] tile)
  col = ``k_step * 32 * FRAG_BYTES / elem_bytes`` (in elements)

Combined with ``vec_load``'s ``dtype=DType.B32`` override, this emits:
  ``ld.shared.v{N}.b32 {rN}, [smem + per-lane-offset + tile-static-offset];``
"""

from __future__ import annotations

import quark.lang as qk
from quark.blocks.dsl import BlockContext
from quark.ir import DType, SharedRegion, Value


def emit_shuffled_bfrag_load(
    ctx: BlockContext,
    b_smem_lane: SharedRegion,
    *,
    frag_regs: int,
    n_tile: int,
    k_step: int,
) -> Value:
    """Emit one vectorized B-fragment load from shuffled smem.

    Args:
      ctx: the BlockContext (provides ``ctx.const()`` for deduplicated
        constants — row and col indices across multiple (n_tile, k_step)
        pairs share ConstOps when they evaluate to the same value).
      b_smem_lane: B smem tile view with per-warp + per-lane ``dyn_offset``
        pre-computed (see module docstring).
      frag_regs: number of b32 registers per fragment (from ``MmaShape.b_regs``).
        - 2 for bf16 / fp16 m16n8k16 (→ ``ld.shared.v2.b32``, 8 B of
          fragment data per load).
        - 4 for 8-bit m16n8k32 (→ ``ld.shared.v4.b32``, 16 B / frag —
          the max vectorized form).
        - 1 for 8-bit m16n8k16 (→ ``ld.shared.b32`` scalar). The shuffled
          smem layout still wins here because each lane's fragment is
          still one contiguous 4-byte cell — the load is scalar, not
          vectorized, but the addressing is identical.
      n_tile: output n-tile index within the warp's N-partition.
      k_step: K-step index within the current BK chunk (``kk // mma_k``).

    Returns:
      A width-``frag_regs`` b32 Value, ready to pass to ``qk.mma()`` as B.
    """
    elem_bytes = b_smem_lane.dtype.bytes
    FRAG_BYTES = frag_regs * 4

    row = n_tile * 8
    # col in tensor-element units: multiplied by elem_bytes by the lowerer
    # to get the byte offset, so we divide by elem_bytes here.
    col_bytes = k_step * 32 * FRAG_BYTES
    assert col_bytes % elem_bytes == 0, (
        f"shuffled bfrag col_bytes={col_bytes} not divisible by elem_bytes={elem_bytes}"
    )
    col = col_bytes // elem_bytes

    if frag_regs not in (1, 2, 4):
        raise NotImplementedError(
            f"shuffled bfrag load supports frag_regs in {{1, 2, 4}}; got "
            f"{frag_regs}. Kernel is_valid() should reject other values."
        )

    # bctx.const → Builder.const → region-scoped CSE: a matching const
    # from an outer region is reused when it dominates; otherwise a
    # fresh ConstOp is emitted in the current region.
    row_c = ctx.const(DType.U32, row)
    col_c = ctx.const(DType.U32, col)

    # VecLoadOp requires width >= 2; for a single-reg fragment (fp8
    # m16n8k16) emit a scalar b32 load from the same per-lane address.
    if frag_regs == 1:
        return qk.load(b_smem_lane, row_c, col_c, dtype=DType.B32)
    return qk.vec_load(
        b_smem_lane,
        row_c,
        col_c,
        width=frag_regs,
        dtype=DType.B32,
    )
