"""Epilogue blocks — output paths for the MoE kernels.

Two epilogue types:

1. **SiluCastEpilogue** (MoE inproj): apply SiLU element-wise to
   the f32 accumulators, cast to out_dtype, store contiguously to
   gmem at h[grp_start + row, n_base + col].

2. **AtomicScatterEpilogue** (MoE outproj): scale each accumulator
   by slot_weights[row], then atomically add via atom.global.add.f32
   into output[token_ids[row], n_base + col]. Per-row token_ids
   and weights are read from smem caches.

Both operate on the same accumulator layout: MT × NT width-4 f32
vecs with per-lane cd_offsets.
"""

from __future__ import annotations

import math

from popcorn.ir import Builder, DType, GlobalTensor, SharedRegion, Value

# SiLU: x * sigmoid(x) = x / (1 + exp(-x))
# For the f32 → f32 path, we compute:
#   sig = rcp_approx(1 + ex2_approx(-x * log2(e)))
#   silu = x * sig
# This is the same fast-math expansion the existing mma/math.py uses.
_LOG2E = math.log2(math.e)


def emit_silu_cast_epilogue(
    b: Builder,
    *,
    acc_results: list[Value],
    g_out: GlobalTensor,
    row_base: Value,
    col_base: Value,
    MT: int,
    NT: int,
    cd_offsets: tuple[tuple[int, int], ...],
    out_dtype: DType,
    shape_id: str = "m16n8k16_bf16",
    m_stride: int = 16,
    n_stride: int = 8,
) -> None:
    """SiLU + cast + contiguous store to gmem via frag_for_each.

    ``m_stride`` / ``n_stride`` match the MMA shape's m/n (default 16/8;
    8/8 for m8n8k8).
    """
    log2e = b.const(DType.F32, _LOG2E)
    one = b.const(DType.F32, 1.0)

    for mt in range(MT):
        for nt in range(NT):
            idx = mt * NT + nt
            acc_v = acc_results[idx]
            mt_row_base = b.const(DType.U32, mt * m_stride)
            nt_col_base = b.const(DType.U32, nt * n_stride)

            def fn(
                elem,
                row,
                col,
                mt_row_base=mt_row_base,
                nt_col_base=nt_col_base,
            ):
                neg_x_log2e = b.neg(b.mul(elem, log2e))
                exp_val = b.ex2_approx(neg_x_log2e)
                denom = b.add(one, exp_val)
                sig = b.rcp_approx(denom)
                silu_val = b.mul(elem, sig)
                if out_dtype != DType.F32:
                    silu_val = b.convert(silu_val, out_dtype)
                gmem_row = b.add(row_base, b.add(mt_row_base, row))
                gmem_col = b.add(col_base, b.add(nt_col_base, col))
                b.store(g_out, silu_val, gmem_row, gmem_col)

            b.frag_for_each(shape_id, acc_v, fn, cd_offsets)


def emit_atomic_scatter_epilogue(
    b: Builder,
    *,
    acc_results: list[Value],
    g_out: GlobalTensor,
    index_smem: SharedRegion,
    weight_smem: SharedRegion | None,
    col_base: Value,
    MT: int,
    NT: int,
    cd_offsets: tuple[tuple[int, int], ...],
    shape_id: str = "m16n8k16_bf16",
    m_stride: int = 16,
    n_stride: int = 8,
) -> None:
    """Weighted atomic scatter-add to gmem (MoE outproj epilogue).

    ``m_stride`` / ``n_stride`` match the MMA shape's m/n (default 16/8
    for m16n8 shapes; 8/8 for m8n8k8).
    """
    for mt in range(MT):
        for nt in range(NT):
            idx = mt * NT + nt
            acc_v = acc_results[idx]
            mt_row_base = b.const(DType.U32, mt * m_stride)
            nt_col_base = b.const(DType.U32, nt * n_stride)

            def fn(
                elem,
                row,
                col,
                mt_row_base=mt_row_base,
                nt_col_base=nt_col_base,
            ):
                row_in_block = b.add(mt_row_base, row)
                token_id = b.load(index_smem, row_in_block)
                token_id_u32 = (
                    b.bitcast(token_id, DType.U32) if index_smem.dtype == DType.S32 else token_id
                )
                if weight_smem is not None:
                    weight = b.load(weight_smem, row_in_block)
                    elem = b.mul(elem, weight)
                gmem_col = b.add(col_base, b.add(nt_col_base, col))
                b.atomic_rmw(g_out, "add", elem, token_id_u32, gmem_col)

            b.frag_for_each(shape_id, acc_v, fn, cd_offsets)
