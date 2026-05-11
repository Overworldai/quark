"""owl_attn iter→kv_offset smem table.

The kernel walks a flat iter sequence ``[0, N_TOTAL)`` whose entries
each map to a kv_offset inside one of ``max_segments`` segments. The
mapping is computed once at kernel start and cached in smem so the
hot loop's ``produce`` callback can fetch it with a single
``ld.shared.b32`` instead of re-running a max_segs-deep cmp+select
chain on every iter.

This is hot-path code: keep it small + free of helpers the kernel
doesn't already use.
"""

from __future__ import annotations

from typing import Any

import quark.lang as qk
from quark.ir import DType


def emit_kv_offset_table(
    *,
    bctx: Any,
    seg_starts_u: list,
    cum_u: list,
    n_real_total: Any,
    KvTile: int,
    KvTile_u: Any,
    zero_u: Any,
    N_TOTAL: int,
    n_threads_per_cta: int,
):
    """Allocate + cooperatively fill ``kv_off_table[N_TOTAL]`` in smem.

    Each thread handles iters ``tid + k * n_threads`` for k = 0, 1, …
    until the table is fully populated. Iters past ``n_real_total``
    clamp to kv_offset=0 (a safe address; consume's mask zeroes their
    contribution). Returns the smem region.
    """
    max_segs = len(seg_starts_u)

    def _chain(iter_u):
        kv_off = seg_starts_u[0] + (iter_u - cum_u[0]) * KvTile_u
        for si in range(1, max_segs):
            take = qk.cmp("ge", iter_u, cum_u[si])
            alt = seg_starts_u[si] + (iter_u - cum_u[si]) * KvTile_u
            kv_off = qk.select(take, alt, kv_off)
        valid = qk.cmp("lt", iter_u, n_real_total)
        return qk.select(valid, kv_off, zero_u)

    kv_off_smem = qk.smem_alloc("kv_off_table", DType.U32, (N_TOTAL,))
    n_setup_chunks = (N_TOTAL + n_threads_per_cta - 1) // n_threads_per_cta
    N_TOTAL_u = bctx.c(N_TOTAL, dtype=DType.U32)
    for chunk in range(n_setup_chunks):
        iter_local = bctx.tid + bctx.c(chunk * n_threads_per_cta, dtype=DType.U32)
        in_range = qk.cmp("lt", iter_local, N_TOTAL_u)
        kv_off = _chain(iter_local)
        qk.store(kv_off_smem, kv_off, iter_local, pred=in_range)
    qk.barrier("block")
    return kv_off_smem
