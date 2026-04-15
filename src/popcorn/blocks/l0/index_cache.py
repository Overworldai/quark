"""IndexCache — cooperatively load a small 1D array from gmem to smem.

Used by the MoE kernels to cache `token_ids[grp_start..+BM]` (int32)
and `slot_weights[grp_start..+BM]` (float32) into smem once per work
item, so the gathered tile loader and the scatter epilogue can read
per-row indices from smem instead of hitting gmem repeatedly.

Pattern: each of n_threads threads loads one element (predicated off
if BM < n_threads). After the cache fill + barrier, every thread can
read `smem_cache[row]` for any `row < BM`.
"""

from __future__ import annotations

from popcorn.ir import Builder, DType, GlobalTensor, SharedRegion, Value


def emit_index_cache(
    b: Builder,
    *,
    gmem: GlobalTensor,
    smem: SharedRegion,
    count: int,
    gmem_base_idx: Value,
    tid: Value,
    n_threads: int,
) -> None:
    """Cooperatively load `count` elements from gmem[gmem_base_idx..]
    into smem[0..count-1].

    The gmem tensor is 1D (shape=(N,), stride=(1,)). The smem tensor
    is also 1D (shape=(count,), stride=(1,)). Each thread loads one
    element; excess threads are predicated off.

    After calling this, issue a `b.barrier("block")` before reading
    the cached values.
    """
    if count <= n_threads:
        # Single pass: thread `tid` loads element `tid` if tid < count.
        pred = b.cmp("lt", tid, b.const(DType.U32, count))
        gmem_idx = b.add(gmem_base_idx, tid)
        val = b.load(gmem, gmem_idx, pred=pred)
        b.store(smem, val, tid, pred=pred)
    else:
        # Multi-pass: each thread loads ceil(count/n_threads) elements.
        per_thread = (count + n_threads - 1) // n_threads
        for i in range(per_thread):
            flat = b.add(tid, b.const(DType.U32, i * n_threads))
            pred = b.cmp("lt", flat, b.const(DType.U32, count))
            gmem_idx = b.add(gmem_base_idx, flat)
            val = b.load(gmem, gmem_idx, pred=pred)
            b.store(smem, val, flat, pred=pred)
