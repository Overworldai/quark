"""MoE out-projection numpy reference.

``output[token_ids[slot]] += h[slot] @ W_out[expert].T * slot_weights[slot]``

Each work_list entry covers BM=32 output slots under one expert.
Scatter-adds into ``output[M, D]`` — multiple slots can map to the
same row (that's the top-k accumulation). f32 accumulator throughout.
"""

from __future__ import annotations

import numpy as np

from popcorn.ir import DType
from popcorn.runtime.npconv import astype_numpy, to_f32_numpy

_REF_BM = 32


def _roundtrip_through(arr_f32: np.ndarray, target: DType) -> np.ndarray:
    carrier = astype_numpy(arr_f32, target)
    return to_f32_numpy(carrier, dtype_hint=target.value)


def moe_outproj_reference_numpy(
    spec, *, h_in, W_out, token_ids, slot_weights, work_list, output=None
):
    del output
    a_hint = spec.a_dtype.value
    b_hint = spec.b_dtype.value

    h = to_f32_numpy(h_in, dtype_hint=a_hint)
    w = to_f32_numpy(W_out, dtype_hint=b_hint)
    tok = to_f32_numpy(token_ids, dtype_hint="s32").astype(np.int64)
    wts = to_f32_numpy(slot_weights, dtype_hint="f32")
    wl = to_f32_numpy(work_list, dtype_hint="s32").astype(np.int64).reshape(-1, 2)

    compute_dt = spec.compute_dtype_resolved
    if compute_dt is not spec.a_dtype:
        h = _roundtrip_through(h, compute_dt)
    if compute_dt is not spec.b_dtype:
        w = _roundtrip_through(w, compute_dt)

    D, H = spec.D, spec.H
    n_experts = spec.n_experts
    w3 = w.reshape(n_experts, D, H)

    out = np.zeros((spec.M, D), dtype=np.float32)
    for grp_start, expert in wl:
        gs, e = int(grp_start), int(expert)
        if gs + _REF_BM > wts.shape[0]:
            continue  # tail fragment the kernel also skips
        proj = (h[gs : gs + _REF_BM] @ w3[e].T) * wts[gs : gs + _REF_BM, None]
        # Scatter-add: np.add.at is the in-place analogue of index_add_.
        np.add.at(out, tok[gs : gs + _REF_BM], proj)
    return out
