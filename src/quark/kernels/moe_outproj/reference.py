"""MoE out-projection numpy reference (per-slot partial layout).

``partials[slot, :] = h[slot] @ W_out[expert].T`` for every slot in the
work-list whose entry has ``expert >= 0``. Sentinel chunks (expert=-1)
are skipped — those slots are unused by the downstream ``moe_reduce``
kernel which only gathers via ``token_slot_table``.

f32 accumulator throughout, narrowed to ``spec.out_dtype`` on store.
"""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import astype_numpy, to_f32_numpy, zeros_for_dtype


def moe_outproj_reference_numpy(spec, *, h_in, W_out, work_list, partials=None):
    del partials
    a_hint = spec.a_dtype.value
    b_hint = spec.b_dtype.value

    h = to_f32_numpy(h_in, dtype_hint=a_hint)
    w = to_f32_numpy(W_out, dtype_hint=b_hint)
    wl = to_f32_numpy(work_list, dtype_hint="s32").astype(np.int64).reshape(-1, 2)

    bm = int(wl[1, 0] - wl[0, 0]) if wl.shape[0] >= 2 else int(spec.total_slots)

    D, H = spec.D, spec.H
    n_experts = spec.n_experts
    w3 = w.reshape(n_experts, D, H)

    # Output shape matches the kernel's TENSORS["partials"] role="out".
    out_f32 = np.zeros((spec.total_slots, D), dtype=np.float32)
    for grp_start, expert in wl:
        gs, e = int(grp_start), int(expert)
        if e < 0:
            continue
        if gs + bm > spec.total_slots:
            continue
        out_f32[gs : gs + bm] = h[gs : gs + bm] @ w3[e].T

    # Narrow to out_dtype carrier so the bench correctness gate sees
    # the same dtype the kernel writes.
    return astype_numpy(out_f32, spec.out_dtype)


# ``zeros_for_dtype`` is exported in case downstream callers need to
# pre-allocate a partials buffer of the right dtype.
__all__ = ["moe_outproj_reference_numpy", "zeros_for_dtype"]
