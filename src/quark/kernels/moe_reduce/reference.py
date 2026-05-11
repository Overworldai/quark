"""moe_reduce numpy reference.

``out[m, d] = sum over k of slot_weights[idx] * partials[idx, d]``
where ``idx = token_slot_table[m, k]``.

f32 accumulator throughout; narrowed to ``spec.out_dtype`` on store.
"""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import astype_numpy, to_f32_numpy


def moe_reduce_reference_numpy(spec, *, partials, slot_weights, token_slot_table, out=None):
    del out
    p_hint = spec.partials_dtype.value
    p = to_f32_numpy(partials, dtype_hint=p_hint).reshape(spec.total_slots, spec.D)
    w = to_f32_numpy(slot_weights, dtype_hint="f32").reshape(spec.total_slots)
    t = (
        to_f32_numpy(token_slot_table, dtype_hint="s32")
        .astype(np.int64)
        .reshape(spec.M, spec.top_k)
    )

    out_f32 = np.zeros((spec.M, spec.D), dtype=np.float32)
    for m in range(spec.M):
        for k in range(spec.top_k):
            idx = int(t[m, k])
            out_f32[m] += float(w[idx]) * p[idx]

    return astype_numpy(out_f32, spec.out_dtype)
