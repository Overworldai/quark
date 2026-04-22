"""MoE in-projection numpy reference.

``h = SiLU(X[token_ids] @ W_in[expert].T)`` with f32 accumulation.
``work_list`` is a flat ``[(grp_start, expert)] * n_work_items`` array
of s32 pairs; each entry covers ``BM`` consecutive output slots.

The BM step here is a **test-fixture constant** (32) baked into
``make_tensors_numpy``, matching the old torch reference. Autotune
configs that pick ``BM != 32`` will have their work_list shape mismatch
the kernel's TensorDecl — that's a pre-existing corner, unchanged by
this port.
"""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import astype_numpy, to_f32_numpy

_REF_BM = 32


def moe_inproj_reference_numpy(spec, *, X, W_in, token_ids, work_list, H_out=None):
    del H_out
    a_hint = spec.a_dtype.value
    b_hint = spec.b_dtype.value

    # Reference stays in f32 — no compute-dtype round-trip. The autotune
    # correctness gate uses a relaxed fp8 cos-sim budget to absorb the
    # kernel's narrow-cast drift.
    x = to_f32_numpy(X, dtype_hint=a_hint)
    w = to_f32_numpy(W_in, dtype_hint=b_hint)
    tok = to_f32_numpy(token_ids, dtype_hint="s32").astype(np.int64)
    wl = to_f32_numpy(work_list, dtype_hint="s32").astype(np.int64).reshape(-1, 2)

    H, D = spec.H, x.shape[-1]
    n_experts = spec.n_experts
    w3 = w.reshape(n_experts, H, D)

    h = np.zeros((spec.total_slots, H), dtype=np.float32)
    for grp_start, expert in wl:
        gs, e = int(grp_start), int(expert)
        tids = tok[gs : gs + _REF_BM]
        x_g = x[tids]
        h[gs : gs + _REF_BM] = x_g @ w3[e].T

    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    h = h / (1.0 + np.exp(-h))
    return astype_numpy(h, spec.out_dtype)
