"""AdaGateResidual numpy reference — ``out = x + gate_bmcast * y``."""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import astype_numpy, to_f32_numpy


def ada_gate_residual_reference_numpy(spec, *, X, Y, gate, Out=None):
    del Out
    hint = spec.dtype.value
    x = to_f32_numpy(X, dtype_hint=hint)
    y = to_f32_numpy(Y, dtype_hint=hint)
    g = to_f32_numpy(gate, dtype_hint=hint)

    B, G = spec.B, spec.G
    M = B // G
    # Broadcast gate [G, D] → [B, D] by repeating each row M times.
    g_bm = np.repeat(g, M, axis=0)
    out = x + g_bm * y
    return astype_numpy(out, spec.dtype)
