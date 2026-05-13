"""Numpy reference for per-token symmetric int8 quant."""

from __future__ import annotations

import numpy as np

from quark.runtime.npconv import astype_numpy, to_f32_numpy


def kv_quantize_reference_numpy(spec, *, K_in, K_out_s8=None, K_scales=None):
    """Per-token symmetric s8 quant. Supports row-major (K-cache)
    and transposed (Vt-cache) layouts via ``spec.transposed``."""
    del K_out_s8, K_scales
    a_hint = spec.in_dtype.value
    x = to_f32_numpy(K_in, dtype_hint=a_hint)

    if not spec.transposed:
        # K-cache: [num_tokens, Dh]. Reduce per row.
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
        abs_max = np.abs(x).max(axis=1)
        scale = np.where(abs_max == 0, 1.0, abs_max / 127.0).astype(np.float32)
        q = np.round(x / scale[:, None]).clip(-127, 127).astype(np.int8)
        return {"K_out_s8": q, "K_scales": scale}
    else:
        # Vt-cache: [n_heads * Dh, num_tokens]. Reduce per column,
        # but only within each head's Dh-block. Reshape to
        # [n_heads, Dh, num_tokens] for the per-(head, token) scale.
        x = x.reshape(spec.n_heads, spec.Dh, spec.num_tokens)
        abs_max = np.abs(x).max(axis=1)  # → [n_heads, num_tokens]
        scale = np.where(abs_max == 0, 1.0, abs_max / 127.0).astype(np.float32)
        # Broadcast scale over Dh: [n_heads, 1, num_tokens] for division.
        q = np.round(x / scale[:, None, :]).clip(-127, 127).astype(np.int8)
        q = q.reshape(spec.n_heads * spec.Dh, spec.num_tokens)
        return {
            "K_out_s8": q,
            # Scales flattened to [n_heads * num_tokens] for storage.
            "K_scales": scale.reshape(spec.n_heads * spec.num_tokens),
        }
