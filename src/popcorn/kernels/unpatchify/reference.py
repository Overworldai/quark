"""Unpatchify numpy reference — matmul + rearrange."""

from __future__ import annotations

from popcorn.runtime.npconv import astype_numpy, to_f32_numpy


def unpatchify_reference_numpy(spec, *, X, W, Bias, Out=None):
    del Out
    hint = spec.dtype.value
    x = to_f32_numpy(X, dtype_hint=hint)
    w = to_f32_numpy(W, dtype_hint=hint)
    h = x @ w.T  # [M, C*ph*pw]
    if spec.has_bias:
        h = h + to_f32_numpy(Bias, dtype_hint=hint)

    B, C = spec.B, spec.C
    Hp, Wp = spec.Hp, spec.Wp
    ph, pw = spec.ph, spec.pw
    h = h.reshape(B, Hp, Wp, C, ph, pw).transpose(0, 3, 1, 4, 2, 5)
    out = h.reshape(B, C * spec.H * spec.W)
    return astype_numpy(out, spec.dtype)
