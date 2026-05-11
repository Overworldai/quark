"""Patchify numpy reference — reshape + permute + matmul."""

from __future__ import annotations

from quark.runtime.npconv import astype_numpy, to_f32_numpy


def patchify_reference_numpy(spec, *, X, W, Out=None):
    del Out
    hint = spec.dtype.value
    x = to_f32_numpy(X, dtype_hint=hint)
    w = to_f32_numpy(W, dtype_hint=hint)

    B, C = spec.B, spec.C
    ph, pw = spec.ph, spec.pw
    Hp, Wp = spec.Hp, spec.Wp
    x = x.reshape(B, C, Hp, ph, Wp, pw)
    x = x.transpose(0, 2, 4, 1, 3, 5)  # [B, Hp, Wp, C, ph, pw]
    x_flat = x.reshape(B * Hp * Wp, C * ph * pw)
    out = x_flat @ w.T
    return astype_numpy(out, spec.dtype)
