"""ValueResidualPacked numpy reference — lerp V cols only, copy Q/K."""

from __future__ import annotations

from popcorn.runtime.npconv import astype_numpy, to_f32_numpy


def value_residual_packed_reference_numpy(spec, *, QKV_curr, QKV_first, lamb, Out=None):
    del Out
    hint = spec.dtype.value
    curr = to_f32_numpy(QKV_curr, dtype_hint=hint)
    first = to_f32_numpy(QKV_first, dtype_hint=hint)
    lamb_f = to_f32_numpy(lamb, dtype_hint="f32")[0]

    out = curr.copy()
    vo, vw = spec.v_col_offset, spec.v_width
    out[:, vo : vo + vw] = curr[:, vo : vo + vw] + lamb_f * (
        first[:, vo : vo + vw] - curr[:, vo : vo + vw]
    )
    return astype_numpy(out, spec.dtype)
