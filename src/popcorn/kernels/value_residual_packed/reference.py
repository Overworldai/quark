"""ValueResidualPacked reference."""

from __future__ import annotations

from popcorn.backend import PT


def value_residual_packed_reference_for_spec(kernel, QKV_curr, QKV_first, lamb):
    s = kernel.spec
    out_dt = s.dtype.backend
    curr_f = PT.astype(QKV_curr, PT.float32)
    first_f = PT.astype(QKV_first, PT.float32)
    lamb_f = PT.astype(lamb, PT.float32)

    # Copy Q/K columns unchanged, lerp V columns.
    if PT._is_mx(curr_f):
        import mlx.core as mx

        v_curr = curr_f[:, s.v_col_offset : s.v_col_offset + s.v_width]
        v_first = first_f[:, s.v_col_offset : s.v_col_offset + s.v_width]
        v_out = v_curr + lamb_f * (v_first - v_curr)
        out = mx.concatenate(
            [
                curr_f[:, : s.v_col_offset],
                v_out,
                curr_f[:, s.v_col_offset + s.v_width :],
            ],
            axis=1,
        )
    else:
        out = curr_f.clone()
        vo = s.v_col_offset
        vw = s.v_width
        out[:, vo : vo + vw] = curr_f[:, vo : vo + vw] + lamb_f * (
            first_f[:, vo : vo + vw] - curr_f[:, vo : vo + vw]
        )

    return PT.astype(out, out_dt)
