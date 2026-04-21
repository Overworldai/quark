"""AdaGateResidual reference — ``out = x + gate_bmcast * y``."""

from __future__ import annotations

from popcorn.backend import PT
from popcorn.ir import DType


def ada_gate_residual_reference(X, Y, gate, *, dtype: DType | str = DType.BF16):
    out_dt = DType(dtype) if isinstance(dtype, str) else dtype
    B = int(X.shape[0])
    G = int(gate.shape[0])
    M = B // G
    assert G * M == B, f"X rows ({B}) not divisible by gate groups ({G})"

    X_f = PT.astype(X, PT.float32)
    Y_f = PT.astype(Y, PT.float32)
    G_f = PT.astype(gate, PT.float32)

    if PT._is_mx(G_f):
        import mlx.core as mx

        g_bm = mx.repeat(G_f, M, axis=0)
    else:
        g_bm = G_f.repeat_interleave(M, dim=0)

    out = X_f + g_bm * Y_f
    return PT.astype(out, out_dt.backend)


def ada_gate_residual_reference_for_spec(kernel, X, Y, gate):
    s = kernel.spec
    return ada_gate_residual_reference(X, Y, gate, dtype=s.dtype)
