"""AdaGateResidual baselines."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def ada_gate_residual_baselines(kernel, tensors: dict) -> list[Baseline]:
    from popcorn.backend import IS_METAL

    X = tensors["X"]
    Y = tensors["Y"]
    gate = tensors["gate"]
    G = int(gate.shape[0])
    M = int(X.shape[0]) // G

    if IS_METAL:
        import mlx.core as mx

        def _mx():
            g_bm = mx.repeat(gate, M, axis=0)
            out = X + g_bm * Y
            mx.eval(out)

        return [Baseline("mx.ada_gate_residual", _mx)]

    def _torch():
        g_bm = gate.repeat_interleave(M, dim=0)
        _ = X + g_bm * Y

    return [Baseline("torch.ada_gate_residual", _torch)]
