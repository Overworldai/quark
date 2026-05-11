"""``quark.functional.linear`` — ``y = x @ W^T + bias`` via qf.gemm."""

from __future__ import annotations

from quark.functional.gemm import gemm


def linear(x, weight, *, bias=None, activation=None):
    """Linear projection: ``y = x @ weight.T [+ bias]``.

    Thin wrapper around ``qf.gemm`` with clearer naming for model code.
    ``weight``: ``[out_features, in_features]``. ``bias``: optional ``[out_features]``.
    """
    return gemm(x, weight, bias=bias, activation=activation)
