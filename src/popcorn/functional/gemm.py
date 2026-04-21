"""``popcorn.functional.gemm`` — torch/MLX-native C = A @ B^T.

Signature:

    C = pcf.gemm(A, B, *, out_dtype=None, compute_dtype=None, b_shuffled=False)

A: [M, K], B: [N, K] (weight, stored transposed so K is fast axis).
Returns C: [M, N] in ``out_dtype`` (defaults to A's dtype).
"""

from __future__ import annotations

from popcorn.functional._dispatch import call_with_bindings, make_autotune
from popcorn.kernels import get

_GemmCls = None


def _cls():
    """Lazy import so module import order doesn't matter."""
    global _GemmCls
    if _GemmCls is None:
        _GemmCls = get("gemm")
    return _GemmCls


def _gemm_impl(
    A,
    B,
    *,
    out_dtype=None,
    compute_dtype=None,
    b_shuffled=False,
    activation=None,
    bias=None,
    out=None,
):
    cls = _cls()
    has_bias = bias is not None
    spec = cls.spec_from_tensors(
        A,
        B,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        b_shuffled=b_shuffled,
        activation=activation,
        has_bias=has_bias,
    )
    provided = {"A": A, "B": B}
    if has_bias:
        provided["Bias"] = bias
    if out is not None:
        provided["Out"] = out
    auto_alloc_names: tuple[str, ...] = ()
    if not has_bias:
        auto_alloc_names = auto_alloc_names + ("Bias",)
    if out is None:
        auto_alloc_names = auto_alloc_names + ("Out",)
    result = call_with_bindings(
        cls,
        spec,
        provided=provided,
        auto_alloc=auto_alloc_names,
        like=A,
    )
    return result["Out"]


def gemm(
    A,
    B,
    out_dtype=None,
    compute_dtype=None,
    b_shuffled=False,
    activation=None,
    bias=None,
    out=None,
):
    """Compute ``C = A @ B.T``.

    ``out``: optional pre-allocated output buffer to write into. When
    provided, skips the auto_alloc path — lets callers (e.g. Linear
    layers) reuse a cached buffer keyed by M and avoid per-call allocs.
    """
    return _gemm_impl(
        A,
        B,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        b_shuffled=b_shuffled,
        activation=activation,
        bias=bias,
        out=out,
    )


gemm.autotune = make_autotune(_gemm_impl, _cls)  # ty: ignore[unresolved-attribute]
