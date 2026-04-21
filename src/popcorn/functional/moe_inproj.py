"""``popcorn.functional.moe_inproj`` — MoE gather + SiLU in-projection.

Signature:

    H_out = pcf.moe_inproj(X, W_in, token_ids, work_list,
                           *, n_experts, top_k=2,
                           out_dtype=None, compute_dtype=None)

X: ``[M, D]``. W_in: ``[n_experts * H, D]``. token_ids: ``[M*top_k]``.
work_list: ``[total_slots // BM * 2]``. Returns H_out: ``[M*top_k, H]``.
"""

from __future__ import annotations

from popcorn.functional._dispatch import call_with_bindings, make_autotune
from popcorn.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("moe_inproj")
    return _Cls


def _impl(X, W_in, token_ids, work_list, *, n_experts, top_k=2, out_dtype=None, compute_dtype=None):
    cls = _cls()
    spec = cls.spec_from_tensors(
        X,
        W_in,
        token_ids,
        work_list,
        n_experts=n_experts,
        top_k=top_k,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
    )
    result = call_with_bindings(
        cls,
        spec,
        provided={"X": X, "W_in": W_in, "token_ids": token_ids, "work_list": work_list},
        auto_alloc=("H_out",),
        like=X,
    )
    return result["H_out"]


def moe_inproj(
    X,
    W_in,
    token_ids,
    work_list,
    n_experts,
    top_k=2,
    out_dtype=None,
    compute_dtype=None,
):
    return _impl(
        X,
        W_in,
        token_ids,
        work_list,
        n_experts=n_experts,
        top_k=top_k,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
    )


moe_inproj.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
