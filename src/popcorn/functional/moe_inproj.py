"""``popcorn.functional.moe_inproj`` — MoE gather + SiLU in-projection.

Signature:

    H_out = pcf.moe_inproj(X, W_in, token_ids, work_list,
                           *, n_experts, top_k=2,
                           out_dtype=None, compute_dtype=None)

X: ``[M, D]``. W_in: ``[n_experts * H, D]``. token_ids: ``[M*top_k]``.
work_list: ``[total_slots // BM * 2]``. Returns H_out: ``[M*top_k, H]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from popcorn.backend import PT
from popcorn.functional._dispatch import call_with_bindings, make_autotune, torch_op
from popcorn.kernels import get

if TYPE_CHECKING:
    import torch

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


@torch_op("popcorn::moe_inproj", mutates_args=())
def moe_inproj(
    X: torch.Tensor,
    W_in: torch.Tensor,
    token_ids: torch.Tensor,
    work_list: torch.Tensor,
    n_experts: int,
    top_k: int = 2,
    out_dtype: Optional[str] = None,
    compute_dtype: Optional[str] = None,
) -> torch.Tensor:
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


@moe_inproj.register_fake
def _(X, W_in, token_ids, work_list, n_experts, top_k=2, out_dtype=None, compute_dtype=None):
    import torch

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
    cfg = cls.CONFIG_CLS.default_for(spec)
    out_decl = next(d for d in cls.TENSORS if d.name == "H_out")
    shape = out_decl.shape(spec, cfg)
    dtype = PT.ir_dtype_to_backend(out_decl.dtype(spec, cfg))
    return torch.empty(tuple(shape), dtype=dtype, device=X.device)


@moe_inproj.register_autograd
def _(ctx, grad_out):
    raise NotImplementedError("popcorn.functional.moe_inproj: backward not implemented.")


moe_inproj.autotune = make_autotune(_impl, _cls)
