"""``popcorn.functional.moe_outproj`` — MoE atomic-scatter out-projection.

Signature:

    out = pcf.moe_outproj(h_in, W_out, token_ids, slot_weights, work_list,
                          *, M, n_experts, top_k=2,
                          out_dtype="bf16", compute_dtype=None)

h_in: ``[M*top_k, H]``. W_out: ``[n_experts * D, H]``. Returns ``[M, D]``
in f32 (the kernel output is always f32; caller casts as needed).
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
        _Cls = get("moe_outproj")
    return _Cls


def _impl(
    h_in,
    W_out,
    token_ids,
    slot_weights,
    work_list,
    *,
    M,
    n_experts,
    top_k=2,
    out_dtype="bf16",
    compute_dtype=None,
):
    cls = _cls()
    spec = cls.spec_from_tensors(
        h_in,
        W_out,
        token_ids,
        slot_weights,
        work_list,
        M=M,
        n_experts=n_experts,
        top_k=top_k,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
    )
    result = call_with_bindings(
        cls,
        spec,
        provided={
            "h_in": h_in,
            "W_out": W_out,
            "token_ids": token_ids,
            "slot_weights": slot_weights,
            "work_list": work_list,
        },
        auto_alloc=("output",),
        like=h_in,
    )
    return result["output"]


@torch_op("popcorn::moe_outproj", mutates_args=())
def moe_outproj(
    h_in: torch.Tensor,
    W_out: torch.Tensor,
    token_ids: torch.Tensor,
    slot_weights: torch.Tensor,
    work_list: torch.Tensor,
    M: int,
    n_experts: int,
    top_k: int = 2,
    out_dtype: str = "bf16",
    compute_dtype: Optional[str] = None,
) -> torch.Tensor:
    return _impl(
        h_in,
        W_out,
        token_ids,
        slot_weights,
        work_list,
        M=M,
        n_experts=n_experts,
        top_k=top_k,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
    )


@moe_outproj.register_fake
def _(
    h_in,
    W_out,
    token_ids,
    slot_weights,
    work_list,
    M,
    n_experts,
    top_k=2,
    out_dtype="bf16",
    compute_dtype=None,
):
    import torch

    cls = _cls()
    spec = cls.spec_from_tensors(
        h_in,
        W_out,
        token_ids,
        slot_weights,
        work_list,
        M=M,
        n_experts=n_experts,
        top_k=top_k,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
    )
    cfg = cls.CONFIG_CLS.default_for(spec)
    out_decl = next(d for d in cls.TENSORS if d.name == "output")
    shape = out_decl.shape(spec, cfg)
    dtype = PT.ir_dtype_to_backend(out_decl.dtype(spec, cfg))
    return torch.empty(tuple(shape), dtype=dtype, device=h_in.device)


@moe_outproj.register_autograd
def _(ctx, grad_out):
    raise NotImplementedError("popcorn.functional.moe_outproj: backward not implemented.")


moe_outproj.autotune = make_autotune(_impl, _cls)
