"""``popcorn.functional.gemm`` — torch/MLX-native C = A @ B^T.

Signature:

    C = pcf.gemm(A, B, *, out_dtype=None, compute_dtype=None, b_shuffled=False)

A: [M, K], B: [N, K] (weight, stored transposed so K is fast axis).
Returns C: [M, N] in ``out_dtype`` (defaults to A's dtype).

The public ``gemm(...)`` is decorated as a ``torch.library.custom_op``
on CUDA so it's traceable under ``torch.compile``; on Metal the
decorator is a pass-through. Either way callers just do ``pcf.gemm(A,
B)`` — no per-call type dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from popcorn.backend import PT
from popcorn.functional._dispatch import call_with_bindings, make_autotune, torch_op
from popcorn.kernels import get

if TYPE_CHECKING:
    import torch

_GemmCls = None


def _cls():
    """Lazy import so module import order doesn't matter."""
    global _GemmCls
    if _GemmCls is None:
        _GemmCls = get("gemm")
    return _GemmCls


def _gemm_impl(A, B, *, out_dtype=None, compute_dtype=None, b_shuffled=False):
    cls = _cls()
    spec = cls.spec_from_tensors(
        A, B, out_dtype=out_dtype, compute_dtype=compute_dtype, b_shuffled=b_shuffled
    )
    result = call_with_bindings(cls, spec, provided={"A": A, "B": B}, auto_alloc=("Out",), like=A)
    return result["Out"]


@torch_op("popcorn::gemm", mutates_args=())
def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: Optional[str] = None,
    compute_dtype: Optional[str] = None,
    b_shuffled: bool = False,
) -> torch.Tensor:
    return _gemm_impl(A, B, out_dtype=out_dtype, compute_dtype=compute_dtype, b_shuffled=b_shuffled)


@gemm.register_fake
def _(A, B, out_dtype=None, compute_dtype=None, b_shuffled=False):
    import torch

    cls = _cls()
    spec = cls.spec_from_tensors(
        A, B, out_dtype=out_dtype, compute_dtype=compute_dtype, b_shuffled=b_shuffled
    )
    cfg = cls.CONFIG_CLS.default_for(spec)
    out_decl = next(d for d in cls.TENSORS if d.name == "Out")
    shape = out_decl.shape(spec, cfg)
    dtype = PT.ir_dtype_to_backend(out_decl.dtype(spec, cfg))
    return torch.empty(tuple(shape), dtype=dtype, device=A.device)


@gemm.register_autograd
def _(ctx, grad_out):
    raise NotImplementedError(
        "popcorn.functional.gemm: backward not implemented. This kernel is inference-only."
    )


gemm.autotune = make_autotune(_gemm_impl, _cls)
