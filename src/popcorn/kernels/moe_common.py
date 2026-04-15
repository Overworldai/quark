"""Shared MoE kernel helpers — baselines, tensor prep, etc.

Pulled out of the per-kernel ``kernel.py`` files so the ``moe_inproj``
and ``moe_outproj`` kernels can share the flashinfer fused-moe baseline
setup without each one bloating past the 500-line file budget.
"""

from __future__ import annotations

from typing import Any

import torch as _torch

from popcorn.kernels.base import Baseline


def flashinfer_fused_moe_baselines(
    *,
    M: int,
    D: int,
    H: int,
    n_experts: int,
    top_k: int,
    dtype: _torch.dtype,
    W_in_source: _torch.Tensor | None = None,
    W_out_source: _torch.Tensor | None = None,
    x_source: _torch.Tensor | None = None,
) -> list[Baseline]:
    """Build a ``flashinfer/2`` baseline that runs one call of
    ``cutlass_fused_moe``. The call covers the in-proj AND out-proj in
    one fused op, so the result uses ``reported_time_scale=0.5`` to
    report a per-projection time directly comparable to a single
    popcorn MoE kernel's time.

    Returns ``[]`` when flashinfer isn't importable.

    Either side (``W_in_source`` / ``W_out_source`` / ``x_source``) may
    be ``None`` — an untuned random tensor of matching shape stands in.
    That's the common case: ``moe_inproj`` only has W_in, passes
    ``W_out_source=None``; ``moe_outproj`` only has W_out.
    """
    try:
        from flashinfer import ActivationType, cutlass_fused_moe
    except ImportError:
        return []

    def _weight(src: _torch.Tensor | None, shape: tuple[int, ...]) -> _torch.Tensor:
        dst = _torch.randn(shape, device="cuda", dtype=dtype)
        if src is not None:
            dst.copy_(src.view(shape))
        return dst

    W_in = _weight(W_in_source, (n_experts, H, D))
    W_out = _weight(W_out_source, (n_experts, D, H))
    x = (
        x_source[:M].contiguous()
        if x_source is not None
        else _torch.randn(M, D, device="cuda", dtype=dtype)
    )
    expert_fi = (_torch.arange(M * top_k, device="cuda") % n_experts).view(M, top_k).int()
    weights_fi = _torch.rand(M, top_k, device="cuda", dtype=_torch.float32)
    fi_out = _torch.empty(M, D, device="cuda", dtype=dtype)

    def fi_fn() -> Any:
        cutlass_fused_moe(
            input=x,
            token_selected_experts=expert_fi,
            token_final_scales=weights_fi,
            fc1_expert_weights=W_in,
            fc2_expert_weights=W_out,
            output=fi_out,
            output_dtype=dtype,
            quant_scales=[],
            tune_max_num_tokens=M,
            activation_type=ActivationType.Silu,
        )

    return [Baseline("flashinfer/2", fi_fn, reported_time_scale=0.5)]
