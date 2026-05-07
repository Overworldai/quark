"""``quark.functional.moe_reduce`` — gather + weighted sum.

Signature:

    out = pcf.moe_reduce(partials, slot_weights, token_slot_table,
                         *, n_experts, out_dtype="bf16", out=None)

``partials[total_slots, D]``: per-slot bf16 partial outputs from
``pcf.moe_outproj``. ``slot_weights[total_slots]``: f32 per-slot
softmax weights from the router. ``token_slot_table[M, top_k]``: s32
inverse-index from the router that maps each token to its top_k slots.

Returns ``[M, D]`` in ``out_dtype``. ``out=`` reuses a cached buffer
when provided.
"""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("moe_reduce")
    return _Cls


def _impl(
    partials,
    slot_weights,
    token_slot_table,
    *,
    n_experts,
    out_dtype="bf16",
    out=None,
):
    cls = _cls()
    spec = cls.spec_from_tensors(
        partials,
        slot_weights,
        token_slot_table,
        n_experts=n_experts,
        out_dtype=out_dtype,
    )
    provided: dict = {
        "partials": partials,
        "slot_weights": slot_weights,
        "token_slot_table": token_slot_table,
    }
    auto_alloc: tuple[str, ...] = ()
    if out is not None:
        provided["out"] = out
    else:
        auto_alloc = ("out",)
    result = call_with_bindings(
        cls,
        spec,
        provided=provided,
        auto_alloc=auto_alloc,
        like=partials,
    )
    return result["out"]


def moe_reduce(
    partials,
    slot_weights,
    token_slot_table,
    *,
    n_experts,
    out_dtype="bf16",
    out=None,
):
    return _impl(
        partials,
        slot_weights,
        token_slot_table,
        n_experts=n_experts,
        out_dtype=out_dtype,
        out=out,
    )


moe_reduce.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
