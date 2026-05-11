"""``quark.functional.moe_router_shared`` — shared-experts routing.

Signature:

    token_ids, slot_weights, counts, work_list = qf.moe_router_shared(
        logits, *, E, top_k, capacity,
        token_ids_out=None, slot_weights_out=None,
        counts_out=None, work_list_out=None,
        cum_probs_out=None, chosen_experts_out=None)

Picks K experts globally by cumulative softmax preference and routes
every token to those same K experts. Buffer layout matches
``qf.moe_router`` so the same ``moe_inproj`` / ``moe_outproj`` calls
can consume the result. Unlike the balanced router, ``work_list`` is
*emitted* here (not precomputed at MoE block init) since its content
depends on which K experts are chosen this call.

``cum_probs`` and ``chosen_experts`` are workspace tensors used for
inter-thread communication; callers don't need to read them, but
``nn.MoE`` keeps them as cached buffers so they aren't reallocated
each call.
"""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune, split_provided_io
from quark.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("moe_router_shared")
    return _Cls


def _impl(
    logits,
    *,
    E,
    top_k,
    capacity,
    token_ids_out=None,
    slot_weights_out=None,
    counts_out=None,
    work_list_out=None,
    cum_probs_out=None,
    chosen_experts_out=None,
):
    cls = _cls()
    spec = cls.spec_from_tensors(logits, E=E, top_k=top_k, capacity=capacity)
    provided, auto_alloc = split_provided_io(
        {"logits": logits},
        {
            "token_ids": token_ids_out,
            "slot_weights": slot_weights_out,
            "counts": counts_out,
            "work_list": work_list_out,
            "cum_probs": cum_probs_out,
            "chosen_experts": chosen_experts_out,
        },
    )
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=logits)
    return (
        result["token_ids"],
        result["slot_weights"],
        result["counts"],
        result["work_list"],
    )


def moe_router_shared(
    logits,
    E,
    top_k,
    capacity,
    token_ids_out=None,
    slot_weights_out=None,
    counts_out=None,
    work_list_out=None,
    cum_probs_out=None,
    chosen_experts_out=None,
):
    return _impl(
        logits,
        E=E,
        top_k=top_k,
        capacity=capacity,
        token_ids_out=token_ids_out,
        slot_weights_out=slot_weights_out,
        counts_out=counts_out,
        work_list_out=work_list_out,
        cum_probs_out=cum_probs_out,
        chosen_experts_out=chosen_experts_out,
    )


moe_router_shared.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
