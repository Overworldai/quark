"""``quark.functional.moe_router_correct`` — purely-correct routing.

Signature:

    token_ids, slot_weights, counts, work_list = pcf.moe_router_correct(
        logits, *, E, top_k, capacity,
        token_ids_out=None, slot_weights_out=None, counts_out=None,
        work_list_out=None, offsets_out=None, token_slot_table_out=None)

Each token routes to its actual top-K experts (no capacity bound, no
substitution). Slots are sorted by expert with each expert's run padded
to BM=32; chunks past the active region are marked with ``expert = -1``
(sentinel) so the inproj/outproj kernels skip them.

``capacity`` must satisfy ``E*capacity >= M*top_k + E*(BM-1)`` and be a
multiple of BM=32. ``nn.MoE(routing='correct')`` picks this for you.

``token_slot_table[M, top_k]`` is the inverse of ``token_ids``: for
each token ``t``, ``token_slot_table[t, k]`` is the slot in the
expert-major output where token ``t``'s k-th expert partial lives.
The downstream ``moe_reduce`` kernel reads it to gather without a
scatter / atomic.
"""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune, split_provided_io
from quark.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("moe_router_correct")
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
    offsets_out=None,
    token_slot_table_out=None,
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
            "offsets": offsets_out,
            "token_slot_table": token_slot_table_out,
        },
    )
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=logits)
    return (
        result["token_ids"],
        result["slot_weights"],
        result["counts"],
        result["work_list"],
        result["token_slot_table"],
    )


def moe_router_correct(
    logits,
    E,
    top_k,
    capacity,
    token_ids_out=None,
    slot_weights_out=None,
    counts_out=None,
    work_list_out=None,
    offsets_out=None,
    token_slot_table_out=None,
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
        offsets_out=offsets_out,
        token_slot_table_out=token_slot_table_out,
    )


moe_router_correct.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
