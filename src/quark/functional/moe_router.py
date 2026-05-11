"""``quark.functional.moe_router`` — capacity-bounded top-K dispatch.

Signature:

    token_ids, slot_weights, counts = qf.moe_router(
        logits, *, E, top_k, capacity,
        token_ids_out=None, slot_weights_out=None, counts_out=None)

``logits``: ``[M, E]`` f32. Returns ``(token_ids[E*C], slot_weights[E*C],
counts[E])`` ready to feed straight into ``qf.moe_inproj`` and
``qf.moe_outproj``. The ``work_list`` those kernels also need is
deterministic given (E, C, BM=32) — the caller builds it once at MoE
block init and reuses, so it isn't returned here.

``*_out`` kwargs let callers reuse pre-allocated buffers (the
``nn.MoE`` block does this for all three to avoid per-call allocs in
the hot path). Pass any combination — unset outputs are auto-allocated.
"""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune, split_provided_io
from quark.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("moe_router")
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
):
    cls = _cls()
    spec = cls.spec_from_tensors(logits, E=E, top_k=top_k, capacity=capacity)
    provided, auto_alloc = split_provided_io(
        {"logits": logits},
        {
            "token_ids": token_ids_out,
            "slot_weights": slot_weights_out,
            "counts": counts_out,
        },
    )
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=logits)
    return result["token_ids"], result["slot_weights"], result["counts"]


def moe_router(
    logits,
    E,
    top_k,
    capacity,
    token_ids_out=None,
    slot_weights_out=None,
    counts_out=None,
):
    return _impl(
        logits,
        E=E,
        top_k=top_k,
        capacity=capacity,
        token_ids_out=token_ids_out,
        slot_weights_out=slot_weights_out,
        counts_out=counts_out,
    )


moe_router.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
