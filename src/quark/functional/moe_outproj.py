"""``quark.functional.moe_outproj`` — MoE per-slot partial out-projection.

Signature:

    partials = pcf.moe_outproj(h_in, W_out, work_list,
                               *, M, n_experts, top_k=2,
                               out_dtype="bf16", compute_dtype=None,
                               out=None)

h_in: ``[total_slots, H]``. W_out: ``[n_experts * D, H]``. Returns
``[total_slots, D]`` in ``out_dtype`` (bf16 by default) — one row per
slot, ``partials[s, :] = h_in[s] @ W_out[expert(s)].T`` for slots in
``work_list`` whose expert >= 0; sentinel chunks are skipped.

The per-token sum (``slot_weights[s] * partials[s, :]`` summed over
each token's top_k slots) is finished by ``pcf.moe_reduce``, gathering
via the router's ``token_slot_table``.

``out``: optional pre-allocated ``[total_slots, D]`` output buffer.
``nn.MoE`` reuses a cached buffer across calls.
"""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

_Cls = None


def _cls():
    global _Cls
    if _Cls is None:
        _Cls = get("moe_outproj")
    return _Cls


def _impl(
    h_in,
    W_out,
    work_list,
    *,
    M,
    n_experts,
    top_k=2,
    out_dtype="bf16",
    compute_dtype=None,
    out=None,
):
    cls = _cls()
    spec = cls.spec_from_tensors(
        h_in,
        W_out,
        work_list,
        M=M,
        n_experts=n_experts,
        top_k=top_k,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
    )
    provided: dict = {"h_in": h_in, "W_out": W_out, "work_list": work_list}
    auto_alloc: tuple[str, ...] = ()
    if out is not None:
        provided["partials"] = out
    else:
        auto_alloc = ("partials",)
    result = call_with_bindings(
        cls,
        spec,
        provided=provided,
        auto_alloc=auto_alloc,
        like=h_in,
    )
    return result["partials"]


def moe_outproj(
    h_in,
    W_out,
    work_list,
    M,
    n_experts,
    top_k=2,
    out_dtype="bf16",
    compute_dtype=None,
    out=None,
):
    return _impl(
        h_in,
        W_out,
        work_list,
        M=M,
        n_experts=n_experts,
        top_k=top_k,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        out=out,
    )


moe_outproj.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
