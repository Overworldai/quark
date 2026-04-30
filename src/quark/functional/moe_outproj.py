"""``quark.functional.moe_outproj`` — MoE atomic-scatter out-projection.

Signature:

    out = pcf.moe_outproj(h_in, W_out, token_ids, slot_weights, work_list,
                          *, M, n_experts, top_k=2,
                          out_dtype="bf16", compute_dtype=None, out=None)

h_in: ``[M*top_k, H]``. W_out: ``[n_experts * D, H]``. Returns ``[M, D]``
in f32 (the kernel output is always f32; caller casts as needed).

``out``: optional pre-allocated f32 output buffer. ``nn.MoE`` reuses a
cached buffer to avoid per-call ``cuMemAllocAsync``. The kernel does
atomic-add into this buffer, so callers reusing it must ``zero_()``
between invocations — auto-alloc gives a fresh zero buffer for free.
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
    token_ids,
    slot_weights,
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
        token_ids,
        slot_weights,
        work_list,
        M=M,
        n_experts=n_experts,
        top_k=top_k,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
    )
    provided: dict = {
        "h_in": h_in,
        "W_out": W_out,
        "token_ids": token_ids,
        "slot_weights": slot_weights,
        "work_list": work_list,
    }
    auto_alloc: tuple[str, ...] = ()
    if out is not None:
        provided["output"] = out
    else:
        auto_alloc = ("output",)
    result = call_with_bindings(
        cls,
        spec,
        provided=provided,
        auto_alloc=auto_alloc,
        like=h_in,
    )
    return result["output"]


def moe_outproj(
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
    out=None,
):
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
        out=out,
    )


moe_outproj.autotune = make_autotune(_impl, _cls)  # ty: ignore[unresolved-attribute]
