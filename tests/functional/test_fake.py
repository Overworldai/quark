"""``torch.library.custom_op`` fake-fn shape/dtype/device sanity.

Runs the meta-device fake fn for every registered op and asserts the
output shape matches what ``kernel.make_tensors`` would produce.
Skipped on Metal — the torch custom ops aren't registered there.
"""

from __future__ import annotations

import pytest

from popcorn.backend import IS_METAL

if IS_METAL:
    pytest.skip("torch custom ops are CUDA-only", allow_module_level=True)

import torch
import torch.library

if not torch.cuda.is_available():
    pytest.skip("no CUDA device available", allow_module_level=True)

import popcorn.functional  # noqa: F401 — triggers registrations
from popcorn.kernels import get


def _fake_call(op_name, *args, **kwargs):
    """Invoke a custom op's fake impl on the meta device.

    Wraps ``torch.library.opcheck`` pattern: run the op under
    ``FakeTensorMode`` and return the meta result.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    op = getattr(torch.ops.popcorn, op_name)
    with FakeTensorMode():
        meta_args = tuple(a.to("meta") if isinstance(a, torch.Tensor) else a for a in args)
        return op(*meta_args, **kwargs)


def _to_meta(t):
    return t.to("meta") if isinstance(t, torch.Tensor) else t


def test_gemm_fake():
    cls = get("gemm")
    p = cls.problems()[0]
    tensors = cls.make_tensors(p.params)
    spec = cls.SPEC_CLS(**p.params)
    out = _fake_call(
        "gemm",
        tensors["A"],
        tensors["B"],
        spec.out_dtype,
        spec.compute_dtype,
        spec.b_shuffle,
    )
    assert tuple(out.shape) == tuple(tensors["Out"].shape)


def test_attention_fake():
    cls = get("attn")
    p = cls.problems()[0]
    tensors = cls.make_tensors(p.params)
    out = _fake_call(
        "attention",
        tensors["Q"],
        tensors["K"],
        tensors["V_t"],
        p.params["B"],
        p.params["n_kv_heads"],
        p.params["gqa_ratio"],
        p.params["seq_len"],
        p.params["kv_len"],
    )
    assert tuple(out.shape) == tuple(tensors["output"].shape)


def test_moe_inproj_fake():
    cls = get("moe_inproj")
    p = cls.problems()[0]
    tensors = cls.make_tensors(p.params)
    spec = cls.SPEC_CLS(**p.params)
    out = _fake_call(
        "moe_inproj",
        tensors["X"],
        tensors["W_in"],
        tensors["token_ids"],
        tensors["work_list"],
        spec.n_experts,
        spec.top_k,
        spec.out_dtype,
        spec.compute_dtype,
    )
    assert tuple(out.shape) == tuple(tensors["H_out"].shape)


def test_moe_outproj_fake():
    cls = get("moe_outproj")
    p = cls.problems()[0]
    tensors = cls.make_tensors(p.params)
    spec = cls.SPEC_CLS(**p.params)
    out = _fake_call(
        "moe_outproj",
        tensors["h_in"],
        tensors["W_out"],
        tensors["token_ids"],
        tensors["slot_weights"],
        tensors["work_list"],
        spec.M,
        spec.n_experts,
        spec.top_k,
        spec.out_dtype,
        spec.compute_dtype,
    )
    assert tuple(out.shape) == tuple(tensors["output"].shape)
