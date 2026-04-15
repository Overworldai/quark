"""``torch.compile(fullgraph=True)`` over every popcorn.functional op.

Each test compiles a thin wrapper around a ``pcf.*`` call and asserts
the compiled path's cos_sim matches the eager path. Skipped on Metal
— MPS + torch.compile is not yet a supported combination here.

"""

from __future__ import annotations

import pytest

from popcorn.backend import IS_METAL

if IS_METAL:
    pytest.skip("torch.compile path is CUDA-only; Metal uses eager MLX", allow_module_level=True)

import torch

if not torch.cuda.is_available():
    pytest.skip("test_compile requires a CUDA device", allow_module_level=True)

import popcorn.functional as pcf
from popcorn.correctness import check_correctness
from popcorn.kernels import get


def _skip_fp8(params: dict) -> bool:
    """Some fp8 problems require sm_120 mma; skip on older GPUs."""
    fp8 = {"e4m3", "e5m2"}
    has_fp8 = any(isinstance(v, str) and v in fp8 for v in params.values())
    if not has_fp8:
        return False
    # Check cc; m16n8k16 fp8 needs sm_120, m16n8k32 fp8 needs sm_89.
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) < (8, 9)


def _first_valid_problem(kernel_cls):
    for p in kernel_cls.problems():
        if not _skip_fp8(p.params):
            return p
    return kernel_cls.problems()[0]


def _check(out, ref, kernel_cls, out_dtype_ir):
    cr = check_correctness(
        out,
        ref,
        out_dtype=out_dtype_ir,
        threshold=kernel_cls.correctness_threshold(out_dtype_ir),
    )
    assert cr.passed, f"{kernel_cls.NAME}: cos_sim={cr.cos_sim:.4f} below threshold"


def test_gemm_compile():
    cls = get("gemm")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    spec = cls.SPEC_CLS(**p.params)
    kernel = cls.from_problem(p.params)

    def fn(A, B):
        return pcf.gemm(
            A,
            B,
            out_dtype=spec.out_dtype,
            compute_dtype=spec.compute_dtype,
            b_shuffled=spec.b_shuffle,
        )

    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(tensors["A"], tensors["B"])
    ref = kernel.reference(tensors["A"], tensors["B"])
    _check(out, ref, cls, spec.out_dtype)


def test_attention_compile():
    cls = get("attn")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    kernel = cls.from_problem(p.params)
    spec = cls.SPEC_CLS(**p.params)

    def fn(Q, K, V_t):
        return pcf.attention(
            Q,
            K,
            V_t,
            B=p.params["B"],
            n_kv_heads=p.params["n_kv_heads"],
            gqa_ratio=p.params["gqa_ratio"],
            seq_len=p.params["seq_len"],
            kv_len=p.params["kv_len"],
        )

    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(tensors["Q"], tensors["K"], tensors["V_t"])
    ref = kernel.reference(tensors["Q"], tensors["K"], tensors["V_t"])
    _check(out, ref, cls, spec.a_dtype)


def test_moe_inproj_compile():
    cls = get("moe_inproj")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    kernel = cls.from_problem(p.params)
    spec = kernel.spec

    def fn(X, W_in, token_ids, work_list):
        return pcf.moe_inproj(
            X,
            W_in,
            token_ids,
            work_list,
            n_experts=spec.n_experts,
            top_k=spec.top_k,
            out_dtype=spec.out_dtype,
            compute_dtype=spec.compute_dtype,
        )

    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(tensors["X"], tensors["W_in"], tensors["token_ids"], tensors["work_list"])
    ref = kernel.reference(
        tensors["X"], tensors["W_in"], tensors["token_ids"], tensors["work_list"]
    )
    _check(out, ref, cls, spec.out_dtype)


def test_moe_outproj_compile():
    from popcorn.ir import DType

    cls = get("moe_outproj")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    kernel = cls.from_problem(p.params)
    spec = kernel.spec

    def fn(h_in, W_out, token_ids, slot_weights, work_list):
        return pcf.moe_outproj(
            h_in,
            W_out,
            token_ids,
            slot_weights,
            work_list,
            M=spec.M,
            n_experts=spec.n_experts,
            top_k=spec.top_k,
            out_dtype=spec.out_dtype,
            compute_dtype=spec.compute_dtype,
        )

    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(
        tensors["h_in"],
        tensors["W_out"],
        tensors["token_ids"],
        tensors["slot_weights"],
        tensors["work_list"],
    )
    ref = kernel.reference(
        tensors["h_in"],
        tensors["W_out"],
        tensors["token_ids"],
        tensors["slot_weights"],
        tensors["work_list"],
    )
    _check(out, ref, cls, DType.F32)


def test_owl_attn_compile():
    cls = get("owl_attn")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    kernel = cls.from_problem(p.params)
    spec = kernel.spec

    def fn(Q, K_cache, Vt_cache, cos, sin, segments, n_segments):
        return pcf.owl_attn(
            Q,
            K_cache,
            Vt_cache,
            cos,
            sin,
            segments,
            n_segments,
            B=spec.B,
            n_kv_heads=spec.n_kv_heads,
            gqa_ratio=spec.gqa_ratio,
            H_spatial=spec.H_spatial,
            W_spatial=spec.W_spatial,
            num_buckets=spec.num_buckets,
            pinned_dilation=spec.pinned_dilation,
            out_dtype=spec.out_dtype,
            compute_dtype=spec.compute_dtype,
            max_segments=spec.max_segments,
        )

    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(
        tensors["Q"],
        tensors["K_cache"],
        tensors["Vt_cache"],
        tensors["cos"],
        tensors["sin"],
        tensors["segments"],
        tensors["n_segments"],
    )
    ref = kernel.reference(
        tensors["Q"],
        tensors["K_cache"],
        tensors["Vt_cache"],
        tensors["cos"],
        tensors["sin"],
        tensors["segments"],
        tensors["n_segments"],
    )
    _check(out, ref, cls, spec.out_dtype)


def test_kv_cache_update_compile():
    cls = get("kv_cache_update")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    kernel = cls.from_problem(p.params)
    spec = kernel.spec

    def fn(K, V, cos, sin, frame_t, Vt_cache, segments, n_segments, K_cache):
        return pcf.kv_cache_update(
            K,
            V,
            cos,
            sin,
            frame_t,
            Vt_cache,
            segments,
            n_segments,
            K_cache,
            B=spec.B,
            n_kv_heads=spec.n_kv_heads,
            H_spatial=spec.H_spatial,
            W_spatial=spec.W_spatial,
            num_buckets=spec.num_buckets,
            pinned_dilation=spec.pinned_dilation,
            kv_dtype=spec.kv_dtype,
            max_segments=spec.max_segments,
        )

    compiled = torch.compile(fn, fullgraph=True)
    K_cache_out, _Vt, _segs, _nsegs = compiled(
        tensors["K"],
        tensors["V"],
        tensors["cos"],
        tensors["sin"],
        tensors["frame_t"],
        tensors["Vt_cache"],
        tensors["segments"],
        tensors["n_segments"],
        tensors["K_cache"],
    )
    ref = kernel.reference(
        tensors["K"],
        tensors["V"],
        tensors["cos"],
        tensors["sin"],
        tensors["frame_t"],
        tensors["Vt_cache"],
        tensors["segments"],
        tensors["n_segments"],
    )
    _check(K_cache_out, ref, cls, spec.kv_dtype)
