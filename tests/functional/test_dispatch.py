"""End-to-end smoke tests for every ``popcorn.functional`` wrapper.

For each kernel we:
1. Pull the first dtype-compatible ``Problem``.
2. Build tensors via ``cls.make_tensors(problem.params)``.
3. Call the corresponding ``pcf.<name>`` wrapper with positional
   tensor args + scalar kwargs from the problem params.
4. Validate cos_sim against ``kernel.reference(*inputs)``.

Mirrors ``tests/kernels/test_smoke.py`` but exercises the functional
surface (spec_from_tensors + AutotuneCache.lookup_or_search + dispatch + allocation)
rather than the raw Launcher path.
"""

from __future__ import annotations

import popcorn.functional as pcf
from popcorn.backend import IS_METAL
from popcorn.correctness import check_correctness
from popcorn.kernels import get


def _skip_fp8_on_metal(params: dict) -> bool:
    if not IS_METAL:
        return False
    fp8 = {"e4m3", "e5m2"}
    return any(isinstance(v, str) and v in fp8 for v in params.values())


def _first_valid_problem(kernel_cls):
    for p in kernel_cls.problems():
        if not _skip_fp8_on_metal(p.params):
            return p
    return kernel_cls.problems()[0]


def _check(out, ref, out_dtype, kernel_cls):
    from popcorn.backend import PT

    out_ir = PT.backend_dtype_to_ir(out.dtype)
    threshold = kernel_cls.correctness_threshold(out_ir)
    cr = check_correctness(out, ref, out_dtype=out_ir, threshold=threshold)
    assert cr.passed, f"{kernel_cls.NAME}: cos_sim={cr.cos_sim:.4f} below threshold"


def test_gemm_smoke():
    cls = get("gemm")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    spec = cls.SPEC_CLS(**p.params)
    kernel = cls.from_problem(p.params)
    out = pcf.gemm(
        tensors["A"],
        tensors["B"],
        out_dtype=spec.out_dtype,
        compute_dtype=spec.compute_dtype,
        b_shuffled=spec.b_shuffle,
    )
    ref = kernel.reference(tensors["A"], tensors["B"])
    _check(out, ref, spec.out_dtype, cls)


def test_attention_smoke():
    cls = get("attn")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    kernel = cls.from_problem(p.params)
    out = pcf.attention(
        tensors["Q"],
        tensors["K"],
        tensors["V_t"],
        B=p.params["B"],
        n_kv_heads=p.params["n_kv_heads"],
        gqa_ratio=p.params["gqa_ratio"],
        seq_len=p.params["seq_len"],
        kv_len=p.params["kv_len"],
    )
    ref = kernel.reference(tensors["Q"], tensors["K"], tensors["V_t"])
    spec = cls.SPEC_CLS(**p.params)
    _check(out, ref, spec.a_dtype, cls)


def test_owl_attn_smoke():
    cls = get("owl_attn")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    kernel = cls.from_problem(p.params)
    spec = kernel.spec
    out = pcf.owl_attn(
        tensors["Q"],
        tensors["K_cache"],
        tensors["Vt_cache"],
        tensors["cos"],
        tensors["sin"],
        tensors["segments"],
        tensors["n_segments"],
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
    ref = kernel.reference(
        tensors["Q"],
        tensors["K_cache"],
        tensors["Vt_cache"],
        tensors["cos"],
        tensors["sin"],
        tensors["segments"],
        tensors["n_segments"],
    )
    _check(out, ref, spec.out_dtype, cls)


def test_moe_inproj_smoke():
    cls = get("moe_inproj")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    kernel = cls.from_problem(p.params)
    spec = kernel.spec
    out = pcf.moe_inproj(
        tensors["X"],
        tensors["W_in"],
        tensors["token_ids"],
        tensors["work_list"],
        n_experts=spec.n_experts,
        top_k=spec.top_k,
        out_dtype=spec.out_dtype,
        compute_dtype=spec.compute_dtype,
    )
    ref = kernel.reference(
        tensors["X"], tensors["W_in"], tensors["token_ids"], tensors["work_list"]
    )
    _check(out, ref, spec.out_dtype, cls)


def test_moe_outproj_smoke():
    cls = get("moe_outproj")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    kernel = cls.from_problem(p.params)
    spec = kernel.spec
    out = pcf.moe_outproj(
        tensors["h_in"],
        tensors["W_out"],
        tensors["token_ids"],
        tensors["slot_weights"],
        tensors["work_list"],
        M=spec.M,
        n_experts=spec.n_experts,
        top_k=spec.top_k,
        out_dtype=spec.out_dtype,
        compute_dtype=spec.compute_dtype,
    )
    ref = kernel.reference(
        tensors["h_in"],
        tensors["W_out"],
        tensors["token_ids"],
        tensors["slot_weights"],
        tensors["work_list"],
    )
    # moe_outproj output is always f32
    from popcorn.ir import DType

    _check(out, ref, DType.F32, cls)


def test_kv_cache_update_smoke():
    cls = get("kv_cache_update")
    p = _first_valid_problem(cls)
    tensors = cls.make_tensors(p.params)
    kernel = cls.from_problem(p.params)
    spec = kernel.spec

    K_cache, Vt_cache, segments, n_segments = pcf.kv_cache_update(
        tensors["K"],
        tensors["V"],
        tensors["cos"],
        tensors["sin"],
        tensors["frame_t"],
        tensors["Vt_cache"],
        tensors["segments"],
        tensors["n_segments"],
        tensors["K_cache"],
        B=spec.B,
        n_kv_heads=spec.n_kv_heads,
        H_spatial=spec.H_spatial,
        W_spatial=spec.W_spatial,
        num_buckets=spec.num_buckets,
        pinned_dilation=spec.pinned_dilation,
        kv_dtype=spec.kv_dtype,
        max_segments=spec.max_segments,
    )

    # The kernel's reference returns only K_cache (OUTPUT_IDX); check that.
    ref_K_cache = kernel.reference(
        tensors["K"],
        tensors["V"],
        tensors["cos"],
        tensors["sin"],
        tensors["frame_t"],
        tensors["Vt_cache"],
        tensors["segments"],
        tensors["n_segments"],
    )
    _check(K_cache, ref_K_cache, spec.kv_dtype, cls)
