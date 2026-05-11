"""``quark.functional.owl_attn`` — segment-sparse flash attention.

Signature:

    out = qf.owl_attn(Q, K_cache, Vt_cache, cos, sin, segments, n_segments,
                       *, B, n_kv_heads, gqa_ratio,
                       H_spatial, W_spatial, num_buckets, pinned_dilation,
                       out_dtype=None, compute_dtype=None, max_segments=3)

The seven tensor inputs match the kernel's ``TENSORS`` declaration order;
the allocated output is shape ``[B * n_q_heads * tpf, Dh]`` in ``out_dtype``.
"""

from __future__ import annotations

from quark.functional._dispatch import call_with_bindings, make_autotune
from quark.kernels import get

_OwlAttnCls = None


def _cls():
    global _OwlAttnCls
    if _OwlAttnCls is None:
        _OwlAttnCls = get("owl_attn")
    return _OwlAttnCls


_DUMMY_FRAME_T = None


# ────────────────────────────────────────────────────────────────────────────
# Per-shape ``(WM, WN)`` picker for the NAX multi-simdgroup attention path.
#
# Mirrors ``functional.gemm._NAX_PER_SHAPE`` — each entry is a known shape
# with a sweep-validated (WM, WN) pair; unknown shapes fall back to the safe
# single-simdgroup default. Populate via ``scripts/sweep_attn_simdgroups.py``
# when adding a new shape; the standard quark autotune machinery doesn't
# cover this path because ``kernels/owl_attn/nax.py`` is a stand-alone
# IR-emitted module that bypasses the ``Kernel/Spec/Config`` framework
# (see the docstring at the top of that file). Wiring NAX into the framework
# autotune is the long-term cleanup; until then this hand-coded table plus
# the env-var override ``QUARK_NAX_ATTN_WM`` / ``QUARK_NAX_ATTN_WN`` is how
# we steer the simdgroup count.
#
# Key: ``(tpf, n_q_heads, n_kv_heads, num_buckets)`` — the spec axes that
# determine the per-frame work and KV cache shape. Value: ``(WM, WN)``.
# ────────────────────────────────────────────────────────────────────────────

_NAX_ATTN_PER_SHAPE: dict[tuple[int, int, int, int], tuple[int, int]] = {
    # Waypoint-1.5-1B 720P (M5 Max, ``scripts/sweep_attn_simdgroups.py``
    # 2026-05-05). Sweep over n_simdgroups ∈ {1, 2, 4, 8, 16} found
    # n_simd=2 wins; clean 4-pair A/B vs n_simd=4 confirms (245.1 vs
    # 247.0 ms median saturated, 4.10 vs 4.06 LFPS, consistent across
    # all 4 runs). Vs n_simd=1 baseline (clean cool-SoC pair): -5 to
    # -7 ms saturated, +2-3% LFPS. Larger n_simd values show diminishing
    # returns past 2 — register pressure rises faster than the L2 dedupe
    # of K/V gmem reads pays off at 64 threads/TG, which is the sweet
    # spot here.
    (512, 32, 16, 16): (2, 1),
    # Waypoint-1.5-1B 360P (same sweep). Multi-simd doesn't beat the
    # single-simd path — sweep median sat is 63.3 ms (n=1), 65.3 (n=2),
    # 64.5 (n=4), 63.6 (n=8); all variants within ±2 ms which is at
    # the noise floor for this resolution. Keep ``(1, 1)`` until a
    # bigger win shows up — the 360p attention working set is small
    # enough that L2 dedupe and launch-overhead reduction don't have
    # enough headroom to matter.
    (128, 32, 16, 16): (1, 1),
}


def _pick_attn_config(
    *, tpf: int, n_q_heads: int, n_kv_heads: int, num_buckets: int
) -> tuple[int, int]:
    """Look up the sweep-validated ``(WM, WN)`` for this attention shape.

    Returns ``(1, 1)`` (legacy single-simdgroup) for any shape not in
    the table — never regress an unknown caller. The dispatch site
    additionally falls back to ``(1, 1)`` if ``tpf`` doesn't divide
    ``16 * WM * WN``, so callers with small ``tpf`` (e.g. controller
    GEMMs, owl_attn tests) get the safe path automatically.
    """
    return _NAX_ATTN_PER_SHAPE.get((tpf, n_q_heads, n_kv_heads, num_buckets), (1, 1))


def _owl_attn_impl(
    Q,
    K_cache,
    Vt_cache,
    segments,
    n_segments,
    *,
    B,
    n_kv_heads,
    gqa_ratio,
    H_spatial,
    W_spatial,
    num_buckets,
    pinned_dilation,
    out_dtype=None,
    compute_dtype=None,
    max_segments=3,
    packed_qkv=False,
    quilt_factor=1,
    quilt_offset=0,
    frame_t=None,
    out=None,
):
    cls = _cls()
    spec = cls.spec_from_tensors(
        Q,
        K_cache,
        Vt_cache,
        segments,
        n_segments,
        B=B,
        n_kv_heads=n_kv_heads,
        gqa_ratio=gqa_ratio,
        H_spatial=H_spatial,
        W_spatial=W_spatial,
        num_buckets=num_buckets,
        pinned_dilation=pinned_dilation,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        max_segments=max_segments,
        packed_qkv=packed_qkv,
        quilt_factor=quilt_factor,
        quilt_offset=quilt_offset,
    )
    if frame_t is None:
        # Cache a process-wide zero sentinel so callers that don't pass
        # an explicit frame_t don't pay a ``cuMemAllocAsync`` per call.
        global _DUMMY_FRAME_T
        if _DUMMY_FRAME_T is None:
            from quark.nn.module import _tensor

            _DUMMY_FRAME_T = _tensor([0], dtype="s32")
        frame_t = _DUMMY_FRAME_T
    provided = {
        "Q": Q,
        "K_cache": K_cache,
        "Vt_cache": Vt_cache,
        "segments": segments,
        "n_segments": n_segments,
        "frame_t": frame_t,
    }
    auto_alloc: tuple[str, ...] = ("output",)
    if out is not None:
        provided["output"] = out
        auto_alloc = ()
    result = call_with_bindings(cls, spec, provided=provided, auto_alloc=auto_alloc, like=Q)
    return result["output"]


def _try_nax_owl_attn(
    Q,
    K_cache,
    Vt_cache,
    segments,
    n_segments,
    *,
    B,
    n_kv_heads,
    gqa_ratio,
    H_spatial,
    W_spatial,
    num_buckets,
    pinned_dilation,
    max_segments,
    frame_t,
    packed_qkv: bool = False,
):
    """NAX fast path for owl_attn on M5+ Metal.

    Returns the output ``QuarkTensor`` on hit, or ``None`` to fall
    through to the framework path. The actual kernel + dispatch live
    in ``kernels/owl_attn/nax.py`` — this function only does the gate
    checks, layout detection, and ``BK`` autoselection that depend on
    the live input tensors (and so can't be baked into a spec).
    """
    import os
    import sys

    if sys.platform != "darwin":
        return None
    if os.environ.get("QUARK_DISABLE_NAX") == "1":
        return None

    from quark.device import current_device

    if not current_device().caps.supports_nax:
        return None

    from quark.ir import DType

    if DType.from_backend(Q) is not DType.BF16:
        return None

    tpf = H_spatial * W_spatial
    n_q_heads = n_kv_heads * gqa_ratio
    if packed_qkv:
        # Packed: last dim is (n_q + 2*n_kv) * Dh. Recover per-head Dh.
        packed_per_token = int(Q.shape[-1])
        denom = n_q_heads + 2 * n_kv_heads
        if packed_per_token % denom != 0:
            return None
        Dh = packed_per_token // denom
    else:
        Dh = int(Q.shape[-1])
    capacity = num_buckets * tpf + tpf

    # The IR-emitted NAX attention path (``kernels/owl_attn/nax.py``)
    # supports inline Q-RoPE on the packed-QKV ``token_first`` layout
    # — the kernel reads each (q_tile, q_head) tile out of the packed
    # buffer, RoPE-rotates in fp32, stages to smem, and runs GEMM1
    # against that smem region. That eliminates the standalone
    # ``slice_q_from_packed`` + ``apply_q_rope`` pre-pass kernels
    # (or their fused ``slice_and_rope_q`` variant) the caller used to
    # run before every dispatch. For non-packed callers (Q already a
    # ``[n_q*tpf, Dh]`` head_first buffer, used by unit tests) the
    # pre-pass branch is bypassed and the kernel runs against the
    # caller's already-RoPE'd Q.

    from quark.kernels.owl_attn.nax import BQ, NaxAttnSpec, dispatch_nax_attn

    if tpf % BQ != 0 or Dh != 64:
        return None

    # Layout detection — the IR-emitted kernel supports three call
    # shapes:
    #   ``head_first`` : Q.shape == (n_q_heads * tpf, Dh) — pre-RoPE'd.
    #   ``token_first``: Q.shape == (tpf, n_q_heads, Dh) — reshape to
    #                    (tpf, n_q_heads * Dh) before dispatch.
    #   ``packed``     : Q.shape == (tpf, (n_q+2*n_kv) * Dh), the
    #                    packed-QKV buffer; routed through inline-RoPE
    #                    (token_first output layout).
    inline_q_rope = False
    if packed_qkv and H_spatial is not None and W_spatial is not None:
        # Packed QKV → inline-RoPE. The kernel reads Q directly.
        layout = "token_first"
        inline_q_rope = True
    else:
        q_shape = tuple(int(s) for s in Q.shape)
        if len(q_shape) >= 2 and q_shape[0] == n_q_heads * tpf:
            layout = "head_first"
        elif len(q_shape) == 3 and q_shape[0] == tpf:
            layout = "token_first"
            Q = Q.reshape(tpf, n_q_heads * Dh)
        elif len(q_shape) == 2 and q_shape == (tpf, n_q_heads * Dh):
            # 2D ``(tpf, n_q*Dh)`` already in token_first layout.
            layout = "token_first"
        else:
            return None

    # BK is pinned at 32. The kernel's K-loop is fixed at 32 kv_pos
    # per chunk (k_frag0 reads rows 0..16, k_frag1 reads rows 16..32);
    # raising BK doesn't extend that coverage, it just makes the kernel
    # *skip* kv_pos 32..BK of every chunk. A safer BK>32 path would
    # need a second K-loop over BK / 32 sub-blocks — out of scope here.
    BK = 32
    if (num_buckets * tpf) % BK != 0:
        return None

    # Multi-simdgroup attention: each threadgroup hosts ``WM*WN``
    # simdgroups, each handling its own 16-row Q slice. Cuts threadgroup
    # launches by ``WM*WN`` and lets Apple's L2 dedupe redundant K/V
    # gmem reads (all simdgroups in one TG hit identical KV positions
    # in lockstep). ``_pick_attn_config`` looks the (WM, WN) up in a
    # per-shape table populated from sweeps; unknown shapes get the
    # safe ``WM=WN=1`` default. Env vars ``QUARK_NAX_ATTN_WM`` /
    # ``QUARK_NAX_ATTN_WN`` override the table for debugging / sweep.
    try:
        env_wm = os.environ.get("QUARK_NAX_ATTN_WM")
        env_wn = os.environ.get("QUARK_NAX_ATTN_WN")
        if env_wm is not None or env_wn is not None:
            wm = int(env_wm) if env_wm is not None else 1
            wn = int(env_wn) if env_wn is not None else 1
        else:
            wm, wn = _pick_attn_config(
                tpf=tpf,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                num_buckets=num_buckets,
            )
    except ValueError:
        wm = wn = 1
    if tpf % (16 * wm * wn) != 0:
        wm = wn = 1

    spec = NaxAttnSpec(
        BK=BK,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        gqa_ratio=gqa_ratio,
        tpf=tpf,
        capacity=capacity,
        Dh=Dh,
        max_segments=max_segments,
        layout=layout,
        inline_q_rope=inline_q_rope,
        H_spatial=H_spatial if inline_q_rope else 0,
        W_spatial=W_spatial if inline_q_rope else 0,
        WM=wm,
        WN=wn,
    )
    out = dispatch_nax_attn(
        spec=spec,
        Q=Q,
        K_cache=K_cache,
        Vt_cache=Vt_cache,
        segments=segments,
        n_segments=n_segments,
        frame_t=frame_t,
    )
    # The dispatch picks the output shape from the spec (head_first
    # gives ``(n_q*tpf, Dh)``; token_first / inline_q_rope both give
    # ``(tpf, n_q*Dh)``) — out_proj's downstream layout expectation
    # already matches, no per-call reshape needed.
    return out


def owl_attn(
    Q,
    K_cache,
    Vt_cache,
    segments,
    n_segments,
    B,
    n_kv_heads,
    gqa_ratio,
    H_spatial,
    W_spatial,
    num_buckets,
    pinned_dilation,
    out_dtype=None,
    compute_dtype=None,
    max_segments=3,
    packed_qkv=False,
    quilt_factor=1,
    quilt_offset=0,
    frame_t=None,
    *,
    out=None,
):
    # NAX fast path on Metal.
    if frame_t is None:
        global _DUMMY_FRAME_T
        if _DUMMY_FRAME_T is None:
            from quark.runtime.tensor import QuarkTensor

            _DUMMY_FRAME_T = QuarkTensor.zeros(1, dtype="s32")
        frame_t = _DUMMY_FRAME_T
    nax = _try_nax_owl_attn(
        Q,
        K_cache,
        Vt_cache,
        segments,
        n_segments,
        B=B,
        n_kv_heads=n_kv_heads,
        gqa_ratio=gqa_ratio,
        H_spatial=H_spatial,
        W_spatial=W_spatial,
        num_buckets=num_buckets,
        pinned_dilation=pinned_dilation,
        max_segments=max_segments,
        frame_t=frame_t,
        packed_qkv=packed_qkv,
    )
    if nax is not None:
        return nax

    return _owl_attn_impl(
        Q,
        K_cache,
        Vt_cache,
        segments,
        n_segments,
        B=B,
        n_kv_heads=n_kv_heads,
        gqa_ratio=gqa_ratio,
        H_spatial=H_spatial,
        W_spatial=W_spatial,
        num_buckets=num_buckets,
        pinned_dilation=pinned_dilation,
        out_dtype=out_dtype,
        compute_dtype=compute_dtype,
        max_segments=max_segments,
        packed_qkv=packed_qkv,
        quilt_factor=quilt_factor,
        quilt_offset=quilt_offset,
        frame_t=frame_t,
        out=out,
    )


owl_attn.autotune = make_autotune(_owl_attn_impl, _cls)  # ty: ignore[unresolved-attribute]
