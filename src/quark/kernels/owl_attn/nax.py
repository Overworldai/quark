"""NAX flash attention for owl_attn — IR-emitted, M5+ Metal only.

Single-simdgroup BQ=16 / BD=64 flash attention using Apple's
``MetalPerformancePrimitives::matmul2d`` (NAX) m16n32k16 bf16→f32 MMA.
Built from quark IR and lowered through MslLowerer. Compile-time
specialization on a ``NaxAttnSpec`` (BK, head counts, layout,
``inline_q_rope``) keeps the emitted MSL constexpr — no runtime params
struct.

Why this file lives outside the standard ``OwlAttnKernel`` machinery
─────────────────────────────────────────────────────────────────────

The rest of ``kernels/owl_attn/`` follows the usual quark kernel
layout: ``spec.py`` / ``config.py`` / ``kernel.py`` (an
``@kernel``-decorated ``OwlAttnKernel(Kernel)``) / ``reference.py`` /
``problems.py`` / ``baselines.py``. This file is a second, parallel
implementation that bypasses that machinery entirely. The salient
differences:

* **Entry point.** ``OwlAttnKernel`` is a registered Kernel class
  consumed by the framework (autotune sweeps, fuzz, bench-cli). This
  file exposes free functions ``build_nax_attn_module`` /
  ``compile_nax_attn`` / ``dispatch_nax_attn`` and is consumed only by
  ``functional/owl_attn.py``'s Metal fast-path.
* **Spec / config.** ``OwlAttnSpec`` + ``AttnConfig`` define a
  multi-knob autotune space (KvTile, NCW, MTiles, KvPad, …). Here
  ``NaxAttnSpec`` is a hand-rolled dataclass that bakes the few useful
  config knobs (BK, layout, inline_q_rope) directly into the spec —
  no autotune.
* **IR construction.** ``kernel.py`` is built on the high-level DSL —
  ``KernelContext.tensor``, ``Stage.staged``, ``PipelineBody``,
  ``MmaBody``, ``Accumulators``, ``qk.smem_alloc`` — all auto-plumbed
  by the ``@kernel`` decorator's BlockContext. Here we go
  Builder-direct: ``b = Builder("nax_attn"); b.begin_function(...);
  b.smem_alloc(...); b.barrier(...)``. There's even a small
  ``_RopeBctxAdapter`` class to fake just enough of ``BlockContext``
  for ``emit_rope_cos_sin`` to work.
* **Lowering.** Standard kernels go through ``Kernel.compile()`` →
  ``Launcher`` → autotune cache → ``MslLowerer``. Here we run
  ``MslLowerer(current_device().caps).lower_module(module)`` directly
  and call ``_md.compile`` on the resulting source.
* **Dispatch.** Standard kernels dispatch through
  ``Launcher.compile(...).launch(...)`` which handles
  persistent_outs / output handles / scalar packing.
  ``dispatch_nax_attn`` calls ``_md.queue_launch`` with a hardcoded
  ``_PLAN`` and ``_TG``.

The DSL was designed around the CUDA simdgroup_matrix load → smem →
MMA → smem → store flow. NAX ``matmul2d`` is a different shape:
direct cooperative-tensor loads from gmem to registers, a 2D warp
layout (WM × WN) instead of the simdgroup-matrix per-warp N-partition
that ``Accumulators`` assumes, and a ``vec<float, 8>[2]`` per-lane
fragment — the same layout that exposed the latent vec-arith
lowering bug fixed in commit ``95d58df``. Folding NAX into the
framework path means teaching ``Stage`` / ``PipelineBody`` /
``Accumulators`` about all of that, which is a real IR/blocks
refactor.

Wiring
──────

The ``OwlAttnKernel`` framework path is the CUDA implementation; on
Metal it never runs — the fast-path in
``functional.owl_attn._try_nax_owl_attn`` claims the call first when
``caps.supports_nax``, builds a ``NaxAttnSpec``, and dispatches via
``dispatch_nax_attn`` here. ``compile_nax_attn`` is ``lru_cache``'d
on the spec, so the per-call hot path is a dict lookup plus the
``_md.queue_launch`` invocation.

End-state cleanup
─────────────────

The right shape is to delete this file and move the body into
``kernel.py:build_metal()`` next to a ``build()`` for CUDA, the way
``gemm/kernel.py`` does. That requires:

  * Teaching ``Stage`` / ``PipelineBody`` / ``Accumulators`` about
    NAX's WM × WN warp layout and direct-from-gmem fragment loads.
  * Adding a NAX shape entry to ``AttnConfig``'s autotune space (so
    ``BK``, ``layout``, ``inline_q_rope`` get tuned through the
    standard machinery).
  * Folding ``NaxAttnSpec`` back into ``OwlAttnSpec`` (or making the
    NAX-only fields optional on the unified spec).
  * Replacing the hand-rolled ``compile_nax_attn`` /
    ``dispatch_nax_attn`` / ``_PLAN`` / ``_TG`` with a normal
    ``Launcher.compile(...).launch(...)`` call.

That's a multi-day project against the IR / blocks layer, not the
owl_attn surface — left as a follow-up. Until then, the docstring on
``__init__.py`` flags the dual layout so readers don't get
surprised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from quark.drivers import _metal_dispatch as _md
from quark.ir import BufferType, Builder, DType
from quark.ir.mma_registry import _BY_SHAPE_ID
from quark.ir.module import ParamAttrs
from quark.ir.tensor import GlobalTensor
from quark.lower.msl import MslLowerer

# ── Tile constants matching world_engine's optimal config for Dh=64 ──
BQ = 16  # Q rows per threadgroup (single simdgroup → 16 rows)
BD = 64  # head dimension
KU = 16  # NAX fragment unit
_MSL_4_0 = 262144

# NAX cd_offsets for one 16×16 fragment: 2 rows × 4 cols per lane
# (kElemRows=2, kElemCols=4). Used by frag_reduce on the f32 acc.
_NAX_CD_OFFS = tuple((r * 8, c) for r in range(2) for c in range(4))

# NAX shape used by both GEMMs (per-MmaOp transpose_b knob switches the
# matmul2d_descriptor between Q@K^T and S@V).
_NAX_SHAPE_ID = "m16n32k16_nax_bf16"


@dataclass(frozen=True)
class NaxAttnSpec:
    """Compile-time-fixed parameters for the IR-emitted NAX attn kernel.

    Baked in at IR-build time — the emitter inlines these as constants
    so the Apple compiler can const-fold all the loop bounds and stride
    math (originally a runtime ``AttnParams`` struct in the legacy
    hand-written ``drivers/nax_attn.py``, now retired).
    """

    BK: int  # KV chunk size (32, 64, 128 ...)
    n_q_heads: int
    n_kv_heads: int
    gqa_ratio: int
    tpf: int  # tokens per frame = seq_len for the K/V cache
    capacity: int  # = num_buckets * tpf + tpf
    Dh: int = 64
    max_segments: int = 3
    # Q (and output) layout — selects which compile-time stride pair
    # the emitter bakes into the load_matrix / store_matrix address
    # math. ``head_first``: Q is ``[n_q_heads * tpf, Dh]`` (each head's
    # tokens are contiguous). ``token_first``: Q is reshaped from
    # ``[tpf, n_q_heads, Dh]`` to ``[tpf, n_q_heads * Dh]`` — each
    # token's heads are contiguous (the Waypoint forward bench layout).
    layout: str = "head_first"
    # Inline Q-RoPE: when True the kernel reads Q from the packed QKV
    # buffer ``[tpf, (n_q + 2*n_kv)*Dh]`` and applies ortho-RoPE in
    # smem before GEMM1 — eliminates the ``slice_q_from_packed`` +
    # ``apply_q_rope`` pre-pass kernels (or their fused
    # ``slice_and_rope_q`` variant) the caller would otherwise have to
    # run. Currently only supported with ``layout="token_first"``
    # because that's the bench's actual call shape; head_first inline
    # would need a different packed-row stride and we'd want to
    # validate it under fuzz before turning it on. Requires
    # ``H_spatial`` and ``W_spatial`` to be set (the RoPE freq formula
    # needs both to compute spatial position from token index).
    inline_q_rope: bool = False
    H_spatial: int = 0
    W_spatial: int = 0

    # Multi-simdgroup composition. ``n_simdgroups = WM * WN`` simdgroups
    # share one threadgroup; each simdgroup independently processes its
    # own ``BQ=16`` Q-row slice. Mirrors MLX's ``attention_nax`` template
    # parameters (see ``steel/attn/kernels/steel_attention_nax.h``). The
    # default ``WM=WN=1`` is the legacy single-simdgroup path; raising
    # one cuts threadgroup launches by the same factor and lets Apple's
    # L2 dedupe redundant K/V gmem reads across the simdgroups (which
    # all hit the same KV positions in lockstep). Per-simdgroup Q tile
    # remains 16×Dh — the wider BQ effective tile is composed by tiling
    # simdgroups, NOT by widening the per-MMA fragment.
    #
    # Constraints: WM and WN are independent per the MLX convention but
    # we collapse them to a single ``n_simdgroups`` axis here since the
    # owl_attn segment-sparse pattern doesn't need a 2D simdgroup grid
    # (a 1D row partition over BQ_total = n_simdgroups × 16 suffices).
    # Keeping the WM/WN names so the field is recognisable to MLX-fluent
    # readers; pass either as 1 to disable that axis.
    WM: int = 1
    WN: int = 1

    def __post_init__(self):
        if self.Dh != 64:
            raise NotImplementedError(
                f"NaxAttnSpec: Dh={self.Dh} — only 64 is supported (matches BD)"
            )
        if self.layout not in ("head_first", "token_first"):
            raise ValueError(
                f"NaxAttnSpec: layout must be 'head_first' or 'token_first', got {self.layout!r}"
            )
        if self.inline_q_rope:
            if self.layout != "token_first":
                raise NotImplementedError(
                    f"NaxAttnSpec: inline_q_rope requires layout='token_first', got {self.layout!r}"
                )
            if self.H_spatial * self.W_spatial != self.tpf:
                raise ValueError(
                    f"NaxAttnSpec: inline_q_rope needs H_spatial*W_spatial==tpf, "
                    f"got H={self.H_spatial}, W={self.W_spatial}, tpf={self.tpf}"
                )
        if self.WM < 1 or self.WN < 1:
            raise ValueError(f"NaxAttnSpec: WM/WN must be >= 1, got WM={self.WM} WN={self.WN}")
        n_simd = self.WM * self.WN
        if self.tpf % (BQ * n_simd) != 0:
            raise ValueError(
                f"NaxAttnSpec: tpf={self.tpf} must be divisible by "
                f"BQ*WM*WN = {BQ}*{self.WM}*{self.WN} = {BQ * n_simd}"
            )

    @property
    def n_simdgroups(self) -> int:
        return self.WM * self.WN

    @property
    def BQ_total(self) -> int:
        """Total Q rows per threadgroup = ``BQ * n_simdgroups``."""
        return BQ * self.n_simdgroups

    @property
    def TD(self) -> int:
        """Number of K-tiles along Dh in GEMM1 (Dh / KU)."""
        return self.Dh // KU

    @property
    def TK(self) -> int:
        """Number of K-tiles along BK in GEMM2 (BK / KU)."""
        return self.BK // KU

    @property
    def scale_log2e(self) -> float:
        """Softmax scale (1/sqrt(Dh)) × log2(e) — pre-multiplied so the
        per-element exp2((x - m) * scale_log2e) is one mul + one exp2."""
        return (1.0 / math.sqrt(self.Dh)) * math.log2(math.e)


# ---------------------------------------------------------------------------
# Module builder — emits the IR for one (NaxAttnSpec)-specialized kernel.
# ---------------------------------------------------------------------------


def build_nax_attn_module(spec: NaxAttnSpec):
    """Build the IR module for the NAX attention kernel.

    Returns the populated ``Module`` ready for ``MslLowerer.lower_module``.

    Parameter order matches the dispatch plan in ``compile_and_dispatch``
    below — Q, K_cache, Vt_cache, segments, n_segments, frame_t, output.
    """
    b = Builder("nax_attn")
    b.register_shape(_BY_SHAPE_ID[_NAX_SHAPE_ID].shape)

    fn = b.begin_function("nax_owl_attn")

    # ── Param declarations. Q/output shape depends on layout:
    #    head_first  → ``[n_q_heads * tpf, Dh]`` — each head's tokens
    #                  are contiguous (default for unit tests).
    #    token_first → ``[tpf, n_q_heads * Dh]`` — each token's heads
    #                  are contiguous (Waypoint forward bench layout).
    #
    #    With ``inline_q_rope=True`` Q is the packed QKV buffer
    #    ``[tpf, (n_q + 2*n_kv) * Dh]``. The Q region still occupies
    #    the first ``n_q*Dh`` columns of each row, so the per-warp
    #    ``q_col_base = q_head * Dh`` math is unchanged; only the row
    #    stride differs and that's encoded in the GlobalTensor's
    #    auto-derived stride.
    #
    #    The 2D shape choice picks ``stride[0]`` automatically. The
    #    lane access math reads that stride, so the same load_matrix
    #    code works for both layouts as long as q_row_base /
    #    q_col_base are derived accordingly (see _emit_kernel_body). ──
    if spec.inline_q_rope:
        # Packed: Q is the start of a packed QKV row. Output is
        # token_first ``[tpf, n_q_heads * Dh]`` (post-attention).
        q_shape = (
            spec.tpf,
            (spec.n_q_heads + 2 * spec.n_kv_heads) * spec.Dh,
        )
        out_shape = (spec.tpf, spec.n_q_heads * spec.Dh)
    elif spec.layout == "head_first":
        q_shape = (spec.n_q_heads * spec.tpf, spec.Dh)
        out_shape = q_shape
    else:  # token_first, sliced (legacy pre-pass path)
        q_shape = (spec.tpf, spec.n_q_heads * spec.Dh)
        out_shape = q_shape
    Q = _declare_tensor(b, fn, "Q", DType.BF16, shape=q_shape, readonly=True)
    K_cache = _declare_tensor(
        b,
        fn,
        "K_cache",
        DType.BF16,
        shape=(spec.n_kv_heads * spec.capacity, spec.Dh),
        readonly=True,
    )
    Vt_cache = _declare_tensor(
        b,
        fn,
        "Vt_cache",
        DType.BF16,
        shape=(spec.n_kv_heads * spec.Dh, spec.capacity),
        readonly=True,
    )
    segments = _declare_tensor(
        b,
        fn,
        "segments",
        DType.S32,
        shape=(spec.max_segments * 2,),
        readonly=True,
    )
    n_segments = _declare_tensor(
        b,
        fn,
        "n_segments",
        DType.S32,
        shape=(1,),
        readonly=True,
    )
    # frame_t is read by the inline Q-RoPE pre-pass when
    # ``spec.inline_q_rope`` is set; otherwise the param stays bound
    # purely to keep the launcher's binding plan stable across the
    # two compile-time variants. Either way it never costs anything
    # at runtime — Apple's compiler drops the unused load.
    frame_t = _declare_tensor(
        b,
        fn,
        "frame_t",
        DType.S32,
        shape=(1,),
        readonly=True,
    )
    # Output mirrors Q's layout (for the inline_q_rope path the output
    # is a separate post-attention buffer ``[tpf, n_q*Dh]`` because the
    # input was the wider packed buffer).
    output = _declare_tensor(
        b,
        fn,
        "output",
        DType.BF16,
        shape=out_shape,
        readonly=False,
    )

    # ── Body ──
    _emit_kernel_body(
        b,
        spec,
        Q=Q,
        K_cache=K_cache,
        Vt_cache=Vt_cache,
        segments=segments,
        n_segments=n_segments,
        frame_t=frame_t,
        output=output,
    )

    b.end_function()
    return b.module


def _declare_tensor(
    b: Builder,
    fn,
    name: str,
    dtype: DType,
    *,
    shape: tuple[int, ...],
    readonly: bool,
) -> GlobalTensor:
    """Declare a kernel parameter and wrap it in a GlobalTensor view.

    Mirrors blocks/dsl/kernel_context.py's ``KernelContext.tensor`` —
    inlined here so this module doesn't depend on the high-level DSL.
    """
    b.param(name, BufferType(dtype), attrs=ParamAttrs(readonly=readonly))
    # Row-major stride: stride[i] = product of shape[i+1:].
    stride = tuple(_product(shape[i + 1 :]) for i in range(len(shape)))
    return GlobalTensor(
        dtype=dtype,
        shape=shape,
        stride=stride,
        name=name,
        param=fn.params[len(fn.params) - 1],
    )


def _product(xs) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def _zero_frag(b: Builder, *, width: int, name: str = "zero_frag"):
    """Build a width-N f32 fragment of zeros backed by N *distinct* lane
    locals declared at the *current* IR insertion point (so loop-local
    inits stay loop-local).

    Two pitfalls this navigates:
      1. ``vec_build([b.const(0.0)] * N)`` aliases all N components to
         one CSE'd const — fatal as an MMA C operand because the MMA
         drains overwrite that single lane local N times.
      2. The MSL lowerer's ``_hoist_consts`` pulls *every* ConstOp to
         function scope to avoid C-scoping issues with CSE'd consts
         used in epilogues. So a chunk-loop-local fresh ConstOp gets
         hoisted out and the "fresh zero" becomes a kernel-scope
         persistent variable shared across loop iterations (S no
         longer resets to 0 between chunks).

    The fix: build N distinct *arith* ops (each ``mul(zero, zero)``)
    that the hoister leaves alone. Each ArithOp produces a fresh Value
    bound to a fresh lane local at the emit site, declared inside the
    enclosing loop body. Bypass CSE by constructing ArithOps directly.
    """
    from quark.ir import op as _op
    from quark.ir.types import ValueShape

    zero_const = b.const(DType.F32, 0.0)  # one CSE'd const is fine
    scalars = []
    for i in range(width):
        s = b._fresh(ValueShape(DType.F32), f"{name}_init{i}")
        # mul(0, 0) = 0, but the ArithOp is non-hoistable and bypassing
        # CSE here gives us N distinct ArithOp Values → N distinct lane
        # locals at this insertion point.
        b._emit(
            _op.ArithOp(
                results=(s,),
                operands=(zero_const, zero_const),
                attrs={"kind": "mul"},
            )
        )
        scalars.append(s)
    return b.vec_build(scalars, name=name)


# ---------------------------------------------------------------------------
# Kernel body emitter — IR-emitted segment-sparse flash attention.
# ---------------------------------------------------------------------------


def _emit_kernel_body(
    b: Builder,
    spec: NaxAttnSpec,
    *,
    Q: GlobalTensor,
    K_cache: GlobalTensor,
    Vt_cache: GlobalTensor,
    segments: GlobalTensor,
    n_segments: GlobalTensor,
    frame_t: GlobalTensor,
    output: GlobalTensor,
) -> None:
    """Emit the kernel body for the IR-emitted NAX attention kernel.

    Originally translated from a hand-written MSL reference at
    ``drivers/nax_attn.py:nax_owl_attn`` (now retired). The IR path
    is the production kernel; if you're tracing what this code does
    against a hand-written reference, MLX's
    ``scatter_sdpa_seq_staged_impl`` is the closest equivalent still
    in tree.

    Grid: (n_q_tiles_outer, n_q_heads, 1) where n_q_tiles_outer =
    tpf / (BQ * WM * WN). Each threadgroup covers BQ_total = BQ*WM*WN
    Q rows; within the threadgroup, simdgroup ``s`` (0..WM*WN-1)
    handles rows ``[s*BQ, (s+1)*BQ)``. The whole simdgroup-group reads
    K and V from the same gmem positions on each chunk, so Apple's L2
    dedupes the redundant fetches naturally — same arithmetic, ¼ the
    threadgroup launches at WM*WN=4.
    Block: 32 * WM * WN threads = WM*WN simdgroups
    Per (q_tile_outer, q_head, simd_group_id):
        load Q rows [q_off:q_off+BQ], iterate segments × chunks,
        accumulate O via online softmax, store O. Each simdgroup runs
        the full attention loop on its own Q slice.
    """
    # ── Grid coords + lane / simdgroup ids. ──
    # Drop tgid.z (always 0 for our 2D grid); access tgid.x and tgid.y
    # via Builder.block_idx which lowers to threadgroup_position_in_grid.
    q_tile_outer = b.block_idx("x", name="q_tile_outer")
    q_head = b.block_idx("y", name="q_head")
    _lane = b.lane_id(name="lane")  # used by NAX visitor's coord preamble
    sgid = b.subgroup_id(name="sgid")  # simdgroup_index_in_threadgroup

    # ── Per-(q_tile, q_head, sgid) base offsets. ──
    # The outer threadgroup tile covers BQ_total = BQ * WM * WN rows;
    # each simdgroup picks its own 16-row slice via ``sgid * BQ``.
    # If WM=WN=1 the outer tile collapses to BQ and ``sgid * BQ = 0``,
    # i.e. identical to the legacy single-simdgroup layout.
    BQ_c = b.const(DType.U32, BQ)
    BQ_total_c = b.const(DType.U32, spec.BQ_total)
    q_tile_base = b.mul(q_tile_outer, BQ_total_c, name="q_tile_base")
    sg_q_off = b.mul(sgid, BQ_c, name="sg_q_off")
    q_off = b.add(q_tile_base, sg_q_off, name="q_off")  # row in [0, tpf)
    # kv_head = q_head / gqa_ratio (integer divide).
    gqa_c = b.const(DType.U32, spec.gqa_ratio)
    kv_head = b.div(q_head, gqa_c, name="kv_head")

    # ── Per-head base offsets in K_cache and Vt_cache (in *rows*). ──
    # K_cache[kv_head * capacity + kv_pos, d_idx]
    cap_c = b.const(DType.U32, spec.capacity)
    k_head_base = b.mul(kv_head, cap_c, name="k_head_base")
    # Vt_cache[kv_head * Dh + d_idx, kv_pos]
    Dh_c = b.const(DType.U32, spec.Dh)
    vt_head_base = b.mul(kv_head, Dh_c, name="vt_head_base")
    # Q row/col bases — depend on layout (see NaxAttnSpec.layout):
    #   head_first  → Q is [n_q_heads*tpf, Dh]; row=q_head*tpf+q_off,
    #                 col=0. The lane reads at row+fm, col+fn (16x16).
    #   token_first → Q is [tpf, n_q_heads*Dh]; row=q_off, col=q_head*Dh.
    #                 The lane reads at row+fm (advances along tpf
    #                 axis) and col+fn (within the head's Dh slot).
    if spec.layout == "head_first":
        tpf_c = b.const(DType.U32, spec.tpf)
        q_head_base = b.mul(q_head, tpf_c, name="q_head_base")
        q_row_base = b.add(q_head_base, q_off, name="q_row_base")
        q_col_base = b.const(DType.U32, 0)
    else:  # token_first
        q_row_base = q_off
        q_col_base = b.mul(q_head, Dh_c, name="q_col_base")

    # ── Inline Q-RoPE pre-pass (when ``spec.inline_q_rope``). ──
    # Read the current (q_tile, q_head) tile out of the packed QKV
    # gmem buffer, apply ortho-RoPE, write the rotated halved-layout
    # Q to a 16×Dh smem region. GEMM1 below switches its A-fragment
    # source from gmem ``Q.view(...)`` to that smem region, eliminating
    # the slice + RoPE pre-pass kernels the caller used to run.
    q_view_for_gemm = None
    if spec.inline_q_rope:
        # Pass ``sg_q_off`` only when there's > 1 simdgroup; the
        # single-simd path keeps writing to ``q_smem[qrow, c]`` exactly
        # as before so existing tests stay byte-identical.
        rope_sg_off = sg_q_off if spec.n_simdgroups > 1 else None
        q_view_for_gemm = _emit_q_rope_prepass(
            b,
            spec,
            Q_packed=Q,
            frame_t=frame_t,
            q_off=q_off,
            q_col_base=q_col_base,
            sg_q_off=rope_sg_off,
        )

    # ── Read n_segments[0]. ── Runtime int — drives the segment loop bound.
    zero_u32 = b.const(DType.U32, 0)
    n_seg = b.load(n_segments, zero_u32, name="n_seg")
    n_seg_u32 = b.convert(n_seg, DType.U32, name="n_seg_u32")

    # ── Initialize O accumulator (2 width-16 f32 frags = 16×64 total). ──
    # Distinct lane locals (see _zero_frag for why one CSE'd const
    # would corrupt the MMA drain pattern).
    O_n0 = _zero_frag(b, width=16, name="O_n0")
    O_n1 = _zero_frag(b, width=16, name="O_n1")

    # ── Online softmax state: per-row max and sum. ──
    # 2 row classes per lane (kElemRows=2). Initial max = -inf, sum = 0.
    neg_inf = b.const(DType.F32, -1e30)
    max0 = neg_inf
    max1 = neg_inf
    zero_f = b.const(DType.F32, 0.0)
    sum0 = zero_f
    sum1 = zero_f

    # ── Segment loop: for seg in 0..n_seg ──
    # Loop carries: O_n0, O_n1, max0, max1, sum0, sum1.
    # Bound is dynamic (read from n_segments). Body unrolls the chunk
    # loop based on each segment's runtime length.
    seg_start_init = b.const(DType.U32, 0)
    one_u32 = b.const(DType.U32, 1)

    # NOTE: the for_loop carry mechanism + GEMM1/GEMM2 emission is
    # implemented in _emit_segment_loop (next phase). For this initial
    # checkpoint, drop the body so the module lowers cleanly and the
    # signature/scaffolding can be validated.
    _emit_segment_loop(
        b,
        spec,
        Q=Q,
        K_cache=K_cache,
        Vt_cache=Vt_cache,
        segments=segments,
        output=output,
        q_row_base=q_row_base,
        q_col_base=q_col_base,
        k_head_base=k_head_base,
        vt_head_base=vt_head_base,
        n_seg=n_seg_u32,
        O_init=(O_n0, O_n1),
        sm_init=(max0, max1, sum0, sum1),
        seg_start_init=seg_start_init,
        one_u32=one_u32,
        q_smem_view=q_view_for_gemm,
    )


def _emit_segment_loop(
    b: Builder,
    spec: NaxAttnSpec,
    *,
    Q,
    K_cache,
    Vt_cache,
    segments,
    output,
    q_row_base,
    q_col_base,
    k_head_base,
    vt_head_base,
    n_seg,
    O_init,
    sm_init,
    seg_start_init,
    one_u32,
    q_smem_view=None,
) -> None:
    """Outer segment loop. For each segment, read its (start, length)
    pair, run an inner chunk loop of length // BK iterations, threading
    the (O_n0, O_n1, max0, max1, sum0, sum1) carry through.

    Loop bounds that depend on segment metadata stay as real IR
    for_loops; the K-loops inside GEMM1 / GEMM2 stay Python-unrolled
    (fixed at compile time).
    """
    O_n0, O_n1 = O_init
    max0, max1, sum0, sum1 = sm_init
    BK_c = b.const(DType.U32, spec.BK)
    two_u32 = b.const(DType.U32, 2)

    # ── Carries: (O_n0, O_n1, max0, max1, sum0, sum1). ──
    carries = (O_n0, O_n1, max0, max1, sum0, sum1)
    with b.for_loop(seg_start_init, n_seg, one_u32, iv_name="seg", carried=carries) as (
        seg,
        (O0_in, O1_in, m0_in, m1_in, s0_in, s1_in),
    ):
        # Read this segment's (start, length) pair.
        seg2 = b.mul(seg, two_u32, name="seg2")
        seg2p1 = b.add(seg2, one_u32, name="seg2p1")
        seg_start_s32 = b.load(segments, seg2, name="seg_start")
        seg_len_s32 = b.load(segments, seg2p1, name="seg_len")
        seg_start = b.convert(seg_start_s32, DType.U32)
        seg_len = b.convert(seg_len_s32, DType.U32)
        n_chunks = b.div(seg_len, BK_c, name="n_chunks")

        # ── Inner chunk loop: 0..n_chunks. Same carry shape. ──
        with b.for_loop(
            seg_start_init,
            n_chunks,
            one_u32,
            iv_name="chunk",
            carried=(O0_in, O1_in, m0_in, m1_in, s0_in, s1_in),
        ) as (chunk, (O0_c, O1_c, m0_c, m1_c, s0_c, s1_c)):
            # kv_off = seg_start + chunk * BK. The K_cache view starts
            # at row=k_head_base+kv_off and the Vt_cache view at
            # col=kv_off; both with stride matching the cache layout.
            chunk_BK = b.mul(chunk, BK_c, name="chunk_BK")
            kv_off = b.add(seg_start, chunk_BK, name="kv_off")
            k_row = b.add(k_head_base, kv_off, name="k_row")  # absolute K row

            # ── GEMM1: S = Q @ K^T, accumulated over TD K-tiles. ──
            # ``q_smem_view`` is set when ``inline_q_rope`` is on — the
            # RoPE pre-pass already staged the rotated Q tile into a
            # 16×Dh smem region and we feed that to GEMM1 instead of
            # the gmem Q view.
            S = _emit_gemm1(
                b,
                spec,
                Q=Q,
                K_cache=K_cache,
                q_row_base=q_row_base,
                q_col_base=q_col_base,
                k_row_base=k_row,
                q_smem_view=q_smem_view,
            )

            # ── Softmax + O rescale (online). Returns new (O0, O1, m0, m1,
            #    s0, s1) and the post-softmax S (cast to bf16 implicitly at
            #    GEMM2 input via the visitor). ──
            S_p, m0_n, m1_n, s0_n, s1_n, O0_n, O1_n = _emit_softmax(
                b,
                spec,
                S=S,
                m0_in=m0_c,
                m1_in=m1_c,
                s0_in=s0_c,
                s1_in=s1_c,
                O0_in=O0_c,
                O1_in=O1_c,
            )

            # ── GEMM2: O += S_p @ V. Vt rows are per-head Dh slots,
            #    cols are kv_pos; the per-head row base is vt_head_base. ──
            O0_out, O1_out = _emit_gemm2(
                b,
                spec,
                Vt_cache=Vt_cache,
                vt_row_base=vt_head_base,
                vt_col_base=kv_off,
                S_p=S_p,
                O_n0=O0_n,
                O_n1=O1_n,
            )

            b.yield_(O0_out, O1_out, m0_n, m1_n, s0_n, s1_n)

        # Inner loop's results become the segment loop's per-iter values.
        O0_seg, O1_seg, m0_seg, m1_seg, s0_seg, s1_seg = b.last_results
        b.yield_(O0_seg, O1_seg, m0_seg, m1_seg, s0_seg, s1_seg)

    # Final carries after both loops.
    O0_f, O1_f, _m0, _m1, s0_f, s1_f = b.last_results

    # ── Normalize O by 1/sum, then store to output. ──
    _emit_normalize_and_store(
        b,
        spec,
        output=output,
        q_row_base=q_row_base,
        q_col_base=q_col_base,
        O_n0=O0_f,
        O_n1=O1_f,
        sum0=s0_f,
        sum1=s1_f,
    )


# ---------------------------------------------------------------------------
# Q-RoPE pre-pass: read packed-QKV Q tile → ortho-RoPE in fp32 → write
# halved-layout bf16 to ``q_smem`` (16 × Dh). Eliminates the standalone
# ``slice_q_from_packed`` + ``apply_q_rope`` (or fused
# ``slice_and_rope_q``) kernels that the caller used to run before
# every owl_attn dispatch.
# ---------------------------------------------------------------------------


class _RopeBctxAdapter:
    """Tiny duck-typed adapter so ``emit_rope_cos_sin`` can run inside
    this kernel emitter without a real :class:`BlockContext`. The only
    method it uses on its ``bctx`` arg is ``.c(value, dtype=...)``,
    which forwards straight to ``Builder.const``. Everything else
    inside ``emit_rope_cos_sin`` (``qk.cmp``, ``qk.if_``, ``qk.cos``,
    ``qk.ex2_approx``, ``qk.convert``, ``qk.yield_``,
    ``qk.last_results``) routes through the active-builder ContextVar
    that ``Builder.begin_function`` already publishes for us.
    """

    def __init__(self, bld: Builder):
        self.bld = bld

    def c(self, value, dtype=None):
        if dtype is None:
            if isinstance(value, float):
                dtype = DType.F32
            elif value < 0:
                dtype = DType.S32
            else:
                dtype = DType.U32
        return self.bld.const(dtype, value)


def _emit_q_rope_prepass(
    b: Builder,
    spec: NaxAttnSpec,
    *,
    Q_packed: GlobalTensor,
    frame_t: GlobalTensor,
    q_off,
    q_col_base,
    sg_q_off=None,
):
    """Cooperative load + ortho-RoPE of one (q_tile, q_head) tile of
    Q from the packed QKV gmem buffer into a 16×Dh smem region per
    simdgroup. Each simdgroup independently runs this function on its
    own slice; the smem allocation backs ALL simdgroups in the
    threadgroup (sized BQ_total × Dh) so writes to disjoint row
    ranges never collide.

    Layout in / out (matches the convention ``kv_cache_update`` uses
    on the K side, so the Q·K dot products see consistent rotation):
      * gmem (read, packed, interleaved): ``Q_packed[token, q_head*Dh + 2k]``,
        ``Q_packed[token, q_head*Dh + 2k+1]`` — pair (x0, x1).
      * smem (write, halved): ``q_smem[sg_q_off + token_local, k] = x0*cv - x1*sv``,
        ``q_smem[sg_q_off + token_local, k + Dh/2] = x1*cv + x0*sv``.

    Threading: 32 lanes × ``per_thr`` pairs each cover the full
    BQ × Dh/2 = 16 × 32 = 512 pair grid. ``per_thr = 16`` is the
    natural fit for one 32-thread simdgroup; if BQ or Dh ever change
    we'll need to retile.

    Returns the smem view rooted at this simdgroup's row base so
    GEMM1 can use it as the A-fragment source
    (``load_matrix(q_smem, ..., row=0, col=col_off)``) without
    knowing about ``sg_q_off``.
    """
    from quark.lang.rope import emit_rope_cos_sin

    half_Dh = spec.Dh // 2
    n_pairs = BQ * half_Dh  # 16 * 32 = 512
    n_threads = 32
    if n_pairs % n_threads != 0:
        raise ValueError(f"Q-RoPE pre-pass: pairs={n_pairs} not divisible by n_threads={n_threads}")
    per_thr = n_pairs // n_threads

    # Smem destination — sized for ALL simdgroups in the threadgroup.
    # Each simdgroup writes to its own ``[sg_q_off:sg_q_off+BQ, :]``
    # band; the regions are disjoint so no cross-simdgroup barrier is
    # needed before each simdgroup reads its own slice (a final
    # block-wide barrier still ensures all writes are visible before
    # GEMM1 fires).
    q_smem = b.smem_alloc("Q_rope_smem", DType.BF16, shape=(spec.BQ_total, spec.Dh))

    # Constants used inside the per-thread loop.
    half_Dh_c = b.const(DType.U32, half_Dh)
    W_c = b.const(DType.U32, spec.W_spatial)
    per_thr_c = b.const(DType.U32, per_thr)
    one_u32 = b.const(DType.U32, 1)
    zero_u32 = b.const(DType.U32, 0)
    two_u32 = b.const(DType.U32, 2)

    # Per-lane id (0..31) and the absolute frame index needed by RoPE.
    tid = b.lane_id(name="rope_tid")
    frame_t_s32 = b.load(frame_t, zero_u32, name="frame_t_s")
    frame_t_u = b.convert(frame_t_s32, DType.U32, name="frame_t_u")

    bctx = _RopeBctxAdapter(b)

    # Python-unroll ``per_thr`` iterations per lane. Each iter does
    # two gmem reads (packed pair) + RoPE + two smem writes (halved).
    for i in range(per_thr):
        i_c = b.const(DType.U32, i)
        flat = b.add(b.mul(tid, per_thr_c, name=f"rope_tid_x_pt_{i}"), i_c, name=f"rope_flat_{i}")
        qrow = b.div(flat, half_Dh_c, name=f"rope_qrow_{i}")
        c_idx = b.rem(flat, half_Dh_c, name=f"rope_cidx_{i}")
        c2 = b.mul(c_idx, two_u32, name=f"rope_c2_{i}")
        c2p1 = b.add(c2, one_u32, name=f"rope_c2p1_{i}")

        # Spatial position for this token. ``q_off`` is the per-block
        # token base; ``qrow`` is the row within the BQ tile.
        spatial_idx = b.add(q_off, qrow, name=f"rope_spatial_{i}")
        h_idx = b.div(spatial_idx, W_c, name=f"rope_h_{i}")
        w_idx = b.rem(spatial_idx, W_c, name=f"rope_w_{i}")

        # Packed gmem reads. Q region of each row sits at the first
        # n_q*Dh cols, head h at cols [h*Dh, (h+1)*Dh). The packed-row
        # stride is encoded in Q_packed.stride[0] from the kernel's
        # param shape, so we just hand (row, col) to the LoadOp.
        col_a = b.add(q_col_base, c2, name=f"rope_col_a_{i}")
        col_b = b.add(q_col_base, c2p1, name=f"rope_col_b_{i}")
        x0_bf = b.load(Q_packed, spatial_idx, col_a, name=f"rope_x0_bf_{i}")
        x1_bf = b.load(Q_packed, spatial_idx, col_b, name=f"rope_x1_bf_{i}")
        x0 = b.convert(x0_bf, DType.F32, name=f"rope_x0_{i}")
        x1 = b.convert(x1_bf, DType.F32, name=f"rope_x1_{i}")

        # Inline RoPE freq → cos/sin (computes in fp32).
        cv, sv = emit_rope_cos_sin(
            bctx,
            h_idx=h_idx,
            w_idx=w_idx,
            frame_t=frame_t_u,
            c_idx=c_idx,
            H=spec.H_spatial,
            W=spec.W_spatial,
            Dh=spec.Dh,
        )

        # Rotation. y0 lives at smem col=c_idx, y1 at smem col=c_idx+half_Dh.
        y0 = x0 * cv - x1 * sv
        y1 = x1 * cv + x0 * sv
        y0_bf = b.convert(y0, DType.BF16, name=f"rope_y0_bf_{i}")
        y1_bf = b.convert(y1, DType.BF16, name=f"rope_y1_bf_{i}")

        # Smem subscript expects ``q_smem[row, col] = value``. Shift
        # the destination row by ``sg_q_off`` so each simdgroup writes
        # to its own band of the shared smem region.
        c_idx_dst1 = b.add(c_idx, half_Dh_c, name=f"rope_c_dst1_{i}")
        if sg_q_off is not None:
            qrow_global = b.add(sg_q_off, qrow, name=f"rope_qrow_g_{i}")
        else:
            qrow_global = qrow
        q_smem[qrow_global, c_idx] = y0_bf
        q_smem[qrow_global, c_idx_dst1] = y1_bf

    # Barrier so all simdgroups' writes are visible before the
    # downstream GEMM1 fragment loads. Each simdgroup only reads from
    # its own band (the slice written by its own lanes), so a
    # simdgroup-scope barrier would suffice; using a block-scope
    # barrier here is the same cost on M5+ (one HW barrier instruction)
    # and matches the smem allocation's threadgroup scope.
    b.barrier("block")
    # Return a view rooted at this simdgroup's row base so the GEMM1
    # fragment loads see ``q_smem_view[row=0..BQ, col=...]`` regardless
    # of which simdgroup is reading. SharedRegion.view takes an *element*
    # offset, not (row, col); compute ``sg_q_off * Dh`` so the row shift
    # walks the stride-Dh row-major storage.
    if sg_q_off is not None:
        Dh_c = b.const(DType.U32, spec.Dh)
        elem_off = b.mul(sg_q_off, Dh_c, name="q_smem_elem_off")
        return q_smem.view(dyn_offset=elem_off, shape=(BQ, spec.Dh))
    return q_smem


# ---------------------------------------------------------------------------
# Per-chunk: GEMM1 (Q @ K^T → S, width-16 f32 acc).
# ---------------------------------------------------------------------------


def _emit_gemm1(
    b: Builder,
    spec: NaxAttnSpec,
    *,
    Q: GlobalTensor,
    K_cache: GlobalTensor,
    q_row_base,
    q_col_base,
    k_row_base,
    q_smem_view=None,
):
    """S = Q @ K^T, accumulated over TD K-tiles. Returns the width-16
    f32 S accumulator (one MMA per K-tile, Python-unrolled).

    The first MMA uses ``accumulate=False`` (matmul2d's ``multiply``
    mode) so we don't need to zero-init S — the first MMA writes it
    directly. Subsequent MMAs accumulate into the running S.

    ``q_smem_view``: when set (the ``inline_q_rope`` path), Q-fragment
    loads come from this 16×Dh smem region — already RoPE-rotated by
    the pre-pass — instead of the gmem ``Q.view(...)``. The smem view
    has no per-warp offsets to fold in (one tile per block), so the
    fragment row=0 / col=col_off addressing is identical to the gmem
    case modulo source.
    """
    if q_smem_view is not None:
        Q_warp = q_smem_view
    else:
        # The view's row + col bases are folded into the load address by
        # ``_visit_load_matrix_nax`` — they account for the (q_head, q_off)
        # offset under the chosen layout (head_first vs token_first).
        Q_warp = Q.view(row=q_row_base, col=q_col_base)
    K_warp = K_cache.view(row=k_row_base)

    # ``S`` is undefined before the first MMA; pass a dummy carrier.
    # In ``accumulate=False`` mode the visitor skips the C input copy,
    # so the dummy is never read — but the IR still requires a
    # well-typed Value. Reuse one zero-frag worth of fresh lane locals
    # so the visitor sees 16 distinct names; it never writes to them.
    S = _zero_frag(b, width=16, name="S_init")

    # Python-unroll TD iterations. We tested swapping this for an IR
    # for_loop with a constant bound — same runtime on Apple's compiler
    # (it unrolls either form equivalently). The GEMM kernel ran the
    # same A/B and confirmed the result.
    for id_ in range(spec.TD):
        col_off = b.const(DType.U32, id_ * KU)
        q_frag = b.load_matrix(
            Q_warp,
            _NAX_SHAPE_ID,
            which="a",
            row=0,
            col=col_off,
            name=f"q_frag_{id_}",
        )
        k_frag = b.load_matrix(
            K_warp,
            _NAX_SHAPE_ID,
            which="b",
            row=0,
            col=col_off,
            name=f"k_frag_{id_}",
        )
        S = b.mma(
            _NAX_SHAPE_ID,
            q_frag,
            k_frag,
            S,
            transpose_b=True,
            accumulate=(id_ != 0),  # first MMA writes S, rest accumulate
            name=f"S_{id_}",
        )
    return S


# ---------------------------------------------------------------------------
# Per-chunk: online softmax + O rescale.
# ---------------------------------------------------------------------------


# slot → row class for a width-16 NAX C-frag (kElemRows=2, kElemCols=4,
# stacked twice for c_regs=16).
#   slots 0-3 → class 0 (dr=0)
#   slots 4-7 → class 1 (dr=8)
#   slots 8-11 → class 0 (second sub-frag's first row band)
#   slots 12-15 → class 1
_SLOT_TO_ROW_CLASS = (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1)


def _emit_softmax(b: Builder, spec: NaxAttnSpec, *, S, m0_in, m1_in, s0_in, s1_in, O0_in, O1_in):
    """Online softmax pass over S (width-16 f32 acc). Returns the
    post-exp S (still width-16 f32) plus new (max0, max1, sum0, sum1)
    and rescaled (O0, O1).

    Steps:
      1. Scale S by 1/sqrt(Dh) * log2(e).
      2. New per-row max = max(running_max, row_reduce(S, max)).
      3. S = exp2(S - new_max) per row class.
      4. factor = exp2(running_max - new_max); update running_max.
      5. running_sum = running_sum * factor + row_reduce(S, add).
      6. O = O * factor (row-broadcast over width-16 frags).
    """
    # Step 1: scale S by scale_log2e (per-element constant multiply).
    scale_c = b.const(DType.F32, spec.scale_log2e)
    S_scaled = b.frag_apply(
        _NAX_SHAPE_ID,
        S,
        lambda elem: b.mul(elem, scale_c),
        name="S_scaled",
    )

    # Step 2: per-row max via NAX frag_reduce, then carry-merge.
    new_m0_local, new_m1_local = b.frag_reduce(
        _NAX_SHAPE_ID,
        S_scaled,
        kind="max",
        axis="row",
        cd_offsets=_NAX_CD_OFFS,
        name="new_max",
    )
    new_m0 = b.max(m0_in, new_m0_local, name="new_max0")
    new_m1 = b.max(m1_in, new_m1_local, name="new_max1")

    # Step 3: S = exp2(S - new_max) per row class.
    S_exp = b.frag_apply(
        _NAX_SHAPE_ID,
        S_scaled,
        lambda elem, sel: b.ex2_approx(b.sub(elem, sel)),
        selectors=(new_m0, new_m1),
        slot_to_selector_idx=_SLOT_TO_ROW_CLASS,
        name="S_exp",
    )

    # Step 4: factor = exp2(running_max - new_max). Update running_max.
    factor0 = b.ex2_approx(b.sub(m0_in, new_m0), name="factor0")
    factor1 = b.ex2_approx(b.sub(m1_in, new_m1), name="factor1")

    # Step 5: scaled_sum = running_sum * factor; then add row-reduced S.
    s0_scaled = b.mul(s0_in, factor0, name="sum0_scaled")
    s1_scaled = b.mul(s1_in, factor1, name="sum1_scaled")
    add0, add1 = b.frag_reduce(
        _NAX_SHAPE_ID,
        S_exp,
        kind="add",
        axis="row",
        cd_offsets=_NAX_CD_OFFS,
        name="row_sum",
    )
    new_s0 = b.add(s0_scaled, add0, name="new_sum0")
    new_s1 = b.add(s1_scaled, add1, name="new_sum1")

    # Step 6: O *= factor (row-broadcast across the width-16 O frags).
    O0_scaled = b.frag_apply(
        _NAX_SHAPE_ID,
        O0_in,
        lambda elem, sel: b.mul(elem, sel),
        selectors=(factor0, factor1),
        slot_to_selector_idx=_SLOT_TO_ROW_CLASS,
        name="O0_scaled",
    )
    O1_scaled = b.frag_apply(
        _NAX_SHAPE_ID,
        O1_in,
        lambda elem, sel: b.mul(elem, sel),
        selectors=(factor0, factor1),
        slot_to_selector_idx=_SLOT_TO_ROW_CLASS,
        name="O1_scaled",
    )

    return S_exp, new_m0, new_m1, new_s0, new_s1, O0_scaled, O1_scaled


# ---------------------------------------------------------------------------
# Per-chunk: GEMM2 (O += S_p @ V, two width-16 N-tile accumulators).
# ---------------------------------------------------------------------------


def _emit_gemm2(
    b: Builder,
    spec: NaxAttnSpec,
    *,
    Vt_cache: GlobalTensor,
    vt_row_base,
    vt_col_base,
    S_p,
    O_n0,
    O_n1,
):
    """O[16, 64] += S_p[16, BK] @ V[BK, 64]. Two N-tile MMAs (each
    covering 16×32 of O), TK K-tiles each. Python-unrolled inner loops.

    S_p is width-16 covering 16×BK; sliced into TK width-8 halves
    (one per K-tile). V is loaded from Vt_cache as width-16 per
    (n_tile, k_tile) pair.
    """
    Vt_warp = Vt_cache.view(row=vt_row_base, col=vt_col_base)

    # Slice S_p into TK width-8 halves (one per K-tile of GEMM2).
    S_slices = [
        b.frag_slice(S_p, start=ik * 8, length=8, name=f"S_p_k{ik}") for ik in range(spec.TK)
    ]

    O_outs = [O_n0, O_n1]
    # Outer N-tile loop (2 iterations: O[:, 0:32] and O[:, 32:64]).
    for nt in range(2):
        n_row = nt * 2 * KU  # row offset into Vt_warp (Vt[n_dim, k_pos])
        for ik in range(spec.TK):
            col_off_v = b.const(DType.U32, ik * KU)
            row_off_v = b.const(DType.U32, n_row)
            v_frag = b.load_matrix(
                Vt_warp,
                _NAX_SHAPE_ID,
                which="b",
                row=row_off_v,
                col=col_off_v,
                name=f"v_frag_n{nt}_k{ik}",
            )
            O_outs[nt] = b.mma(
                _NAX_SHAPE_ID,
                S_slices[ik],
                v_frag,
                O_outs[nt],
                transpose_b=True,
                name=f"O_n{nt}_k{ik}",
            )
    return O_outs[0], O_outs[1]


# ---------------------------------------------------------------------------
# Epilogue: normalize O by 1/sum and store to output.
# ---------------------------------------------------------------------------


def _emit_normalize_and_store(
    b: Builder,
    spec: NaxAttnSpec,
    *,
    output: GlobalTensor,
    q_row_base,
    q_col_base,
    O_n0,
    O_n1,
    sum0,
    sum1,
):
    """O /= sum (row-broadcast) then store to output via store_matrix."""
    one_f = b.const(DType.F32, 1.0)
    rcp0 = b.div(one_f, sum0, name="rcp0")
    rcp1 = b.div(one_f, sum1, name="rcp1")

    O0_norm = b.frag_apply(
        _NAX_SHAPE_ID,
        O_n0,
        lambda elem, sel: b.mul(elem, sel),
        selectors=(rcp0, rcp1),
        slot_to_selector_idx=_SLOT_TO_ROW_CLASS,
        name="O0_norm",
    )
    O1_norm = b.frag_apply(
        _NAX_SHAPE_ID,
        O_n1,
        lambda elem, sel: b.mul(elem, sel),
        selectors=(rcp0, rcp1),
        slot_to_selector_idx=_SLOT_TO_ROW_CLASS,
        name="O1_norm",
    )

    # Store: each width-16 O frag covers 16 rows × 32 cols, so the two
    # tiles cover col=[0:32] and col=[32:64] within this head's slot.
    # ``q_col_base`` shifts the col base for token_first layout (where
    # heads are concatenated along the col axis).
    out_warp = output.view(row=q_row_base, col=q_col_base)
    for _nt, O_frag in enumerate(((O0_norm, 0), (O1_norm, 32))):
        frag, col = O_frag
        b.store_matrix(
            out_warp,
            frag,
            _NAX_SHAPE_ID,
            which="d",
            row=0,
            col=col,
        )


# ---------------------------------------------------------------------------
# Compile + dispatch
# ---------------------------------------------------------------------------

# Constant binding plan + threadgroup shape. Fixed by the kernel
# signature: 6 input buffers, 1 output buffer at slot 6, no params
# struct (everything is constexpr in the emitted MSL). Materialized
# once at module load — every dispatch reuses the same tuple instead
# of rebuilding a list of 7 entries on each call.
_PLAN: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),  # Q          → buffer(0)
    (0, 1, 1),  # K_cache    → buffer(1)
    (0, 2, 2),  # Vt_cache   → buffer(2)
    (0, 3, 3),  # segments   → buffer(3)
    (0, 4, 4),  # n_segments → buffer(4)
    (0, 5, 5),  # frame_t    → buffer(5)
    (1, 0, 6),  # output     → buffer(6)
)
_TG: tuple[int, int, int] = (32, 1, 1)


@dataclass(frozen=True)
class _DispatchMeta:
    """Per-spec dispatch precomputes — built once per cache hit, not
    per call. Caches everything spec-derived (grid, output shape /
    strides / nbytes) so ``dispatch_nax_attn`` does no spec arithmetic
    and no list construction in the hot path."""

    pipeline: object
    grid: tuple[int, int, int]
    tg: tuple[int, int, int]
    out_nbytes: int
    out_shape: tuple[int, int]
    out_strides: tuple[int, int]
    smem_bytes: int


@lru_cache(maxsize=8)
def compile_nax_attn(spec: NaxAttnSpec) -> _DispatchMeta:
    """Build, lower, compile, and pack a per-spec dispatch bundle.

    Cached on the spec — every unique (BK, n_q_heads, n_kv_heads, ...)
    yields its own pipeline + precomputed dispatch fields. First call
    pays the Metal compile (a few ms); subsequent calls are a single
    dict lookup.
    """
    module = build_nax_attn_module(spec)
    from quark.device import current_device

    lowered = MslLowerer(current_device().caps).lower_module(module)
    from quark.drivers.metal_harness import TensorParam, build_kernel_source

    inputs = [
        TensorParam(name="Q", dtype="bfloat"),
        TensorParam(name="K_cache", dtype="bfloat"),
        TensorParam(name="Vt_cache", dtype="bfloat"),
        TensorParam(name="segments", dtype="int"),
        TensorParam(name="n_segments", dtype="int"),
        TensorParam(name="frame_t", dtype="int"),
    ]
    outputs = [TensorParam(name="output", dtype="bfloat")]
    # max_threads_per_threadgroup is 32 × n_simdgroups so Apple's
    # compiler picks per-thread register allocation matched to the
    # actual TG size. The single-simdgroup default (n_simdgroups=1)
    # stays at 32 — matches the hand-written kernel's
    # ``[[kernel, max_total_threads_per_threadgroup(32)]]`` decl.
    threads_per_tg = 32 * spec.n_simdgroups
    full_source, _layout = build_kernel_source(
        name="nax_owl_attn",
        body=lowered.source,
        inputs=inputs,
        outputs=outputs,
        scalars=[],
        header=lowered.header or "",
        max_threads_per_threadgroup=threads_per_tg,
    )
    pipeline = _md.compile(full_source, "nax_owl_attn", _MSL_4_0)
    # Each threadgroup covers BQ_total Q rows (= BQ × n_simdgroups);
    # the X axis carries threads-per-block × n_q_tiles_outer.
    n_q_tiles_outer = spec.tpf // spec.BQ_total
    # Output shape mirrors the kernel's ``output`` param declaration in
    # ``build_nax_attn_module``: head_first / token_first sliced both
    # produce ``(n_q*tpf, Dh)`` (head_first) or ``(tpf, n_q*Dh)``
    # (token_first); inline_q_rope always produces ``(tpf, n_q*Dh)``.
    if spec.inline_q_rope or spec.layout == "token_first":
        out_shape = (spec.tpf, spec.n_q_heads * spec.Dh)
        out_strides = (spec.n_q_heads * spec.Dh, 1)
    else:  # head_first
        out_shape = (spec.n_q_heads * spec.tpf, spec.Dh)
        out_strides = (spec.Dh, 1)
    out_nbytes = out_shape[0] * out_shape[1] * 2  # bf16 = 2 bytes
    # Grid is in *threads* (Metal's dispatchThreads convention used by
    # _md.queue_launch); divide by threads_per_tg to get the threadgroup
    # count Apple's runtime sees. ``_TG`` adapts via ``threads_per_tg``.
    return _DispatchMeta(
        pipeline=pipeline,
        grid=(n_q_tiles_outer * threads_per_tg, spec.n_q_heads, 1),
        tg=(threads_per_tg, 1, 1),
        out_nbytes=out_nbytes,
        out_shape=out_shape,
        out_strides=out_strides,
        smem_bytes=int(lowered.smem_bytes),
    )


def dispatch_nax_attn(
    *,
    spec: NaxAttnSpec,
    Q,
    K_cache,
    Vt_cache,
    segments,
    n_segments,
    frame_t,
):
    """Run the IR-emitted NAX attention kernel and return the output
    ``QuarkTensor``. All spec-derived precomputes (grid, output shape,
    pipeline) are cached on the spec — the hot path here is just
    handle/ptr extraction + ``queue_launch``.
    """
    import numpy as np

    from quark.runtime.tensor import QuarkTensor, _MetalStorage

    meta = compile_nax_attn(spec)

    # Handle/ptr/size lists. Parameter order MUST match
    # build_nax_attn_module: Q, K, Vt, segments, n_segments, frame_t.
    handles = [0] * 6
    ptrs = [0] * 6
    nbytes_list = [0] * 6
    for i, arr in enumerate((Q, K_cache, Vt_cache, segments, n_segments, frame_t)):
        h = getattr(arr, "metal_handle", None)
        if h is not None:
            handles[i] = h
        else:
            narr = np.ascontiguousarray(arr)
            handles[i] = -1
            ptrs[i] = narr.ctypes.data
            nbytes_list[i] = narr.nbytes

    out_handle, out_ptr = _md.queue_launch(
        meta.pipeline,
        handles,
        ptrs,
        nbytes_list,
        meta.out_nbytes,
        _PLAN,
        meta.grid,
        meta.tg,
        meta.smem_bytes,
    )
    return QuarkTensor(
        _MetalStorage(out_handle, out_ptr, meta.out_nbytes),
        meta.out_shape,
        meta.out_strides,
        0,
        "bf16",
    )
