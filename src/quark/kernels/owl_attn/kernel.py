"""owl_attn — segment-sparse flash attention with ortho-RoPE on Q.

EXEMPT FROM 500-LINE RULE

Consumer of the kv_cache_update kernel. Per call:
  Inputs:
    Q          [B*Hq*tpf, Dh]                a_dtype    (pre-RoPE)
    K_cache    [B*Hk*capacity, Dh]           kv_dtype   (post-RoPE; cached)
    Vt_cache   [B*Hk*Dh, capacity]           kv_dtype   (transposed cache)
    cos        [tpf, Dh//2]                  f32        (per-frame slice)
    sin        [tpf, Dh//2]                  f32
    segments   [B*max_segments*2]            s32        ((start, length) pairs)
    n_segments [B]                           s32        (== 2 in steady state)
  Output:
    output     [B*Hq*tpf, Dh]                out_dtype

Kernel architecture (extends `attn`):
  Grid: (n_q_tiles, B*Hk, 1) ; n_q_tiles = tpf / BlockQRows
  Block: NumWarps*32 threads ; NumWarps = gqa_ratio * NCW

  1. cp.async Q → smem (per-warp slice).
  2. cp.async cos / sin → smem (block-shared, BlockQRows × Dh//2).
  3. async_wait + barrier.
  4. Q-RoPE pass: each thread reads pairs from Q_smem (a_dtype),
     applies fp32 ortho-RoPE with concat-output layout, casts back
     to a_dtype, writes y0/y1 to Q_smem at (r, c) and (r, c+half_Dh).
     Same convention K_cache uses (kv_cache_update writes K with concat).
  5. barrier; load_matrix → register fragments.
  6. Per segment (python-unrolled), inner for_loop over KvTile chunks:
     cp.async K (then GEMM1), cp.async V^T (then softmax + GEMM2).
     Accumulators carry through all segment loops.
  7. ``qk.store_acc(per_warp=True, row_scale=l)`` epilogue.
"""

from __future__ import annotations

import math
from typing import ClassVar

import quark.lang as qk
from quark.blocks import (
    Accumulators,
    Carry,
    IterCtx,
    MmaBody,
    PipelineBody,
    SmemTile,
    Stage,
    TensorDecl,
)
from quark.ir import DType
from quark.kernels.attn.online_softmax_block import OnlineSoftmax
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.owl_attn.baselines import owl_attn_baselines
from quark.kernels.owl_attn.config import AttnConfig as OwlAttnConfig
from quark.kernels.owl_attn.iter_table import emit_kv_offset_table
from quark.kernels.owl_attn.problems import owl_attn_problems
from quark.kernels.owl_attn.reference import owl_attn_reference_numpy
from quark.kernels.owl_attn.spec import OwlAttnSpec


@kernel(
    "owl_attn",
    spec=OwlAttnSpec,
    config=OwlAttnConfig,
    output_idx=-1,
    problems=owl_attn_problems,
    baselines=owl_attn_baselines,
    reference=owl_attn_reference_numpy,
)
class OwlAttnKernel(Kernel):
    # Parameter manifest — every tensor shape derives from spec only
    # (no config-dependent layouts), so the lambdas ignore config.
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("Q", dtype=lambda s, c: s.a_dtype, shape=lambda s, c: (s.q_rows, s.q_cols)),
        TensorDecl(
            "K_cache",
            dtype=lambda s, c: s.kv_dtype,
            shape=lambda s, c: (s.B * s.n_kv_heads * s.capacity, s.Dh),
        ),
        TensorDecl(
            "Vt_cache",
            dtype=lambda s, c: s.kv_dtype,
            shape=lambda s, c: (s.B * s.n_kv_heads * s.Dh, s.capacity),
        ),
        TensorDecl("segments", dtype=DType.S32, shape=lambda s, c: (s.B * s.max_segments * 2,)),
        TensorDecl("n_segments", dtype=DType.S32, shape=lambda s, c: (s.B,)),
        TensorDecl("frame_t", dtype=DType.S32, shape=lambda s, c: (1,)),
        TensorDecl(
            "output",
            dtype=lambda s, c: s.out_dtype,
            shape=lambda s, c: (s.out_rows, s.out_cols),
            role="out",
        ),
    ]
    # An extra in-register fp32 RoPE pass on Q + bf16 K_cache/Vt_cache
    # loads + bf16 MMA accumulators (followed by softmax over a 2k-8k KV)
    # accumulate ~2-3e-2 of cosine drift vs the torch baseline (which keeps
    # Q in fp32 throughout flex_attention). 0.97 is more than sufficient
    # signal for an attention output — anything wrong with the kernel
    # math drops well below this.
    CORRECTNESS_THRESHOLD = 0.97

    spec: OwlAttnSpec
    config: OwlAttnConfig

    def _block_q_rows(self) -> int:
        try:
            m = self._mma_cfg().shape.m
        except KeyError:
            m = 16
        return self.config.NCW * self.config.MTiles * m

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        try:
            mma = self._mma_cfg()
        except KeyError:
            return False
        # k=8 bf16 MMAs (m16n8k8, m8n8k8) are slower than m16n8k16 on
        # every owl_attn shape and the PTX m16n8k8 path trips NaNs in
        # the autotune sweep — filter them out so autotune doesn't
        # waste time on a dominated/broken path.
        if mma.shape.k == 8:
            return False
        # NCW=1 broken: every other 8-row Vt group lands wrong (odd
        # nt-tiles ~100x off ref). n_threads=64 cp.async quirk;
        # scripts/diag_owl_attn_ncw.py reproes.
        if c.NCW == 1:
            return False
        if s.Dh % mma.shape.m != 0 or s.Dh % mma.shape.n != 0 or s.Dh % 2 != 0:
            return False
        if s.capacity % c.KvTile != 0:
            return False
        if c.KvTile % mma.shape.n != 0 or c.KvTile % mma.mma_k != 0:
            return False
        # Segments must be tile-aligned: each segment.length % KvTile == 0.
        # That holds when num_buckets*tpf and tpf are both multiples of KvTile.
        if s.tpf % c.KvTile != 0 or s.L % c.KvTile != 0:
            return False
        bqr = self._block_q_rows()
        if s.tpf % bqr != 0:
            return False
        if c.KvTile not in (8, 16, 32, 64, 128, 256):
            return False
        n_threads = math.prod(self.block())
        if n_threads > 1024:
            return False
        # K/V smem live in compute_dtype regardless of cache dtype. Pad rules
        # follow the smem element size: 16B granularity for cp.async means
        # `(stride_elems * elem_bytes) % 16 == 0` and `pad ∈ {0, 8 if 2B
        # else 16}` (matches the GEMM kernel's pad_nonzero rule).
        compute_b = s.compute_dtype_resolved.bytes
        pad_nonzero = 16 if compute_b == 1 else 8
        if c.KvPad not in (0, pad_nonzero):
            return False
        # if c.MTiles > 1:
        #     return False
        # cp.async alignment for K/V smem (when no cast — kv_ir == compute_ir).
        if s.kv_dtype is s.compute_dtype_resolved:
            k_lines = c.KvTile * s.Dh * compute_b // 16
            if k_lines > n_threads and k_lines % n_threads != 0:
                return False
            v_lines = s.Dh * c.KvTile * compute_b // 16
            if v_lines > n_threads and v_lines % n_threads != 0:
                return False
            if ((s.Dh + c.KvPad) * compute_b) % 16 != 0:
                return False
            if ((c.KvTile + c.KvPad) * compute_b) % 16 != 0:
                return False
        else:
            # Scalar cast load: tile total elements must be divisible by n_threads.
            if (c.KvTile * s.Dh) % n_threads != 0:
                return False
            if (s.Dh * c.KvTile) % n_threads != 0:
                return False
        # Q-RoPE pass: total pairs (BlockQRows × Dh//2) divisible by n_threads.
        if (bqr * (s.Dh // 2)) % n_threads != 0:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        s = self.spec
        bqr = self._block_q_rows()
        n_q_tiles = s.tpf // bqr
        return (n_q_tiles, s.B * s.n_kv_heads, 1)

    def block(self) -> tuple[int, int, int]:
        return ((self.spec.gqa_ratio * self.config.NCW) * 32, 1, 1)

    def flops(self) -> int:
        s = self.spec
        # Conservative estimate: 2 GEMMs over full capacity (some segments
        # may be empty in early frames; this is a rough number for bench).
        return 2 * 2 * s.B * s.n_q_heads * s.tpf * s.capacity * s.Dh

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        """Build random-but-shape-valid inputs for the attention kernel.

        owl_attn *consumes* K_cache / Vt_cache / segments — it doesn't
        produce them. Test data here is generated directly, not by
        simulating the upstream kv_cache_update pipeline (which would
        churn the allocator with redundant RoPE + set_slice work
        irrelevant to this kernel's correctness). Segments declare a
        single contiguous valid range ``[0, capacity)`` so the kernel
        attends over the full cache, matching what the real upstream
        pipeline would produce at steady state (``num_buckets * tpf``
        ring + ``tpf`` tail = ``capacity``).
        """
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = OwlAttnSpec(**problem)
        B, Hk, tpf, Dh = spec.B, spec.n_kv_heads, spec.tpf, spec.Dh
        cap = spec.capacity
        rng = np.random.default_rng(seed)

        Q_shape = (spec.q_rows, spec.q_cols)
        Q = astype_numpy((rng.standard_normal(Q_shape) * 0.3).astype(np.float32), spec.a_dtype)
        K_cache = astype_numpy(
            (rng.standard_normal((B * Hk * cap, Dh)) * 0.3).astype(np.float32), spec.kv_dtype
        )
        Vt_cache = astype_numpy(
            (rng.standard_normal((B * Hk * Dh, cap)) * 0.3).astype(np.float32), spec.kv_dtype
        )

        L = spec.L
        seg_rows = [[0, L], [L, tpf]] + [[0, 0]] * (spec.max_segments - 2)
        seg_per_batch = [v for row in seg_rows for v in row]
        segments = np.array([seg_per_batch] * B, dtype=np.int32).reshape(B * spec.max_segments * 2)
        n_segments = np.full(B, 2, dtype=np.int32)

        test_frame_t = spec.num_buckets * spec.pinned_dilation
        frame_t = np.array([test_frame_t], dtype=np.int32)
        output = zeros_for_dtype((spec.out_rows, spec.out_cols), spec.out_dtype)

        return {
            "Q": Q,
            "K_cache": K_cache,
            "Vt_cache": Vt_cache,
            "segments": segments,
            "n_segments": n_segments,
            "frame_t": frame_t,
            "output": output,
        }

    @classmethod
    def spec_from_tensors(
        cls,
        Q,
        K_cache,
        Vt_cache,
        segments,
        n_segments,
        *,
        B: int,
        n_kv_heads: int,
        gqa_ratio: int,
        H_spatial: int,
        W_spatial: int,
        num_buckets: int,
        pinned_dilation: int,
        out_dtype: DType | str | None = None,
        compute_dtype: DType | str | None = None,
        max_segments: int = 3,
        packed_qkv: bool = False,
    ) -> OwlAttnSpec:
        """Derive an ``OwlAttnSpec`` from the kernel-input tensors."""

        if Q.ndim != 2:
            raise ValueError(f"pcf.owl_attn: Q must be rank-2 (flat), got {Q.shape}")
        if packed_qkv:
            qkv_dim = int(Q.shape[1])
            n_q_heads = n_kv_heads * gqa_ratio
            Dh = qkv_dim // (n_q_heads + 2 * n_kv_heads)
        else:
            Dh = int(Q.shape[1])
        tpf = H_spatial * W_spatial
        capacity = num_buckets * tpf + tpf
        if tuple(K_cache.shape) != (B * n_kv_heads * capacity, Dh):
            raise ValueError(
                f"pcf.owl_attn: K_cache shape {tuple(K_cache.shape)} != "
                f"({B * n_kv_heads * capacity}, {Dh})"
            )
        if tuple(Vt_cache.shape) != (B * n_kv_heads * Dh, capacity):
            raise ValueError(
                f"pcf.owl_attn: Vt_cache shape {tuple(Vt_cache.shape)} != "
                f"({B * n_kv_heads * Dh}, {capacity})"
            )
        if int(segments.shape[0]) != B * max_segments * 2:
            raise ValueError(
                f"pcf.owl_attn: segments shape {tuple(segments.shape)} != ({B * max_segments * 2},)"
            )
        if int(n_segments.shape[0]) != B:
            raise ValueError(f"pcf.owl_attn: n_segments shape {tuple(n_segments.shape)} != ({B},)")
        a_dt = DType.from_backend(Q.dtype)
        return OwlAttnSpec(
            B=B,
            n_kv_heads=n_kv_heads,
            gqa_ratio=gqa_ratio,
            H_spatial=H_spatial,
            W_spatial=W_spatial,
            num_buckets=num_buckets,
            pinned_dilation=pinned_dilation,
            Dh=Dh,
            a_dtype=a_dt,
            kv_dtype=DType.from_backend(K_cache.dtype),
            out_dtype=DType.coerce(out_dtype) or a_dt,
            compute_dtype=DType.coerce(compute_dtype),
            max_segments=max_segments,
            packed_qkv=packed_qkv,
        )

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {
            # KvTile=16 is the smallest Metal-compilable; is_valid rejects
            # combos where the spec alignment doesn't hold.
            "KvTile": [8, 16, 32, 64, 128, 256],
            "MTiles": [1, 2, 4, 8],
            "NCW": [1, 2, 4, 8],
            # KvPad: 8B granule for bf16/f16, 16B for fp8 (is_valid gates).
            "KvPad": [0, 8, 16],
            "n_stages": [1, 2],  # run_pipeline tops out at double-buffer.
        }

    # ── build() ──

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        Dh = s.Dh
        half_Dh = Dh // 2
        KvTile = c.KvTile
        MTiles = c.MTiles
        NCW = c.NCW
        GQA = s.gqa_ratio
        NumWarps = GQA * NCW
        mma_cfg = self._mma_cfg()
        m_tile = mma_cfg.shape.m
        n_tile = mma_cfg.shape.n
        BlockQRows = NCW * MTiles * m_tile
        NK = KvTile // n_tile
        N_DH = Dh // n_tile
        max_segs = s.max_segments

        # Compute dtype drives both MMAs and the smem tile element type.
        # Defaults to a_dtype (no cast on Q). For mixed-precision (e.g. Q in
        # bf16, MMA in e4m3, output in bf16 — the GEMM kernel pattern), Q is
        # cast during the RoPE pass and K/V are cast during TileLoad.
        compute_ir_dtype = s.compute_dtype_resolved
        kv_cast = compute_ir_dtype if s.kv_dtype is not compute_ir_dtype else None
        compute_is_fp8 = compute_ir_dtype in (DType.E4M3, DType.E5M2)

        bctx = self.bctx
        ctx = self.ctx
        g_q, g_kc, g_vtc = g.Q, g.K_cache, g.Vt_cache
        # cos/sin tables removed — computed inline via emit_rope_cos_sin.
        g_sg, g_out = g.segments, g.output

        n_threads = NumWarps * 32
        tid = bctx.tid

        # Grid decomposition.
        q_tile_idx = ctx.block_idx("x")
        bh_idx = ctx.block_idx("y")

        # Auto-CSE in Builder dedupes every repeat use of these
        # expressions; operator overloading keeps the math readable.
        warp_id = bctx.warp_id
        gqa_idx = warp_id // NCW
        m_idx = warp_id % NCW

        kv_h = bh_idx % s.n_kv_heads
        b_idx = bh_idx // s.n_kv_heads
        q_head = kv_h * GQA + gqa_idx

        # Token-row offset within the frame (shared by Q load and output).
        token_tile_offset = q_tile_idx * BlockQRows + m_idx * (MTiles * m_tile)

        if s.packed_qkv:
            # Packed QKV: Q input is [B*tpf, qkv_dim].
            # Row = batch * tpf + token offset (no head multiplier).
            # Col = q_head * Dh (head's column in the packed buffer).
            q_row_warp = b_idx * s.tpf + token_tile_offset
            q_col_base = q_head * Dh
            # Output is [B*tpf, n_q_heads*Dh].
            out_row_warp = q_row_warp
            out_col_base = q_head * Dh
        else:
            # Legacy layout: Q is [B*nq*tpf, Dh].
            q_row_warp = (b_idx * s.n_q_heads + q_head) * s.tpf + token_tile_offset
            q_col_base = bctx.c(0)
            out_row_warp = q_row_warp
            out_col_base = bctx.c(0)

        # K/V cache row bases (per (B, head)).
        k_row_base = bh_idx * s.capacity
        vt_row_base = bh_idx * Dh

        # frame_t for inline RoPE computation.
        frame_t_val = qk.load(g.frame_t, bctx.c(0, dtype=DType.S32))
        frame_t_u = qk.convert(frame_t_val, DType.U32)

        # ── Smem allocs ──
        # K/V smem live in `compute_ir_dtype` regardless of the gmem cache
        # dtype — TileLoad casts on the way in for narrower caches.
        lcs = mma_cfg.lane_col_step
        # Q cp.async destination — always in a_ir_dtype (cp.async can't cast).
        # Over-allocate to ``NumWarps * MTiles * 16`` rows (not just
        # ``BlockQRows = NCW * MTiles * 16``) so the epilogue's
        # ``qk.store_acc(per_warp=True)`` can reuse this region as its
        # per-warp staging buffer — Q is dead by the time the epilogue runs
        # (finished during RoPE + ldmatrix, both before the KV loop
        # opens). The extra rows waste ``(GQA-1) * BlockQRows`` Q
        # positions during the cp.async load but save a full fresh
        # O_stage allocation (saves exactly those same rows, net zero
        # at the one-GQA cost and a win for GQA>1 where Q is shared
        # across the GQA group but the epilogue writes per-warp).
        q_in_smem = qk.smem_alloc(
            "Q_in_smem",
            s.a_dtype,
            (NumWarps * MTiles * m_tile, Dh),
            pad=c.KvPad,
        )
        # Q ldmatrix source — in compute_ir_dtype. Always a separate
        # buffer from q_in_smem (aliasing previously caused a read/write
        # race on Metal: thread T1's y1 write to col half_Dh raced with
        # thread T_half's x read from col half_Dh since lanes across
        # warps aren't in lockstep).
        #
        # Sized ``NumWarps * MTiles * m_tile`` (= ``GQA * BlockQRows``),
        # matching q_in_smem — NOT ``BlockQRows``. The RoPE pass writes
        # one row per physical Q row, iterating ``n_phys_pairs =
        # NumWarps * warp_rows * half_Dh`` so ``qrow`` reaches
        # ``NumWarps * warp_rows - 1``. Likewise the ldmatrix view
        # offsets by ``warp_id * warp_rows * stride_elems`` for warp
        # ids up to ``NumWarps - 1``. Sizing to BlockQRows is correct
        # only when GQA=1; for GQA>1 the writes (and ldmatrix reads)
        # spill past the slot into whatever's adjacent (K/Vt smem),
        # producing data-dependent garbage in the warps that handle
        # ``gqa_idx >= 1``. is_valid rejects configs whose total smem
        # exceeds the device cap, so larger configs that newly overflow
        # after this resize get filtered there rather than silently.
        q_smem = qk.smem_alloc(
            "Q_compute_smem",
            compute_ir_dtype,
            (NumWarps * MTiles * m_tile, Dh),
            pad=c.KvPad,
        )
        # K/V smem tiles live on a single-stage Stage so PipelineBody.run
        # handles cp.async commit + wait + barrier for us. Segment-sparse
        # iteration is Python-unrolled below: one PipelineBody per segment,
        # carry threaded across.
        stages = Stage.staged(
            c.n_stages,
            k=SmemTile.spec(compute_ir_dtype, (KvTile, Dh), pad=c.KvPad, lane_col_step=lcs),
            vt=SmemTile.spec(compute_ir_dtype, (Dh, KvTile), pad=c.KvPad, lane_col_step=lcs),
        )
        # cos/sin computed inline via emit_rope_cos_sin (no tables).

        # cp.async Q (per-warp slice; QRegisterLoad below handles it AFTER
        # we apply RoPE in smem — so pre-emit just the cp.async + cos/sin
        # loads here and run ldmatrix later).
        # Q load — replicate QRegisterLoad's cp.async path manually so we
        # can run a RoPE pass between load and ldmatrix.
        warp_rows = MTiles * m_tile
        stride_elems = Dh + c.KvPad
        warp_dyn = warp_id * (warp_rows * stride_elems)
        # ``warp_dyn`` is reused later (epilogue staging); keep the explicit
        # binding. Equivalent to ``q_in_smem.warp_view(rows=warp_rows,
        # warp_id=warp_id)``.
        warp_smem_in = q_in_smem.view(dyn_offset=warp_dyn, shape=(warp_rows, Dh), name="Q_warp_in")
        lane_id = tid % 32
        # Q tile: per-warp cooperative cp.async. Warp-local 32-thread
        # split; one line per lane per iteration.
        warp_smem_in.copy_from(
            g_q.tile(row=q_row_warp, col=q_col_base, shape=(warp_rows, Dh)),
            tid=lane_id,
            n_threads=32,
            async_load=True,
        )
        # (cos/sin computed inline — no table load needed)
        qk.async_commit()
        qk.async_wait(0)
        qk.barrier("block")

        # ── Q-RoPE pass: read q_in_smem (a_dtype) → fp32 RoPE → cast to
        # compute_ir_dtype → write q_smem (compute_ir_dtype). When compute
        # is fp8 (e4m3 / e5m2), pairs of (c, c+1) outputs are packed via
        # packed_convert into one b16 store per pair (same trick the
        # kv_cache_update kernel uses for K). cos/sin rows are shared
        # across the GQA dim — for physical q row R, the cos row is
        # ((R / warp_rows) % NCW) * warp_rows + (R % warp_rows).
        n_phys_pairs = NumWarps * warp_rows * half_Dh
        if n_phys_pairs % n_threads != 0:
            raise ValueError(
                f"Q-RoPE: physical pair count {n_phys_pairs} not divisible by n_threads={n_threads}"
            )
        per_thr = n_phys_pairs // n_threads
        half_Dh_c = bctx.c(half_Dh)
        warp_rows_c = bctx.c(warp_rows)

        # Inline RoPE: compute cos/sin from (h, w, frame_t) per element.
        from quark.lang.rope import emit_rope_cos_sin

        q_tile_base = q_tile_idx * bctx.c(BlockQRows)
        W_c = bctx.c(s.W_spatial)

        def _rope_one(qrow_v, c_idx_v):
            """Compute (y0_f32, y1_f32) for one (qrow, c) pair."""
            warp_of_row = qrow_v // warp_rows_c
            m_idx_w = warp_of_row % NCW
            r_local = qrow_v % warp_rows_c
            local_token = m_idx_w * warp_rows_c + r_local
            # Absolute spatial index in [0, tpf).
            spatial_idx = q_tile_base + local_token
            h_idx = spatial_idx // W_c
            w_idx = spatial_idx % W_c
            c2 = c_idx_v * 2
            c2p1 = c2 + 1
            x0 = qk.convert(q_in_smem[qrow_v, c2], DType.F32)
            x1 = qk.convert(q_in_smem[qrow_v, c2p1], DType.F32)
            cv, sv = emit_rope_cos_sin(
                bctx,
                h_idx=h_idx,
                w_idx=w_idx,
                frame_t=frame_t_u,
                c_idx=c_idx_v,
                H=s.H_spatial,
                W=s.W_spatial,
                Dh=Dh,
            )
            y0 = x0 * cv - x1 * sv
            y1 = x1 * cv + x0 * sv
            return y0, y1

        if compute_is_fp8:
            if per_thr % 2 != 0:
                raise ValueError(f"Q-RoPE fp8: per_thr={per_thr} must be even (need pair grouping)")
            for i in range(0, per_thr, 2):
                # Two adjacent (qrow, c) pairs on the same physical Q row.
                flat_a = tid * per_thr + i
                qrow = flat_a // half_Dh_c
                c_a = flat_a % half_Dh_c
                c_b = c_a + 1
                y0_a, y1_a = _rope_one(qrow, c_a)
                y0_b, y1_b = _rope_one(qrow, c_b)
                y0_pk = qk.packed_convert(y0_a, y0_b, compute_ir_dtype)
                y1_pk = qk.packed_convert(y1_a, y1_b, compute_ir_dtype)
                q_smem[qrow, c_a] = y0_pk
                q_smem[qrow, c_a + half_Dh_c] = y1_pk
        else:
            for i in range(per_thr):
                flat = tid * per_thr + i
                qrow = flat // half_Dh_c
                c_idx = flat % half_Dh_c
                y0, y1 = _rope_one(qrow, c_idx)
                q_smem[qrow, c_idx] = qk.convert(y0, compute_ir_dtype)
                q_smem[qrow, c_idx + half_Dh_c] = qk.convert(y1, compute_ir_dtype)

        qk.barrier("block")

        # ── ldmatrix → Q register fragments ──
        lane_off = bctx.gid * stride_elems + bctx.tig * mma_cfg.lane_col_step
        warp_lane = q_smem.view(
            dyn_offset=warp_dyn + lane_off,
            shape=(warp_rows, Dh),
            name="Q_warp_lane",
        )
        KK_STEPS = Dh // mma_cfg.mma_k
        q_frags: list[list] = []
        for mt in range(MTiles):
            mt_frags = []
            for kk_step in range(KK_STEPS):
                kk = kk_step * mma_cfg.mma_k
                frag = qk.load_matrix(
                    warp_lane,
                    mma_cfg.shape_id,
                    which="a",
                    row=mt * m_tile,
                    col=kk,
                    reg_offsets=mma_cfg.a_offsets,
                )
                mt_frags.append(frag)
            q_frags.append(mt_frags)

        # ── Accumulators + sub-blocks ──
        acc_width = mma_cfg.shape.c_regs
        o_acc = Accumulators(MT=MTiles, NT=N_DH, width=acc_width)
        s_acc = Accumulators(MT=MTiles, NT=NK, width=acc_width)
        # fp8 GEMM2 path: softmax round-trips P through smem because the
        # PTX ISA fp8 A-fragment layout (4 consecutive k-cols per lane)
        # doesn't line up same-lane with the f32 accumulator layout
        # (2 cols per lane across 2 adjacent lanes). Softmax writes
        # f32→fp8 packed stores into P_smem; the kernel reloads via
        # load_matrix into the fp8 A-frag layout for GEMM2. One block
        # barrier per KV iter between the stores and the ldmatrix reads.
        if compute_is_fp8:
            p_stride_elems = KvTile + c.KvPad
            p_warp_rows = MTiles * m_tile
            p_warp_dyn = warp_id * (p_warp_rows * p_stride_elems)
            p_lane_off = bctx.gid * p_stride_elems + bctx.tig * mma_cfg.lane_col_step
            p_row_base = warp_id * p_warp_rows
            p_smem = qk.smem_alloc(
                "P_smem",
                compute_ir_dtype,
                (NumWarps * p_warp_rows, KvTile),
                pad=c.KvPad,
            )
            p_warp_lane = p_smem.view(
                dyn_offset=p_warp_dyn + p_lane_off,
                shape=(p_warp_rows, KvTile),
                name="P_warp_lane",
            )
        else:
            p_smem = None
            p_warp_lane = None
            p_row_base = 0
        # GEMM1: A is the register Q tile, B is K smem.
        # GEMM2: A is the P fragment from online softmax, B is V^T smem.
        # ``shape`` defaults to ``active_bctx().mma_cfg``; ``K_inner``
        # is inferred from ``b.shape[1]`` // mma_k at call time.
        mma1 = MmaBody(acc=s_acc)
        softmax = OnlineSoftmax(
            s_acc=s_acc,
            o_acc=o_acc,
            scale=1.0 / math.sqrt(Dh),
            p_smem=p_smem,
            p_row_base=p_row_base,
        )
        mma2 = MmaBody(acc=o_acc)

        # Reuse Q_in_smem as the staging buffer — Q is finished (ldmatrix
        # into q_frags above) before the KV loop, so by the time the
        # epilogue writes it the buffer is dead. Q_in_smem was
        # over-allocated above to match the per-warp staging size and
        # carries the same ``pad=c.KvPad`` stride the cooperative
        # ``vec_store`` epilogue path expects. Saves a full
        # ``NumWarps * MTiles * 16 * Dh * bf16.bytes`` region of smem.
        # The actual scale + cast + per-warp staged store happens via
        # ``qk.store_acc(..., per_warp=True, row_scale=l_vals)`` after
        # the segment loop closes; here we just keep ``q_in_smem`` /
        # ``out_row_warp`` / ``warp_id`` in scope.

        n_rc = len({dr for dr, _ in mma_cfg.cd_offsets})
        n_ml = MTiles * n_rc

        # Typed carry — O + per-(mt, rc) m / l scalars. See attn/kernel.py
        # for the same pattern.
        carry_spec = Carry(
            o=o_acc,
            m=(n_ml, -1e30, DType.F32),
            l=(n_ml, 0.0, DType.F32),
        )

        # ── Flat KV iteration across all segments ──────────────────────
        # PipelineBody with compile-time ``n_iters = capacity / KvTile``
        # so n_stages=2 works (double-buffered cp.async needs the half
        # iter count for prologue/epilogue placement). iter→kv_offset
        # is precomputed into smem; consume's validity check skips the
        # whole iter when invalid (segment lengths are tile-aligned).
        N_TOTAL = s.capacity // KvTile
        seg_base = b_idx * (max_segs * 2)

        KvTile_u = qk.const(DType.U32, KvTile)
        zero_u = qk.const(DType.U32, 0)

        # Read all segment metadata up front (the only g_sg gmem reads).
        seg_starts_u: list = []
        cum_u: list = [zero_u]
        for seg_idx in range(max_segs):
            seg_off = seg_idx * 2
            seg_start = g_sg[seg_base + seg_off]
            seg_len = g_sg[seg_base + seg_off + 1]
            seg_starts_u.append(qk.convert(seg_start, DType.U32))
            n_chunks_u = qk.convert(seg_len, DType.U32) // KvTile_u
            cum_u.append(cum_u[-1] + n_chunks_u)
        n_real_total = cum_u[max_segs]

        # ── Cooperative iter→kv_offset table fill ──
        # 4 bytes × N_TOTAL ≈ 1 KB at our shape. Replaces a per-iter
        # chain of cmp+select with a single ld.shared.b32 in produce.
        kv_off_smem = emit_kv_offset_table(
            bctx=bctx,
            seg_starts_u=seg_starts_u,
            cum_u=cum_u,
            n_real_total=n_real_total,
            KvTile=KvTile,
            KvTile_u=KvTile_u,
            zero_u=zero_u,
            N_TOTAL=N_TOTAL,
            n_threads_per_cta=NumWarps * 32,
        )

        def produce(ictx: IterCtx) -> None:
            stage = ictx.stage
            # Single smem load — replaces the segment-chain compute that
            # ran every iter before. Uniform across the warp; lowers to
            # one ld.shared.b32.
            kv_offset = qk.load(kv_off_smem, ictx.iter_idx)
            stage.k.load_from(g_kc, row=k_row_base + kv_offset, cast=kv_cast)
            stage.vt.load_from(g_vtc, row=vt_row_base, col=kv_offset, cast=kv_cast)

        mma_shape_id = mma_cfg.shape_id

        GEMM2_K_STEPS = KvTile // mma_cfg.mma_k

        n_o = MTiles * N_DH

        def _reload_p_frags_fp8():
            """fp8 GEMM2 path — reload P from smem into the fp8 A-frag."""
            qk.barrier("block")
            return [
                [
                    qk.load_matrix(
                        p_warp_lane,
                        mma_shape_id,
                        which="a",
                        row=mt * m_tile,
                        col=k_step * mma_cfg.mma_k,
                        reg_offsets=mma_cfg.a_offsets,
                    )
                    for k_step in range(GEMM2_K_STEPS)
                ]
                for mt in range(MTiles)
            ]

        # Segment lengths are tile-aligned → an iter is ALL valid or
        # ALL invalid (no sub-fragment qualifier). Wrap consume in a
        # uniform ``if_(valid)``: skip mma1+softmax+mma2 entirely for
        # invalid iters, AND drop the per-cell ``select(valid, x, -inf)``
        # mask from the taken arm (it sat on the mma1→softmax chain).
        def consume(ictx: IterCtx) -> Carry:
            carry = ictx.carry
            valid = qk.cmp("lt", ictx.iter_idx, n_real_total)
            flat_carry = list(carry.o) + list(carry.m) + list(carry.l)
            with bctx.bld.if_(valid, carried=flat_carry) as (then_in, else_in, arms):
                with arms.then_():
                    inner_o = list(then_in[:n_o])
                    inner_m = list(then_in[n_o : n_o + n_ml])
                    inner_l = list(then_in[n_o + n_ml :])
                    s_vals = mma1(a=q_frags, b=ictx.stage.k, acc=s_acc.init())
                    o_vals, m_vals, l_vals, p_frags = softmax(
                        s_acc_vals=s_vals, o_vals=inner_o, m_vals=inner_m, l_vals=inner_l
                    )
                    if compute_is_fp8:
                        p_frags = _reload_p_frags_fp8()
                    new_o = mma2(a=p_frags, b=ictx.stage.vt, acc=o_vals)
                    qk.yield_(*new_o, *m_vals, *l_vals)
                with arms.else_():
                    qk.yield_(*else_in)
            results = bctx.bld.last_results
            carry.o = list(results[:n_o])
            carry.m = list(results[n_o : n_o + n_ml])
            carry.l = list(results[n_o + n_ml :])
            return carry

        current_carry: Carry = PipelineBody(
            stages=stages,
            produce=produce,
            consume=consume,
            carry=carry_spec,
        ).run(n_iters=N_TOTAL, n_stages=c.n_stages)

        # ``current_carry`` is a Carry rebound to the final values across
        # all segments. ``o_acc.results`` is already populated by the last
        # PipelineBody.run via _stash_results_on_carry.
        final = current_carry

        # Barrier before the epilogue: the per-warp staged store
        # scatters to ``q_in_smem`` which aliases the (now-dead) Q_in
        # region. A barrier here ensures no warp is still in the
        # yield-sync tail of the KV loop while another has advanced
        # to scattering to the same buffer.
        qk.barrier("block")

        # Epilogue: O /= l, scale + cast + per-warp staged → vec_store.
        # Outer ``qk.for_range`` doesn't run ``_stash_results_on_carry``
        # the way ``run_pipeline`` does, so populate ``o_acc.results``
        # directly from the bound Carry here.
        o_acc.results = list(final.o)
        qk.store_acc(
            g_out,
            o_acc,
            row=out_row_warp,
            col=out_col_base,
            cast=s.out_dtype,
            staging_smem=q_in_smem,
            per_warp=True,
            warp_id=warp_id,
            kv_pad=c.KvPad,
            row_scale=final.l,
            gmem_col_base_full=out_col_base,
        )
