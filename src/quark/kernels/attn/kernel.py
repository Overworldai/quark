"""Flash attention kernel with GQA, multi-MTile, double-buffered KV pipeline.

Architecture (matches the old owl_attn kernel):
  Grid: (n_q_tiles, B * n_kv_heads, 1)
  Block: (gqa_ratio * NCW * 32) threads

  Warp decomposition:
    warp_id = gqa_idx * NCW + m_idx
    gqa_idx ∈ [0, gqa_ratio) — which Q head within the GQA group
    m_idx ∈ [0, NCW) — which M-tile block within a head

  Per warp: MTiles × 16 Q rows (each warp handles its own Q head)
  BlockQRows = NCW * MTiles * 16

  Pipeline:
    1. cp.async Q → smem, extract register fragments
    2. Double-buffered KV: prefetch chunk N+1 while computing chunk N
    3. GEMM1: S = Q_regs @ K_smem^T
    4. Online softmax + P fragment build
    5. GEMM2: O += P @ V_smem
    6. Epilogue: O /= l, store f32

Tensor layouts:
  Q:   [B * n_q_heads * seq_len, Dh]    bf16
  K:   [B * n_kv_heads * kv_len, Dh]    bf16
  V_t: [B * n_kv_heads * Dh, kv_len]    bf16 (transposed)
  Out: [B * n_q_heads * seq_len, Dh]    f32
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
from quark.kernels.attn.baselines import attn_baselines
from quark.kernels.attn.config import AttnConfig
from quark.kernels.attn.online_softmax_block import OnlineSoftmax
from quark.kernels.attn.problems import attn_problems
from quark.kernels.attn.reference import attn_reference_numpy
from quark.kernels.attn.spec import AttnSpec
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.gemm.mma_shapes import lookup_mma


@kernel(
    "attn",
    spec=AttnSpec,
    config=AttnConfig,
    problems=attn_problems,
    baselines=attn_baselines,
    reference=attn_reference_numpy,
)
class AttnKernel(Kernel):
    # bf16 MMA accumulates drift ~2-3e-2 from the fp32 flex_attention
    # reference — same story as owl_attn. Relax so valid tuned configs
    # aren't dropped on fourth-decimal numerical noise.
    CORRECTNESS_THRESHOLD = 0.99

    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("Q", dtype=lambda s, c: s.a_dtype, shape=lambda s, c: (s.total_q, s.Dh)),
        TensorDecl(
            "K",
            dtype=lambda s, c: s.b_dtype,
            shape=lambda s, c: (s.B * s.n_kv_heads * s.kv_len, s.Dh),
        ),
        TensorDecl(
            "V_t",
            dtype=lambda s, c: s.b_dtype,
            shape=lambda s, c: (s.B * s.n_kv_heads * s.Dh, s.kv_len),
        ),
        TensorDecl(
            "output",
            dtype=lambda s, c: s.a_dtype,
            shape=lambda s, c: (s.total_q, s.Dh),
            role="out",
        ),
    ]

    spec: AttnSpec
    config: AttnConfig

    # MMA site uses the raw a_dtype / b_dtype axes (no compute_dtype
    # fork on attn); override ``mma_sites`` to reflect that. ``_mma_cfg``
    # also overrides its fallback to match — the base default uses
    # ``spec.compute_dtype_resolved`` which doesn't exist on AttnSpec.

    def _mma_cfg(self):
        main_shape = self.config.main_shape
        if main_shape:
            from quark.ir.mma_registry import ALL_SHAPES

            for cfg in ALL_SHAPES:
                if cfg.shape_id == main_shape:
                    return cfg
            raise KeyError(f"AttnKernel: main_shape={main_shape!r} not in registry")
        return lookup_mma(self.spec.a_dtype, self.spec.b_dtype)

    @classmethod
    def mma_sites(cls, spec) -> list:
        from quark.kernels.base import MmaSite

        if spec is None:
            return []
        return [MmaSite(name="main", a_dtype=spec.a_dtype, b_dtype=spec.b_dtype)]

    def _n_warps(self) -> int:
        return self.spec.gqa_ratio * self.config.NCW

    def _block_q_rows(self) -> int:
        # Uses shape.m from the resolved MMA shape so m8n8k8 paths get
        # 8-row Q tiles. Caller-side ``MTiles`` is the number of M-tiles
        # per warp; each tile covers ``shape.m`` query rows.
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
        if s.Dh % mma.shape.m != 0 or s.Dh % mma.shape.n != 0:
            return False
        # KvTile must be a multiple of shape.n (GEMM1 N tiles) AND
        # shape.k (GEMM2 inner-K tiles — P × V iterates inner_k = KvTile // mma_k).
        if c.KvTile % mma.shape.n != 0 or c.KvTile % mma.mma_k != 0:
            return False
        if s.kv_len % c.KvTile != 0:
            return False
        bqr = self._block_q_rows()
        if s.seq_len % bqr != 0:
            return False
        if c.KvTile not in (8, 16, 32, 64, 128, 256):
            return False
        # n_stages=2 requires even KV-iter count (pipeline works in pairs)
        # and KvTile small enough that two copies of K+V stages still fit.
        if c.n_stages == 2 and (s.kv_len // c.KvTile) % 2 != 0:
            return False
        if c.n_stages not in (1, 2):
            return False
        n_warps = self._n_warps()
        if n_warps * 32 > 1024:
            return False
        n_threads = n_warps * 32
        # cp.async K tile: KvTile * Dh * 2 bytes / 16 bytes per line
        k_lines = c.KvTile * s.Dh * 2 // 16
        if k_lines > n_threads and k_lines % n_threads != 0:
            return False
        # cp.async V tile: Dh * KvTile * 2 bytes / 16 bytes per line
        v_lines = s.Dh * c.KvTile * 2 // 16
        if v_lines > n_threads and v_lines % n_threads != 0:
            return False
        # Smem alignment for cp.async
        if ((s.Dh + c.KvPad) * 2) % 16 != 0:
            return False
        if ((c.KvTile + c.KvPad) * 2) % 16 != 0:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        s = self.spec
        bqr = self._block_q_rows()
        n_q_tiles = s.seq_len // bqr
        return (n_q_tiles, s.B * s.n_kv_heads, 1)

    def block(self) -> tuple[int, int, int]:
        return (self._n_warps() * 32, 1, 1)

    def flops(self) -> int:
        s = self.spec
        # 2 GEMMs × (q_rows × kv_cols × Dh) per head
        return 2 * 2 * s.B * s.n_q_heads * s.seq_len * s.kv_len * s.Dh

    # ── Registry classmethods ──

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = AttnSpec(**problem)
        B, nqh, nkvh = spec.B, spec.n_q_heads, spec.n_kv_heads
        sl, kvl, Dh = spec.seq_len, spec.kv_len, spec.Dh
        rng = np.random.default_rng(seed)
        return {
            "Q": astype_numpy(
                (rng.standard_normal((B * nqh * sl, Dh)) * 0.3).astype(np.float32), spec.a_dtype
            ),
            "K": astype_numpy(
                (rng.standard_normal((B * nkvh * kvl, Dh)) * 0.3).astype(np.float32), spec.b_dtype
            ),
            "V_t": astype_numpy(
                (rng.standard_normal((B * nkvh * Dh, kvl)) * 0.3).astype(np.float32), spec.b_dtype
            ),
            "output": zeros_for_dtype((B * nqh * sl, Dh), spec.a_dtype),
        }

    @classmethod
    def spec_from_tensors(
        cls,
        Q,
        K,
        V_t,
        *,
        B: int,
        n_kv_heads: int,
        gqa_ratio: int,
        seq_len: int,
        kv_len: int,
    ) -> AttnSpec:
        """Derive an ``AttnSpec`` from flat kernel-shape tensors.

        Q: ``[B * n_q_heads * seq_len, Dh]``
        K: ``[B * n_kv_heads * kv_len, Dh]``
        V_t: ``[B * n_kv_heads * Dh, kv_len]`` (V transposed so ``kv_len``
            is the fast axis).

        The batch / head / seq dims can't be recovered from a flat Q
        shape alone — the caller passes them explicitly. ``Dh`` is
        taken from Q's trailing dim.
        """

        if Q.ndim != 2:
            raise ValueError(f"pcf.attention: Q must be rank-2 (flat), got {Q.shape}")
        Dh = int(Q.shape[1])
        return AttnSpec(
            B=B,
            n_kv_heads=n_kv_heads,
            gqa_ratio=gqa_ratio,
            seq_len=seq_len,
            kv_len=kv_len,
            Dh=Dh,
            a_dtype=DType.from_backend(Q),
            b_dtype=DType.from_backend(K),
        )

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        # KvTile=8 is the SIMD-native minimum on both CUDA (one m16n8
        # tile maps exactly 8 kv cols) and Metal (8x8 simdgroup_matrix
        # natural tile). Empirically, small KvTile + n_stages=2 often
        # wins because the extra outer iterations give double-buffering
        # more latency to hide. is_valid_for filters KvTile + n_stages
        # combos that blow smem budget.
        return {
            "KvTile": [8, 16, 32, 64, 128],
            "MTiles": [1, 2],
            "NCW": [1, 2],
            "KvPad": [0, 8],
            "n_stages": [1, 2],
        }

    # ── build() ──

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        Dh = s.Dh
        KvTile = c.KvTile
        MTiles = c.MTiles
        NCW = c.NCW
        GQA = s.gqa_ratio
        mma_cfg = self._mma_cfg()
        # Tile geometry from the resolved MMA shape. MTiles is in units
        # of shape.m query rows; NK / N_DH count shape.n-wide tiles.
        m_tile = mma_cfg.shape.m
        n_tile = mma_cfg.shape.n
        BlockQRows = NCW * MTiles * m_tile
        NK = KvTile // n_tile  # S n-tiles
        N_DH = Dh // n_tile  # O n-tiles
        KV_CHUNKS = s.kv_len // KvTile

        bctx = self.bctx
        ctx = self.ctx
        g_q, g_k, g_vt, g_out = g.Q, g.K, g.V_t, g.output

        # Grid decomposition — all hoisted (loop-invariant)
        q_tile_idx = ctx.block_idx("x")
        bh_idx = ctx.block_idx("y")  # b * n_kv_heads + kv_h

        # Warp decomposition. Auto-CSE dedupes the repeated bh_idx
        # divisions; operator overloading keeps the math readable.
        warp_id = bctx.warp_id
        gqa_idx = warp_id // NCW
        m_idx = warp_id % NCW

        kv_h = bh_idx % s.n_kv_heads
        b_idx = bh_idx // s.n_kv_heads
        q_head = kv_h * GQA + gqa_idx

        # Q row base for this warp
        q_row_warp = (b_idx * s.n_q_heads + q_head) * s.seq_len + (
            q_tile_idx * BlockQRows + m_idx * (MTiles * m_tile)
        )

        # K/V row bases (shared across GQA group)
        k_row_base = bh_idx * s.kv_len
        vt_row_base = bh_idx * Dh

        out_row_warp = q_row_warp

        # ── Smem: Q + output staging (kernel-lifetime); K/V per stage ──
        lcs = mma_cfg.lane_col_step
        q_smem = qk.smem_alloc("Q_smem", s.a_dtype, (BlockQRows, Dh), pad=c.KvPad)

        # cp.async Q → per-warp smem → register fragments. Q-frags are
        # resident in registers for the whole KV loop — the standard
        # flash-attn Q-in-registers pattern. ``q_smem`` is dead after
        # this point; the smem layout pass aliases it with O_stage.
        q_frags = qk.q_register_load(
            g_q,
            smem=q_smem,
            q_row=q_row_warp,
            warp_id=warp_id,
            MT=MTiles,
            Dh=Dh,
            kv_pad=c.KvPad,
        )

        # One (K, Vt) smem pair per physical pipeline stage. run_pipeline
        # rotates through them; at n_stages=2 the producer can prefetch
        # chunk N+1 while the consumer works on N.
        stages = Stage.staged(
            c.n_stages,
            k=SmemTile.spec(s.b_dtype, (KvTile, Dh), pad=c.KvPad, lane_col_step=lcs),
            vt=SmemTile.spec(s.b_dtype, (Dh, KvTile), pad=c.KvPad, lane_col_step=lcs),
        )

        # ── Accumulators per warp ──
        acc_width = mma_cfg.shape.c_regs
        o_acc = Accumulators(MT=MTiles, NT=N_DH, width=acc_width)
        s_acc = Accumulators(MT=MTiles, NT=NK, width=acc_width)
        # GEMM1: A is the register Q tile (q_frags[mt][kk]). B = K smem.
        # GEMM2: A is the P fragment from online softmax. B = V^T smem.
        # ``shape`` defaults to ``active_bctx().mma_cfg``; ``K_inner``
        # is inferred from ``b.shape[1]`` // mma_k at call time.
        mma1 = MmaBody(acc=s_acc)
        softmax = OnlineSoftmax(s_acc=s_acc, o_acc=o_acc, scale=1.0 / math.sqrt(Dh))
        mma2 = MmaBody(acc=o_acc)

        # Row classes per m-tile = distinct dr values in cd_offsets.
        # m16n8 → 2 (dr ∈ {0, 8}); m8n8k8 → 1 (dr ∈ {0}).
        n_rc = len({dr for dr, _ in mma_cfg.cd_offsets})
        n_ml = MTiles * n_rc

        # Typed carry: O is a vec-accumulator grid; m / l are per-(mt, rc)
        # scalar arrays initialized to -inf / 0 respectively. Carry.split
        # unpacks ictx.carry back into the three named groups inside
        # consume.
        carry = Carry(
            o=o_acc,
            m=(n_ml, -1e30, DType.F32),
            l=(n_ml, 0.0, DType.F32),
        )

        def produce(ictx: IterCtx) -> None:
            stage = ictx.stage
            kv_offset = ictx.iter_idx * KvTile
            # K and V are loaded together in this consume-shape. Doubles
            # the peak smem working set vs the sequential K-then-V pattern
            # but lets run_pipeline's double-buffer prefetch work cleanly.
            stage.k.load_from(g_k, row=k_row_base + kv_offset)
            stage.vt.load_from(g_vt, row=vt_row_base, col=kv_offset)

        def consume(ictx: IterCtx) -> Carry:
            stage = ictx.stage
            carry = ictx.carry  # Carry rebound to this iter's values

            # GEMM1: S = Q_regs @ K^T. A is the pre-loaded Q register
            # tile (indexed [mt][inner_k]); B is K smem.
            s_vals = mma1(a=q_frags, b=stage.k, acc=s_acc.init())
            # Online softmax — pure register math, no barrier needed.
            o_vals, m_vals, l_vals, p_frags = softmax(
                s_acc_vals=s_vals,
                o_vals=carry.o,
                m_vals=carry.m,
                l_vals=carry.l,
            )
            # GEMM2: O += P @ V. A is the P fragment from online
            # softmax (one width-N Value per (mt, kk_step)); B is V^T smem.
            carry.o = mma2(a=p_frags, b=stage.vt, acc=o_vals)
            carry.m = m_vals
            carry.l = l_vals
            return carry

        final = PipelineBody(
            stages=stages,
            produce=produce,
            consume=consume,
            carry=carry,
        ).run(n_iters=KV_CHUNKS, n_stages=c.n_stages)

        # Epilogue: O /= l, scale + cast + per-warp staged → vec_store gmem.
        # ``final`` is the Carry rebound to loop-end values — read ``.l``
        # directly. ``o_acc.results`` was populated by the pipeline
        # (_stash_results_on_carry). Staging smem is auto-allocated (AUTO
        # lifetime — the layout pass aliases it over dead upstream
        # regions); warp_id defaults to bctx.warp_id.
        qk.store_acc(
            g_out,
            o_acc,
            row=out_row_warp,
            col=0,
            cast=DType.BF16,
            per_warp=True,
            row_scale=final.l,
        )
