"""OwlAttnIntKernel — int8 attention on Intel Xe2.

PHASE 3.3a — online softmax dump path (in addition to 3.2 score
dump). The build body dispatches on ``config.phase``:

  - ``"qk_scores"``    — Phase 3.2: dump dequantized QK scores
                          ``s_f32`` ``(q_rows, capacity)`` f32.
  - ``"softmax_probs"``— Phase 3.3a: same as 3.2 through QK MMA,
                          then per-row max via subgroup_reduce,
                          subtract, exp, subgroup_reduce(sum), and
                          divide to produce normalized softmax
                          probabilities ``P`` ``(q_rows, capacity)``.
  - ``"full"``         — Phase 3.3b+: AV MMA + output cast (TODO).

Phase 3.2 algorithm:
  1. Load Q (bf16) gmem → registers.
  2. Per Q row: cross-lane absmax, derive Q_scale, quantize to s8
     in smem; lane 0 writes Q_scale[m] to a 1D smem slot.
  3. K-loop over capacity in KvTile=32 chunks:
       * cooperative load K_s8 (KvTile × Dh) + K_scales (KvTile) → smem
       * MMA: s_s32 += Q_s8 @ K_s8^T via m8n16k32 s8/s8/s32
       * scratch s_s32 → smem; per-lane scatter dequant
            ``s_f32 = s_s32 * Q_scale[m] * K_scale[n]``.

Phase 3.3a adds, after the dequant scatter is loaded back into
per-lane registers (layout: lane c, register r → s_f32[row r, col c]):
  4. m[r] = subgroup_reduce(max, s_f32_lane[r])
  5. e[r] = exp_approx(s_f32_lane[r] - m[r])
  6. l[r] = subgroup_reduce(sum, e[r])
  7. P[row r, col=lane] = e[r] / l[r], written to output gmem.

Reference: ``reference.py`` computes both score and probability
references; the active one is selected by ``spec.phase``-mirroring
config check in the smoke harness.

"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.owl_attn_int8.baselines import owl_attn_int8_baselines
from quark.kernels.owl_attn_int8.config import (
    PHASE_FULL,
    PHASE_QK_SCORES,
    PHASE_SOFTMAX_PROBS,
    VALID_PHASES,
    OwlAttnIntConfig,
)
from quark.kernels.owl_attn_int8.problems import owl_attn_int8_problems
from quark.kernels.owl_attn_int8.reference import owl_attn_int8_reference_numpy
from quark.kernels.owl_attn_int8.spec import OwlAttnIntSpec


def _output_shape(s, c):
    if c.phase in (PHASE_QK_SCORES, PHASE_SOFTMAX_PROBS):
        # Dumped QK scores / softmax probabilities: (q_rows, capacity) f32.
        return (s.q_rows, s.capacity)
    return (s.out_rows, s.out_cols)


def _output_dtype(s, c):
    if c.phase in (PHASE_QK_SCORES, PHASE_SOFTMAX_PROBS):
        return DType.F32
    return s.out_dtype


@kernel(
    "owl_attn_int8",
    spec=OwlAttnIntSpec,
    config=OwlAttnIntConfig,
    output_idx=-1,
    problems=owl_attn_int8_problems,
    baselines=lambda kernel, tensors: owl_attn_int8_baselines(tensors),
    reference=owl_attn_int8_reference_numpy,
)
class OwlAttnIntKernel(Kernel):
    # Phase 3.3b TENSORS — adds Vt_s8 + V_scales for the AV MMA + V_scale
    # fold. Phase 3.4 will add segments / n_segments / frame_t when segment-
    # sparse iteration + RoPE land.
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("Q",         dtype=lambda s, c: s.a_dtype,
                   shape=lambda s, c: (s.q_rows, s.q_cols)),
        TensorDecl(
            "K_s8",     dtype=lambda s, c: DType.S8,
            shape=lambda s, c: (s.B * s.n_kv_heads * s.capacity, s.Dh),
        ),
        TensorDecl(
            "K_scales", dtype=lambda s, c: s.scale_dtype,
            shape=lambda s, c: (s.B * s.n_kv_heads * s.capacity,),
        ),
        TensorDecl(
            "Vt_s8",    dtype=lambda s, c: DType.S8,
            shape=lambda s, c: (s.B * s.n_kv_heads * s.Dh, s.capacity),
        ),
        TensorDecl(
            "V_scales", dtype=lambda s, c: s.scale_dtype,
            shape=lambda s, c: (s.B * s.n_kv_heads * s.capacity,),
        ),
        TensorDecl("frame_t", dtype=DType.S32, shape=lambda s, c: (1,)),
        TensorDecl(
            "output", dtype=_output_dtype, shape=_output_shape, role="out",
        ),
    ]

    spec: OwlAttnIntSpec
    config: OwlAttnIntConfig

    # MMA site is fixed to int8 m8n16k32.
    @classmethod
    def mma_sites(cls, spec) -> list:
        from quark.kernels.base import MmaSite
        return [MmaSite(name="main", a_dtype=DType.S8, b_dtype=DType.S8)]

    def _mma_cfg(self):
        from quark.ir.mma_registry import _BY_SHAPE_ID
        cfg = _BY_SHAPE_ID.get("m8n16k32_intel_s8_s32")
        if cfg is None:
            raise RuntimeError(
                "OwlAttnIntKernel: m8n16k32_intel_s8_s32 not in registry"
            )
        return cfg

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if c.phase not in VALID_PHASES:
            return False
        mma = self._mma_cfg()
        m_tile = mma.shape.m  # 8
        n_tile = mma.shape.n  # 16
        k_tile = mma.shape.k  # 32
        # AV MMA: N axis = Dh, must be multiple of n_tile=16.
        if c.phase == PHASE_FULL and s.Dh % n_tile != 0:
            return False
        if c.KvTile != k_tile:
            return False
        if c.MTiles < 1 or c.NCW < 1:
            return False
        # Phase 3.6a: allow NCW in {1, 2, 4, 8}. Multi-warp shares K/V
        # smem loads between warps; each warp owns BlockQRows/NCW rows.
        if c.NCW not in (1, 2, 4, 8):
            return False
        # Each warp owns ``c.MTiles * m_tile`` Q rows. tpf must hold
        # a whole multiple of BlockQRows.
        bqr = c.NCW * c.MTiles * m_tile
        if s.tpf % bqr != 0:
            return False
        # Dh must be a multiple of MMA k_tile (== KvTile).
        if s.Dh % k_tile != 0:
            return False
        # KvTile divides n_tile cleanly (== 32/16 = 2 n-tiles per K-iter).
        if c.KvTile % n_tile != 0:
            return False
        # Capacity must be a whole multiple of KvTile.
        if s.capacity % c.KvTile != 0:
            return False
        # Dh must be a multiple of sgs so Q quant lanes split evenly.
        sgs = self.resolve_subgroup_size()
        if s.Dh % sgs != 0:
            return False
        return True

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {
            "KvTile": [32], "MTiles": [1], "NCW": [1, 2, 4], "KvPad": [0],
            "n_stages": [1], "phase": [PHASE_FULL],
        }

    def grid(self) -> tuple[int, int, int]:
        s, c = self.spec, self.config
        m_tile = self._mma_cfg().shape.m
        bqr = c.NCW * c.MTiles * m_tile
        # Mirror OwlAttn grid: (n_q_tiles, B * n_kv_heads, gqa_ratio_for_unpacked).
        # For unpacked Q the q-head axis collapses into the q-row index
        # ``q_row_base = (b*n_q_heads + q_head)*tpf + token_off``, so we
        # need an extra grid axis to cover all q_heads. Single-warp +
        # gqa_ratio=1 → grid.z = 1.
        return (s.tpf // bqr, s.B * s.n_kv_heads * s.gqa_ratio, 1)

    def block(self) -> tuple[int, int, int]:
        # Phase 3.5a: gqa heads are expanded onto grid.y (each WG handles
        # one (b, kv_h, q_head_in_gqa) triple), not onto warps within a
        # block. Block has c.NCW warps regardless of gqa_ratio.
        n_warps = self.config.NCW
        return (n_warps * self.resolve_subgroup_size(), 1, 1)

    def flops(self) -> int:
        s = self.spec
        # QK only (Phase 3.2): 2 ops/element across all (Q, capacity, Dh).
        return 2 * s.B * s.n_q_heads * s.tpf * s.capacity * s.Dh

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED, cfg=None) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = OwlAttnIntSpec(**problem)
        cfg = cfg if cfg is not None else OwlAttnIntConfig()
        rng = np.random.default_rng(seed)
        q_np = rng.standard_normal((spec.q_rows, spec.q_cols)).astype(np.float32) * 0.02
        k_s8 = rng.integers(
            -64, 64,
            size=(spec.B * spec.n_kv_heads * spec.capacity, spec.Dh),
            dtype=np.int8,
        )
        k_scales = rng.uniform(
            0.001, 0.01,
            size=(spec.B * spec.n_kv_heads * spec.capacity,),
        ).astype(np.float32)
        vt_s8 = rng.integers(
            -64, 64,
            size=(spec.B * spec.n_kv_heads * spec.Dh, spec.capacity),
            dtype=np.int8,
        )
        v_scales = rng.uniform(
            0.001, 0.01,
            size=(spec.B * spec.n_kv_heads * spec.capacity,),
        ).astype(np.float32)
        out_shape = _output_shape(spec, cfg)
        out_dtype = _output_dtype(spec, cfg)
        # Use a steady-state frame_t (matches OwlAttn's test fixture).
        frame_t = np.array([spec.num_buckets * spec.pinned_dilation],
                           dtype=np.int32)
        return {
            "Q":          astype_numpy(q_np, spec.a_dtype.value),
            "K_s8":       k_s8,
            "K_scales":   k_scales,
            "Vt_s8":      vt_s8,
            "V_scales":   v_scales,
            "frame_t":    frame_t,
            "output":     zeros_for_dtype(out_shape, out_dtype.value),
        }

    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        bctx = self.bctx
        mma_cfg = self._mma_cfg()
        shape_id = mma_cfg.shape.name      # "m8n16k32_intel_s8_s32"
        m_tile = mma_cfg.shape.m            # 8
        n_tile = mma_cfg.shape.n            # 16
        k_tile = mma_cfg.shape.k            # 32

        Dh = s.Dh
        KvTile = c.KvTile                   # 32 == k_tile
        capacity = s.capacity
        # Phase 3.5a: gqa heads are expanded onto grid.y, not warps.
        # NumWarps == c.NCW (block has NCW warps regardless of gqa).
        NumWarps = c.NCW
        sgs = self.resolve_subgroup_size()
        n_threads = NumWarps * sgs

        BlockQRows = c.NCW * c.MTiles * m_tile
        warp_q_rows = c.MTiles * m_tile
        n_k_iters = capacity // KvTile      # python-side unroll count
        NK = KvTile // n_tile               # 2 n-tiles per KvTile iter
        KK = Dh // k_tile                   # k-tiles per Dh (Dh=64 → 2)

        # ── Grid decomposition ──
        # Mirror OwlAttn grid but expand grid.y to also cover the
        # gqa_ratio axis (unpacked Q layout, gqa_ratio=1 first cut).
        q_tile_idx = qk.block_idx("x")
        bhq_idx = qk.block_idx("y")          # b * n_kv_heads * gqa_ratio + ...

        warp_id = bctx.warp_id
        gqa_idx = warp_id // bctx.c(c.NCW, dtype=DType.U32)
        m_idx = warp_id % bctx.c(c.NCW, dtype=DType.U32)

        # Unflatten bhq_idx into (b, kv_h, gqa_in_grid). Since
        # n_warps_per_block = 1 we don't fold gqa across warps; instead
        # the grid.y stride covers it.
        n_q_heads_total = s.n_kv_heads * s.gqa_ratio
        # bhq_idx = b * (n_kv_heads * gqa_ratio) + kv_h * gqa_ratio + g
        q_head_global = bhq_idx % bctx.c(n_q_heads_total, dtype=DType.U32)
        b_idx = bhq_idx // bctx.c(n_q_heads_total, dtype=DType.U32)
        kv_h = q_head_global // bctx.c(s.gqa_ratio, dtype=DType.U32)
        # q_head = q_head_global (full head idx within batch)
        q_head = q_head_global

        token_tile_offset = (
            q_tile_idx * bctx.c(BlockQRows, dtype=DType.U32)
            + m_idx * bctx.c(c.MTiles * m_tile, dtype=DType.U32)
        )

        # Q row + col bases — branch on packed_qkv:
        #   Unpacked: Q = (B*n_q_heads*tpf, Dh)        — row encodes (b, q_head, t).
        #   Packed:   Q = (B*tpf, n_q_heads*Dh + 2*n_kv_heads*Dh)
        #                  — row encodes (b, t); col encodes q_head head-slice.
        # ``out_row_base`` / ``out_col_base`` mirror the same layout.
        if s.packed_qkv:
            q_row_base = (
                b_idx * bctx.c(s.tpf, dtype=DType.U32) + token_tile_offset
            )
            q_col_base = q_head * bctx.c(s.Dh, dtype=DType.U32)
            out_row_base = q_row_base
            out_col_base = q_head * bctx.c(s.Dh, dtype=DType.U32)
        else:
            q_row_base = (
                (b_idx * bctx.c(s.n_q_heads, dtype=DType.U32) + q_head)
                * bctx.c(s.tpf, dtype=DType.U32)
                + token_tile_offset
            )
            q_col_base = bctx.c(0, dtype=DType.U32)
            out_row_base = q_row_base
            out_col_base = bctx.c(0, dtype=DType.U32)
        # K_s8 row base: (b * n_kv_heads + kv_h) * capacity
        k_row_base = (
            (b_idx * bctx.c(s.n_kv_heads, dtype=DType.U32) + kv_h)
            * bctx.c(capacity, dtype=DType.U32)
        )

        # ── Smem allocs ──
        q_s8_smem = qk.smem_alloc("Q_s8_smem", DType.S8, (BlockQRows, Dh))
        q_scale_smem = qk.smem_alloc("Q_scale_smem", DType.F32, (BlockQRows,))
        k_s8_smem = qk.smem_alloc("K_s8_smem", DType.S8, (KvTile, Dh))
        k_scale_smem = qk.smem_alloc("K_scale_smem", DType.F32, (KvTile,))
        s_s32_smem = qk.smem_alloc("S_s32_smem", DType.S32, (BlockQRows, KvTile))
        # Phase 3.3b+ extras: V tile, P quant scratch, O scratch.
        # Phase 3.4a keeps p_scale and l in registers (carry, not smem)
        # since they're per-row + broadcast-identical across lanes.
        if c.phase == PHASE_FULL:
            vt_s8_smem = qk.smem_alloc("Vt_s8_smem", DType.S8, (Dh, KvTile))
            p_s8_smem = qk.smem_alloc("P_s8_smem", DType.S8, (BlockQRows, KvTile))
            o_s32_smem = qk.smem_alloc("O_s32_smem", DType.S32, (BlockQRows, Dh))

        # Common constants.
        zero_u = bctx.c(0, dtype=DType.U32)
        zero_f = bctx.c(0.0, dtype=DType.F32)
        one_f = bctx.c(1.0, dtype=DType.F32)
        c127 = bctx.c(127.0, dtype=DType.F32)
        c_neg127 = bctx.c(-127.0, dtype=DType.F32)
        c_half = bctx.c(0.5, dtype=DType.F32)
        c_neg_half = bctx.c(-0.5, dtype=DType.F32)
        sgs_c = bctx.c(sgs, dtype=DType.U32)

        # ── 1. Q quant-on-load ──
        # One warp = one SIMD32 wave. For each of warp_q_rows rows the
        # wave cooperatively (a) reads ``Dh`` elements, (b) computes
        # cross-lane absmax, (c) derives scale = absmax / 127, (d)
        # quantizes to s8 and stores to q_s8_smem, (e) lane 0 stores
        # the f32 scale to q_scale_smem.
        lane = bctx.tid % sgs_c
        epl_q = Dh // sgs                        # = 2 at Dh=64
        epl_q_c = bctx.c(epl_q, dtype=DType.U32)
        warp_smem_row_base = warp_id * bctx.c(warp_q_rows, dtype=DType.U32)

        if c.phase == PHASE_FULL:
            # Phase 3.5b: Q-side ortho-RoPE on-load.
            # OwlAttn's gmem Q is laid out interleaved (pairs at cols
            # 2c, 2c+1); RoPE rotates each pair into (y0, y1) which
            # land at the "concat" output positions (c, c+half_Dh).
            # We apply RoPE in f32 BEFORE absmax/quantize so the int8
            # path's Q row matches what OwlAttn's bf16 kernel computes.
            from quark.lang.rope import emit_rope_cos_sin

            half_Dh = Dh // 2
            half_Dh_c = bctx.c(half_Dh, dtype=DType.U32)
            pairs_per_lane = half_Dh // sgs
            if pairs_per_lane * sgs != half_Dh:
                raise RuntimeError(
                    f"Phase 3.5b RoPE: half_Dh={half_Dh} must be a "
                    f"multiple of sgs={sgs}"
                )

            frame_t_v = qk.load(g.frame_t, bctx.c(0, dtype=DType.S32))
            frame_t_u = qk.convert(frame_t_v, DType.U32)
            W_c = bctx.c(s.W_spatial, dtype=DType.U32)
            two_u = bctx.c(2, dtype=DType.U32)

            for r in range(warp_q_rows):
                r_c = bctx.c(r, dtype=DType.U32)
                qrow_smem = warp_smem_row_base + r_c
                qrow_gmem = q_row_base + r_c

                # Spatial index for this Q row's RoPE — token offset
                # within one frame in row-major (h, w) order. For
                # NCW=1, MTiles=1, single-warp: just
                # ``token_tile_offset + r`` (no m_idx fold).
                spatial_idx = token_tile_offset + r_c
                h_idx = spatial_idx // W_c
                w_idx = spatial_idx % W_c

                # Read all (x0, x1) pairs this lane owns, apply RoPE
                # in f32, collect (y0, y1) values for absmax + quant.
                y0_vals: list = []
                y1_vals: list = []
                c_idx_list: list = []
                for pp in range(pairs_per_lane):
                    c_idx = (
                        lane * bctx.c(pairs_per_lane, dtype=DType.U32)
                        + bctx.c(pp, dtype=DType.U32)
                    )
                    c0 = c_idx * two_u
                    c1 = c0 + bctx.c(1, dtype=DType.U32)
                    x0 = qk.convert(
                        g.Q[qrow_gmem, q_col_base + c0], DType.F32
                    )
                    x1 = qk.convert(
                        g.Q[qrow_gmem, q_col_base + c1], DType.F32
                    )
                    cv, sv = emit_rope_cos_sin(
                        bctx, h_idx=h_idx, w_idx=w_idx,
                        frame_t=frame_t_u, c_idx=c_idx,
                        H=s.H_spatial, W=s.W_spatial, Dh=Dh,
                    )
                    y0 = x0 * cv - x1 * sv
                    y1 = x1 * cv + x0 * sv
                    y0_vals.append(y0)
                    y1_vals.append(y1)
                    c_idx_list.append(c_idx)

                # Per-lane absmax over all (y0, y1) the lane holds.
                lane_max = qk.max(qk.abs_(y0_vals[0]), qk.abs_(y1_vals[0]))
                for pp in range(1, pairs_per_lane):
                    lane_max = qk.max(lane_max, qk.abs_(y0_vals[pp]))
                    lane_max = qk.max(lane_max, qk.abs_(y1_vals[pp]))
                abs_max = qk.subgroup_reduce("max", lane_max)

                cmp_zero = qk.cmp("eq", abs_max, zero_f)
                scale = qk.select(
                    cmp_zero, one_f, qk.div(abs_max, c127)
                )
                inv_scale = qk.div(one_f, scale)

                # Quantize + store. ``y0`` goes to col ``c_idx``;
                # ``y1`` to col ``c_idx + half_Dh`` (concat layout).
                for pp in range(pairs_per_lane):
                    c_idx = c_idx_list[pp]
                    for y, col_v in (
                        (y0_vals[pp], c_idx),
                        (y1_vals[pp], c_idx + half_Dh_c),
                    ):
                        x_scaled = y * inv_scale
                        is_neg = qk.cmp("lt", x_scaled, zero_f)
                        x_rnd = x_scaled + qk.select(
                            is_neg, c_neg_half, c_half
                        )
                        x_clamped = qk.max(
                            c_neg127, qk.min(c127, x_rnd)
                        )
                        q_i = qk.convert(x_clamped, DType.S8)
                        q_s8_smem[qrow_smem, col_v] = q_i

                is_lane0 = qk.cmp("eq", lane, zero_u)
                qk.store(q_scale_smem, scale, qrow_smem, pred=is_lane0)
        else:
            # Phase 3.2/3.3a (diagnostic phases): no RoPE pass.
            for r in range(warp_q_rows):
                r_c = bctx.c(r, dtype=DType.U32)
                qrow_smem = warp_smem_row_base + r_c
                qrow_gmem = q_row_base + r_c

                vals = []
                for i in range(epl_q):
                    col_in_head = lane * epl_q_c + bctx.c(i, dtype=DType.U32)
                    v = g.Q[qrow_gmem, q_col_base + col_in_head]
                    vals.append(qk.convert(v, DType.F32))

                lane_max = qk.abs_(vals[0])
                for i in range(1, epl_q):
                    lane_max = qk.max(lane_max, qk.abs_(vals[i]))
                abs_max = qk.subgroup_reduce("max", lane_max)

                cmp_zero = qk.cmp("eq", abs_max, zero_f)
                scale = qk.select(cmp_zero, one_f, qk.div(abs_max, c127))
                inv_scale = qk.div(one_f, scale)

                for i in range(epl_q):
                    col = lane * epl_q_c + bctx.c(i, dtype=DType.U32)
                    x = vals[i] * inv_scale
                    is_neg = qk.cmp("lt", x, zero_f)
                    x_rnd = x + qk.select(is_neg, c_neg_half, c_half)
                    x_clamped = qk.max(c_neg127, qk.min(c127, x_rnd))
                    q_i = qk.convert(x_clamped, DType.S8)
                    q_s8_smem[qrow_smem, col] = q_i

                is_lane0 = qk.cmp("eq", lane, zero_u)
                qk.store(q_scale_smem, scale, qrow_smem, pred=is_lane0)

        qk.barrier("block")

        # ── 2. K-loop ──
        # Per-warp accumulator grid: MTiles × NK s32 fragments.
        warp_n_tile_base = warp_id * bctx.c(NK, dtype=DType.U32)
        # PHASE_FULL pre-K-loop init: per-row online softmax carry +
        # per-lane running o_f32 accumulator + AV-MMA-side constants.
        if c.phase == PHASE_FULL:
            NK_av = Dh // n_tile           # n-tiles along the AV N axis
            K_steps_av = KvTile // k_tile  # AV K-inner steps per KvTile
            vt_row_base = (
                (b_idx * bctx.c(s.n_kv_heads, dtype=DType.U32) + kv_h)
                * bctx.c(Dh, dtype=DType.U32)
            )
            # Per-row carry — values broadcast-identical across lanes
            # (sourced from subgroup_reduce within a warp). One Python
            # entry per row WITHIN A WARP (warp_q_rows entries). Each
            # warp's lanes carry their own warp's row values.
            neg_inf_f = bctx.c(-1e30, dtype=DType.F32)
            m_per_row = [neg_inf_f for _ in range(warp_q_rows)]
            l_per_row = [bctx.c(0.0, dtype=DType.F32)
                         for _ in range(warp_q_rows)]
            # Per-lane running o_f32 register array. With the gemm-int
            # scatter layout (Dh=64, KvTile=32, sgs=32, NumWarps=1):
            #   flat = lane + i*32; row = flat // Dh = i // 2;
            #   col  = flat %  Dh = lane + (i%2)*32.
            # So slots 2k and 2k+1 share row k.
            epl_o_init = (BlockQRows * Dh) // n_threads
            o_f32_per_lane = [bctx.c(0.0, dtype=DType.F32)
                              for _ in range(epl_o_init)]
            # Dh divisor used in both the per-iter scatter and the
            # post-K-loop epilogue scatter.
            dh_c = bctx.c(Dh, dtype=DType.U32)
            KvTile_c = bctx.c(KvTile, dtype=DType.U32)
            kt_c = bctx.c(KvTile, dtype=DType.U32)

            # ── PHASE 3.4b — runtime for_range K-loop ──
            # The Python-unrolled K-loop below blows IR at the saturated
            # KV shape (n_k_iters=272 × ~50 ops/iter = ~14k IR ops). The
            # runtime loop emits one body and threads (m_per_row +
            # l_per_row + o_f32_per_lane) through ``carried=`` so the
            # IR stays compact.
            carry_init = tuple(m_per_row + l_per_row + o_f32_per_lane)
            with qk.for_range(
                0, n_k_iters, 1, iv_name="k", carried=carry_init,
            ) as (iv, body_carried):
                # Unpack carry from the for-loop frame (warp-local).
                m_per_row_b = list(body_carried[0:warp_q_rows])
                l_per_row_b = list(
                    body_carried[warp_q_rows:2 * warp_q_rows]
                )
                o_f32_per_lane_b = list(body_carried[2 * warp_q_rows:])

                kv_off_c = iv * KvTile_c   # runtime KV offset

                # ── 2a. K_s8 cooperative load ──
                n_k_elems = KvTile * Dh
                if n_k_elems % n_threads != 0:
                    raise RuntimeError(
                        f"OwlAttnInt FULL: KvTile*Dh={n_k_elems} not "
                        f"divisible by n_threads={n_threads}"
                    )
                epl_k = n_k_elems // n_threads
                for i in range(epl_k):
                    flat = bctx.tid + bctx.c(i * n_threads, dtype=DType.U32)
                    row_local = flat // dh_c
                    col_local = flat % dh_c
                    gmem_row = k_row_base + kv_off_c + row_local
                    k_s8_smem[row_local, col_local] = g.K_s8[gmem_row, col_local]

                # ── 2b. K_scales cooperative load ──
                if KvTile <= n_threads:
                    tid_u = bctx.tid
                    in_range = qk.cmp("lt", tid_u, KvTile_c)
                    ksc_val = g.K_scales[k_row_base + kv_off_c + tid_u]
                    qk.store(k_scale_smem, ksc_val, tid_u, pred=in_range)
                else:
                    raise NotImplementedError("KvTile > n_threads")

                qk.barrier("block")

                # ── 2c. QK MMA ──
                zero_s32 = bctx.c(0, dtype=DType.S32)
                c_frags = [
                    qk.vec_build([zero_s32] * mma_cfg.shape.c_regs)
                    for _ in range(c.MTiles * NK)
                ]
                # Phase 3.6a: per-warp partitioning. Each warp's QK MMA
                # produces its own M-slice (rows warp_smem_row_base..
                # +warp_q_rows). K (= N axis of QK MMA) is SHARED across
                # warps: both warps' MMAs read the same b_frags.
                for kk in range(KK):
                    kk_col = bctx.c(kk * k_tile, dtype=DType.U32)
                    a_frags = [
                        qk.load_matrix(
                            q_s8_smem, shape_id, which="a",
                            row=warp_smem_row_base
                                + bctx.c(mt * m_tile, dtype=DType.U32),
                            col=kk_col, reg_offsets=mma_cfg.a_offsets,
                        )
                        for mt in range(c.MTiles)
                    ]
                    b_frags = [
                        qk.load_matrix(
                            k_s8_smem, shape_id, which="b",
                            row=bctx.c(nt * n_tile, dtype=DType.U32),
                            col=kk_col, reg_offsets=mma_cfg.b_offsets,
                        )
                        for nt in range(NK)
                    ]
                    for mt in range(c.MTiles):
                        for nt in range(NK):
                            idx = mt * NK + nt
                            c_frags[idx] = qk.mma(
                                shape_id, a_frags[mt], b_frags[nt], c_frags[idx]
                            )

                # ── 2d. Dump c_frags → s_s32_smem ──
                for mt in range(c.MTiles):
                    for nt in range(NK):
                        idx = mt * NK + nt
                        qk.store_matrix(
                            s_s32_smem, c_frags[idx], shape_id, "d",
                            row=warp_smem_row_base
                                + bctx.c(mt * m_tile, dtype=DType.U32),
                            col=bctx.c(nt * n_tile, dtype=DType.U32),
                            reg_offsets=mma_cfg.cd_offsets,
                        )
                qk.barrier("block")

                # ── 2e. Per-warp scatter → s_f32 register array ──
                # Per-warp slice: warp_q_rows × KvTile. With sgs lanes:
                # lane c, iter i: flat=c+i*sgs; row_in_warp=flat//KvTile;
                # col=flat%KvTile; full smem row = warp_smem_row_base +
                # row_in_warp.
                n_elems_warp = warp_q_rows * KvTile
                epl_s_b = n_elems_warp // sgs
                s_f32_lane_b = []
                row_locals_b = []
                col_locals_b = []
                for i in range(epl_s_b):
                    flat = lane + bctx.c(i * sgs, dtype=DType.U32)
                    row_in_warp = flat // kt_c
                    col_local = flat % kt_c
                    row_local = warp_smem_row_base + row_in_warp
                    s32_val = s_s32_smem[row_local, col_local]
                    q_sc = qk.load(q_scale_smem, row_local)
                    k_sc = qk.load(k_scale_smem, col_local)
                    f32_val = qk.convert(s32_val, DType.F32) * q_sc * k_sc
                    s_f32_lane_b.append(f32_val)
                    row_locals_b.append(row_local)
                    col_locals_b.append(col_local)

                # ── 2f. Online softmax with carry update ──
                rescale_per_row_b: list = []
                e_per_lane_b: list = []
                for i in range(epl_s_b):
                    m_iter = qk.subgroup_reduce("max", s_f32_lane_b[i])
                    m_new = qk.max(m_per_row_b[i], m_iter)
                    rescale_i = qk.exp_approx(m_per_row_b[i] - m_new)
                    e_i = qk.exp_approx(s_f32_lane_b[i] - m_new)
                    l_local_i = qk.subgroup_reduce("sum", e_i)
                    m_per_row_b[i] = m_new
                    l_per_row_b[i] = l_per_row_b[i] * rescale_i + l_local_i
                    rescale_per_row_b.append(rescale_i)
                    e_per_lane_b.append(e_i)

                # ── 2g. V_scale fold + P quant → p_s8_smem ──
                p_scale_per_row_b: list = []
                for i in range(epl_s_b):
                    v_sc = g.V_scales[k_row_base + kv_off_c + col_locals_b[i]]
                    p_scaled = e_per_lane_b[i] * v_sc
                    p_abs_max = qk.subgroup_reduce("max", qk.abs_(p_scaled))
                    p_z = qk.cmp("eq", p_abs_max, zero_f)
                    p_scale = qk.select(p_z, one_f, qk.div(p_abs_max, c127))
                    inv_p_scale = qk.div(one_f, p_scale)
                    x = p_scaled * inv_p_scale
                    is_neg_p = qk.cmp("lt", x, zero_f)
                    x_rnd = x + qk.select(is_neg_p, c_neg_half, c_half)
                    x_clamped = qk.max(c_neg127, qk.min(c127, x_rnd))
                    p_s8 = qk.convert(x_clamped, DType.S8)
                    p_s8_smem[row_locals_b[i], col_locals_b[i]] = p_s8
                    p_scale_per_row_b.append(p_scale)

                qk.barrier("block")

                # ── 2h. Cooperative Vt_s8 load ──
                n_vt_elems = Dh * KvTile
                epl_vt = n_vt_elems // n_threads
                for i in range(epl_vt):
                    flat = bctx.tid + bctx.c(i * n_threads, dtype=DType.U32)
                    row_local = flat // kt_c
                    col_local = flat % kt_c
                    vt_s8_smem[row_local, col_local] = g.Vt_s8[
                        vt_row_base + row_local, kv_off_c + col_local
                    ]

                qk.barrier("block")

                # ── 2i. AV MMA → o_s32_smem ──
                zero_s32_iter = bctx.c(0, dtype=DType.S32)
                o_s32_frags_iter = [
                    qk.vec_build([zero_s32_iter] * mma_cfg.shape.c_regs)
                    for _ in range(c.MTiles * NK_av)
                ]
                # AV MMA — per-warp M-slice; Vt shared across warps.
                for ks in range(K_steps_av):
                    ks_col = bctx.c(ks * k_tile, dtype=DType.U32)
                    a_frags_av = [
                        qk.load_matrix(
                            p_s8_smem, shape_id, which="a",
                            row=warp_smem_row_base
                                + bctx.c(mt * m_tile, dtype=DType.U32),
                            col=ks_col, reg_offsets=mma_cfg.a_offsets,
                        )
                        for mt in range(c.MTiles)
                    ]
                    b_frags_av = [
                        qk.load_matrix(
                            vt_s8_smem, shape_id, which="b",
                            row=bctx.c(nt * n_tile, dtype=DType.U32),
                            col=ks_col, reg_offsets=mma_cfg.b_offsets,
                        )
                        for nt in range(NK_av)
                    ]
                    for mt in range(c.MTiles):
                        for nt in range(NK_av):
                            idx = mt * NK_av + nt
                            o_s32_frags_iter[idx] = qk.mma(
                                shape_id,
                                a_frags_av[mt], b_frags_av[nt],
                                o_s32_frags_iter[idx],
                            )

                for mt in range(c.MTiles):
                    for nt in range(NK_av):
                        idx = mt * NK_av + nt
                        qk.store_matrix(
                            o_s32_smem, o_s32_frags_iter[idx],
                            shape_id, "d",
                            row=warp_smem_row_base
                                + bctx.c(mt * m_tile, dtype=DType.U32),
                            col=bctx.c(nt * n_tile, dtype=DType.U32),
                            reg_offsets=mma_cfg.cd_offsets,
                        )
                qk.barrier("block")

                # ── 2j. Per-warp scatter — dequant + fold into o_f32 carry ──
                # Per-warp slice: warp_q_rows × Dh = 8 × 64 = 512 elems
                # at NCW=2. epl_o = 16 per lane. flat=lane+i*sgs.
                rows_per_lane_block = Dh // sgs  # = 2 at Dh=64 sgs=32
                n_o_elems_warp = warp_q_rows * Dh
                epl_o = n_o_elems_warp // sgs
                for i in range(epl_o):
                    flat = lane + bctx.c(i * sgs, dtype=DType.U32)
                    row_in_warp = flat // dh_c
                    col_local = flat % dh_c
                    row_local = warp_smem_row_base + row_in_warp
                    row_idx_py = i // rows_per_lane_block
                    o_s32_val = o_s32_smem[row_local, col_local]
                    o_tile_f32 = qk.convert(o_s32_val, DType.F32) \
                                 * p_scale_per_row_b[row_idx_py]
                    o_f32_per_lane_b[i] = (
                        o_f32_per_lane_b[i] * rescale_per_row_b[row_idx_py]
                        + o_tile_f32
                    )

                # ── 2k. Yield new carry ──
                qk.yield_(*m_per_row_b, *l_per_row_b, *o_f32_per_lane_b)

            # ── 3. Post-for_range: final carry + output epilogue ──
            # Carry tuple is per-warp (m_per_row + l_per_row over
            # warp_q_rows entries each + o_f32_per_lane over epl_o entries).
            final_carry = self.bctx.bld.last_results
            m_per_row = list(final_carry[0:warp_q_rows])
            l_per_row = list(final_carry[warp_q_rows:2 * warp_q_rows])
            o_f32_per_lane = list(final_carry[2 * warp_q_rows:])

            rows_per_lane_block = Dh // sgs  # = 2 at Dh=64 sgs=32
            n_o_elems_warp_final = warp_q_rows * Dh
            epl_o_final = n_o_elems_warp_final // sgs
            for i in range(epl_o_final):
                flat = lane + bctx.c(i * sgs, dtype=DType.U32)
                row_in_warp = flat // dh_c
                col_local = flat % dh_c
                row_idx_py = i // rows_per_lane_block
                o_final = qk.div(o_f32_per_lane[i], l_per_row[row_idx_py])
                if s.out_dtype is DType.F32:
                    out_val = o_final
                else:
                    out_val = qk.convert(o_final, s.out_dtype)
                gmem_row = out_row_base + row_in_warp
                gmem_col = out_col_base + col_local
                g.output[gmem_row, gmem_col] = out_val
            return  # PHASE_FULL handled — skip the Python K-loop below.

        for k_iter in range(n_k_iters):
            kv_off = k_iter * KvTile  # python-side constant
            kv_off_c = bctx.c(kv_off, dtype=DType.U32)

            # ── 2a. Cooperative load K_s8 (KvTile × Dh) into smem ──
            # n_threads loads (KvTile*Dh)/n_threads elements each.
            n_k_elems = KvTile * Dh
            if n_k_elems % n_threads != 0:
                raise RuntimeError(
                    f"OwlAttnInt: KvTile*Dh={n_k_elems} must be divisible "
                    f"by n_threads={n_threads}"
                )
            epl_k = n_k_elems // n_threads
            dh_c = bctx.c(Dh, dtype=DType.U32)
            for i in range(epl_k):
                flat = bctx.tid + bctx.c(i * n_threads, dtype=DType.U32)
                row_local = flat // dh_c
                col_local = flat % dh_c
                gmem_row = k_row_base + kv_off_c + row_local
                k_s8_smem[row_local, col_local] = g.K_s8[gmem_row, col_local]

            # ── 2b. Load K_scales (KvTile values) into smem ──
            # KvTile=32 == sgs=32: one scale per lane (multi-warp pattern
            # adapts if needed). Use warp 0 (or all warps redundantly).
            if KvTile <= n_threads:
                tid_u = bctx.tid
                in_range = qk.cmp("lt", tid_u, bctx.c(KvTile, dtype=DType.U32))
                k_sc_idx = k_row_base + kv_off_c + tid_u
                # Predicated gmem→smem hop via register.
                # qk.load on gmem isn't strictly needed — we can do
                # `g.K_scales[k_sc_idx]` directly. Guard via predicated
                # store. The OOB lanes' read may be UB; for KvTile==32
                # all lanes are in-range.
                ksc_val = g.K_scales[k_sc_idx]
                qk.store(k_scale_smem, ksc_val, tid_u, pred=in_range)
            else:
                raise NotImplementedError("KvTile > n_threads not yet supported")

            qk.barrier("block")

            # ── 2c. MMA: s_s32 = Q_s8 @ K_s8^T ──
            zero_s32 = bctx.c(0, dtype=DType.S32)
            c_frags = [
                qk.vec_build([zero_s32] * mma_cfg.shape.c_regs)
                for _ in range(c.MTiles * NK)
            ]
            for kk in range(KK):
                kk_col = bctx.c(kk * k_tile, dtype=DType.U32)
                a_frags = [
                    qk.load_matrix(
                        q_s8_smem, shape_id, which="a",
                        row=bctx.c(mt * m_tile, dtype=DType.U32),
                        col=kk_col,
                        reg_offsets=mma_cfg.a_offsets,
                    )
                    for mt in range(c.MTiles)
                ]
                # B convention (from GemmInt): B stored as (N, K_inner)
                # row-major. K_s8 here is (KvTile, Dh) which already
                # matches that — rows = tokens (N axis), cols = Dh
                # (K_inner axis).
                b_frags = [
                    qk.load_matrix(
                        k_s8_smem, shape_id, which="b",
                        row=(warp_n_tile_base + bctx.c(nt, dtype=DType.U32))
                            * bctx.c(n_tile, dtype=DType.U32),
                        col=kk_col,
                        reg_offsets=mma_cfg.b_offsets,
                    )
                    for nt in range(NK)
                ]
                for mt in range(c.MTiles):
                    for nt in range(NK):
                        idx = mt * NK + nt
                        c_frags[idx] = qk.mma(
                            shape_id, a_frags[mt], b_frags[nt], c_frags[idx]
                        )

            # ── 2d. Dump c_frags → s_s32 smem ──
            for mt in range(c.MTiles):
                for nt in range(NK):
                    idx = mt * NK + nt
                    col_abs = (warp_n_tile_base + bctx.c(nt, dtype=DType.U32)) \
                              * bctx.c(n_tile, dtype=DType.U32)
                    qk.store_matrix(
                        s_s32_smem, c_frags[idx],
                        shape_id, "d",
                        row=bctx.c(mt * m_tile, dtype=DType.U32),
                        col=col_abs,
                        reg_offsets=mma_cfg.cd_offsets,
                    )
            qk.barrier("block")

            # ── 2e. Per-lane scatter: dequant ──
            # Layout: BlockQRows rows × KvTile cols, distributed across
            # n_threads=NumWarps*sgs lanes. With NumWarps=1 (Phase 3.2/3.3a
            # single-warp scope), n_threads = sgs = 32. For each lane c:
            #   flat = c + i*32  →  row = i,  col = c (since KvTile == 32).
            # So per-lane register array of size epl_s = BlockQRows holds
            # ``s_f32[row=i, col=lane]`` at index ``i``. This row-per-register
            # layout is what the softmax path below needs to compute per-row
            # reductions via subgroup_reduce across lanes (= across cols).
            n_elems = BlockQRows * KvTile
            if n_elems % n_threads != 0:
                raise RuntimeError(
                    f"OwlAttnInt: BlockQRows*KvTile={n_elems} not "
                    f"divisible by n_threads={n_threads}"
                )
            epl_s = n_elems // n_threads
            kt_c = bctx.c(KvTile, dtype=DType.U32)

            s_f32_lane: list = []
            row_locals: list = []
            col_locals: list = []
            for i in range(epl_s):
                flat = bctx.tid + bctx.c(i * n_threads, dtype=DType.U32)
                row_local = flat // kt_c
                col_local = flat % kt_c
                s32_val = s_s32_smem[row_local, col_local]
                q_sc = qk.load(q_scale_smem, row_local)
                k_sc = qk.load(k_scale_smem, col_local)
                f32_val = qk.convert(s32_val, DType.F32) * q_sc * k_sc
                s_f32_lane.append(f32_val)
                row_locals.append(row_local)
                col_locals.append(col_local)

            if c.phase == PHASE_QK_SCORES:
                # Phase 3.2 — write dequantized scores.
                for i in range(epl_s):
                    gmem_row = q_row_base + row_locals[i]
                    gmem_col = kv_off_c + col_locals[i]
                    g.output[gmem_row, gmem_col] = s_f32_lane[i]

            elif c.phase == PHASE_SOFTMAX_PROBS:
                # Phase 3.3a — per-row softmax across lanes (= across
                # cols). For each register index i (= row), reduce
                # across lanes to compute max, subtract, exp, sum,
                # divide.
                # Note: for single KvTile (capacity == KvTile), this
                # is the standard softmax. For multi-KvTile (Phase 3.4
                # validation), the running m/l carry across iterations
                # will live in this same per-register slot, updated
                # via rescale = exp(m_old - m_new).
                for i in range(epl_s):
                    m_i = qk.subgroup_reduce("max", s_f32_lane[i])
                    e_i = qk.exp_approx(s_f32_lane[i] - m_i)
                    l_i = qk.subgroup_reduce("sum", e_i)
                    p_i = qk.div(e_i, l_i)

                    gmem_row = q_row_base + row_locals[i]
                    gmem_col = kv_off_c + col_locals[i]
                    g.output[gmem_row, gmem_col] = p_i

            elif c.phase == PHASE_FULL:
                # Phase 3.4a — online softmax with running carry across
                # K iters. For the smoke (n_k_iters in {1, 2}) we keep
                # the K-loop Python-unrolled; Phase 3.4b will switch to
                # ``qk.for_range`` for the saturated KV shape (272 iters).
                #
                # Stage 1: per-row online softmax — compute m_new, e_i,
                # rescale (= exp(m_old - m_new)), l_local. The per-row
                # values live in ``m_per_row`` / ``l_per_row`` carry —
                # broadcast-identical across lanes. Per-lane ``e_i`` and
                # ``s_f32`` stay register-local until P quant.
                rescale_per_row: list = []
                e_per_lane: list = []
                for i in range(epl_s):
                    m_iter = qk.subgroup_reduce("max", s_f32_lane[i])
                    m_new = qk.max(m_per_row[i], m_iter)
                    rescale_i = qk.exp_approx(m_per_row[i] - m_new)
                    e_i = qk.exp_approx(s_f32_lane[i] - m_new)
                    l_local_i = qk.subgroup_reduce("sum", e_i)
                    # Update running m, l carry now (before P quant uses
                    # rescale_per_row[i]/l_local_i below).
                    m_per_row[i] = m_new
                    l_per_row[i] = l_per_row[i] * rescale_i + l_local_i
                    rescale_per_row.append(rescale_i)
                    e_per_lane.append(e_i)

                # Stage 2: V_scale fold + P quant → p_s8_smem. Per-row
                # P_scale lives in ``p_scale_per_row`` (carry across
                # this iter's stages 3-5 only — NOT across K iters).
                p_scale_per_row: list = []
                for i in range(epl_s):
                    v_sc = g.V_scales[k_row_base + kv_off_c + col_locals[i]]
                    p_scaled = e_per_lane[i] * v_sc
                    p_abs_max = qk.subgroup_reduce("max", qk.abs_(p_scaled))
                    p_z = qk.cmp("eq", p_abs_max, zero_f)
                    p_scale = qk.select(p_z, one_f, qk.div(p_abs_max, c127))
                    inv_p_scale = qk.div(one_f, p_scale)
                    x = p_scaled * inv_p_scale
                    is_neg_p = qk.cmp("lt", x, zero_f)
                    x_rnd = x + qk.select(is_neg_p, c_neg_half, c_half)
                    x_clamped = qk.max(c_neg127, qk.min(c127, x_rnd))
                    p_s8 = qk.convert(x_clamped, DType.S8)
                    p_s8_smem[row_locals[i], col_locals[i]] = p_s8
                    p_scale_per_row.append(p_scale)

                qk.barrier("block")

                # Stage 3: cooperative Vt_s8 load (Dh × KvTile s8).
                n_vt_elems = Dh * KvTile
                if n_vt_elems % n_threads != 0:
                    raise RuntimeError(
                        f"OwlAttnInt FULL: Dh*KvTile={n_vt_elems} not "
                        f"divisible by n_threads={n_threads}"
                    )
                epl_vt = n_vt_elems // n_threads
                for i in range(epl_vt):
                    flat = bctx.tid + bctx.c(i * n_threads, dtype=DType.U32)
                    row_local = flat // kt_c   # row in [0, Dh)
                    col_local = flat % kt_c    # col in [0, KvTile)
                    vt_s8_smem[row_local, col_local] = g.Vt_s8[
                        vt_row_base + row_local, kv_off_c + col_local
                    ]

                qk.barrier("block")

                # Stage 4: AV MMA — o_s32 = P_s8 @ Vt_s8 (init each iter
                # to 0; cross-iter accumulation happens in o_f32_per_lane
                # after dequant, NOT in the s32 accumulator since each
                # iter has a different P_scale).
                zero_s32_iter = bctx.c(0, dtype=DType.S32)
                o_s32_frags_iter = [
                    qk.vec_build([zero_s32_iter] * mma_cfg.shape.c_regs)
                    for _ in range(c.MTiles * NK_av)
                ]
                for ks in range(K_steps_av):
                    ks_col = bctx.c(ks * k_tile, dtype=DType.U32)
                    a_frags_av = [
                        qk.load_matrix(
                            p_s8_smem, shape_id, which="a",
                            row=bctx.c(mt * m_tile, dtype=DType.U32),
                            col=ks_col,
                            reg_offsets=mma_cfg.a_offsets,
                        )
                        for mt in range(c.MTiles)
                    ]
                    b_frags_av = [
                        qk.load_matrix(
                            vt_s8_smem, shape_id, which="b",
                            row=bctx.c(nt * n_tile, dtype=DType.U32),
                            col=ks_col,
                            reg_offsets=mma_cfg.b_offsets,
                        )
                        for nt in range(NK_av)
                    ]
                    for mt in range(c.MTiles):
                        for nt in range(NK_av):
                            idx = mt * NK_av + nt
                            o_s32_frags_iter[idx] = qk.mma(
                                shape_id,
                                a_frags_av[mt], b_frags_av[nt],
                                o_s32_frags_iter[idx],
                            )

                # Stage 4b: dump o_s32 frags to smem (BlockQRows × Dh).
                for mt in range(c.MTiles):
                    for nt in range(NK_av):
                        idx = mt * NK_av + nt
                        qk.store_matrix(
                            o_s32_smem, o_s32_frags_iter[idx],
                            shape_id, "d",
                            row=bctx.c(mt * m_tile, dtype=DType.U32),
                            col=bctx.c(nt * n_tile, dtype=DType.U32),
                            reg_offsets=mma_cfg.cd_offsets,
                        )
                qk.barrier("block")

                # Stage 5: per-lane scatter — dequant this iter's o_s32
                # by ``P_scale[row]`` and FOLD into running ``o_f32_per_lane``
                # via ``o_f32 = o_f32 * rescale[row] + o_tile_f32``.
                # Layout: lane c, register slot i → flat=c+i*32, row=i//2,
                # col=c+(i%2)*32 (since Dh=64, n_threads=32).
                rows_per_lane_block = Dh // n_threads  # 2
                n_o_elems = BlockQRows * Dh
                epl_o = n_o_elems // n_threads
                for i in range(epl_o):
                    flat = bctx.tid + bctx.c(i * n_threads, dtype=DType.U32)
                    row_local = flat // dh_c
                    col_local = flat % dh_c
                    row_idx_py = i // rows_per_lane_block
                    o_s32_val = o_s32_smem[row_local, col_local]
                    o_tile_f32 = qk.convert(o_s32_val, DType.F32) \
                                 * p_scale_per_row[row_idx_py]
                    o_f32_per_lane[i] = (
                        o_f32_per_lane[i] * rescale_per_row[row_idx_py]
                        + o_tile_f32
                    )

            else:
                raise NotImplementedError(
                    f"OwlAttnInt: phase={c.phase!r} not implemented"
                )

            if k_iter + 1 < n_k_iters:
                qk.barrier("block")

        # ── 3. Post-K-loop: PHASE_FULL output epilogue ──
        if c.phase == PHASE_FULL:
            # Final epilogue: divide each running o_f32 entry by its
            # per-row ``l`` and cast to out_dtype. ``l_per_row`` and
            # ``o_f32_per_lane`` were updated each K-iter above.
            rows_per_lane_block = Dh // n_threads  # 2
            n_o_elems = BlockQRows * Dh
            epl_o = n_o_elems // n_threads
            for i in range(epl_o):
                flat = bctx.tid + bctx.c(i * n_threads, dtype=DType.U32)
                row_local = flat // dh_c
                col_local = flat % dh_c
                row_idx_py = i // rows_per_lane_block
                o_final = qk.div(o_f32_per_lane[i], l_per_row[row_idx_py])
                if s.out_dtype is DType.F32:
                    out_val = o_final
                else:
                    out_val = qk.convert(o_final, s.out_dtype)
                gmem_row = q_row_base + row_local
                gmem_col = col_local
                g.output[gmem_row, gmem_col] = out_val
