"""Unpatchify — GEMM + bias with scatter epilogue to [B, C*H*W].

    X [M, d_model] × W [C*ph*pw, d_model]^T + bias → Out [B, C*H*W]

The epilogue maps each GEMM output element at (token, patch_elem) to
a spatial index (b, c, h, w) and writes directly to the flat image
buffer — no host-side reshape or permute.

Mapping:
    token = m_base + local_row
    b       = token // (Hp * Wp)
    h_tok   = (token % (Hp*Wp)) // Wp
    w_tok   = (token % (Hp*Wp)) % Wp
    patch_elem = n_base + local_col  (range [0, C*ph*pw))
    c       = patch_elem // (ph*pw)
    ph_off  = (patch_elem % (ph*pw)) // pw
    pw_off  = (patch_elem % (ph*pw)) % pw
    flat_col = c*(H*W) + (h_tok*ph + ph_off)*W + (w_tok*pw + pw_off)
"""

from __future__ import annotations

from typing import ClassVar

from popcorn.blocks import (
    Accumulators,
    IterCtx,
    MmaBody,
    PipelineBody,
    SmemPlan,
    TensorDecl,
)
from popcorn.ir import DType
from popcorn.kernels.base import Kernel, MmaSite
from popcorn.kernels.decorator import kernel
from popcorn.kernels.unpatchify.baselines import unpatchify_baselines
from popcorn.kernels.unpatchify.config import UnpatchifyConfig
from popcorn.kernels.unpatchify.problems import unpatchify_problems
from popcorn.kernels.unpatchify.reference import unpatchify_reference_numpy
from popcorn.kernels.unpatchify.spec import UnpatchifySpec


@kernel(
    "unpatchify",
    spec=UnpatchifySpec,
    config=UnpatchifyConfig,
    output_idx=-1,
    problems=unpatchify_problems,
    baselines=unpatchify_baselines,
    reference=unpatchify_reference_numpy,
)
class UnpatchifyKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("X", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.M, s.K)),
        TensorDecl("W", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N, s.K)),
        TensorDecl(
            "Bias",
            dtype=lambda s, c: s.dtype,
            shape=lambda s, c: (s.N,) if s.has_bias else (1,),
        ),
        TensorDecl(
            "Out",
            dtype=lambda s, c: s.dtype,
            shape=lambda s, c: (s.B, s.C * s.H * s.W),
            role="out",
        ),
    ]

    spec: UnpatchifySpec
    config: UnpatchifyConfig

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if s.M % c.BM != 0 or s.N % c.BN != 0 or s.K % c.BK != 0:
            return False
        try:
            mma = self._mma_cfg()
        except KeyError:
            return False
        if not self._validate_gemm_tile(mma):
            return False
        elem_b = s.dtype.bytes
        if c.a_pad and ((c.BK + c.a_pad) * elem_b) % 16 != 0:
            return False
        if c.b_pad and ((c.BK + c.b_pad) * elem_b) % 16 != 0:
            return False
        if c.n_stages == 2 and (s.K // c.BK) % 2 != 0:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        s, c = self.spec, self.config
        return (s.N // c.BN, s.M // c.BM, 1)

    def flops(self) -> int:
        s = self.spec
        return 2 * s.M * s.N * s.K

    @classmethod
    def mma_sites(cls, spec) -> list[MmaSite]:
        if spec is None:
            return []
        dt = spec.compute_dtype_resolved
        return [MmaSite(name="main", a_dtype=dt, b_dtype=dt)]

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {
            "BM": [16, 32, 64, 128],
            "BN": [16, 32, 64, 128],
            "BK": [16, 32, 64],
            "n_warps": [2, 4, 8],
            "n_stages": [1, 2],
            "a_pad": [0, 8],
            "b_pad": [0, 8],
        }

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from popcorn.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = UnpatchifySpec(**problem)
        rng = np.random.default_rng(seed)
        if spec.has_bias:
            bias_np = astype_numpy(rng.standard_normal(spec.N).astype(np.float32), spec.dtype)
        else:
            bias_np = zeros_for_dtype((1,), spec.dtype)
        return {
            "X": astype_numpy(rng.standard_normal((spec.M, spec.K)).astype(np.float32), spec.dtype),
            "W": astype_numpy(rng.standard_normal((spec.N, spec.K)).astype(np.float32), spec.dtype),
            "Bias": bias_np,
            "Out": zeros_for_dtype((spec.B, spec.C * spec.H * spec.W), spec.dtype),
        }

    @classmethod
    def spec_from_tensors(
        cls,
        X,
        W,
        Bias,
        *,
        B: int,
        C: int,
        H: int,
        W_spatial: int,
        ph: int = 2,
        pw: int = 2,
        has_bias: bool = True,
    ):
        d_model = int(X.shape[1])
        dt = DType.from_backend(X.dtype)
        return UnpatchifySpec(
            B=B,
            C=C,
            H=H,
            W=W_spatial,
            ph=ph,
            pw=pw,
            d_model=d_model,
            dtype=dt,
            has_bias=has_bias,
        )

    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        bctx, m_base, n_base = self.bctx, self.m_base, self.n_base
        compute_ir = s.compute_dtype_resolved
        mma_cfg = self._mma_cfg()

        stages = SmemPlan.staged_pairs(
            compute_ir,
            a_shape=(c.BM, c.BK),
            b_shape=(c.BN, c.BK),
            a_pad=c.a_pad,
            b_pad=c.b_pad,
            mma_cfg=mma_cfg,
            n_warps=c.n_warps,
            b_shuffled=False,
            n_stages=c.n_stages,
        )
        acc = Accumulators.from_mma(mma_cfg, BM=c.BM, BN=c.BN, n_warps=c.n_warps)
        mma = MmaBody(acc=acc)
        BN_per_warp = (c.BN // mma_cfg.shape.n // c.n_warps) * mma_cfg.shape.n
        K_outer = s.K // c.BK

        def produce(ictx: IterCtx) -> None:
            plan = ictx.stage
            k_col = ictx.iter_idx * c.BK
            plan.a.load_from(g.X, row=m_base, col=k_col, cast=None)
            plan.b.load_from(g.W, row=n_base, col=k_col, cast=None)

        PipelineBody(
            stages=stages,
            produce=produce,
            consume=mma,
            carry=acc,
        ).run(n_iters=K_outer, n_stages=c.n_stages)

        # ── Scatter epilogue: map (token, patch_elem) → (b, flat_img_col) ──
        # Use the frag_for_each path via store_acc's _emit_scatter_store,
        # but with a custom store target that remaps columns.
        #
        # We emit the scatter manually because the address remap can't be
        # expressed through store_acc's column-linear model.
        from popcorn.lang import convert, frag_for_each, load
        from popcorn.lang import store as _scalar_store

        cfg = bctx.mma_cfg
        shape_id = cfg.shape_id
        m_stride = cfg.shape.m
        n_stride = cfg.shape.n
        MT, NT = acc.MT, acc.NT

        warp_col_base = n_base + bctx.warp_id * BN_per_warp

        Hp = s.Hp
        Wp_val = s.Wp
        HpWp = Hp * Wp_val
        HW = s.H * s.W
        ph, pw = s.ph, s.pw
        pp = ph * pw

        HpWp_c = bctx.c(HpWp)
        Wp_c = bctx.c(Wp_val)
        HW_c = bctx.c(HW)
        W_c = bctx.c(s.W)
        pp_c = bctx.c(pp)
        pw_c = bctx.c(pw)
        ph_c = bctx.c(ph)

        _bias = g.Bias if s.has_bias else None
        assert acc.results is not None

        for mt in range(MT):
            for nt in range(NT):
                idx = mt * NT + nt
                acc_v = acc.results[idx]
                mt_off = bctx.c(mt * m_stride)
                nt_off = bctx.c(nt * n_stride)

                def fn(elem, row, col, *, _mt=mt_off, _nt=nt_off):
                    token = m_base + (_mt + row)
                    patch_elem = warp_col_base + (_nt + col)

                    # Bias add (in f32 accumulator space).
                    if _bias is not None:
                        b_val = load(_bias, patch_elem)
                        b_f32 = convert(b_val, DType.F32) if b_val.dtype != DType.F32 else b_val
                        elem = elem + b_f32

                    # Cast to output dtype.
                    if s.dtype != DType.F32:
                        elem = convert(elem, s.dtype)

                    # Map (token, patch_elem) → (b, flat_img_col).
                    b = token // HpWp_c
                    tok_in_frame = token % HpWp_c
                    h_tok = tok_in_frame // Wp_c
                    w_tok = tok_in_frame % Wp_c

                    c_ch = patch_elem // pp_c
                    k_rem = patch_elem % pp_c
                    ph_off = k_rem // pw_c
                    pw_off = k_rem % pw_c

                    flat_col = c_ch * HW_c + (h_tok * ph_c + ph_off) * W_c + (w_tok * pw_c + pw_off)
                    _scalar_store(g.Out, elem, b, flat_col)

                frag_for_each(shape_id, acc_v, fn, cfg.cd_offsets)
