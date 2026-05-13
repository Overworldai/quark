"""KVQuantizeKernel — per-token symmetric int8 quantization.

For a [num_tokens, Dh] bf16 tensor:
  abs_max = max(|x[token, :]|) per row (Dh elements)
  scale   = abs_max / 127
  q       = round(x / scale).clip(-127, 127)  as int8

WG layout: 1 WG = 1 token. 32 threads (1 SIMD32 wave) cooperatively
own Dh elements (Dh/32 per lane). Cross-lane absmax via
``subgroup_reduce("max", ...)``. Lane 0 writes the scale; all
lanes write their s8 quantized values.

Algorithm validated end-to-end in `/tmp/run_kv_quantize_probe.py`
(byte-identical to numpy reference at multiple shapes; dequant
cos > 0.9999 vs original bf16).
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.kv_cache_quantize.baselines import kv_quantize_baselines
from quark.kernels.kv_cache_quantize.config import KVQuantizeConfig
from quark.kernels.kv_cache_quantize.problems import kv_quantize_problems
from quark.kernels.kv_cache_quantize.reference import kv_quantize_reference_numpy
from quark.kernels.kv_cache_quantize.spec import KVQuantizeSpec


@kernel(
    "kv_quantize",
    spec=KVQuantizeSpec,
    config=KVQuantizeConfig,
    output_idx=-1,
    problems=kv_quantize_problems,
    baselines=lambda kernel, tensors: kv_quantize_baselines(tensors),
    reference=kv_quantize_reference_numpy,
)
class KVQuantizeKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        # Layout depends on ``transposed``:
        #   False: K-cache style [num_tokens, Dh]
        #   True:  Vt-cache style [n_heads * Dh, num_tokens]
        TensorDecl("K_in",      dtype=lambda s, c: s.in_dtype,
                   shape=lambda s, c: (
                       (s.n_heads * s.Dh, s.num_tokens) if s.transposed
                       else (s.num_tokens, s.Dh)
                   )),
        TensorDecl("K_out_s8",  dtype=lambda s, c: DType.S8,
                   shape=lambda s, c: (
                       (s.n_heads * s.Dh, s.num_tokens) if s.transposed
                       else (s.num_tokens, s.Dh)
                   )),
        # Scales: one f32 per (head, token) in transposed mode,
        # one per token otherwise.
        TensorDecl("K_scales",  dtype=lambda s, c: s.scale_dtype,
                   shape=lambda s, c: (
                       (s.n_heads * s.num_tokens,) if s.transposed
                       else (s.num_tokens,)
                   ),
                   role="out"),
    ]

    spec: KVQuantizeSpec
    config: KVQuantizeConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if c.n_warps != 1:
            return False
        # Dh must be divisible by 32 (one wave covers it).
        if s.Dh % 32 != 0:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        # Row-major: 1 WG per token. Transposed: 1 WG per (head, token).
        if self.spec.transposed:
            return (self.spec.num_tokens, self.spec.n_heads, 1)
        return (self.spec.num_tokens, 1, 1)

    def flops(self) -> int:
        # ~3 FLOPs per element (abs + cmp for max + div+round): cheap.
        return 3 * self.spec.num_tokens * self.spec.Dh

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = KVQuantizeSpec(**problem)
        rng = np.random.default_rng(seed)
        if spec.transposed:
            shape = (spec.n_heads * spec.Dh, spec.num_tokens)
            scale_shape = (spec.n_heads * spec.num_tokens,)
        else:
            shape = (spec.num_tokens, spec.Dh)
            scale_shape = (spec.num_tokens,)
        x = rng.standard_normal(shape).astype(np.float32) * 0.5
        return {
            "K_in":     astype_numpy(x, spec.in_dtype.value),
            "K_out_s8": zeros_for_dtype(shape, "s8"),
            "K_scales": zeros_for_dtype(scale_shape, "f32"),
        }

    @classmethod
    def spec_from_tensors(cls, K_in, K_out_s8=None, K_scales=None):
        # Heuristic: if K_in is 2D with shape (N, Dh) where Dh is small,
        # assume row-major. The transposed path's first dim is
        # ``n_heads * Dh`` which is much larger, but we can't infer
        # n_heads from shape alone — callers must construct the spec
        # explicitly for the transposed case.
        if len(K_in.shape) == 2:
            num_tokens, Dh = K_in.shape
            return KVQuantizeSpec(num_tokens=num_tokens, Dh=Dh)
        raise ValueError(
            "KVQuantizeKernel.spec_from_tensors: cannot infer spec for "
            "non-2D input. Construct KVQuantizeSpec explicitly."
        )

    def build(self) -> None:
        from quark.lang import abs_, div, max as qk_max, min as qk_min

        s, c, g = self.spec, self.config, self.g
        bctx = self.bctx
        Dh = s.Dh
        # SIMD32 wave covers Dh elements: each lane handles Dh/32.
        epl = Dh // 32  # elements per lane

        # WG indexing depends on layout:
        #   row-major: grid.x = token  → each WG handles one row.
        #   transposed: grid.x = token, grid.y = head → each WG handles
        #               one (head, token) Dh-vector.
        token = qk.block_idx("x")
        lane = bctx.tid  # SIMD32 single-warp → tid is also lane id
        epl_c = bctx.c(epl, dtype=DType.U32)

        if s.transposed:
            head = qk.block_idx("y")
            # Vt row base for this (head): head * Dh
            row_base = head * bctx.c(Dh, dtype=DType.U32)

        # ── 1. Load per-lane bf16 values, cast to f32 for absmax ──
        vals = []
        for i in range(epl):
            local = lane * epl_c + bctx.c(i, dtype=DType.U32)
            if s.transposed:
                # Vt layout: dh-row across rows, token is the column.
                # Each lane reads epl rows starting at lane*epl.
                v = g.K_in[row_base + local, token]
            else:
                v = g.K_in[token, local]
            vals.append(qk.convert(v, DType.F32))

        # Per-lane absmax across ``epl`` elements.
        lane_max = abs_(vals[0])
        for i in range(1, epl):
            lane_max = qk_max(lane_max, abs_(vals[i]))

        # Cross-lane subgroup reduce → max broadcast to every lane.
        abs_max = qk.subgroup_reduce("max", lane_max)

        # ── 2. scale = abs_max / 127  (guard zero → 1.0) ──
        c127 = bctx.c(127.0, dtype=DType.F32)
        c_neg127 = bctx.c(-127.0, dtype=DType.F32)
        c_half = bctx.c(0.5, dtype=DType.F32)
        c_neg_half = bctx.c(-0.5, dtype=DType.F32)
        zero_f = bctx.c(0.0, dtype=DType.F32)
        one_f = bctx.c(1.0, dtype=DType.F32)
        cmp_zero = qk.cmp("eq", abs_max, zero_f)
        scale = qk.select(cmp_zero, one_f, div(abs_max, c127))

        # ── 3. Quantize: round-to-nearest via add-half-with-sign +
        # truncating cast. Clamp to [-127, 127] before cast. ──
        inv_scale = div(one_f, scale)
        for i in range(epl):
            local = lane * epl_c + bctx.c(i, dtype=DType.U32)
            x = vals[i] * inv_scale
            is_neg = qk.cmp("lt", x, zero_f)
            x_rounded = x + qk.select(is_neg, c_neg_half, c_half)
            # Clamp to s8 range, then truncating cast.
            x_clamped = qk_max(c_neg127, qk_min(c127, x_rounded))
            q_i = qk.convert(x_clamped, DType.S8)
            if s.transposed:
                g.K_out_s8[row_base + local, token] = q_i
            else:
                g.K_out_s8[token, local] = q_i

        # ── 4. Lane 0 writes the f32 scale ──
        is_lane_0 = qk.cmp("eq", lane, bctx.c(0, dtype=DType.U32))
        if s.transposed:
            # Scale index = head * num_tokens + token.
            scale_idx = head * bctx.c(s.num_tokens, dtype=DType.U32) + token
            qk.store(g.K_scales, scale, scale_idx, pred=is_lane_0)
        else:
            qk.store(g.K_scales, scale, token, pred=is_lane_0)
