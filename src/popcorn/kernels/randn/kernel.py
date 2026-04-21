"""randn — Philox4x32-10 + Box-Muller per thread.

Counter-based stateless PRNG: every thread derives its uniform u32s
from ``(flat_output_group, counter_offset, seed0, seed1)`` — no RNG
state to thread through kernel launches, so the kernel is graph-
capturable and repeatable given the same ``counter_offset``. Each
Philox call produces 4 u32s → 4 uniforms → 2 Box-Muller pairs → 4
normals, written with coalesced stores.

Per-output cost:
  * 10 Philox rounds × (2 mul.hi + 2 mul.lo + 3 xor + 2 add) per round
    ~= 70 scalar ops amortized over 4 outputs.
  * Box-Muller: 1 log2.approx, 1 sqrt.approx, 1 sin.approx, 1 cos.approx,
    a handful of muls.

On Ampere / Ada / Hopper this runs at tens of Gsamples/s — a 65 536-
element latent generates in a few microseconds vs ~60 ms for the
Python ``random.gauss`` loop that ``PopcornTensor.randn`` previously
used.

Inputs:
  counter_offset  [1] u32    bumped by the host per launch
  Out             [N]        output, dtype from spec

Grid: (1, N / elems_per_block, 1). Each block's threads cooperatively
fill ``elems_per_block`` consecutive output positions.
"""

from __future__ import annotations

import math
from typing import ClassVar

import popcorn.lang as pop
from popcorn.blocks import TensorDecl
from popcorn.ir import DType
from popcorn.kernels.base import Kernel
from popcorn.kernels.decorator import kernel
from popcorn.kernels.randn.config import RandnConfig
from popcorn.kernels.randn.problems import randn_problems
from popcorn.kernels.randn.reference import randn_reference_for_spec
from popcorn.kernels.randn.spec import RandnSpec

# Philox4x32-10 constants (standard — same values every implementation uses).
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_PHILOX_ROUNDS = 10

# Box-Muller constants.
_NEG_2_LN2 = -2.0 * math.log(2.0)  # converts log2(u) to -2*ln(u)
_TWO_PI = 2.0 * math.pi
# Uniform scale: (x >> 8) * 2^-24 + 2^-25 ∈ [2^-25, 1 - 2^-25] — never
# exactly 0 (so log is safe) and never exactly 1.
_U_SCALE = 2.0**-24
_U_BIAS = 2.0**-25


@kernel(
    "randn",
    spec=RandnSpec,
    config=RandnConfig,
    output_idx=-1,
    problems=randn_problems,
    baselines=lambda: [],
    reference=randn_reference_for_spec,
)
class RandnKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("counter_offset", dtype=DType.U32, shape=lambda s, c: (1,)),
        TensorDecl(
            "Out",
            dtype=lambda s, c: s.dtype,
            shape=lambda s, c: (s.N,),
            role="out",
        ),
    ]
    # Randn has no bitwise reference; correctness gating disabled at
    # the autotune layer via empty tune_space. Any non-NaN output is
    # considered correct.
    CORRECTNESS_THRESHOLD = 0.0

    spec: RandnSpec
    config: RandnConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        n_threads = c.n_warps * 32
        if c.n_warps < 1 or c.n_warps > 32:
            return False
        if c.elems_per_block <= 0:
            return False
        if c.elems_per_block % n_threads != 0:
            return False
        # Each thread produces outputs in groups of 4 (one Philox call).
        if (c.elems_per_block // n_threads) % 4 != 0:
            return False
        if s.N % c.elems_per_block != 0:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        return (1, self.spec.N // self.config.elems_per_block, 1)

    def flops(self) -> int:
        # ~75 ops per output — counted as FLOPs for bench reporting.
        return self.spec.N * 75

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        # No autotune — default config is always used. Randn has no
        # golden reference for the correctness gate to compare against.
        return {}

    @classmethod
    def make_tensors(cls, problem: dict) -> dict:
        from popcorn.backend import PT

        spec = RandnSpec(**problem)
        return {
            "counter_offset": PT.zeros(1, dtype=PT.uint32),
            "Out": PT.zeros(spec.N, dtype=spec.dtype.backend),
        }

    @classmethod
    def spec_from_tensors(cls, counter_offset, Out) -> RandnSpec:
        dt = DType.from_backend(Out.dtype)
        return RandnSpec(N=int(Out.shape[0]), dtype=dt)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _philox_round(
        self, bctx, ctr: tuple, key: tuple, M0_c, M1_c, W0_c, W1_c
    ) -> tuple[tuple, tuple]:
        """One Philox4x32 round. Returns (new_ctr, new_key)."""
        c0, c1, c2, c3 = ctr
        k0, k1 = key
        hi0 = pop.mul_hi(M0_c, c0)
        lo0 = pop.mul(M0_c, c0)
        hi1 = pop.mul_hi(M1_c, c2)
        lo1 = pop.mul(M1_c, c2)
        new_c0 = pop.xor(pop.xor(hi1, c1), k0)
        new_c1 = lo1
        new_c2 = pop.xor(pop.xor(hi0, c3), k1)
        new_c3 = lo0
        new_k0 = pop.add(k0, W0_c)
        new_k1 = pop.add(k1, W1_c)
        return (new_c0, new_c1, new_c2, new_c3), (new_k0, new_k1)

    def _u32_to_uniform(self, bctx, x_u32, scale_c, bias_c):
        """u32 → f32 uniform in [2^-25, 1 - 2^-25]. Safe input to log2.

        ``(x >> 8) * 2^-24 + 2^-25`` — drops the low 8 bits (the Philox
        low bits are the weakest, so we throw those away), casts the
        top 24 bits to f32, and scales. The bias keeps us off zero so
        ``log2(u)`` stays finite even when the raw u32 is 0.
        """
        x_shifted = pop.shr(x_u32, bctx.c(8, dtype=DType.U32))
        x_f = pop.convert(x_shifted, DType.F32)
        return pop.fma(x_f, scale_c, bias_c)

    def _box_muller(self, bctx, u1, u2, neg_2_ln2_c, two_pi_c):
        """(u1, u2) ∈ (0, 1]² → (z1, z2) two independent standard normals.

        Uses the SFU-pipe ``.approx`` variants of lg2 and sqrt — the
        precision loss (~1 ULP) is invisible in a random draw and we
        get a ~4× throughput win over the precise versions.
        """
        log2_u1 = pop.log2_approx(u1)
        # -2*ln(u1) = log2(u1) * (-2 * ln(2)). u1 ≤ 1 → log2 ≤ 0 → arg ≥ 0.
        arg = pop.mul(log2_u1, neg_2_ln2_c)
        r = pop.sqrt_approx(arg)
        theta = pop.mul(u2, two_pi_c)
        z1 = pop.mul(r, pop.cos(theta))
        z2 = pop.mul(r, pop.sin(theta))
        return z1, z2

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_threads = c.n_warps * 32
        epb = c.elems_per_block
        epl = epb // n_threads  # outputs per thread, multiple of 4
        groups = epl // 4  # Philox calls per thread
        out_dtype = s.dtype

        # Constants (materialized once per kernel — CSE reuses them).
        M0_c = bctx.c(_PHILOX_M0, dtype=DType.U32)
        M1_c = bctx.c(_PHILOX_M1, dtype=DType.U32)
        W0_c = bctx.c(_PHILOX_W0, dtype=DType.U32)
        W1_c = bctx.c(_PHILOX_W1, dtype=DType.U32)
        seed0_c = bctx.c(s.seed0 & 0xFFFFFFFF, dtype=DType.U32)
        seed1_c = bctx.c(s.seed1 & 0xFFFFFFFF, dtype=DType.U32)
        zero_u = bctx.c(0, dtype=DType.U32)
        n_threads_c = bctx.c(n_threads, dtype=DType.U32)

        u_scale_c = bctx.c(_U_SCALE, dtype=DType.F32)
        u_bias_c = bctx.c(_U_BIAS, dtype=DType.F32)
        neg_2_ln2_c = bctx.c(_NEG_2_LN2, dtype=DType.F32)
        two_pi_c = bctx.c(_TWO_PI, dtype=DType.F32)

        # Counter-offset scalar — bumped by host per launch.
        ctr_offset = pop.load(g.counter_offset, zero_u)

        # Block-level base index into Out and into the Philox counter
        # space. Each block owns epb outputs; 4 outputs per Philox call
        # means each block consumes epb/4 counter values.
        block = pop.block_idx("y")
        out_base = block * bctx.c(epb, dtype=DType.U32)
        counter_block_base = block * bctx.c(epb // 4, dtype=DType.U32) + ctr_offset

        for gi in range(groups):
            # Unique Philox counter for this (block, tid, group). The
            # coalesced-write layout stores output at
            # ``out_base + i*n_threads + tid`` for i in 0..epl-1, so
            # the group of four iters i=(4gi)..(4gi+3) all share this
            # philox call.
            ctr0 = counter_block_base + bctx.c(gi, dtype=DType.U32) * n_threads_c + bctx.tid
            ctr = (ctr0, zero_u, seed0_c, seed1_c)
            key = (seed0_c, seed1_c)

            for _ in range(_PHILOX_ROUNDS):
                ctr, key = self._philox_round(bctx, ctr, key, M0_c, M1_c, W0_c, W1_c)

            # Uniforms (safe range) and Box-Muller.
            u1 = self._u32_to_uniform(bctx, ctr[0], u_scale_c, u_bias_c)
            u2 = self._u32_to_uniform(bctx, ctr[1], u_scale_c, u_bias_c)
            u3 = self._u32_to_uniform(bctx, ctr[2], u_scale_c, u_bias_c)
            u4 = self._u32_to_uniform(bctx, ctr[3], u_scale_c, u_bias_c)
            z0, z1 = self._box_muller(bctx, u1, u2, neg_2_ln2_c, two_pi_c)
            z2, z3 = self._box_muller(bctx, u3, u4, neg_2_ln2_c, two_pi_c)

            # Coalesced stores: iter i in the unrolled group writes at
            # off = out_base + (4*gi + k) * n_threads + tid  for k in 0..3.
            #                                         ↑ strided by k·n_threads
            # Within a warp (varying tid) this covers 32 consecutive
            # output positions per k — one coalesced transaction.
            for k, z in enumerate((z0, z1, z2, z3)):
                off = out_base + bctx.c(4 * gi + k, dtype=DType.U32) * n_threads_c + bctx.tid
                if out_dtype is not DType.F32:
                    z = pop.convert(z, out_dtype)
                g.Out[off] = z
