"""HeadRMSNorm — per-head RMSNorm on packed QKV tensor.

All threads cooperate to load many (token, head) slices via cp.async
into flat smem. Each warp processes multiple heads sequentially with
full 32-lane reduction. Q/K heads normalized, V heads copied.

Input/output layout: ``[M, D_full]`` where ``D_full = total_heads * Dh``.
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.head_rmsnorm.baselines import head_rmsnorm_baselines
from quark.kernels.head_rmsnorm.config import HeadRMSNormConfig
from quark.kernels.head_rmsnorm.problems import head_rmsnorm_problems
from quark.kernels.head_rmsnorm.reference import head_rmsnorm_reference_numpy
from quark.kernels.head_rmsnorm.spec import HeadRMSNormSpec

_WARP = 32
_CP_BYTES = 16


@kernel(
    "head_rmsnorm",
    spec=HeadRMSNormSpec,
    config=HeadRMSNormConfig,
    output_idx=-1,
    problems=head_rmsnorm_problems,
    baselines=head_rmsnorm_baselines,
    reference=head_rmsnorm_reference_numpy,
)
class HeadRMSNormKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("X", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.M, s.D_full)),
        TensorDecl(
            "Out",
            dtype=lambda s, c: s.dtype,
            shape=lambda s, c: (s.M, s.D_full),
            role="out",
        ),
    ]

    spec: HeadRMSNormSpec
    config: HeadRMSNormConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if c.n_warps < 1 or c.n_warps > 32:
            return False
        if s.Dh % _WARP != 0:
            return False
        vec_elems = _CP_BYTES // s.dtype.bytes
        if s.Dh % vec_elems != 0:
            return False
        n_threads = c.n_warps * _WARP
        total_work = s.M * s.total_heads
        # Each load pass covers n_threads * vec_elems / Dh heads.
        elems_per_load = n_threads * vec_elems
        if elems_per_load % s.Dh != 0:
            return False
        heads_per_load = elems_per_load // s.Dh
        if heads_per_load % c.n_warps != 0:
            return False
        if total_work % heads_per_load != 0:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        s, c = self.spec, self.config
        n_threads = c.n_warps * _WARP
        vec_elems = _CP_BYTES // s.dtype.bytes
        heads_per_load = (n_threads * vec_elems) // s.Dh
        total_work = s.M * s.total_heads
        return (1, total_work // heads_per_load, 1)

    def flops(self) -> int:
        s = self.spec
        norm_heads = s.n_q_heads + s.n_kv_heads
        return s.M * norm_heads * s.Dh * 4 + s.M * s.n_kv_heads * s.Dh

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1, 2, 4, 8]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = HeadRMSNormSpec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "X": astype_numpy(
                rng.standard_normal((spec.M, spec.D_full)).astype(np.float32), spec.dtype
            ),
            "Out": zeros_for_dtype((spec.M, spec.D_full), spec.dtype),
        }

    @classmethod
    def spec_from_tensors(
        cls, X, *, n_q_heads: int, n_kv_heads: int, Dh: int, eps: float = 1.1920929e-07
    ) -> HeadRMSNormSpec:
        M = int(X.shape[0])
        D_full = int(X.shape[1])
        dt = DType.from_backend(X.dtype)
        return HeadRMSNormSpec(
            M=M,
            D_full=D_full,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            Dh=Dh,
            dtype=dt,
            eps=eps,
        )

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        Dh = s.Dh
        n_warps = c.n_warps
        n_threads = n_warps * _WARP
        dtype = s.dtype
        total_heads = s.total_heads
        n_norm_heads = s.n_q_heads + s.n_kv_heads
        epl = Dh // _WARP  # elements per lane per head (e.g. 2 for Dh=64)

        vec_elems = _CP_BYTES // dtype.bytes

        # Cooperative load: all threads load 1 vec each.
        # Total elems per load = n_threads * vec_elems.
        # heads_per_load = total_elems / Dh.
        heads_per_load = (n_threads * vec_elems) // Dh
        heads_per_warp = heads_per_load // n_warps

        # Flat smem: (heads_per_load, Dh).
        flat_smem = qk.smem_alloc("X_flat", dtype, (heads_per_load, Dh), pad=0)

        # Block-level head base.
        block_head_base = qk.block_idx("y") * bctx.c(heads_per_load, dtype=DType.U32)

        lane = bctx.lane_id
        warp_c = bctx.c(_WARP, dtype=DType.U32)
        total_heads_c = bctx.c(total_heads, dtype=DType.U32)
        n_norm_c = bctx.c(n_norm_heads, dtype=DType.U32)
        Dh_c = bctx.c(Dh, dtype=DType.U32)

        # Each thread loads 1 cp.async 16B.
        # flat element = tid * vec_elems → row = flat // Dh, col = flat % Dh.
        flat_elem = bctx.tid * vec_elems
        load_row = flat_elem // Dh_c
        load_col = flat_elem % Dh_c

        # Gmem address: the flat head index maps to (token, head) → row=token, col=head*Dh+local_col.
        flat_head_idx = block_head_base + load_row
        gmem_token = flat_head_idx // total_heads_c
        gmem_head = flat_head_idx % total_heads_c
        gmem_col = gmem_head * Dh_c + load_col

        qk.async_copy(
            dst=flat_smem,
            src=g.X,
            dst_idx=[load_row, load_col],
            src_idx=[gmem_token, gmem_col],
            count=_CP_BYTES,
        )
        qk.async_commit()
        qk.async_wait(0)
        qk.barrier("block")

        # Each warp processes heads_per_warp heads sequentially.
        heads_per_warp_c = bctx.c(heads_per_warp, dtype=DType.U32)
        warp_head_start = bctx.warp_id * heads_per_warp_c

        for hi in range(heads_per_warp):
            local_head = warp_head_start + hi
            # Absolute head index for norm/copy decision.
            abs_head_idx = block_head_base + local_head
            abs_head_in_qkv = abs_head_idx % total_heads_c
            needs_norm = qk.cmp("lt", abs_head_in_qkv, n_norm_c)

            # Reduce from smem.
            local_sum_sq = bctx.c(0.0, dtype=DType.F32)
            x_regs: list = []
            for e in range(epl):
                col = bctx.c(e) * warp_c + lane
                x = qk.convert(flat_smem[local_head, col], DType.F32)
                local_sum_sq = qk.fma(x, x, local_sum_sq)
                x_regs.append(x)

            total = qk.subgroup_reduce("sum", local_sum_sq)
            one_f = bctx.c(1.0, dtype=DType.F32)
            rms_inv = qk.rsqrt_approx(
                total * bctx.c(1.0 / Dh, dtype=DType.F32) + bctx.c(s.eps, dtype=DType.F32)
            )
            rms_inv = qk.select(needs_norm, rms_inv, one_f)

            # Output: scalar store (Dh is small).
            gmem_out_token = abs_head_idx // total_heads_c
            gmem_out_head = abs_head_idx % total_heads_c
            gmem_out_col_base = gmem_out_head * Dh_c
            for e in range(epl):
                col = bctx.c(e) * warp_c + lane
                y = x_regs[e] * rms_inv
                g.Out[gmem_out_token, gmem_out_col_base + col] = (
                    qk.convert(y, dtype) if dtype is not DType.F32 else y
                )
