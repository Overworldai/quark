"""MoE reduce — gather-and-sum the per-slot bf16 partials produced by
``moe_outproj`` into a per-token output, weighted by the router's
``slot_weights``.

    out[m, d] = sum over k of slot_weights[idx] * partials[idx, d]
    where idx = token_slot_table[m, k]

Architecture (warp-per-row):
  - Grid: (1, M // n_warps, 1)
  - Each warp owns one output row ``m``.
  - Each lane handles ``D / WARP`` elements of that row, vec-loaded in
    chunks of 16B (8 bf16). The top_k axis is python-unrolled inside
    the chunk loop so per-chunk we do top_k indirect vec_loads against
    ``partials`` and accumulate in f32 registers.
  - ``token_slot_table[m, k]`` and ``slot_weights[idx]`` are uniform
    across the warp; the lowerer's load coalescing collapses each
    warp's lane redundant loads into a single mem op.
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.device import DEFAULT_SUBGROUP_WIDTH as _WARP
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.moe_reduce.config import MoeReduceConfig
from quark.kernels.moe_reduce.problems import moe_reduce_problems
from quark.kernels.moe_reduce.reference import moe_reduce_reference_numpy
from quark.kernels.moe_reduce.spec import MoeReduceSpec

_CP_BYTES = 16


@kernel(
    "moe_reduce",
    spec=MoeReduceSpec,
    config=MoeReduceConfig,
    output_idx=-1,
    problems=moe_reduce_problems,
    baselines=lambda kernel, tensors: [],
    reference=moe_reduce_reference_numpy,
)
class MoeReduceKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl(
            "partials",
            dtype=lambda s, c: s.partials_dtype,
            shape=lambda s, c: (s.total_slots, s.D),
        ),
        TensorDecl(
            "slot_weights",
            dtype=lambda s, c: DType.F32,
            shape=lambda s, c: (s.total_slots,),
        ),
        TensorDecl(
            "token_slot_table",
            dtype=lambda s, c: DType.S32,
            shape=lambda s, c: (s.M, s.top_k),
        ),
        TensorDecl(
            "out",
            dtype=lambda s, c: s.out_dtype,
            shape=lambda s, c: (s.M, s.D),
            role="out",
        ),
    ]

    spec: MoeReduceSpec
    config: MoeReduceConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if c.n_warps < 1 or c.n_warps > 32:
            return False
        if s.M % c.n_warps != 0:
            return False
        vec_w_in = _CP_BYTES // s.partials_dtype.bytes
        vec_w_out = _CP_BYTES // s.out_dtype.bytes
        # The output vec_store is the granularity that drives the chunk
        # loop; we issue ``vec_w_out / vec_w_in`` partials vec_loads per
        # chunk to feed it. Out width must be a multiple of in width
        # (e.g. f32 partials → bf16 output: 4 → 8, two loads per chunk).
        if vec_w_out % vec_w_in != 0:
            return False
        if s.D % (_WARP * vec_w_out) != 0:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        s, c = self.spec, self.config
        return (1, s.M // c.n_warps, 1)

    def flops(self) -> int:
        s = self.spec
        return 2 * s.M * s.D * s.top_k

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [1, 2, 4, 8]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = MoeReduceSpec(**problem)
        rng = np.random.default_rng(seed)
        # Random partials and slot_weights so the cos-sim check is
        # well-conditioned. token_slot_table draws indices from
        # [0, total_slots) — every entry must be a valid slot index
        # (the real router guarantees this; the test fixture mimics it).
        total_slots = spec.total_slots
        partials = astype_numpy(
            rng.standard_normal((total_slots, spec.D)).astype(np.float32),
            spec.partials_dtype,
        )
        slot_weights = rng.standard_normal((total_slots,)).astype(np.float32) * 0.5
        # Use the deterministic "first M*K slots are token assignments"
        # scheme so the reference and kernel agree on which slots
        # contribute. Per-token indices are the first top_k slots
        # belonging to that token's k-th expert assignment — for the
        # fixture, just stripe.
        token_slot_table = np.zeros((spec.M, spec.top_k), dtype=np.int32)
        for m in range(spec.M):
            for k in range(spec.top_k):
                token_slot_table[m, k] = (m * spec.top_k + k) % total_slots
        return {
            "partials": partials,
            "slot_weights": slot_weights.astype(np.float32),
            "token_slot_table": token_slot_table,
            "out": zeros_for_dtype((spec.M, spec.D), spec.out_dtype),
        }

    @classmethod
    def spec_from_tensors(
        cls,
        partials,
        slot_weights,
        token_slot_table,
        *,
        n_experts: int,
        out_dtype: DType | str = DType.BF16,
    ) -> MoeReduceSpec:
        if partials.ndim != 2:
            raise ValueError(f"moe_reduce: partials must be rank-2; got {partials.shape}")
        if token_slot_table.ndim != 2:
            raise ValueError(
                f"moe_reduce: token_slot_table must be rank-2; got {token_slot_table.shape}"
            )
        total_slots, D = int(partials.shape[0]), int(partials.shape[1])
        if total_slots % n_experts != 0:
            raise ValueError(
                f"moe_reduce: partials.shape[0] ({total_slots}) not divisible by "
                f"n_experts ({n_experts})"
            )
        capacity = total_slots // n_experts
        M, top_k = int(token_slot_table.shape[0]), int(token_slot_table.shape[1])
        if int(slot_weights.shape[0]) != total_slots:
            raise ValueError(
                f"moe_reduce: slot_weights {tuple(slot_weights.shape)} != partials.shape[0] "
                f"({total_slots})"
            )
        return MoeReduceSpec(
            M=M,
            D=D,
            n_experts=n_experts,
            capacity=capacity,
            top_k=top_k,
            partials_dtype=DType.from_backend(partials),
            out_dtype=DType.coerce(out_dtype) or DType.BF16,
        )

    def build(self) -> None:
        """Default (CUDA) path — single warp per row, vec-load loop.

        Metal happens to be fine with the same shape because there's no
        cp.async involved (everything's a plain vec_load), so no
        ``build_metal`` override is needed.

        ``vec_w_out`` (output store width) drives the chunk loop;
        ``vec_w_in`` (partials load width) may be smaller — when
        ``partials_dtype`` is wider than ``out_dtype`` (the f32
        partials → bf16 output path) we issue
        ``vec_w_out / vec_w_in`` partials loads per output chunk.
        """
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        D = s.D
        n_warps = c.n_warps
        partials_dt = s.partials_dtype
        out_dt = s.out_dtype
        vec_w_in = _CP_BYTES // partials_dt.bytes
        vec_w_out = _CP_BYTES // out_dt.bytes
        loads_per_chunk = vec_w_out // vec_w_in
        epl = D // _WARP
        vecs_per_lane = epl // vec_w_out

        lane = bctx.lane_id
        vec_w_out_c = bctx.c(vec_w_out, dtype=DType.U32)

        # Each block owns ``n_warps`` rows; each warp gets one row.
        block_base = qk.block_idx("y") * bctx.c(n_warps, dtype=DType.U32)
        my_row = block_base + bctx.warp_id

        # Hoist the top_k slot-index loads outside the chunk loop —
        # they're uniform across the warp's lanes and across all D
        # chunks for a given row. Loaded once per row instead of
        # ``vecs_per_lane × top_k`` times.
        slot_indices: list = []
        slot_weights_per_k: list = []
        for k in range(s.top_k):
            k_const = bctx.c(k, dtype=DType.U32)
            idx = qk.load(g.token_slot_table, my_row, k_const)
            idx_u = qk.bitcast(idx, DType.U32)
            slot_indices.append(idx_u)
            w = qk.load(g.slot_weights, idx_u)
            slot_weights_per_k.append(w)

        # Walk D in output-vec-sized chunks. For each chunk, accumulate
        # the weighted top_k partials in f32 registers (issuing
        # ``loads_per_chunk`` partials vec_loads per k), then
        # narrow-cast and vec_store.
        for v in range(vecs_per_lane):
            col = (lane + bctx.c(v * _WARP, dtype=DType.U32)) * vec_w_out_c

            # f32 accumulators — one per element of the output vec.
            accs: list = [bctx.c(0.0, dtype=DType.F32) for _ in range(vec_w_out)]

            for k in range(s.top_k):
                idx_u = slot_indices[k]
                w = slot_weights_per_k[k]
                for sub in range(loads_per_chunk):
                    sub_col = col if sub == 0 else col + bctx.c(sub * vec_w_in, dtype=DType.U32)
                    p_vec = qk.vec_load(
                        g.partials, idx_u, sub_col, width=vec_w_in, dtype=partials_dt
                    )
                    for j in range(vec_w_in):
                        j_full = sub * vec_w_in + j
                        p_elem = qk.vec_extract(p_vec, j)
                        p_f = p_elem if partials_dt is DType.F32 else qk.convert(p_elem, DType.F32)
                        accs[j_full] = qk.fma(w, p_f, accs[j_full])

            out_elems = []
            for j in range(vec_w_out):
                if out_dt is DType.F32:
                    out_elems.append(accs[j])
                else:
                    out_elems.append(qk.convert(accs[j], out_dt))
            qk.vec_store(g.out, qk.vec_build(out_elems), my_row, col)
