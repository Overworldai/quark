"""moe_router_correct — purely-correct MoE routing kernel.

Single-block kernel. Each token routes to its actual top-K experts
(no capacity bound, no substitution). Slots are sorted by expert into
the output buffer with each expert's run padded to BM=32. Chunks past
the last filled slot get ``expert = -1`` (sentinel) so inproj/outproj
can early-exit on them.

Phases (one thread per token; ``n_threads == M`` enforced by is_valid):
  0. Cooperative zero-init of token_ids / slot_weights / counts /
     work_list (with sentinel ``-1``) / offsets.
  1. Each thread t: softmax + insertion-sort top-K over logits[t, :].
     For each k, ``atomic_add(counts[expert_k], 1)`` returns the
     thread's slot-within-expert index — saved in registers.
  2. Block barrier.
  3. Leading thread: cumulative offsets per expert (BM-padded), and
     work_list chunks-to-expert for the active region.
  4. Block barrier.
  5. Each thread t: re-uses register tuples (expert_k, weight_k,
     slot_in_e_k) to scatter into ``token_ids[offsets[e] + slot_in_e]``
     and ``slot_weights[...]``, and emit ``token_slot_table[t, k] = final``
     so the downstream reduce kernel can gather without a scatter.
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels._moe_router_dsl import insertion_sort_topp, softmax_topk
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.moe_router_correct.config import MoeRouterCorrectConfig
from quark.kernels.moe_router_correct.problems import moe_router_correct_problems
from quark.kernels.moe_router_correct.reference import moe_router_correct_reference_numpy
from quark.kernels.moe_router_correct.spec import MoeRouterCorrectSpec

_BM = 32
_LOG2E = 1.4426950408889634


@kernel(
    "moe_router_correct",
    spec=MoeRouterCorrectSpec,
    config=MoeRouterCorrectConfig,
    output_idx=-1,
    problems=moe_router_correct_problems,
    baselines=lambda kernel, tensors: [],
    reference=moe_router_correct_reference_numpy,
)
class MoeRouterCorrectKernel(Kernel):
    # Atomic-add ordering is non-deterministic so within an expert the
    # token permutation differs across runs. Validation compares the
    # multiset of (expert, weight, token) emitted, not the within-expert
    # ordering — keep the threshold loose.
    CORRECTNESS_THRESHOLD = 0.95

    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl(
            "logits",
            dtype=lambda s, c: DType.F32,
            shape=lambda s, c: (s.M, s.E),
        ),
        TensorDecl(
            "token_ids",
            dtype=lambda s, c: DType.S32,
            shape=lambda s, c: (s.total_slots,),
            role="out",
        ),
        TensorDecl(
            "slot_weights",
            dtype=lambda s, c: DType.F32,
            shape=lambda s, c: (s.total_slots,),
            role="out",
        ),
        TensorDecl(
            "counts",
            dtype=lambda s, c: DType.S32,
            shape=lambda s, c: (s.E,),
            role="out",
        ),
        TensorDecl(
            "work_list",
            dtype=lambda s, c: DType.S32,
            shape=lambda s, c: (2 * (s.total_slots // _BM),),
            role="out",
        ),
        # Workspace: cumulative starting slot per expert (with BM padding).
        # offsets[E] holds total active slots; downstream tools may inspect.
        TensorDecl(
            "offsets",
            dtype=lambda s, c: DType.S32,
            shape=lambda s, c: (s.E + 1,),
            role="out",
        ),
        # Per-token inverse of token_ids: token_slot_table[t, k] is the
        # slot index in the expert-major output buffer where token ``t``'s
        # k-th top-expert partial lands. Lets the downstream reduce
        # kernel gather without a scatter / atomic.
        TensorDecl(
            "token_slot_table",
            dtype=lambda s, c: DType.S32,
            shape=lambda s, c: (s.M, s.top_k),
            role="out",
        ),
    ]

    spec: MoeRouterCorrectSpec
    config: MoeRouterCorrectConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if not (1 <= c.n_warps <= 32):
            return False
        n_threads = c.n_warps * self._sgs
        # Strict one-thread-per-token: phase-1 reads logits[tid, e]
        # unconditionally; relaxing to n_threads > M would let the
        # extra threads OOB-read past the logits buffer.
        if n_threads != s.M:
            return False
        if s.total_slots % n_threads != 0:
            return False
        if n_threads < s.E:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        return (1, 1, 1)

    def flops(self) -> int:
        s = self.spec
        return s.M * (4 * s.E + s.top_k * s.E + 4 * s.top_k) + s.E * (s.total_slots // _BM)

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [4, 8, 16, 32]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        spec = MoeRouterCorrectSpec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "logits": rng.standard_normal((spec.M, spec.E)).astype(np.float32),
            "token_ids": np.zeros((spec.total_slots,), dtype=np.int32),
            "slot_weights": np.zeros((spec.total_slots,), dtype=np.float32),
            "counts": np.zeros((spec.E,), dtype=np.int32),
            "work_list": np.zeros((2 * (spec.total_slots // _BM),), dtype=np.int32),
            "offsets": np.zeros((spec.E + 1,), dtype=np.int32),
            "token_slot_table": np.zeros((spec.M, spec.top_k), dtype=np.int32),
        }

    @classmethod
    def spec_from_tensors(
        cls,
        logits,
        *,
        E: int,
        top_k: int,
        capacity: int,
    ) -> MoeRouterCorrectSpec:
        if logits.ndim != 2:
            raise ValueError(f"moe_router_correct: logits must be rank-2; got {logits.shape}")
        M, E_in = int(logits.shape[0]), int(logits.shape[1])
        if E_in != E:
            raise ValueError(f"moe_router_correct: logits.shape[1] ({E_in}) != E ({E})")
        return MoeRouterCorrectSpec(M=M, E=E, top_k=top_k, capacity=capacity)

    # ── build() ──

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_threads = c.n_warps * self._sgs
        M, E, K = s.M, s.E, s.top_k
        total_slots = s.total_slots
        n_chunks = total_slots // _BM

        tid = bctx.tid

        one_i32 = bctx.c(1, dtype=DType.S32)
        zero_i32 = bctx.c(0, dtype=DType.S32)
        zero_f32 = bctx.c(0.0, dtype=DType.F32)
        sentinel_i32 = bctx.c(-1, dtype=DType.S32)
        log2e = bctx.c(_LOG2E, dtype=DType.F32)
        neg_inf_f32 = bctx.c(-1.0e30, dtype=DType.F32)
        bm_const = bctx.c(_BM, dtype=DType.S32)
        bm_minus_1 = bctx.c(_BM - 1, dtype=DType.S32)

        # ── Phase 0: cooperative zero-init ─────────────────────────────
        # token_ids / slot_weights: full coverage by inits_per_thread passes.
        inits_per_thread = total_slots // n_threads
        for i in range(inits_per_thread):
            idx = qk.add(tid, bctx.c(i * n_threads, dtype=DType.U32))
            qk.store(g.token_ids, zero_i32, idx)
            qk.store(g.slot_weights, zero_f32, idx)

        # counts[E], offsets[E+1]: first E (or E+1) threads.
        e_pred = qk.cmp("lt", tid, bctx.c(E, dtype=DType.U32))
        qk.store(g.counts, zero_i32, tid, pred=e_pred)
        e1_pred = qk.cmp("lt", tid, bctx.c(E + 1, dtype=DType.U32))
        qk.store(g.offsets, zero_i32, tid, pred=e1_pred)

        # work_list: pre-fill with (grp_start = i*BM, expert = -1). Phase
        # 3 (leading thread) overwrites the active region with valid
        # experts. We need n_chunks pairs => 2*n_chunks entries.
        # Each thread covers ceil(2*n_chunks / n_threads) entries.
        wl_total = 2 * n_chunks
        wl_passes = (wl_total + n_threads - 1) // n_threads
        for i in range(wl_passes):
            offset = i * n_threads
            idx = qk.add(tid, bctx.c(offset, dtype=DType.U32))
            in_range_wl = (
                qk.cmp("lt", idx, bctx.c(wl_total, dtype=DType.U32))
                if offset + n_threads > wl_total
                else None
            )
            # Even index → grp_start; odd index → sentinel.
            # Express this with a select on (tid + offset) parity.
            # Simpler: write grp_start at idx*BM/2 if even, sentinel if odd
            # — but odd/even depends on `idx` value, not unrolled position.
            # We need a runtime select.
            idx_s32 = qk.bitcast(idx, DType.S32)
            is_odd = qk.and_(idx_s32, one_i32)  # 1 if odd, 0 if even
            is_odd_pred = qk.cmp("eq", is_odd, one_i32)
            half_idx = qk.shr(idx_s32, one_i32)  # idx // 2 (chunk number)
            grp_start_for_chunk = qk.mul(half_idx, bm_const)
            value = qk.select(is_odd_pred, sentinel_i32, grp_start_for_chunk)
            if in_range_wl is None:
                qk.store(g.work_list, value, idx)
            else:
                qk.store(g.work_list, value, idx, pred=in_range_wl)

        qk.barrier("block")

        # ── Phase 1: per-token top-K + softmax + atomic count ───────────
        # n_threads >= M so each thread handles exactly one token (predicate
        # masks the threads with tid >= M).
        in_range = qk.cmp("lt", tid, bctx.c(M, dtype=DType.U32))

        # Load all E logits.
        logits_e: list = []
        for e in range(E):
            v = qk.load(g.logits, tid, bctx.c(e, dtype=DType.U32))
            logits_e.append(v)

        # Insertion-sort top-K (descending) + softmax over those K.
        pri_score, pri_idx = insertion_sort_topp(
            bctx, logits_e, top_p=K, neg_inf=neg_inf_f32, zero_i32=zero_i32
        )
        pri_weight = softmax_topk(pri_score, top_k=K, log2e=log2e)

        # Atomic-add to counts[expert_k]; capture slot_in_e in registers
        # for use in phase 5. Gate via if_ since atomic_rmw doesn't take pred.
        slot_in_e_regs: list = [zero_i32 for _ in range(K)]
        with qk.if_(in_range, carried=[*slot_in_e_regs]) as (then_in, else_in, arms):
            with arms.then_():
                new_slots = []
                for k in range(K):
                    e_idx = qk.bitcast(pri_idx[k], DType.U32)
                    s_local = qk.atomic_rmw(g.counts, "add", one_i32, e_idx)
                    new_slots.append(s_local)
                qk.yield_(*new_slots)
            with arms.else_():
                qk.yield_(*else_in)
        results = qk.last_results()
        slot_in_e_regs = list(results)

        qk.barrier("block")

        # ── Phase 2: leading thread → offsets + active work_list ───────
        is_lead = qk.cmp("eq", tid, bctx.c(0, dtype=DType.U32))
        with qk.if_(is_lead, carried=[]) as (_, _, arms):
            with arms.then_():
                # Walk experts in order, accumulating offsets and writing
                # one work_list entry per BM-chunk owned by each expert.
                # ``cum`` mutates across the python-unrolled outer loop only
                # (each iteration extends the if_region.then). ``chunk_i``
                # also mutates inside a runtime for_range body, so it has
                # to be ``carried=`` through that loop.
                cum = zero_i32
                chunk_i = zero_i32
                u32_zero = bctx.c(0, dtype=DType.U32)
                u32_one = bctx.c(1, dtype=DType.U32)
                shift5 = bctx.c(5, dtype=DType.S32)
                for e in range(E):
                    qk.store(g.offsets, cum, bctx.c(e, dtype=DType.U32))
                    e_count = qk.load(g.counts, bctx.c(e, dtype=DType.U32))
                    # padded = ((count + BM-1) >> 5) << 5
                    padded = qk.shl(qk.shr(qk.add(e_count, bm_minus_1), shift5), shift5)
                    # n_chunks_e: U32 for the for_range bound.
                    n_chunks_e_u32 = qk.bitcast(qk.shr(padded, shift5), DType.U32)
                    cum = qk.add(cum, padded)
                    e_const = bctx.c(e, dtype=DType.S32)

                    # Inner runtime loop. Carry ``chunk_i`` so each iteration
                    # gets the running global chunk index, and the loop's
                    # final value comes back via last_results.
                    with qk.for_range(
                        u32_zero,
                        n_chunks_e_u32,
                        u32_one,
                        iv_name="j",
                        carried=(chunk_i,),
                    ) as (_j, carried_in):
                        cur_chunk = carried_in[0]
                        addr_i = qk.add(qk.shl(cur_chunk, one_i32), one_i32)
                        addr_u = qk.bitcast(addr_i, DType.U32)
                        qk.store(g.work_list, e_const, addr_u)
                        next_chunk = qk.add(cur_chunk, one_i32)
                        qk.yield_(next_chunk)
                    chunk_i = qk.last_results()[0]
                # offsets[E] = total active slots
                qk.store(g.offsets, cum, bctx.c(E, dtype=DType.U32))
                qk.yield_()
            with arms.else_():
                qk.yield_()

        qk.barrier("block")

        # ── Phase 3: scatter token_ids/slot_weights + emit
        # token_slot_table[tid, k] = final ─────────────────────
        tid_i = qk.bitcast(tid, DType.S32)
        with qk.if_(in_range, carried=[]) as (_, _, arms):
            with arms.then_():
                for k in range(K):
                    e = pri_idx[k]
                    e_u = qk.bitcast(e, DType.U32)
                    base = qk.load(g.offsets, e_u)
                    final = qk.add(base, slot_in_e_regs[k])
                    final_u = qk.bitcast(final, DType.U32)
                    qk.store(g.token_ids, tid_i, final_u)
                    qk.store(g.slot_weights, pri_weight[k], final_u)
                    # token_slot_table is [M, top_k]; the per-row store
                    # uses ``(tid, k_const)`` indexing so the lowerer
                    # picks up the row stride from the tensor's shape.
                    qk.store(
                        g.token_slot_table,
                        final,
                        tid,
                        bctx.c(k, dtype=DType.U32),
                    )
                qk.yield_()
            with arms.else_():
                qk.yield_()
