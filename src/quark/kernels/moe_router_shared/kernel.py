"""moe_router_shared — shared-experts routing.

Single-block kernel. Picks K experts globally (highest cumulative
softmax preference summed across all M tokens), then routes every
token to those same K experts. Emits the same buffer layout as
``moe_router`` so the same ``moe_inproj`` / ``moe_outproj`` kernels
consume the result without modification.

Phases:
  0. Cooperative zero-init of token_ids / slot_weights / counts /
     work_list / cum_probs / chosen_experts.
  1. Each thread t (where t < M) computes softmax(logits[t, :]) and
     atomic-adds each ``probs[e]`` into ``cum_probs[e]``.
  2. Block barrier.
  3. Leading thread sorts ``cum_probs`` top-K and writes
     ``chosen_experts[K]``.
  4. Block barrier.
  5. Each thread t reloads the K chosen logits, computes softmax over
     them, writes ``slot_weights[k*M + t]`` and ``token_ids[k*M + t]``.
  6. Cooperatively build the ``work_list`` (one (grp_start, expert)
     entry per BM=32-slot chunk); set ``counts[chosen[k]] = M``.

For the typical W1.5 case (M=128, E=16, K=4, n_warps=4 → n_threads=128)
each thread handles exactly one token, and the routing kernel is
~5–10 µs total.
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels._moe_router_dsl import insertion_sort_topp, softmax_topk
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.moe_router_shared.config import MoeRouterSharedConfig
from quark.kernels.moe_router_shared.problems import moe_router_shared_problems
from quark.kernels.moe_router_shared.reference import moe_router_shared_reference_numpy
from quark.kernels.moe_router_shared.spec import MoeRouterSharedSpec

_BM = 32
_LOG2E = 1.4426950408889634


@kernel(
    "moe_router_shared",
    spec=MoeRouterSharedSpec,
    config=MoeRouterSharedConfig,
    output_idx=-1,
    problems=moe_router_shared_problems,
    baselines=lambda kernel, tensors: [],
    reference=moe_router_shared_reference_numpy,
)
class MoeRouterSharedKernel(Kernel):
    # The leading-thread top-K is deterministic; per-token weights are
    # bit-equivalent across threads (no atomic accumulator on the weight
    # path). cum_probs accumulation uses f32 atomic-add, which is
    # non-deterministic in ordering and can drift the K-th vs (K+1)-th
    # ranking on near-tie configs. Keep the threshold loose for that.
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
        # Workspace: cum_probs[E] f32 (atomic-accumulator target).
        TensorDecl(
            "cum_probs",
            dtype=lambda s, c: DType.F32,
            shape=lambda s, c: (s.E,),
            role="out",
        ),
        # Workspace: chosen_experts[K] s32 (broadcasted from leading
        # thread's top-K to all threads via gmem).
        TensorDecl(
            "chosen_experts",
            dtype=lambda s, c: DType.S32,
            shape=lambda s, c: (s.top_k,),
            role="out",
        ),
    ]

    spec: MoeRouterSharedSpec
    config: MoeRouterSharedConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if not (1 <= c.n_warps <= 32):
            return False
        n_threads = c.n_warps * self._sgs
        # Strict one-thread-per-token — phase-1 reads logits[tid, e]
        # unconditionally; relaxing to n_threads > M would let the extra
        # threads OOB-read past the logits buffer. Same reasoning as
        # moe_router_correct.
        if n_threads != s.M:
            return False
        if s.total_slots % n_threads != 0:
            return False
        if n_threads < s.E:
            return False
        # Cooperative work_list build at the end of build() requires one
        # thread per chunk. With n_threads pinned to M, this is satisfied
        # exactly when M >= total_slots/BM (i.e. M*BM >= K*M → BM >= K).
        if n_threads < s.total_slots // 32:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        return (1, 1, 1)

    def flops(self) -> int:
        s = self.spec
        # Per-token softmax (~4*E ops), atomic adds (E), top-K sort
        # (E*K), per-token softmax over K (~4*K ops), slot writes (K).
        return s.M * (4 * s.E + s.E) + s.E * s.top_k + s.M * (4 * s.top_k + s.top_k)

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [4, 8, 16, 32]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        spec = MoeRouterSharedSpec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "logits": rng.standard_normal((spec.M, spec.E)).astype(np.float32),
            "token_ids": np.zeros((spec.total_slots,), dtype=np.int32),
            "slot_weights": np.zeros((spec.total_slots,), dtype=np.float32),
            "counts": np.zeros((spec.E,), dtype=np.int32),
            "work_list": np.zeros((2 * (spec.total_slots // _BM),), dtype=np.int32),
            "cum_probs": np.zeros((spec.E,), dtype=np.float32),
            "chosen_experts": np.zeros((spec.top_k,), dtype=np.int32),
        }

    @classmethod
    def spec_from_tensors(
        cls,
        logits,
        *,
        E: int,
        top_k: int,
        capacity: int,
    ) -> MoeRouterSharedSpec:
        if logits.ndim != 2:
            raise ValueError(f"moe_router_shared: logits must be rank-2; got {logits.shape}")
        M, E_in = int(logits.shape[0]), int(logits.shape[1])
        if E_in != E:
            raise ValueError(f"moe_router_shared: logits.shape[1] ({E_in}) != E ({E})")
        return MoeRouterSharedSpec(M=M, E=E, top_k=top_k, capacity=capacity)

    # ── build() ──

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_threads = c.n_warps * self._sgs
        M, E, K = s.M, s.E, s.top_k
        KM = s.total_slots  # == K * M
        n_chunks = KM // _BM
        chunks_per_chosen = M // _BM  # how many BM-chunks each chosen expert covers

        tid = bctx.tid

        one_i32 = bctx.c(1, dtype=DType.S32)
        zero_i32 = bctx.c(0, dtype=DType.S32)
        zero_f32 = bctx.c(0.0, dtype=DType.F32)
        log2e = bctx.c(_LOG2E, dtype=DType.F32)
        neg_inf_f32 = bctx.c(-1.0e30, dtype=DType.F32)
        M_i32 = bctx.c(M, dtype=DType.S32)
        bm_const = bctx.c(_BM, dtype=DType.S32)

        # ── Phase 0: cooperative zero-init ─────────────────────────────
        # token_ids[K*M], slot_weights[K*M]: each thread does KM/n_threads.
        inits_per_thread = KM // n_threads
        for i in range(inits_per_thread):
            idx = qk.add(tid, bctx.c(i * n_threads, dtype=DType.U32))
            qk.store(g.token_ids, zero_i32, idx)
            qk.store(g.slot_weights, zero_f32, idx)

        # counts[E], cum_probs[E]: first E threads (n_threads >= E by is_valid).
        e_pred = qk.cmp("lt", tid, bctx.c(E, dtype=DType.U32))
        qk.store(g.counts, zero_i32, tid, pred=e_pred)
        qk.store(g.cum_probs, zero_f32, tid, pred=e_pred)

        # chosen_experts[K]: first K threads.
        k_pred = qk.cmp("lt", tid, bctx.c(K, dtype=DType.U32))
        qk.store(g.chosen_experts, zero_i32, tid, pred=k_pred)

        # work_list[2*n_chunks]: zero so unused slots stay safe.
        wl_total = 2 * n_chunks
        wl_full_passes = wl_total // n_threads
        for i in range(wl_full_passes):
            idx = qk.add(tid, bctx.c(i * n_threads, dtype=DType.U32))
            qk.store(g.work_list, zero_i32, idx)
        wl_remainder = wl_total - wl_full_passes * n_threads
        if wl_remainder > 0:
            idx = qk.add(tid, bctx.c(wl_full_passes * n_threads, dtype=DType.U32))
            wl_pred = qk.cmp("lt", idx, bctx.c(wl_total, dtype=DType.U32))
            qk.store(g.work_list, zero_i32, idx, pred=wl_pred)

        qk.barrier("block")

        # ── Phase 1: per-token softmax → atomic-add into cum_probs ─────
        # n_threads >= M (is_valid pinned), so each thread handles exactly
        # one token (or zero, gated by predicate).
        in_range = qk.cmp("lt", tid, bctx.c(M, dtype=DType.U32))

        # Load all E logits for this thread's token.
        logits_e: list = []
        for e in range(E):
            v = qk.load(g.logits, tid, bctx.c(e, dtype=DType.U32))
            logits_e.append(v)

        # max over E for numerical stability.
        max_l = logits_e[0]
        for e in range(1, E):
            gt = qk.cmp("gt", logits_e[e], max_l)
            max_l = qk.select(gt, logits_e[e], max_l)

        # exps and sum.
        exps: list = []
        for e in range(E):
            shifted = qk.sub(logits_e[e], max_l)
            exp_v = qk.ex2_approx(qk.mul(shifted, log2e))
            exps.append(exp_v)
        sum_exps = exps[0]
        for e in range(1, E):
            sum_exps = qk.add(sum_exps, exps[e])
        rcp_sum = qk.rcp_approx(sum_exps)

        # Atomic-add each prob into cum_probs[e]. Gate via if_ since
        # atomic_rmw doesn't accept a predicate.
        with qk.if_(in_range, carried=[]) as (_, _, arms):
            with arms.then_():
                for e in range(E):
                    p = qk.mul(exps[e], rcp_sum)
                    qk.atomic_rmw(g.cum_probs, "add", p, bctx.c(e, dtype=DType.U32))
                qk.yield_()
            with arms.else_():
                qk.yield_()

        qk.barrier("block")

        # ── Phase 2: leading thread picks top-K of cum_probs ───────────
        is_lead = qk.cmp("eq", tid, bctx.c(0, dtype=DType.U32))
        with qk.if_(is_lead, carried=[]) as (_, _, arms):
            with arms.then_():
                # Load all E cum_probs.
                cum_e: list = []
                for e in range(E):
                    v = qk.load(g.cum_probs, bctx.c(e, dtype=DType.U32))
                    cum_e.append(v)

                # Insertion sort top-K (K << E typically).
                _top_score, top_idx = insertion_sort_topp(
                    bctx, cum_e, top_p=K, neg_inf=neg_inf_f32, zero_i32=zero_i32
                )
                del _top_score  # only the indices are written out

                # Write chosen_experts[k] and counts[chosen[k]] = M.
                for k in range(K):
                    qk.store(g.chosen_experts, top_idx[k], bctx.c(k, dtype=DType.U32))
                    qk.store(g.counts, M_i32, qk.bitcast(top_idx[k], DType.U32))
                qk.yield_()
            with arms.else_():
                qk.yield_()

        qk.barrier("block")

        # ── Phase 3: per-token softmax over chosen K, write outputs ────
        # Each thread reloads chosen_experts (small, hits L1).
        chosen: list = []
        for k in range(K):
            v = qk.load(g.chosen_experts, bctx.c(k, dtype=DType.U32))  # s32
            chosen.append(v)

        # Load logits[t, chosen[k]] for k in [0, K).
        chosen_logits: list = []
        for k in range(K):
            chosen_u = qk.bitcast(chosen[k], DType.U32)
            v = qk.load(g.logits, tid, chosen_u)
            chosen_logits.append(v)

        # softmax over K.
        weights = softmax_topk(chosen_logits, top_k=K, log2e=log2e)

        # Write slot_weights[k*M + tid] = weights[k] and token_ids[...] = tid.
        tid_i = qk.bitcast(tid, DType.S32)
        for k in range(K):
            slot_addr_i = qk.add(bctx.c(k * M, dtype=DType.S32), tid_i)
            slot_addr_u = qk.bitcast(slot_addr_i, DType.U32)
            qk.store(g.slot_weights, weights[k], slot_addr_u, pred=in_range)
            qk.store(g.token_ids, tid_i, slot_addr_u, pred=in_range)

        # ── Phase 4: build work_list cooperatively ─────────────────────
        # n_chunks entries; each is (grp_start, expert_id) = 2 s32. The
        # chosen expert for chunk i is chosen[i // chunks_per_chosen].
        # ``is_valid`` pins ``n_threads >= n_chunks`` so one thread per
        # chunk always works; the fallback (n_threads < n_chunks) is
        # rejected upstream so we never reach build() with that combo.
        wl_pred = qk.cmp("lt", tid, bctx.c(n_chunks, dtype=DType.U32))
        with qk.if_(wl_pred, carried=[]) as (_, _, arms):
            with arms.then_():
                # k_pos = tid // chunks_per_chosen. Static-unroll the
                # power-of-two division as a nested ge-cmp/select chain
                # (K is small, typically ≤ 16).
                k_pos_i = bctx.c(0, dtype=DType.S32)
                for k in range(1, K):
                    threshold = bctx.c(k * chunks_per_chosen, dtype=DType.U32)
                    ge = qk.cmp("ge", tid, threshold)
                    k_pos_i = qk.select(ge, bctx.c(k, dtype=DType.S32), k_pos_i)
                expert_id = qk.load(g.chosen_experts, qk.bitcast(k_pos_i, DType.U32))

                # grp_start = tid * BM
                tid_s32 = qk.bitcast(tid, DType.S32)
                grp_start = qk.mul(tid_s32, bm_const)

                # work_list[2*tid] = grp_start; work_list[2*tid + 1] = expert
                addr0_i = qk.mul(tid_s32, bctx.c(2, dtype=DType.S32))
                addr0_u = qk.bitcast(addr0_i, DType.U32)
                qk.store(g.work_list, grp_start, addr0_u)
                addr1_i = qk.add(addr0_i, one_i32)
                addr1_u = qk.bitcast(addr1_i, DType.U32)
                qk.store(g.work_list, expert_id, addr1_u)
                qk.yield_()
            with arms.else_():
                qk.yield_()
