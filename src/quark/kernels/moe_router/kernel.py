"""moe_router — capacity-bounded top-K dispatch with full-fallback substitution.

Single-block kernel that consumes router logits ``[M, E]`` and writes
expert-major-sorted ``token_ids[E*C]`` / ``slot_weights[E*C]`` plus the
per-expert ``counts[E]`` gmem workspace it uses for atomic capacity
tracking.

Phases:
  1. Cooperative zero-init of counts / token_ids / slot_weights (gmem).
  2. Per-token routing — each thread handles ``M / n_threads`` tokens:
       a. Load logits[t, 0..E-1] into registers.
       b. Insertion-sort to fully order all E priorities (descending).
       c. Softmax over the top-K priorities only — those are the
          original-rank weights stored regardless of which expert
          actually fills each slot.
       d. For each k in [0, K), greedy claim via atomicAdd on the
          per-expert counter; on overflow, fall through to *any*
          non-claimed expert with capacity, in priority order.

The runtime sees this kernel as the single op that turns "router GEMM
output" into "what moe_inproj/moe_outproj want as input metadata".
``work_list`` is *not* emitted here — its content is deterministic
given (E, C, BM) and the runtime builds it once at MoE-block init.

Constraints (enforced in ``is_valid``):
  * M, E*C both divisible by ``n_threads`` (clean cooperative partition).
  * Single block — M ≤ 1024 in practice.
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels._moe_router_dsl import insertion_sort_topp, softmax_topk
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.moe_router.config import MoeRouterConfig
from quark.kernels.moe_router.problems import moe_router_problems
from quark.kernels.moe_router.reference import moe_router_reference_numpy
from quark.kernels.moe_router.spec import MoeRouterSpec

_LOG2E = 1.4426950408889634


@kernel(
    "moe_router",
    spec=MoeRouterSpec,
    config=MoeRouterConfig,
    output_idx=-1,
    problems=moe_router_problems,
    baselines=lambda kernel, tensors: [],
    reference=moe_router_reference_numpy,
)
class MoeRouterKernel(Kernel):
    # Atomic dispatch is non-deterministic — within an expert, the slot
    # ordering depends on which thread's atomic resolves first. The
    # numpy reference resolves in token-id order. Validation compares
    # multisets of (expert, weight, token) emitted, not the within-
    # expert permutation, so the cos-sim check passes loose. Keep the
    # threshold relaxed because the bench gate is per-output-element.
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
        # Per-expert atomic counters — kernel zeros them at entry, so the
        # runtime doesn't need a separate memset launch. Declared as an
        # output role so the runtime allocates persistent storage; the
        # contents at exit aren't useful downstream.
        TensorDecl(
            "counts",
            dtype=lambda s, c: DType.S32,
            shape=lambda s, c: (s.E,),
            role="out",
        ),
    ]

    spec: MoeRouterSpec
    config: MoeRouterConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if not (1 <= c.n_warps <= 32):
            return False
        n_threads = c.n_warps * self._sgs
        # Single-block design caps M.
        if n_threads < s.M:
            return False
        if n_threads % s.M != 0 and s.M % n_threads != 0:
            return False
        if (s.E * s.capacity) % n_threads != 0:
            return False
        # Need enough threads to issue at least one atomic counter init
        # per expert in the leading-thread phase.
        if n_threads < s.E:
            return False
        return True

    def grid(self) -> tuple[int, int, int]:
        return (1, 1, 1)

    def flops(self) -> int:
        s = self.spec
        # Approximate: full E-priority insertion sort (E*E compares) + softmax
        # over K (K exps + K divs) + K greedy claims with up-to-E candidates.
        # Dominated by per-token work, scaled by M.
        return s.M * (s.E * s.E + 4 * s.top_k + s.top_k * s.E)

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [4, 8, 16, 32]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        spec = MoeRouterSpec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "logits": rng.standard_normal((spec.M, spec.E)).astype(np.float32),
            "token_ids": np.zeros((spec.total_slots,), dtype=np.int32),
            "slot_weights": np.zeros((spec.total_slots,), dtype=np.float32),
            "counts": np.zeros((spec.E,), dtype=np.int32),
        }

    @classmethod
    def spec_from_tensors(
        cls,
        logits,
        *,
        E: int,
        top_k: int,
        capacity: int,
    ) -> MoeRouterSpec:
        if logits.ndim != 2:
            raise ValueError(f"moe_router: logits must be rank-2; got {logits.shape}")
        M, E_in = int(logits.shape[0]), int(logits.shape[1])
        if E_in != E:
            raise ValueError(f"moe_router: logits.shape[1] ({E_in}) != E ({E})")
        return MoeRouterSpec(M=M, E=E, top_k=top_k, capacity=capacity)

    # ── build() ──

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_threads = c.n_warps * self._sgs
        M, E, K, C = s.M, s.E, s.top_k, s.capacity
        # All E priorities are sorted; the fallback path can try every
        # non-claimed expert, so per-slot trial set spans all of pri_idx.
        P = E
        EC = E * C

        tid = bctx.tid

        one_i32 = bctx.c(1, dtype=DType.S32)
        zero_i32 = bctx.c(0, dtype=DType.S32)
        zero_f32 = bctx.c(0.0, dtype=DType.F32)
        true_p = bctx.c(True, dtype=DType.PRED)
        false_p = bctx.c(False, dtype=DType.PRED)
        sentinel_i32 = bctx.c(-1, dtype=DType.S32)
        cap_i32 = bctx.c(C, dtype=DType.S32)
        log2e = bctx.c(_LOG2E, dtype=DType.F32)
        neg_inf_f32 = bctx.c(-1.0e30, dtype=DType.F32)

        # ── Phase 0: cooperative zero-init ─────────────────────────────
        # counts[E]: first E threads each zero one entry.
        if n_threads > E:
            init_count_pred = qk.cmp("lt", tid, bctx.c(E, dtype=DType.U32))
            qk.store(g.counts, zero_i32, tid, pred=init_count_pred)
        else:
            qk.store(g.counts, zero_i32, tid)

        # token_ids[E*C], slot_weights[E*C]: each thread zeros EC/n_threads entries.
        inits_per_thread = EC // n_threads
        for i in range(inits_per_thread):
            idx = qk.add(tid, bctx.c(i * n_threads, dtype=DType.U32))
            qk.store(g.token_ids, zero_i32, idx)
            qk.store(g.slot_weights, zero_f32, idx)

        qk.barrier("block")

        # ── Phase 1: per-token routing ─────────────────────────────────
        # Each thread handles tokens_per_thread (or 0 if M < n_threads,
        # in which case is_valid pinned n_threads >= M and we predicate).
        if n_threads <= M:
            tokens_per_thread = M // n_threads
            need_predicate = False
        else:
            tokens_per_thread = 1
            need_predicate = True

        for tk in range(tokens_per_thread):
            tid_token_u = qk.add(tid, bctx.c(tk * n_threads, dtype=DType.U32))
            tid_token_i = qk.bitcast(tid_token_u, DType.S32)

            in_range = (
                qk.cmp("lt", tid_token_u, bctx.c(M, dtype=DType.U32)) if need_predicate else true_p
            )

            # Step A: load logits[token, 0..E-1] into registers.
            logits_e = []
            for e in range(E):
                v = qk.load(g.logits, tid_token_u, bctx.c(e, dtype=DType.U32))
                logits_e.append(v)

            # Step B: insertion sort — top-P (score, idx) pairs.
            pri_score, pri_idx = insertion_sort_topp(
                bctx, logits_e, top_p=P, neg_inf=neg_inf_f32, zero_i32=zero_i32
            )

            # Step C: softmax over the top-K scores only. Fallback
            # candidates (positions K..E-1) don't enter the softmax —
            # their slot weight, if used, is the original-rank
            # ``pri_weight[k]``. Matches world_engine's
            # ``weights = softmax(scores.topk(top_k))`` so per-token
            # weights sum to 1.0 (modulo any structural drops).
            pri_weight = softmax_topk(pri_score, top_k=K, log2e=log2e)

            # Step D: greedy claim with full-fallback substitution +
            # per-token dedup. claimed_e[k] is the expert this token's
            # k-th slot won, or sentinel if no expert had room (only
            # possible when K is close to E and the bipartite assignment
            # is structurally infeasible — unreachable for typical MoE
            # configs where K << E).
            claimed_e: list = [sentinel_i32 for _ in range(K)]

            for k in range(K):
                done_k = false_p
                # Try every expert in priority order. pri_idx[k] is the
                # k-th-best (the "intended" choice); if it's full, fall
                # through to any non-claimed expert with room. Dedup via
                # claimed_e prevents a token from doubling up on one
                # expert. Note: the slot weight stored is always the
                # original-rank softmax weight pri_weight[k], regardless
                # of which expert ultimately fills the slot.
                trial_indices = [k] + [i for i in range(P) if i != k]
                for i in trial_indices:
                    cand_e = pri_idx[i]
                    # already_claimed = OR_{j<k} (cand_e == claimed_e[j])
                    already_claimed = false_p
                    for j in range(k):
                        eq = qk.cmp("eq", cand_e, claimed_e[j])
                        already_claimed = qk.or_(already_claimed, eq)
                    not_done = qk.xor(done_k, true_p)
                    not_claimed = qk.xor(already_claimed, true_p)
                    should_attempt = qk.and_(qk.and_(not_done, not_claimed), in_range)

                    # Gate the atomic+stores behind should_attempt to
                    # avoid wasting capacity on dedup-rejected attempts.
                    with qk.if_(should_attempt, carried=[done_k, claimed_e[k]]) as (
                        then_in,
                        else_in,
                        arms,
                    ):
                        with arms.then_():
                            slot = qk.atomic_rmw(g.counts, "add", one_i32, cand_e)
                            ok = qk.cmp("lt", slot, cap_i32)
                            addr_i = qk.add(qk.mul(cand_e, cap_i32), slot)
                            addr_u = qk.bitcast(addr_i, DType.U32)
                            qk.store(g.token_ids, tid_token_i, addr_u, pred=ok)
                            qk.store(g.slot_weights, pri_weight[k], addr_u, pred=ok)
                            new_claimed = qk.select(ok, cand_e, then_in[1])
                            qk.yield_(ok, new_claimed)
                        with arms.else_():
                            qk.yield_(else_in[0], else_in[1])
                    results = qk.last_results()
                    done_k = results[0]
                    claimed_e[k] = results[1]
