"""MoeRouterSpec — capacity-bounded top-K router with full-fallback substitution.

Consumes router logits ``[M, E]`` (f32) and emits the metadata three
the moe_inproj / moe_outproj kernels need: expert-major-sorted
``token_ids[E*C]`` and ``slot_weights[E*C]``. ``work_list`` is
deterministic given (E, C, BM) so the runtime builds it once and
reuses; the router kernel doesn't touch it.

Routing contract:
  * Per token, compute softmax over the top-K expert logits.
  * Greedy fill: each token tries to claim its top-K experts in order
    via per-expert atomic counters; on overflow, falls through to *any*
    non-claimed expert with capacity (priority-sorted across all E).
    The token always gets ``top_k`` distinct experts whenever the
    bipartite assignment is feasible (which it always is for K << E
    with ``E*C >= M*top_k``); only adversarial K-near-E configs can
    still hit infeasibility.
  * Slot weight stored is always the original-rank softmax weight so
    per-token weights sum to ~1 regardless of which expert filled the slot.

Capacity ``C`` is set by the caller and must satisfy
``E * C >= M * top_k`` so the bipartite assignment is feasible.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class MoeRouterSpec(KernelSpec):
    M: int  # number of input tokens
    E: int  # number of experts
    top_k: int  # slots claimed per token
    capacity: int  # slots per expert (== total_slots / E)

    def __post_init__(self):
        if self.M <= 0:
            raise ValueError(f"MoeRouterSpec: M must be positive; got {self.M}")
        if self.E <= 0:
            raise ValueError(f"MoeRouterSpec: E must be positive; got {self.E}")
        if not (1 <= self.top_k <= self.E):
            raise ValueError(f"MoeRouterSpec: top_k must be in [1, E={self.E}]; got {self.top_k}")
        if self.capacity <= 0:
            raise ValueError(f"MoeRouterSpec: capacity must be positive; got {self.capacity}")
        if self.M * self.top_k > self.E * self.capacity:
            raise ValueError(
                f"MoeRouterSpec: M*top_k ({self.M * self.top_k}) > E*capacity "
                f"({self.E * self.capacity}); capacity too small to admit all assignments"
            )

    @property
    def total_slots(self) -> int:
        return self.E * self.capacity
