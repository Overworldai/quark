"""MoeRouterSharedSpec — shared-experts routing.

All M tokens route to the *same* K experts, selected globally by
cumulative softmax preference (sum of per-token softmax(logits) across
all tokens; pick the K experts with highest sum). Per-token slot weights
are then ``softmax(logits[t, chosen_experts])`` so each token's K
contributions still sum to 1.

Buffer layout matches ``MoeRouterSpec`` exactly so ``moe_inproj`` /
``moe_outproj`` can be reused unchanged:

  * ``token_ids[K*M]``: ``token_ids[k*M + t] = t`` (every token under
    every chosen expert; total ``K*M = E*capacity`` slots).
  * ``slot_weights[K*M]``: ``slot_weights[k*M + t] =
    softmax(logits[t, chosen_experts])[k]``.
  * ``counts[E]``: ``M`` for each chosen expert, ``0`` for the rest
    (informational; not consumed by inproj/outproj).
  * ``work_list[2*K*M/BM]``: paired ``(grp_start, expert_id)`` entries.
    For shared-experts the layout is ``K`` blocks of ``M`` slots, each
    block labelled with its chosen expert ID (in E-space); ``BM=32``.

This makes shared-experts mode identical in compute cost to a dense
MLP: only ``K`` distinct experts are touched, each does one full GEMM
over all ``M`` tokens, and ``K * (M, H_per_expert) @ (D, H_per_expert).T
= 1 * (M, K*H_per_expert) @ (D, K*H_per_expert).T = 1 * (M, H_dense)
@ (D, H_dense).T``, the dense MLP.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class MoeRouterSharedSpec(KernelSpec):
    M: int  # number of input tokens
    E: int  # number of experts (full pool)
    top_k: int  # K experts shared across all tokens
    capacity: int  # per-expert slot count; must satisfy E * capacity == K * M
    # (so the buffer layout aligns with the balanced router's)

    def __post_init__(self):
        if self.M <= 0:
            raise ValueError(f"MoeRouterSharedSpec: M must be positive; got {self.M}")
        if self.E <= 0:
            raise ValueError(f"MoeRouterSharedSpec: E must be positive; got {self.E}")
        if not (1 <= self.top_k <= self.E):
            raise ValueError(
                f"MoeRouterSharedSpec: top_k must be in [1, E={self.E}]; got {self.top_k}"
            )
        if self.capacity <= 0:
            raise ValueError(f"MoeRouterSharedSpec: capacity must be positive; got {self.capacity}")
        if self.E * self.capacity != self.M * self.top_k:
            raise ValueError(
                f"MoeRouterSharedSpec: E*capacity ({self.E * self.capacity}) "
                f"!= M*top_k ({self.M * self.top_k}); shared-experts layout "
                f"requires exact equality so each chosen expert gets all M tokens"
            )
        # Need each chosen expert's M slots to align cleanly to BM=32 work-list chunks.
        if self.M % 32 != 0:
            raise ValueError(f"MoeRouterSharedSpec: M ({self.M}) must be divisible by BM=32")

    @property
    def total_slots(self) -> int:
        return self.E * self.capacity  # == K * M
