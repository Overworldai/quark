"""MoeRouterCorrectSpec — purely-correct routing.

Each token routes to its actual top-K experts (ranked by router logits)
with NO capacity bound — no substitution, no drops. Per-expert token
counts are imbalanced and vary per call.

Buffer layout: tokens are sorted by their assigned expert into the
``token_ids[]`` array, with each expert's run padded to a multiple of
BM=32 so the kernel can keep its "all BM rows in a chunk share one
expert" invariant. The padding tail of each expert has
``slot_weights = 0`` so it contributes nothing in outproj. Chunks past
the last filled slot are marked with ``expert = -1`` (sentinel) so the
inproj/outproj kernels can early-exit on them.

  * ``token_ids[E*capacity]``   sorted-by-expert; padding tail per expert
  * ``slot_weights[E*capacity]`` non-zero for real entries, 0 for padding
  * ``counts[E]``                actual per-expert token count (no padding)
  * ``work_list[2*total/BM]``    one (grp_start, expert) per BM-chunk;
                                 expert = -1 for chunks past total active
  * ``offsets[E+1]``             cumulative starting slot per expert
                                 (with BM padding); offsets[E] = total
                                 active slots this call

Capacity sizing — buffer must hold the worst case where every expert
has count_e off-BM, padding up to BM-1 each:

    total_slots_max = M*K + E*(BM-1), rounded up to BM
    capacity = total_slots_max / E (rounded up to BM)

For Waypoint15 (M=128, K=4, E=16, BM=32) this is 992 → 1024 = 64*16,
so capacity=64 — exactly 2× the balanced router's 32. A 2× buffer
budget; the actual chunks-iterated each call is always less.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class MoeRouterCorrectSpec(KernelSpec):
    M: int
    E: int
    top_k: int
    capacity: int  # per-expert slot count; total_slots = E * capacity
    # must be >= M*K + E*(BM-1) so the worst-case padded
    # layout fits.

    def __post_init__(self):
        if self.M <= 0:
            raise ValueError(f"MoeRouterCorrectSpec: M must be positive; got {self.M}")
        if self.E <= 0:
            raise ValueError(f"MoeRouterCorrectSpec: E must be positive; got {self.E}")
        if not (1 <= self.top_k <= self.E):
            raise ValueError(
                f"MoeRouterCorrectSpec: top_k must be in [1, E={self.E}]; got {self.top_k}"
            )
        if self.capacity <= 0:
            raise ValueError(
                f"MoeRouterCorrectSpec: capacity must be positive; got {self.capacity}"
            )
        worst_case = self.M * self.top_k + self.E * (32 - 1)
        if self.E * self.capacity < worst_case:
            raise ValueError(
                f"MoeRouterCorrectSpec: E*capacity ({self.E * self.capacity}) "
                f"< worst-case M*top_k + E*(BM-1) ({worst_case}); buffer too small"
            )
        if self.capacity % 32 != 0:
            raise ValueError(
                f"MoeRouterCorrectSpec: capacity ({self.capacity}) must be a multiple of BM=32"
            )

    @property
    def total_slots(self) -> int:
        return self.E * self.capacity
