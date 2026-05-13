"""OwlAttnIntConfig — int8 attention tunables.

KvTile fixed to 32 (= MMA k for int8). Single-warp first cut to
mirror GemmIntKernel's incremental lifting pattern.

``phase`` selects the build-body slice the kernel implements:
  - ``"qk_scores"``    — Phase 3.2: dump dequantized QK scores
                          ``s_f32`` ``(q_rows, capacity)`` f32.
  - ``"softmax_probs"``— Phase 3.3a: dump softmax probabilities ``P``
                          ``(q_rows, capacity)`` f32 (max + exp + l +
                          normalize, no AV yet).
  - ``"full"``         — Phase 3.3b+: full attention output
                          ``(out_rows, out_cols)`` ``out_dtype``.

Default is ``"qk_scores"`` (Phase 3.2). Phase 3.3a flips the
default to ``"softmax_probs"`` once it's validated; Phase 3.3b
flips to ``"full"``.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig

PHASE_QK_SCORES = "qk_scores"
PHASE_SOFTMAX_PROBS = "softmax_probs"
PHASE_FULL = "full"
VALID_PHASES = (PHASE_QK_SCORES, PHASE_SOFTMAX_PROBS, PHASE_FULL)


@dataclass(frozen=True)
class OwlAttnIntConfig(KernelConfig):
    KvTile: int = 32  # = MMA k for int8 m8n16k32
    MTiles: int = 1   # m-tiles per warp (each = 8 Q rows)
    # NCW (consumer warps per WG): default 1 for smoke shapes to validate;
    # autotune picks NCW=4 at the production shape (~1.55× faster than
    # NCW=1 per Phase 3.6a bench).
    NCW: int = 1
    KvPad: int = 0
    n_stages: int = 1  # single-buffered for the first cut
    main_shape: str = "m8n16k32_intel_s8_s32"
    phase: str = PHASE_FULL  # default lifts as each phase lands
