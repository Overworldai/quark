# Conventions

## Files

- 500-line soft target, 800 hard cap.
- Exempt with `EXEMPT FROM 500-LINE RULE` in the module docstring +
  a one-sentence reason (see `backend.py`, `ir/tensor.py`,
  `ir/validator.py`, `owl_attn/kernel.py`, `lower/msl/mma.py`,
  `weight_shuffle.py`, `lang/__init__.py`, `lang/epilogue.py`). The
  pre-commit hook blocks commits that exceed the soft cap without
  the exemption.
- `blocks/dsl/` is a package split across concern-specific submodules
  (context, tensors, accumulators, carry, smem_tile, block_context,
  kernel_context) — kernel authors import from `popcorn.blocks`, not
  the submodules directly.
- One class per file for L1 / L2 blocks.
- Kernel folders split into `spec.py` / `config.py` / `kernel.py` /
  `reference.py` / `problems.py` / `baselines.py` — never collapse.

## Naming

| Kind | Example |
|---|---|
| Problem dims | `M`, `N`, `K` |
| Block tile dims | `BM`, `BN`, `BK` |
| MMA tile counts | `MT = BM // mma.m`, `NT = BN // mma.n` |
| Thread identity | `tid`, `lane_id`, `warp_id`, `gid` (lane >> 2), `tig` (lane & 3) |
| Global tensor | `g.A` (via TENSORS manifest) |
| Smem region | `A_smem` / `q_region` / `A_tile.smem` |
| Per-lane view | `A_lane` / `q_lane` |
| L0 emit functions | `emit_*` |
| Block classes | CamelCase: `MmaBody`, `SmemPlan`, `PipelineBody`, `Accumulators`, `Stage`, `Carry` |
| `pop.*` helpers | snake_case: `pop.store_acc`, `pop.work_list_load`, `pop.index_cache`, `pop.q_register_load`, `pop.silu`, `pop.cast` |
| Dtype | `DType` enum (str + IR + backend bridge). `spec.a_dtype.backend` → torch/mlx dtype; `DType("bf16") is DType.BF16`. |

## Imports

Order: stdlib → `torch` (only in baselines/ci-only code; otherwise
route through `PT`) → `popcorn.lang` → `popcorn.ir` → `popcorn.backend`
→ `popcorn.blocks` → `popcorn.kernels.*`.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import popcorn.lang as pop
from popcorn.backend import PT
from popcorn.blocks import (
    Accumulators, IterCtx, MmaBody, PipelineBody, SmemPlan, SmemTile, Stage, TensorDecl,
)
from popcorn.ir import Builder, DType, Module
from popcorn.ir.mma_registry import ALL_SHAPES, lookup_mma
from popcorn.kernels.base import Kernel, MmaSite
from popcorn.kernels.decorator import kernel
```

## Type hints

- PEP 604: `X | None`, not `Optional[X]`.
- PEP 585: `list[int]`, not `List[int]`.
- `from __future__ import annotations` at the top of every file.
- `ClassVar[T]` for mutable class attributes (ruff RUF012).

## Dataclasses

- `frozen=True` for Spec and Config — required for autotune cache
  hashing. `__post_init__` validates.
- Not frozen for mutable state (Builder internals, BlockContext).
- `field(compare=False)` for non-identity fields on hashable
  classes (see `Lifetime.region`).

## Comments

- Default: **no comments**. Well-named identifiers + short bodies
  do the job.
- Write a comment only when the **why** is non-obvious: a hidden
  constraint, a workaround for a specific bug, a subtle invariant.
- Never restate what the code does.
- Never reference the current task / fix / callers ("used by X",
  "added for Y flow") — those belong in the PR description.

## Forbidden patterns

| Banned | Use instead |
|---|---|
| `emit_raw()` outside `lower/ptx/lower.py` | Builder / block methods |
| `import cupy` at module scope | (just don't) |
| `torch.allclose` / `max_abs` as pass/fail | `check_correctness` (cos_sim) |
| `time.perf_counter` / `torch.cuda.Event` in `tests/` | `tools/bench.py` |
| Star imports | explicit names |
| Mutable default values on dataclass fields | `field(default_factory=...)` |
| `TODO` with no context | `TODO(issue #N): …` |
| Raw `torch.*` / `mlx.*` in references / make_tensors | `popcorn.backend.PT` |
| `getattr(cls, "CORRECTNESS_THRESHOLD", None)` | `cls.correctness_threshold(out_dtype)` |
| `spec_from_tensors` | retired; don't add new callers |
| `global_tensors() -> []` stubs | omit; base default is fine |
| Per-kernel `from_problem` for trivial cases | override `CONFIG_CLS.default_for(spec)` instead |
| `bctx.bld.*` direct calls in `build()` | `import popcorn.lang as pop`; use `pop.mul`, `pop.for_range`, `pop.barrier`, … |
| `KLoop` / `Pipeline` (retired) | `PipelineBody(stages=…, produce=…, consume=…, carry=…).run(n_iters=…, n_stages=…)` |
| Retired dataclass wrappers (`WorkListLoad`, `IndexCache`, `QRegisterLoad`, `GatheredTileLoad`, `NormalizeAndStore`, `Gemm1QK`/`Gemm2PV`) | `pop.work_list_load`, `pop.index_cache`, `pop.q_register_load`, `SmemTile.gather_from`, `pop.store_acc(per_warp=, row_scale=)`, `MmaBody(a=, b=, acc=)` |
| Hardcoded `mma_k` / shape names in config | declare `MmaSite`s on the kernel; autotuner fills per-site `<site>_shape` |

## Performance rules

- Epilogue: **always** frag → smem → max-width `vec_store` (16 B =
  v4.b32 = 8×bf16) for large output tiles. `pop.store_acc(...,
  stage_in_smem=True, staging_smem=...)` is the fast path; direct
  scalar stores stay for tiny per-warp tiles.
- Prefer `smem_tile.load_from(gmem_tensor, row=, col=, cast=)` for
  gmem→smem tile loads. It dispatches through `SharedRegion.copy_from`
  to cp.async (PTX) or sync vec_load (MSL).
- Prefer `RegisterTile.map / reduce_along_cols / convert` over
  `vec_extract` + `vec_build` — the former stays in registers; the
  latter on MSL routes through smem.
- Re-autotune after any kernel-emit refactor. The pre-existing JSON
  configs will be stale.

## Git

- New commits only; never amend a pushed commit.
- Never `--no-verify`; if a hook fails, fix the root cause.
- Pre-commit hook runs ruff format + ruff check + ty + file_size +
  device-free pytest in ~1 second. Budget is 5 seconds; if you
  break it, move the check to CI.
