# Variable Subgroup Size — SPV Stack Scope

Lifting the hardcoded `subgroup_width = 32` assumption so kernels can
opt into SIMD16 (or SIMD8) on Intel Xe2.

## Why

On Battlemage, SIMD32 forces three structural inefficiencies:

1. Only half the Xe-core's Vector Engines are dual-mode SIMD16/SIMD32;
   the SIMD16-only VEs sit idle.
2. The native `m8n16k16` MMA shape has n=16 — half a SIMD32 wave.
3. The 4 KB GRF/HW-thread budget is split across 32 work-items
   (128 B each), so kernels with non-trivial register footprint
   (OwlAttn at MTiles=2) spill. At SIMD16 each work-item sees 256 B
   logically, and Mesa anv's allocator can also opt into 256-reg
   "large GRF" mode — typically doubling the effective register
   budget again.

Expected payoff: OwlAttn **1.8–2.5×** standalone, **1.4–1.7×** LFPS
end-to-end after pipeline-overlap shrinkage. Benefits propagate to
every other SPV kernel.

## Touchpoint Map

### Hard SIMD32 hardcodes (must become parametric)

| file | line(s) | what |
|---|---|---|
| `src/quark/lower/spv/lower.py` | 256 | `_SUBGROUP_WIDTH = 32` global |
| `src/quark/lower/spv/lower.py` | 2803, 3128, 3358, 3650 | `subgroup_width = _SUBGROUP_WIDTH` reads in 4 visitors |
| `src/quark/lower/spv/lower.py` | 3257–3263 | `_visit_frag_reduce` cluster math (`rows_per_slot = subgroup_width // cluster_size`) |
| `src/quark/drivers/_spv_dispatch.cpp` | 928 | `rss.requiredSubgroupSize = 32` |
| `src/quark/lang/memory.py` | 79 | `lane_id = bctx.tid % 32` |
| `src/quark/lang/epilogue.py` | 743 | `n_lanes = 32` in `per_warp` path |
| `src/quark/lang/epilogue.py` | 744 | `thread_idx = bctx.tid % 32` |
| `src/quark/kernels/owl_attn/kernel.py` | 469 | `lane_id = tid % 32` |
| `src/quark/kernels/*/kernel.py` | many | `n_threads = c.n_warps * 32` (18 sites across kernels) |
| `src/quark/kernels/base.py` | 652, 733 | default `block()` returns `n_warps * 32` |
| `src/quark/ir/mma_registry.py` | 545–547, 570–572, 595–597, ... | `a_regs/b_regs/c_regs` (= elems/lane) values are SIMD32-derived |

### MMA fragment count math (must recompute)

For a coopmat tile of size `rows × cols`:

- `n_slots_per_lane = rows * cols / subgroup_width`
  - m8n16 = 128: SIMD32 → 4 slots/lane, SIMD16 → 8 slots/lane
- `slot_idx(lane, s) = lane + s * subgroup_width`
  (formula in `_frag_scratch_slot_idx`)
- Per-warp scratch size unchanged (always `rows * cols` per warp)

### `_visit_frag_reduce` clustered-reduce math

Comment block at line 3245–3263 currently assumes:
```
lanes 0..15  → row class 2*s
lanes 16..31 → row class 2*s + 1
```
i.e. 2 clusters of 16 lanes per slot at SIMD32. At SIMD16 with the same
`m8n16` shape: 1 cluster of 16 lanes per slot, so `rows_per_slot = 1`
and the cluster broadcast leader chain changes. The math
`rows_per_slot = subgroup_width // cluster_size` already generalises;
the leader-lane offset comment needs updating.

### Kernel-side warp ↔ thread arithmetic

`tid % 32` (lane id) and `tid // 32` (warp id) appear in:
- `src/quark/lang/memory.py:79` — Q load per-warp cooperative split
- `src/quark/lang/epilogue.py:744` — `per_warp` epilogue
- `src/quark/kernels/owl_attn/kernel.py:469` — Q-RoPE
- 18 sites computing `n_threads = c.n_warps * 32`

These should use `bctx.lane_id` / `bctx.warp_id` SSA values (which
already exist for `warp_id` via `qk.subgroup_id()`; need to add
`lane_id` via a new builder helper that lowers to
`OpLoad SubgroupLocalInvocationId`).

## Phased Implementation Plan

### Phase 0 — Plumbing (1 day)

Add a `SubgroupSize` capability to the kernel/lowerer pipeline:

1. **`SpirVLowerer`** takes a `subgroup_width: int` param at init
   (default 32 for back-compat). Plumb through `_SpvCtx`.
2. **All `subgroup_width = _SUBGROUP_WIDTH` lines** replaced with
   `subgroup_width = ctx.subgroup_width`.
3. **`MmaConfig`** gets a `regs_for(subgroup_width)` method that
   computes `a_regs/b_regs/c_regs` from `(m, n, k)` and the width.
   Existing fields kept as fallback (default at width=32).
4. **Kernel `block()` and `n_threads`** consult `ctx.caps.subgroup_size`
   (or a new `KernelConfig.subgroup_size` field) instead of hardcoded
   32. Plumbing pattern: kernel reads from active `BlockContext` caps.
5. **Builder gets** `bctx.lane_id` (lazy `qk.subgroup_invocation_id()`).
   Replace `tid % 32` patterns.

**Deliverable:** lowerer + framework accepts a subgroup_width param
but every site still passes 32 → identical SPV output. Validation:
all 77 SPV tests pass byte-for-byte (snapshot the SPV text before/after).

### Phase 1 — Pipeline create wiring (half day)

1. **`_spv_dispatch.cpp::compile`** takes a `subgroup_size` arg
   (default 32). Pass it to `rss.requiredSubgroupSize`.
2. **Python `_sd.compile`** wrapper accepts/forwards the arg.
3. **`Launcher.compile`** plumbs it from the lowered kernel metadata
   (read from `LoweredSpirVKernel`, which gets a new field).
4. **`SpirVLowerer.lower_module`** writes the chosen width into the
   returned `LoweredSpirVKernel`.

**Deliverable:** end-to-end opt-in: a kernel emits at width=16 →
pipeline created with `requiredSubgroupSize=16`. Reverse: width=32
behaves identically to today.

### Phase 2 — `_visit_frag_*` SIMD16 correctness (1–2 days)

The four visitors that use `subgroup_width`:
- `_visit_frag_for_each` (line 2782+)
- `_visit_frag_apply` (line 3635+)
- `_visit_frag_convert` (line 3358+)
- `_visit_frag_reduce` (line 3128+ — multi-class cluster path)

Each has a slot-iteration loop sized by `n_slots_per_lane`. At
SIMD16 with m8n16=128 elements, this doubles from 4 to 8 slots/lane.
The math already uses `subgroup_width` as a variable; just need to
plumb correctly.

The cluster-reduce in `_visit_frag_reduce` needs the comment block
updated and the assertion `rows_per_slot * n_slots == n_results`
re-verified. With `cluster_size = cols = 16` and `subgroup_width = 16`:
`rows_per_slot = 1`, so n_classes math goes from `2 * n_slots` to
`1 * n_slots`. The IR call sites (`OnlineSoftmax` etc.) pass
`n_classes_override` based on cd_offsets; need to recompute that
from the *subgroup width × cluster size* product.

**Tests:**
- SPV unit tests with both width=32 and width=16 paths. Initially
  parameterise just `_visit_frag_for_each` and run a trivial frag-
  identity kernel under both. Add per-visitor coverage.
- Cross-validate: same input, both widths, same output bit-identically.

**Risk:** Mesa anv's actual lane↔(row,col) mapping inside a coopmat
fragment is implementation-private and **may differ at SIMD16 vs
SIMD32**. Spec says it's opaque — we round-trip through smem to
expose the layout. If Mesa packs differently at SIMD16, our
`row = mat_idx / cols, col = mat_idx % cols` assumption breaks.
Mitigation: a probe kernel that writes a known pattern, reads back,
inspects. Run it once per (shape, width) at startup, cache the
layout. Fall back to width=32 if mismatch.

### Phase 3 — Framework lane/warp arithmetic (half day)

1. Add `BlockContext.lane_id` (lazy SSA, `qk.subgroup_invocation_id()`).
2. Replace `tid % 32` → `bctx.lane_id` and `tid // 32` → `bctx.warp_id`
   at 4+ sites in `src/quark/lang/`.
3. Replace `n_threads = c.n_warps * 32` with `c.n_warps *
   ctx.subgroup_size` at 18 kernel sites + base.py.
4. `Kernel.block()` defaults to `(c.n_warps * ctx.subgroup_size, 1, 1)`.

**Risk:** some kernels (e.g. epilogue with `n_lanes = 32` hardcoded
for vec_store stride math) may break at SIMD16 because they assume
4-lane × 8-element vec_store coverage. Audit each `* 32` and
`% 32` site for layout-correctness, not just thread-count
correctness. Some may need actual algorithmic rework, not just
parameter substitution.

### Phase 4 — OwlAttn at SIMD16 (1 day)

1. Configure `AttnConfig` with `subgroup_size=16`.
2. Verify the kernel compiles and produces correct output (cos_sim
   > 0.999 vs SIMD32 reference).
3. Sweep `(NCW, MTiles)` at SIMD16 — expect MTiles=2 to work without
   spilling. Confirm via `INTEL_DEBUG=shaders,spill` output
   (target: `0:0 spills:fills` at MTiles=2 SIMD16, ≤ MTiles=1 SIMD32
   reg count of 67).
4. Benchmark standalone (`/tmp/bench_owlattn_ncw.py` adapted).
5. Run model end-to-end benchmark (`scripts/bench_quark_world_engine.py`)
   and report LFPS change.

### Phase 5 — Other kernels (1–2 days)

Each kernel needs a per-config decision: SIMD16 or SIMD32?

- Heavy register pressure / small-n MMA → SIMD16 likely wins.
- Big throughput-bound GEMM (`GemmKernel`) → SIMD32 may still win
  because it amortizes thread launch better at high arithmetic
  intensity. Bench both.

Build `KernelConfig.subgroup_size` into autotune search space.
Each `(spec, config)` cache key already includes the full config
so this is a free axis — autotune will pick whichever is faster
per shape.

### Phase 6 — Probe + fall-back (half day)

The Mesa-may-relayout-at-SIMD16 risk from Phase 2: ship a startup
probe that confirms the coopmat layout matches our assumption. If
mismatch detected, force `subgroup_size = 32` and emit a one-line
diagnostic.

## Total Effort & Risk

- **Phase 0–1**: 1.5 days. Mechanical. Low risk.
- **Phase 2**: 1–2 days. **Highest risk** (Mesa layout assumption).
- **Phase 3**: 0.5 days. Wide-spread changes; medium risk of subtle
  layout bugs in epilogue vec_store paths.
- **Phase 4**: 1 day. The "is it actually faster" measurement.
- **Phase 5**: 1–2 days. Parallel work across kernels.
- **Phase 6**: 0.5 days. Safety net.

**Total: 5–7 days** for full implementation + validation across
all kernels. Phases 0–4 alone (just enough to ship OwlAttn at
SIMD16) is **3–4 days**.

## Validation

**Correctness:**
- All 77 existing SPV tests pass at both widths.
- Per-kernel cos_sim > 0.999 between SIMD32 and SIMD16 outputs on
  fixed inputs.
- Full end-to-end Waypoint frame: byte-identical or cos_sim > 0.9999.

**Performance:**
- OwlAttn standalone at `/tmp/bench_owlattn_ncw.py`: target
  `< 17 ms/dispatch` at saturation (vs 30 ms today).
- Mesa stats at MTiles=2 SIMD16: target `0:0 spills:fills`.
- End-to-end LFPS at 360p saturation: target ≥ 2.5 (vs 1.63 today).

## Out of scope

- Variable subgroup size *within a single kernel* (mixing SIMD16
  and SIMD32 work). Not supported by KHR_subgroup_size_control
  on Mesa anv anyway.
- Other vendors. Metal (Apple) and PTX (NVIDIA) lowerers stay at
  their respective fixed widths.
- Auto-selection heuristic. We expose the knob; autotune picks.
