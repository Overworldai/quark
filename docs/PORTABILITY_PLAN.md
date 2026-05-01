# Portability Plan — toward a third backend (SPIR-V for Intel iGPUs)

Status: **Phases 1 & 2 landed; Phase 3 scaffolded only.**

Landing map (commit SHAs on branch stack
`refactor/portability-phase-1` → `refactor/portability-phase-2`):

| Phase | Status | Commit |
|---|---|---|
| 1.1 Lowerer ABC + registry        | ✅ landed      | `21034b5` |
| 1.2 `subgroup_width` + caps plumb | ✅ landed      | `b9de2ed` |
| 1.3 Shared `NameAllocator`        | ✅ landed      | `54521cc` |
| 1.4 MmaShape de-stringify         | ✅ landed      | `cd2987f` |
| 2.1 Legalization framework        | ✅ landed      | `a88390a` |
| 2.2 Registrations (keep-path only) | ✅ landed     | `e4e432b` |
| 2.2 `cvt_rn_bf16x2_f32` body      | ✅ real        | `e0c4325` |
| 2.2 `fma_bf16x2` body             | ✅ real        | `cc54795` |
| 2.2 Vector-atomic body            | ✅ real        | `134297f` |
| 2.2 `AsyncCopyOp` family body     | ✅ real        | `fe727fb` |
| 2.2 `SubgroupReduceOp` butterfly  | ✅ real        | `acb29d5` |
| 2.2 `Rewriter` SSA helper         | ✅ landed      | `dac7f00` |
| 2.2 Nested-region walks           | ✅ landed      | `8197cff` |
| 2.2 Per-function shared ID counter | ✅ fixed      | `0578d32` |
| 2.3 Tests                         | ✅ landed (per-rewrite goldens + driver mechanics) |
| 2.3 Idempotency across nested regions | ✅ pinned | `2e149bf` |
| 3.0 SPIR-V backend scaffolding    | ✅ landed (stub only) | `e12b08e` |
| 3.1 Driver + device probing       | ⏳ needs Intel GPU hardware |
| 3.2 `SpirVLowerer` body           | ⏳ needs 3.1 + cooperative-matrix probe |
| 3.3 `FragApplyOp` lowering        | ⏳ needs 3.2 |
| 3.4 Subgroup width decision       | ⏳ needs 3.1 |
| 3.5 Rollout v1 → v3               | ⏳ multi-week on real hardware |

Post-landing hardening (not on the original plan but flowed out of
phase-2 integration):

| Fix | Status | Commit |
|---|---|---|
| PTX ``_visit_shuffle`` clamp from ``caps.subgroup_width``  | ✅ | `ec2ddba` |
| MMA payloads colocated on ``MmaConfig`` as per-``DeviceFamily`` fields (``cuda``/``metal``), gates + payloads paired per backend | ✅ | `b87db86` |

**Phase 2 is substantively complete.** Every registered rewrite has
a real expansion body; the five non-hardware pieces of the plan
are shipped. The ``SubgroupReduceOp`` butterfly migration was the
last one — it fires on CUDA kernels today (rmsnorm / attn /
ada_rmsnorm exercise it) and produces byte-equivalent PTX to the
previous inline butterfly, validated by cos_sim ≈ 0.999996 on
real RTX 4090 kernels. PTX's ``_visit_subgroup_reduce`` stays as
a safety net for direct-to-lowerer paths that bypass legalize.

Since initial phase-2 completion the code has been hardened
against three latent bugs surfaced by probing tests: nested-region
coverage (ops inside ``ForLoopOp`` bodies were skipped by the
top-level-only walk), SSA Value-ID collisions (two rewrites on
nearby-ID ops could allocate overlapping ranges), and the PTX
shuffle-clamp hardcode (wrong for sub-32-wide subgroups). Each
had zero behavior change on wired CUDA but would have been a
miscompile on a sub-warp SPIR-V backend; the fixes land before
the first such backend arrives.

## Why

The current lowerer architecture (`lower/ptx/`, `lower/msl/`) works for
two backends that happen to share most fundamentals: fixed 32-wide
subgroups, GPU-style threadgroup memory, 8×8 or 16×8 MMA tiles.
Adding a third backend — SPIR-V compute shaders targeting Intel Arc /
Xe iGPUs — stress-tests every assumption that was never made explicit:

- Subgroup width is **8, 16, or 32** on Intel and chosen by the
  driver, not the hardware.
- Cooperative matrix sizes are **driver-queried at runtime**, not
  statically registered.
- There is no `cp.async`, no `ldmatrix`, no `fma.rn.bf16x2`.
- The generic "reduce across the subgroup" operation is a
  **first-class opcode** (`OpGroupNonUniformAdd`), not an expanded
  butterfly.

The current code handles these by accident, not by design — things
happen to work on NV and Apple because the authors knew the
assumptions. SPIR-V forces us to name and abstract them.

This document is the staged plan to get there without rewriting the
world. The ordering is chosen so that every step is independently
valuable even if SPIR-V is never shipped.

## Non-goals

- Eliminating `cp.async`, `ldmatrix`, or other NV-specific fast paths
  from the IR. Those remain as optional instructions; the legalization
  pass decides whether to use or expand them.
- Making the IR fully target-neutral. Quark's IR is deliberately
  low-level — the sweet spot is "kernel author writes cp.async and
  gets cp.async on CUDA." The goal is *graceful degradation on
  targets that lack the primitive*, not a universal IR.
- Shipping SPIR-V on day one. Each phase below stands alone; SPIR-V
  is the forcing function, not the deliverable.

---

## Phase 1 — refactors that are no-regret on current hardware

These land independently of SPIR-V and improve the codebase for its
own sake. Each is under a week.

**Landed `21034b5`…`cd2987f`.** Sub-phase sections below describe the
original design; implementation notes call out where reality deviated
from the sketch.

### 1.1 Introduce a `Lowerer` ABC + registry

**File:** new `src/quark/lower/base.py`.

```python
class Lowerer(Protocol):
    def lower_module(self, m: Module) -> LoweredModule: ...
    def lower_function(self, f: Function) -> LoweredFunction: ...

LOWERERS: dict[DeviceFamily, type[Lowerer]] = {}

def register_lowerer(family: DeviceFamily):
    def _deco(cls): LOWERERS[family] = cls; return cls
    return _deco
```

Replace the family-typed branch in `launcher.py:_lower()` (today at
[launcher.py:510-529](quark/src/quark/launcher/launcher.py#L510)) with a single lookup:

```python
def _lower(self, kernel, spec, config):
    family = self.device.family
    lowerer_cls = LOWERERS[family]
    lowerer = lowerer_cls(self.device.caps)
    return lowerer.lower_module(...)
```

Annotate existing backends:

```python
@register_lowerer(DeviceFamily.CUDA)
class PtxLowerer: ...

@register_lowerer(DeviceFamily.METAL)
class MslLowerer: ...
```

**Behavior:** zero change. **Risk:** zero. **Benefit:** SPIR-V is a
one-line `@register_lowerer(DeviceFamily.INTEL_GPU)` away.

**Implementation note:** the decorator ended up wrapping a *factory
function* rather than the Lowerer class directly, because
`PtxLowerer.__init__` takes `(ptx_version, target_sm, caps)` — not a
single `caps` argument. The factory maps caps → ctor kwargs. The
plan's "`@register_lowerer(DeviceFamily.CUDA) class PtxLowerer`"
sketch works for MSL's ctor but not PTX's; both now go through a
small adapter in the backend's `__init__.py`.

### 1.2 Plumb `caps.subgroup_width` into the lowerers

Today `DeviceCaps.warp_size` exists (`device.py:156`) but is consumed
only by `occupancy.py`. Kernels and lowerers bypass it via hardcoded
constants.

**Changes:**

1. Rename `warp_size` → `subgroup_width` in `DeviceCaps` (keep a
   deprecated alias for one release). Name reflects the vendor-
   neutral term in SPIR-V / WebGPU.
2. Pass `caps` to `PtxLowerer.__init__` and stash as `self.caps`.
3. Replace the hardcoded `32` / `31` sites in the shuffle-reduction
   emitters ([ptx/lower.py:770](quark/src/quark/lower/ptx/lower.py#L770)
   and [ptx/lower.py:1438](quark/src/quark/lower/ptx/lower.py#L1438)):

   ```python
   # BEFORE
   for offset in (16, 8, 4, 2, 1):
       ctx.emit(f"shfl.sync.bfly.b32 {other}, {cur}, {offset}, 31, -1;")

   # AFTER
   W = self.caps.subgroup_width
   offset = W // 2
   while offset >= 1:
       ctx.emit(f"shfl.sync.bfly.b32 {other}, {cur}, {offset}, {W-1}, -1;")
       offset //= 2
   ```

4. Delete the per-kernel `_WARP = 32` constants in the 5 kernel
   files ([ada_gate_residual, ada_rmsnorm, rmsnorm, head_rmsnorm,
   value_residual_packed](quark/src/quark/kernels/)). Replace
   with `spec.caps.subgroup_width` at kernel-build time (not import
   time). Current hardware gives W=32 either way → behavior
   unchanged.

**Behavior on NV/Apple:** identical. **Risk:** subtle — the kernel
  validity checks (`s.D % _WARP != 0`) now parameterize on caps; if a
  test harness constructs DeviceCaps without setting subgroup_width,
  it breaks. Guard with an explicit default of 32.

**Implementation note:** step 4 didn't fully land. The `_WARP = 32`
literals in each of the 5 kernel files now import a shared
`DEFAULT_SUBGROUP_WIDTH = 32` constant from `quark.device`, but
the per-kernel `is_valid()` check still reads the module constant
rather than `caps.subgroup_width` — because `is_valid()` runs
before a device context attaches to the kernel. A proper fix
threads caps through `is_valid_for(caps)` into a caps-aware
validity hook; that refactor is tracked as a TODO in the shared
constant's docstring and deferred until a backend with W ≠ 32
(RDNA wave64, Intel non-pinned) actually demands it.

### 1.3 Extract a shared `NameAllocator`

**File:** new `src/quark/lower/_common/name_alloc.py`.

PTX `regs.py` (233 LOC) does two orthogonal things:
- (a) counter-per-class unique-name generation with a prefix table,
- (b) PTX register class mapping (`reg_class(dtype)`, `.reg .f32`
  declarations, `arith_suffix`).

(a) is 100% shareable with MSL and any future backend. (b) stays in
PTX. Move (a) to `lower/_common/name_alloc.py`, make PTX import it
and subclass with the register-class logic. MSL's `names.py`
replaces its own counter with the same class.

**Benefit:** ~80 LOC of duplication gone. Mostly aesthetic; the real
value is that the third backend doesn't write its own counter yet
again.

**Implementation note:** the MSL lowerer's MMA emitter
(`lower/msl/mma.py`) reaches into `ctx.names._names` directly to
re-bind loop-body locals across walks of a `FragApplyOp` body.
That private access survived the refactor via a ``_names`` property
alias on the MSL subclass that returns the shared base's
``_components`` dict. A cleaner fix lifts the re-bind into a public
method on the allocator — deferred until a second use case appears.

### 1.4 Move backend-specific strings off `MmaShape`

Today `MmaShape` carries backend-specific mnemonic strings —
`ptx: str | None` and `msl: str | None` (plus `hip` / `opencl`
placeholders) — embedded in the shape dataclass
([mma_registry.py:88-89](quark/src/quark/ir/mma_registry.py#L88)).
The chip gates (`min_ptx_cc` / `min_metal_gen`) live on `MmaConfig`
a few lines down, not on `MmaShape` — that part stays as-is.
Adding SPIR-V means adding `spv: str | None` to `MmaShape` and
growing the dataclass per backend. This is a leaky abstraction — the
IR has no business knowing what PTX mnemonics look like.

**Changes:**

1. Strip `ptx`, `msl`, `hip`, `opencl` from `MmaShape`. Keep
   `name, m, n, k, a_dtype, b_dtype, acc_dtype, a_regs, b_regs, c_regs`.
2. Add per-backend mnemonic tables:
   ```python
   # src/quark/lower/ptx/mma_mnemonics.py
   PTX_MMA: dict[str, str] = {
       "m16n8k16_bf16": "m16n8k16.row.col.f32.bf16.bf16.f32",
       ...
   }
   ```
   MSL gets an analogous `msl_simdgroup_tiles.py`.
3. Rename `min_ptx_cc` → `min_cuda_cc` on `MmaConfig` for vendor-
   neutral naming. The gate already lives on `MmaConfig` rather than
   `MmaShape`, so no relocation is needed — just the rename.

**Behavior:** zero change. **Benefit:** adding SPIR-V cooperative
matrix doesn't touch `MmaShape` or the IR at all — new shapes just
get registered in `spv_coop_matrix.py`.

---

## Phase 2 — the legalization pass

This is the one architecturally new piece. Phase 1 made it possible;
Phase 2 builds it.

### 2.1 Legalization framework

**File:** new `src/quark/lower/legalize.py`.

Between IR build and lowering, run target-specific rewrites:

```
Module (target-agnostic)
   ↓ legalize(module, caps)
Module' (same IR, ops the target can't do natively are expanded)
   ↓ lower
Target code
```

A legalization is a function `(Op, caps) -> list[Op]` returning a
replacement sequence. The driver walks the module, fires handlers
keyed on `type(op)` and gated on `caps`.

```python
@register_legalization(AsyncCopyOp)
def legalize_async_copy(op, caps):
    if caps.has_async_copy:
        return [op]
    return expand_async_copy_to_vec_loop(op)
```

This mirrors MLIR's conversion-pattern design. The pass runs once
per module before lowering. Legalizations are pure IR → IR
rewrites; they never touch target syntax.

### 2.2 First legalizations

Ship Phase 2 with the rewrites below, each independently testable.
None of them change behavior on NV because the relevant caps flags
are `True` on the targeted generations.

**Fully landed.** All five rewrites have real expansion bodies in
``src/quark/lower/legalizations.py``, guarded by a per-rewrite caps
flag. The registrations first shipped as keep-path-only stubs
(``e4e432b``) and then filled in incrementally:

  * ``cvt_rn_bf16x2_f32`` (``e0c4325``) — 2×Convert + 2×Bitcast + Merge
    (5 ops).
  * ``fma_bf16x2`` (``cc54795``) — 3×Split + 6×Bitcast + 6×Convert +
    2×scalar Fma + 2×Convert + 2×Bitcast + Merge (22 ops).
  * Vector-atomic (``134297f``) — Split + 2×Bitcast + Const(1) + Add +
    2×scalar AtomicRmwOp on adjacent columns (7 ops).
  * ``AsyncCopyOp`` family (``fe727fb``) — VecLoad→VecStore pair;
    commit/wait strip.
  * ``SubgroupReduceOp`` butterfly (``acb29d5``) — ``log2(W)`` Shuffle
    + Arith pairs; ``W = caps.subgroup_width``.

The last one (``SubgroupReduceOp``) actually fires on current CUDA
kernels — rmsnorm / attn / ada_rmsnorm exercise subgroup reduction
on real RTX 4090. The legalize-pass-expanded path emits
byte-equivalent PTX to the PTX lowerer's previous inline butterfly,
validated by unchanged cos_sim ≈ 0.999996 vs. main. Every other
rewrite's expansion branch stays unreached on wired backends
(CUDA has every ``has_*`` flag ``True`` on sm_80+ / sm_90+; Metal
kernels don't emit the relevant ops) but the tests synthesize the
op + fake caps and golden-verify the produced IR.

**Infrastructure:** ``Rewriter`` (``quark.lower.legalize.Rewriter``,
landed ``dac7f00``) handles function-scoped fresh SSA Value IDs so
rewrites can construct intermediate Values without colliding with
the existing op graph. Identity-preservation pattern: the final op
in every expansion chain reuses the original op's result Value so
downstream consumers stay linked without a separate use-replacement
pass.

**Implementation note (op-type variance):** the plan's sketch has
the rewrite fn typed as `(Op, caps) -> RewriteResult` and concrete
rewrites parameterized on `ArithOp` / `AsyncCopyOp` / etc. Python
allows covariant parameter narrowing at runtime but `ty` flags it
as `invalid-argument-type`. Widened every rewrite to take a base
`Op`; rewrites that care about a specific op type check
`isinstance` or read `op.attrs` inside. Minor readability loss
vs. type-narrowing, but avoids a per-rewrite `# ty: ignore`.

| Op | Target lacks? | Rewrite |
|---|---|---|
| `AsyncCopyOp` + `AsyncCopyCommit/Wait` | `not caps.has_async_copy` | → cooperative `VecLoad` → `VecStore` loop, strip commit/wait |
| `ArithOp(kind="fma_bf16x2")` (shipped) | `not caps.has_fma_bf16x2` | → 2× `Fma(f32)` with convert and extract |
| `ArithOp(kind="cvt_rn_bf16x2_f32")` (shipped) | `not caps.has_fma_bf16x2` | → 2× `convert(f32→bf16)` + `vec_build_packed_b32` |
| `AtomicRmwOp` with `atomic_type in ("bf16x2","f16x2")` (shipped) | `(dtype,2) not in caps.atomic_add_vector` | → expand to two scalar atomic adds, or fall back to f32 RMW in the scatter epilogue's non-paired path |
| `SubgroupReduceOp` (native path) | `caps.has_native_subgroup_reduce` | → keep op; lowerer emits `OpGroupNonUniformAdd` / `simd_sum`. Otherwise expand to butterfly chain (today's PTX behavior). |

The bf16x2 IR ops and the paired bf16x2 scatter epilogue
(`_emit_scatter_store_paired_bf16x2`) are already in tree — the
legalization rewrites above are how they degrade on non-NV backends.

**Caps flag split.** Main currently exposes
`caps.atomic_add_vector: frozenset[(DType, lanes)]` (set on
CUDA sm_90+ for `(BF16, 2)` and sm_70+ for `(F16, 2)`). It does
**not** expose a flag for scalar `fma.rn.bf16x2` (Ampere+ / sm_80+);
the PTX lowerer assumes availability. Add two named flags so
legalization can read them uniformly:

- `has_fma_bf16x2` — gates scalar `fma.rn.bf16x2` and
  `cvt.rn.bf16x2.f32` (CUDA sm_80+).
- `has_atomic_add_bf16x2` — gates `red.global.add.noftz.bf16x2`
  (CUDA sm_90+). Equivalent to `(BF16, 2) in atomic_add_vector`;
  keep the frozenset for generality but also expose the named
  bool for readable call sites.

**Behavior on NV:** `has_async_copy=True`, `has_fma_bf16x2=True`
on sm_80+, `has_atomic_add_bf16x2=True` on sm_90+,
`has_native_subgroup_reduce=False`. All rewrites either keep the
op or expand `SubgroupReduce` to the butterfly the PTX lowerer
already emits — net: zero change.

**Behavior on Apple:** `has_async_copy=False` (Metal has no
equivalent), `has_fma_bf16x2=False`, `has_atomic_add_bf16x2=False`.
The MSL backend today has no `AsyncCopyOp` visitor — kernels that
emit `async_copy` don't hit a Metal fallback because Metal-targeted
builds of those kernels don't use the op. Moving the expansion to
a pre-lowering legalization pass unifies the story: kernels emit
the IR they want, and the pass expands it on any backend that
lacks the primitive. This is more a design improvement than a
cleanup of duplicated code.

### 2.3 Tests

Legalization is pure IR-to-IR. Golden-file test for each rewrite:

```
tests/lower/test_legalize.py
  input IR (Python-constructed module)
  caps with feature=False
  → expected IR output
```

Run on CI without any GPU. This is the leverage — the hard logic
(async_copy expansion) gets tested once, not twice (MSL + future
SPIR-V).

**Landed state:** `tests/lower/test_legalize.py` (10 tests) covers
the driver mechanics — registry ops, rewrite splicing, caps
forwarding, self-match avoidance. `tests/lower/test_rewriter.py`
(8 tests) covers the SSA fresh-Value helper including a regression
test for the Value-ID collision bug (`0578d32`). And
`tests/lower/test_legalizations.py` (19 tests) covers each
registered rewrite with three patterns:

  * keep-path: op survives on ``_CUDA_SM90`` fake caps;
  * expansion-path: golden-IR walk asserts the produced op chain
    by count-per-kind + identity-preservation on the final op's
    result Value;
  * properties: idempotency (``legalize`` twice == once),
    nested-region walks (rewrites fire inside ``ForLoopOp``
    bodies), and combined idempotency + nested regions.

For the ``SubgroupReduceOp`` butterfly rewrite the golden is
parameterized on ``caps.subgroup_width`` — W=32 produces 5
iterations with params (16, 8, 4, 2, 1), W=8 produces 3
iterations with (4, 2, 1). The PTX lowerer's inline butterfly
is still covered by ``tests/lower/ptx/test_frag_reduce.py``
(direct-to-lowerer path, bypasses legalize).

---

## Phase 3 — SPIR-V backend (the payoff)

**Current status: §3.0 scaffolding landed (`e12b08e`); §3.1–3.5
need Intel Arc / Xe iGPU hardware and a chosen runtime.** The
scaffolding reserves `DeviceFamily.INTEL_GPU` in the enum and
registers a `SpirVLowerer` stub that raises `NotImplementedError`
on `lower_module`. Every phase-1/2 refactor that assumes
`INTEL_GPU` exists (legalization pass, caps-split, lowerer
registry) is self-consistent.

### 3.0 Scaffolding (landed)

**File:** new `src/quark/lower/spv/{__init__,lower}.py`.

- `DeviceFamily.INTEL_GPU` enum value present.
- `SpirVLowerer` + `LoweredSpirVKernel` stub classes; both raise
  `NotImplementedError` with messages pointing at this document.
- `@register_lowerer(DeviceFamily.INTEL_GPU)` factory registered at
  package-import time so `get_lowerer(DeviceFamily.INTEL_GPU, caps)`
  returns the stub (never a KeyError).
- `tests/lower/test_spv_skeleton.py` pins the expected surface.

Everything below (3.1+) is the real engineering work.

### 3.1 Driver + device probing

**File:** new `src/quark/drivers/spv.py`.

Wraps the chosen Vulkan / Level Zero runtime (probably vulkan-headers
+ `vulkan` Python bindings or `pyopencl` with SPIR-V intermediate
— decision deferred to prototype).

Must populate `DeviceCaps` by:
- Querying `VkPhysicalDeviceSubgroupProperties` for subgroup width.
- Querying `VkPhysicalDeviceCooperativeMatrixPropertiesKHR` for
  supported MMA shapes (Intel's set differs from NV's — e.g., Arc
  supports different k's).
- Setting flags: `has_async_copy=False`, `has_fma_bf16x2=False`,
  `has_atomic_add_bf16x2=False`, `has_native_subgroup_reduce=True`,
  `subgroup_width=<queried>`
  (32 on Arc, 8/16/32 on iGPU — pin via `reqd_sub_group_size(32)`
  for the v1, see §3.3).

### 3.2 `SpirVLowerer`

**File:** new `src/quark/lower/spv/lower.py` + `visitors.py` +
`types.py`.

Same shape as `MslLowerer`. Implement ~45 visitor handlers for the
existing op set. Thanks to Phase 2, several ops are pre-expanded:
- `AsyncCopyOp` → already rewritten to `VecLoad`/`VecStore`.
- `ArithOp(kind="fma_bf16x2")` / `ArithOp(kind="cvt_rn_bf16x2_f32")`
  → already rewritten to `Fma(f32)` / paired `convert(f32→bf16)` +
  `vec_build_packed_b32`.
- `SubgroupReduceOp` → single `OpGroupNonUniformAdd`.

The tricky ones that DON'T have Phase-2 rewrites:
- `MmaOp` / `LoadMatrixOp` / `StoreMatrixOp` — map to
  `OpCooperativeMatrixMulAddKHR` + `OpCooperativeMatrixLoadKHR` +
  `OpCooperativeMatrixStoreKHR`. Shape tables populated in Phase 1.4.
- `FragApplyOp` — see §3.3 below.
- `FragForEachOp` / `FragReduceOp` — declared unsupported (kernel
  rejected at `is_valid_for`); v1 ships without flash-attention on
  SPIR-V.

### 3.3 `FragApplyOp` lowering — the hard case

`FragApply` lets kernels transform accumulator elements between
matmuls (FP8 dequant scale, bias add). It's used by the MLP epilogue
and by some GEMM variants.

SPIR-V has no PTX-style "loop over per-lane registers" model.
`VK_KHR_cooperative_matrix` exposes element access via
`OpCooperativeMatrixLengthKHR` + index-based mutation, but the
lane→element mapping is driver-private.

**V1 strategy:** emit the `OpCooperativeMatrixLengthKHR` loop for
element-wise body functions that don't depend on (row, col)
position. This covers the "scale by a scalar" and "apply sigmoid"
cases. For any body function that reads `row_var` or `col_var`
(FragForEachOp territory), reject the kernel. This is the 80%
solution; the remaining 20% (coordinate-aware epilogues) needs
either a restructured kernel or a more elaborate legalization.

**Tracked:** `FragApplyOp.body` must be analyzed during
`is_valid_for` — if the body references block_context coord
variables, flag the kernel as unsupported on SPIR-V.

### 3.4 Subgroup width decision for Intel

Ship v1 with `reqd_sub_group_size(32)` pinned on all kernels. On Arc
this is optimal; on Xe-LP iGPU it sacrifices some perf but gives a
single-variant-per-kernel build pipeline. Revisit per-kernel after
benchmarking.

See [the subgroup-width analysis in the session notes] for tradeoff
detail.

### 3.5 Rollout

- v1: Smoke tests pass on Intel Arc A770 / A380 and one Xe iGPU
  (e.g., Arc Graphics in Meteor Lake). Kernel coverage: the
  elementwise/normalization set (silu, ada_rmsnorm,
  ada_gate_residual, value_residual_packed), and GEMM at one shape.
- v2: Full GEMM autotune space. Flash-attention excluded.
- v3: Flash-attention via restructured FragReduce lowering (TBD).

---

## Risk / open questions

- **Intel cooperative_matrix shape support**: Need to prototype
  what shapes Arc actually reports. If it's only m8n8k16 and we've
  autotuned against m16n8k16 elsewhere, config porting is non-trivial.
  Owner: whoever does 3.1, first thing, before committing to 3.2.
- **Driver maturity**: Intel's SPIR-V compute drivers on Linux
  (Mesa's ANV or Intel's proprietary) have known reliability
  issues on non-standard workloads. Expect at least one week of
  chasing driver bugs during 3.2.
- **`reqd_sub_group_size` rejection**: Some drivers may refuse
  SIMD32 for certain kernels (high register pressure). Need a
  runtime fallback to let the driver choose, which reopens the
  "variant per width" question. Revisit if this happens in practice.
- **Legalization correctness for `AsyncCopyOp`**: Today the Apple
  fallback is inline in the MSL visitor. Replacing it with a
  pre-lowering rewrite must produce byte-identical output to
  avoid perf regressions. Validate with MSL perf numbers before
  and after Phase 2 on a pipelined kernel (e.g.,
  `ada_gate_residual` smem path).

## Effort estimate

| Phase | Work | Calendar |
|---|---|---|
| 1.1 Lowerer ABC | 0.5 day | day 1 |
| 1.2 Subgroup width plumbing | 1 day | days 2–3 |
| 1.3 Shared NameAllocator | 0.5 day | day 4 |
| 1.4 MmaShape de-stringify | 1 day | days 5–6 |
| 2.1 Legalization framework | 1 day | days 7–8 |
| 2.2 First three rewrites + tests | 2 days | days 9–12 |
| 3.1 SPIR-V driver + probe | 2 days | days 13–16 |
| 3.2 SpirVLowerer core (non-MMA) | 3 days | days 17–22 |
| 3.3 MMA + FragApply | 3 days | days 23–28 |
| 3.4–3.5 Integration / kernel coverage | 3 days | days 29–32 |

Phase 1 + 2 combined: ~2 weeks. SPIR-V v1 after that: ~1 month.
This assumes one engineer and no novel driver bugs. Plan for 1.5×.

## Sequencing constraint

**Phase 1.2 and 1.4 must land before any SPIR-V work starts.**
Phase 2 is recommended before SPIR-V but the legalization pass
can be deferred and expanded inline in the SpirVLowerer if
schedule pressure demands — at the cost of duplicating the
async-copy-fallback logic between MSL and SPIR-V backends.

Phase 1.1 and 1.3 are pure refactors and can happen in any order.

## Out of scope for this plan

- HIP backend for AMD RDNA/CDNA. The architecture changes here
  (legalization pass, `caps.subgroup_width`, backend registry)
  make HIP strictly easier later, but the work is separate.
- WebGPU / WGSL backend. WebGPU's subgroup extension is even
  less mature than SPIR-V's.
- CPU AMX backend. Radically different programming model; would
  need its own plan.
