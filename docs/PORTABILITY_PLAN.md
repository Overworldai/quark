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

## What changed after Phase 2 — context for the Phase 3 revision

The plan as originally drafted ran on assumptions that the
``remove-mlx`` branch invalidates. Three structural shifts reshape
what "an Intel target" means in practice:

1. **MLX is gone.** The Metal driver
   (``src/quark/drivers/metal.py``, 488L) talks to ``Metal.framework``
   through a metal-cpp + nanobind C++ extension
   (``drivers/_metal_dispatch``). Buffers are bound by raw
   ``MTLBuffer`` handles; host tensors are numpy arrays with a
   quark-dtype tag. There is zero framework dependency in the
   kernel-side path. SPIR-V's analogue is a parallel C extension
   wrapping Vulkan — same shape, different runtime — not a
   Python-level binding into an existing framework.

2. **``Engine`` is the consumer surface, not ``quark.functional``.**
   ``engine/base.py`` factors out a polymorphic
   ``Engine.__new__(model_uri, ...)`` that constructs ``EngineCUDA``
   or ``EngineMetal`` based on the host platform. Biome and other
   consumers import ``quark.Engine`` and never see the kernel
   framework. The original plan only required a new ``Lowerer``
   slot; the revised plan also requires a new ``EngineIntel``
   subclass that owns:

   * host tensor format (numpy/torch — the choice differs from
     CUDA's torch and Metal's numpy);
   * VAE strategy on Intel (no Apple Neural Engine; candidates are
     OpenVINO, oneDNN, or torch via XPU/SYCL);
   * graph-capture / lazy-dispatch idioms (``quark.lazy()`` is a
     Metal-side construct mapping to deferred ``MTLCommandBuffer``
     commits — Vulkan has command buffers but a different lifetime
     model).

3. **NAX is the precedent for kernels that bypass the framework.**
   ``kernels/owl_attn/nax.py`` is hand-written outside the IR-emit
   path because the per-fragment cooperative-tensor layout doesn't
   fit the framework's ``Kernel/Spec/Config`` shape. Phase 3 must
   decide whether the SPIR-V port pursues a parallel hand-written
   ``OpCooperativeMatrix*KHR`` flash-attention or accepts an
   attention regression for v1. With NAX in tree, v1 now defaults
   to "no flash-attention on SPIR-V at all" until §3.6 lands; the
   plan's earlier "FragForEachOp / FragReduceOp rejection" line
   stays accurate but understates the gap — the only attention
   path on production Apple Silicon is the NAX one, not the
   IR-framework's ``owl_attn/kernel.py``, so SPIR-V doesn't get
   attention "for free" once the lowerer is real.

Two more changes inform the work but aren't blockers:

* The Phase 2 ``AsyncCopyOp`` legalization is unreached on
  production. CUDA has ``has_async_copy=True``; the MSL lowerer
  has no ``AsyncCopyOp`` visitor (Metal-targeted kernels don't
  emit the op). SPIR-V will be the first backend to actually run
  the expansion path. Validate the rewrite on synthetic IR first
  — the in-tree golden tests pin its shape, not its end-to-end
  correctness.
* The ``SubgroupReduceOp`` butterfly legalization fires on
  current CUDA (``acb29d5``); the SPIR-V native-keep path is a
  one-line ``OpGroupNonUniformAdd`` emit. The visitor is trivial;
  the work in §3.2 is the surrounding ops.

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

**Current status: §3.0 scaffolding landed (`e12b08e`); §3.1–3.7
need Intel Arc / Xe iGPU hardware and a chosen runtime.** The
scaffolding reserves `DeviceFamily.INTEL_GPU` in the enum and
registers a `SpirVLowerer` stub that raises `NotImplementedError`
on `lower_module`. Every phase-1/2 refactor that assumes
`INTEL_GPU` exists (legalization pass, caps-split, lowerer
registry) is self-consistent.

The original plan stopped at §3.5 ("rollout"). The revision
expands to §3.7 to cover the two surfaces the MLX removal made
visible: a Vulkan-side host-runtime C extension that mirrors
``drivers/_metal_dispatch`` (§3.1), and a third ``Engine``
subclass that owns Intel-side host I/O (§3.4). Without those,
the lowerer compiles SPIR-V into the void — there is no caller.

### Sub-phase map

| § | Work | Status | Blocker |
|---|---|---|---|
| 3.0 | Scaffolding (registry slot, stub lowerer, skeleton tests) | ✅ landed `e12b08e` | — |
| 3.1 | Runtime choice + ``drivers/_spv_dispatch`` C ext + ``drivers/spv.py`` Python wrapper + ``DeviceCaps`` probe + compile + launch | ✅ landed `a585e1e` (probe + EngineIntel) + `54449a1` (compile/launch) | command-buffer accumulation perf optim deferred |
| 3.2 | ``SpirVLowerer`` body — ~45 visitors + ``LoweredSpirVKernel`` artifact | 🟡 first cut landed (this commit) — ``vec_add``-class kernels round-trip end-to-end through Vulkan; remaining ~40 visitors expand incrementally | each new visitor is one entry in ``_DISPATCH`` |
| 3.3 | ``FragApplyOp`` lowering (coord-free body via ``OpCooperativeMatrixLengthKHR``) | ⏳ | 3.2 |
| 3.4 | ``EngineIntel`` subclass — host tensor format, VAE choice, lazy-dispatch idiom | 🟡 stub landed `a585e1e` (constructs + caps); inference paths blocked on 3.2 + OpenVINO TAEHV | 3.2 + VAE backend |
| 3.5 | Subgroup width policy (``reqd_sub_group_size`` pin vs driver-chosen + per-width variants) | ⏳ | 3.1 (probe data — captured) |
| 3.6 | NAX-equivalent path for flash-attention (hand-written ``OpCooperativeMatrix*KHR`` outside the framework) | ⏳ | 3.4 + microbench data |
| 3.7 | Rollout v1 → v3 across kernel cohorts | ⏳ | hardware + calendar |

### §3.0 Scaffolding (landed)

**File:** `src/quark/lower/spv/{__init__,lower}.py`,
`tests/lower/test_spv_skeleton.py`.

- ``DeviceFamily.INTEL_GPU = "intel_gpu"`` reserved in
  ``device.py:45``.
- ``SpirVLowerer`` + ``LoweredSpirVKernel`` stubs raise
  ``NotImplementedError(match="PORTABILITY_PLAN.md")``.
- Factory registered at package import; ``get_lowerer(INTEL_GPU,
  caps)`` returns the stub instead of raising ``KeyError``.
- 5 skeleton tests pin the surface (registry slot, error message
  shape, factory return type).

**Followup landed during §3.1 prep:**

- ``ChipGeneration.INTEL_XE_LPG`` / ``INTEL_XE2`` / ``INTEL_XE3``
  reserved in ``device.py``; ``is_intel_gpu`` property + a
  ``chip_gen_from_intel_info`` mapper that infers the gen from the
  Vulkan ``deviceID`` upper nibble (with name-string fallback for
  early dev-kit firmware).
- ``MmaConfig`` extended with ``min_intel_gpu_gen`` + ``intel_gpu``
  fields; four shapes registered for Battlemage (Xe2+) coopmat:
  ``m8n16k16_intel_{bf16,f16}_{bf16,f16,f32}``.
- ``shapes_for_chip(INTEL_XE3) ==  shapes_for_chip(INTEL_XE2) ==
  {those four}``; Xe-LPG (Meteor Lake) currently empty until we
  re-probe on real hardware.
- Test coverage: 6 new cases in ``tests/launcher/test_device.py``
  (chip-gen mapping, payload routing, cross-family negative).

Everything below is the real engineering work.

### §3.1 Runtime choice + driver + device probe

**Decision needed first.** The original plan kept "Vulkan vs
Level Zero vs pyopencl" open. Pick before writing code:

| Option | Verdict | Reasoning |
|---|---|---|
| **Vulkan + ``VK_KHR_cooperative_matrix``** | **Pick this.** | Cross-vendor (NV / AMD / Intel; Apple via MoltenVK in principle, though MoltenVK lacks cooperative_matrix today). KHR cooperative_matrix shipped Q3 2023 and ships in Mesa ANV + Intel proprietary. First-class subgroup ops, mature loader (``volk``). Locks SPIR-V to nothing IHV-specific — the same path that admits AMD RDNA later. |
| Level Zero + zeKernel | Skip for now. | Intel-only; lower dispatch latency than Vulkan but the SPIR-V→ELF path is less documented and ``ze_experimental_oneapi_subgroup_extensions`` is moving. Reconsider as a side-channel optimisation if Vulkan dispatch shows in profiles. |
| pyopencl + SPIR-V intermediate | No. | OpenCL on Intel iGPUs is being deprecated upstream (Intel NEO is moving to Level Zero). cooperative_matrix isn't exposed via cl_intel extensions on the path we'd target. |

**Files (new):**

- ``src/quark/drivers/_spv_dispatch/`` — nanobind C++ extension.
  Mirror ``drivers/_metal_dispatch``'s structure: ``module.cpp``
  declares the Python bindings, ``runtime.{cpp,h}`` owns the
  ``VkInstance`` / ``VkPhysicalDevice`` / ``VkDevice`` /
  ``VkQueue`` / command-pool singletons. Use ``volk`` for the
  Vulkan loader (header-only; doesn't need the LunarG SDK on the
  build host).
- ``src/quark/drivers/spv.py`` — Python wrapper. Mirrors
  ``drivers/metal.py`` (488L) at the interface level:
  ``SpvCompiledModule`` dataclass holds ``pipeline``,
  ``pipeline_layout``, ``descriptor_set_layout``, ``smem_bytes``;
  ``SpvDriver.compile(spirv_blob, entry, smem_bytes)`` runs
  ``vkCreateShaderModule`` + ``vkCreateComputePipelines``;
  ``SpvDriver.launch(...)`` records into a
  ``VkCommandBuffer`` and submits at the same trigger points
  the Metal driver uses (sync, output read, ops threshold).
- ``src/quark/runtime/spv.py`` — runtime helpers analogous to
  ``runtime/cuda.py`` and ``runtime/metal.py`` (queue probing,
  device enumeration, host-pointer pinning via
  ``VK_EXT_host_image_copy`` or BAR-mapped buffers).

**Caps populated from Vulkan:**

| Cap | Source | Captured (Battlemage / Xe3 PTL, Mesa 26.0.3) |
|---|---|---|
| ``subgroup_width`` | ``VkPhysicalDeviceSubgroupProperties.subgroupSize`` | **32** (matches Apple + NV — single-variant SIMD32 pin works) |
| MMA shapes | ``VkPhysicalDeviceCooperativeMatrixPropertiesKHR`` | 6 shapes, all subgroup scope: bf16/f16 m=8 n=16 k=16; s8/u8 m=8 n=16 k=32 |
| ``has_async_copy`` | (no Vulkan equivalent of ``cp.async``) | ``False`` — legalisation expansion path |
| ``has_fma_bf16x2`` | (legalize to scalar fma + bitcast) | ``False`` — already shipped |
| ``has_atomic_add_bf16x2`` | (legalize to two scalar atomic adds) | ⚠ ``False``, **and** ``shaderBufferFloat16AtomicAdd = NO`` — the scalar fallback fails too. See "newly-confirmed risks" below. |
| ``has_atomic_add_f32_smem`` | (used by some reduce kernels) | ⚠ ``False`` on Battlemage — smem→buffer escape needed for f32 atomic reductions, mark v2 |
| ``has_native_subgroup_reduce`` | ``OpGroupNonUniformAdd`` | ``True`` — single-op emit |
| ``maxPushConstantsSize`` | ``VkPhysicalDeviceLimits`` | **256 bytes** — narrower than CUDA's 4096; param packs > 256B spill to UBO |
| ``shaderBFloat16Type`` / ``CooperativeMatrix`` | ``VkPhysicalDeviceShaderBfloat16FeaturesKHR`` | **yes / yes** — bf16 path is fully native |

Captured via ``scripts/spirv/probe_coopmat.c`` on the Intel
Panther Lake dev-kit (`xe3-devbox`). Full output persisted at
``scripts/spirv/probe_output.battlemage_ptl.txt`` and digested in
``scripts/spirv/probe_coopmat.md``.

**Pre-3.2 cooperative-matrix shape gap — resolved.** Intel exposes
``m=8, n=16, k=16`` (vs Apple NAX m=16, n=32, k=16; PTX m=16, n=8,
k=16). The bf16 input path is intact — both ``bf16/bf16 → bf16``
and ``bf16/bf16 → f32`` ship. Four MmaConfig entries registered
in ``ir/mma_registry``:

* ``m8n16k16_intel_bf16_f32`` (the standard "compute_dtype=bf16,
  acc=f32" path)
* ``m8n16k16_intel_bf16_bf16`` (bf16 acc — opt-in for memory-bound
  kernels)
* ``m8n16k16_intel_f16_f32``
* ``m8n16k16_intel_f16_f16``

Tile multiples on Intel: ``BM % 8 == 0``, ``BN % 16 == 0``,
``BK % 16 == 0`` (vs Apple ``BM % 16, BN % 32``). The autotune
cache will not hit on first run from the Apple-tuned configs —
fresh autotune campaign on Intel hardware needed in §3.7 v1.

**Tests:** ``tests/drivers/test_spv_probe.py`` (skipped on
non-Intel CI). On Intel CI: probe success, caps shape, plus a
"compile a 5-line ``OpAdd`` kernel and dispatch over numpy
inputs" smoke. ``tests/drivers/_spv_dispatch/`` for the C
extension's own unit tests (mirror the metal-cpp side).

### §3.2 ``SpirVLowerer`` body

**Files (new):** ``src/quark/lower/spv/lower.py`` (replaces the
stub), ``visitors.py``, ``types.py``, ``mma_emit.py``.

Mirror ``MslLowerer``'s shape: a ``_SpvCtx`` dataclass per
function (lines list, ``names: NameAlloc``, smem-byte
accumulator, op stack, ``frag_values`` map, ``uses_coopmat``
flag); a ``_DISPATCH`` table from ``Op`` type to visitor
function; ``lower_function(fn, module)`` walks the ops and
emits SPIR-V text (or directly to a ``SpvBuilder`` that
produces the binary blob — see below).

**SPIR-V emission shape.** Two options:

- **Text→assembler.** Emit human-readable SPIR-V assembly,
  shell out to ``spirv-as`` (Khronos tools) at compile time.
  Easier to debug; the binary is also readable as text via
  ``spirv-dis``. Adds an external tool dependency.
- **Direct binary.** Build the SPIR-V binary in-process via a
  small Python ``SpvBuilder`` (or via the C extension if
  performance matters). No external tool; harder to debug
  early on.

Recommend **text→assembler** for the prototype, swap to direct
binary in v2 if compile latency hurts.

**Visitor inventory (from current op surface, ~45 entries):**

Pre-expanded by Phase 2 — visitor is trivial pass-through or
keep-path:

* ``AsyncCopyOp`` / ``AsyncCopyCommitOp`` / ``AsyncCopyWaitOp``
  → never reach the lowerer (legalize expands to vec
  load/store).
* ``ArithOp(kind="fma_bf16x2")`` → never reaches lowerer.
* ``ArithOp(kind="cvt_rn_bf16x2_f32")`` → never reaches lowerer.
* ``AtomicRmwOp(bf16x2)`` → legalize splits to two scalar
  atomic adds; visitor emits ``OpAtomicFAdd`` (requires
  ``VK_EXT_shader_atomic_float`` for f32, ``VK_EXT_shader_
  atomic_float2`` for f16/bf16 — the spike must confirm
  availability).
* ``SubgroupReduceOp`` (native-keep path) → single
  ``OpGroupNonUniformAdd`` / ``OpGroupNonUniformFMin`` /
  ``OpGroupNonUniformFMax`` per call.

Standard scalar / vector ops — direct mapping:

* ``ArithOp`` (add/sub/mul/div/min/max/and/or/xor/shl/shr/...)
  → ``OpFAdd`` / ``OpFSub`` / ``OpIAdd`` / ... per dtype.
* ``CmpOp`` → ``OpFOrd*`` / ``OpFUnord*`` / ``OpI*``.
* ``SelectOp`` → ``OpSelect``.
* ``ConvertOp`` → ``OpFConvert`` / ``OpUConvert`` /
  ``OpConvertFToU`` / ``OpConvertUToF`` etc., dtype matrix.
* ``BitcastOp`` → ``OpBitcast``.
* ``MathOp`` (cos/sin/exp2/log2/sqrt/rsqrt) → ``OpExtInst
  GLSL.std.450 Cos/Sin/Exp2/Log2/Sqrt/InverseSqrt``. Note: the
  ``ex2_approx`` precise/approximate split disappears — Vulkan
  has no separate "approximate" path; ``GLSL.std.450 Exp2`` is
  the only option, and the ULP guarantee is loose enough
  (4 ULP per Vulkan spec) that the existing precise paths
  expecting fp32 ULP-tight semantics will see noisier output.
  Validate against the rmsnorm / softmax cos_sim threshold
  during the §3.7 v1 rollout; bake an ``f32_exp2_strict``
  legalize-time expansion (range reduction + polynomial) if
  needed. Tracked.
* ``VecBuildOp`` / ``VecExtractOp`` / ``VecBuildPackedB32Op``
  → ``OpCompositeConstruct`` / ``OpCompositeExtract`` /
  ``OpBitcast`` of an ``OpVectorShuffle``.
* ``LoadOp`` / ``StoreOp`` (scalar gmem) → ``OpLoad`` /
  ``OpStore`` against the ``StorageBuffer`` storage class.
* ``VecLoadOp`` / ``VecStoreOp`` → vec load/store. Watch
  alignment: SPIR-V requires the load operand to carry
  ``Aligned`` decoration matching the source pointer's actual
  alignment, and the kernel-author IR doesn't carry alignment
  promises. Default to ``Aligned 4`` for ``b32`` carriers; emit
  ``Aligned 16`` only when the source ``GlobalTensor.stride``
  divides 16. Mismatch is a validation-layer error on debug
  drivers and silent corruption on release drivers.

Threadgroup memory:

* ``SmemAllocOp`` → ``OpVariable Workgroup`` of an
  ``OpTypeArray`` sized in elements. SPIR-V requires the array
  size to be a constant; quark's ``smem_layout`` pass already
  produces compile-time sizes.
* ``BarrierOp("block")`` → ``OpControlBarrier Workgroup
  Workgroup AcquireRelease | WorkgroupMemory``.

Thread / lane identity:

* ``thread_idx`` / ``block_idx`` / ``block_dim`` →
  ``GlobalInvocationId`` / ``WorkgroupId`` / ``WorkgroupSize``
  built-in inputs.
* ``lane_id`` → ``SubgroupLocalInvocationId``.
* ``subgroup_id`` → ``SubgroupId``.
* ``ShuffleOp(kind="bfly")`` → ``OpGroupNonUniformShuffleXor``
  (subgroup scope). The PTX butterfly mask becomes the
  ``XorMask`` operand.

Cooperative matrix (the new path):

* ``LoadMatrixOp`` →
  ``OpCooperativeMatrixLoadKHR <ptr>, <stride>, <layout>,
  MakePointerAvailable | NonPrivatePointer``. Layout argument
  picks ``RowMajor`` / ``ColumnMajor`` from the
  ``LoadMatrixOp.attrs["layout"]``. Stride is in elements.
* ``StoreMatrixOp`` → ``OpCooperativeMatrixStoreKHR``.
* ``MmaOp`` → ``OpCooperativeMatrixMulAddKHR <a>, <b>, <c>``;
  optional ``MatrixASignedComponentsKHR`` /
  ``MatrixBSignedComponentsKHR`` operands for fp8 / int8
  signedness disambiguation if Intel reports those KType
  variants.
* ``StoreMatrixGateResidualOp`` (the bf16 epilogue used by
  ``GemmKernel._build_metal_nax_*``) → no direct SPIR-V
  analogue. Phase 1 considered legalizing this but the
  rewrite would break the per-fragment lane layout the
  cooperative-matrix path relies on. Instead: extract
  fragments to scalars via the ``OpCooperativeMatrixLength
  KHR`` element loop, fold the gate+residual scalar math, and
  store back. Treat as a §3.3 case (coord-free FragApply over
  the accumulator); reject the kernel if the gate broadcast
  reads ``row_var``.

Fragment ops — see §3.3:

* ``FragApplyOp`` → ``OpCooperativeMatrixLengthKHR`` element
  loop, gated on coord-free body.
* ``FragForEachOp`` / ``FragReduceOp`` → reject in
  ``is_valid_for(caps)`` for v1.
* ``FragConvertOp`` → emit a fresh cooperative matrix of the
  target dtype with ``OpCooperativeMatrixLengthKHR`` element
  copy + ``OpFConvert`` per element. The lane→element mapping
  is driver-private but the operation is per-element so it
  doesn't need to be coherent with anything.

Control flow:

* ``ForLoopOp`` → ``OpLoopMerge`` + ``OpBranchConditional`` on
  the bound check; carries handled via ``OpPhi`` at the loop
  header.
* ``IfRegionOp`` → ``OpSelectionMerge`` + ``OpBranchConditional``.
* ``WhileLoopOp`` → ``OpLoopMerge`` + back-edge.

The control-flow visitors are the part most likely to surface
``smem_layout``-pass assumptions baked into MSL. Specifically,
MSL's ``threadgroup`` decls live at function-entry scope; the
SPIR-V ``Workgroup`` storage-class variables also have to be
function-scoped (SPIR-V validation rejects them inside
structured control flow). The hoist already happens for MSL
(see ``OVERVIEW.md``); confirm it's idempotent under SPIR-V
emission.

**Artifact:** ``LoweredSpirVKernel`` carries ``binary: bytes``
(SPIR-V words), ``entry: str`` (the entry point name —
``main`` by convention), ``smem_bytes: int``, ``binding_layout:
list[BindingSlot]`` (the descriptor-set / binding indices for
each tensor parameter). Mirrors ``LoweredMslKernel`` shape.

**Tests:** ``tests/lower/spv/test_*_visitor.py``, one file per
visitor cluster. Two layers:

1. *IR-to-SPIR-V text* goldens — pin the assembly produced for
   a small synthetic IR. CI runs without hardware.
2. *Compile + dispatch + numerical compare* — gated on Intel
   CI. Compile via ``spirv-as`` or the in-process builder,
   dispatch via the §3.1 driver, compare against the PTX or
   numpy reference at the CORRECTNESS_THRESHOLD per-kernel.

### §3.3 ``FragApplyOp`` lowering — the hard case

``FragApply`` lets kernels transform accumulator elements
between matmuls (FP8 dequant scale, bias add, silu fused into
GEMM epilogue). It's used by the MLP epilogue and by some GEMM
variants. The Phase 2 plan covered the basic case; the post-MLX
state adds two complications:

1. **The Apple side already has a working FragApply** that the
   MSL lowerer handles via per-lane scalar arithmetic over the
   NAX cooperative-tensor layout. That lane→element mapping is
   exposed because Apple's NAX is documented; SPIR-V's KHR
   cooperative matrix mapping is **driver-private**, so the
   "loop over per-lane registers" model the MSL lowerer uses
   doesn't translate. Ask via ``OpCooperativeMatrixLengthKHR``
   for the lane's element count and index by integer; never
   inspect the (row, col) coord.

2. **NAX's hand-written attention** (``kernels/owl_attn/nax.py``)
   reaches into the per-fragment layout directly to fuse the
   inline-Q-RoPE (``_emit_q_rope_prepass``). The same fusion in
   SPIR-V cannot use ``OpCooperativeMatrixLengthKHR`` element
   indexing because the rotation needs (row, col) for the freq
   formula. v1 doesn't ship inline-Q-RoPE on SPIR-V at all (see
   §3.6); FragApply only handles the coord-free cases.

**V1 strategy (unchanged from original plan):**

```
loop i in 0..OpCooperativeMatrixLengthKHR(C):
    e := OpCompositeExtract C, i        # implementation-defined order
    e' := apply body(e, scalar_selectors)
    C  := OpCompositeInsert e', C, i
```

The body must be a pure scalar computation over ``e`` plus
scalar selectors (``scale_log2e``, ``rcp_sum``, etc.). It must
not reference ``row_var`` or ``col_var`` from the surrounding
``BlockContext``.

**``is_valid_for`` analysis hook.** Walk
``FragApplyOp.body`` at kernel-build time; if any op references
a ``BlockContext``-rooted Value that traces back to a coord
read, mark the kernel as ``valid=False`` for SPIR-V. The walk
is target-agnostic and lives in
``src/quark/blocks/coord_analysis.py`` (new); it's also useful
for AMD HIP later. Plus a ``Spec.compatible_with(caps)`` shim
on every kernel using ``FragApply`` to surface a clear
diagnostic instead of a kernel-launch failure.

### §3.4 ``EngineIntel`` subclass — the new requirement

The original plan stopped at "lowerer + driver + caps" because
``Engine`` didn't exist. ``Engine`` now does, and Biome consumes
it as the public surface. SPIR-V isn't usable until there's an
``EngineIntel`` mirror.

**File:** new ``src/quark/engine/intel.py`` (alongside
``cuda.py`` and ``metal.py``).

Things to decide explicitly:

| Question | EngineCUDA | EngineMetal | EngineIntel |
|---|---|---|---|
| Host tensor format on the kernel boundary | torch (with ``.data_ptr()``) | numpy + tagged-dtype carrier | **Pick:** start with numpy + tagged carrier (matches Metal; works without torch on Linux); upgrade to torch-on-XPU later if perf needs it |
| Lazy / graph-capture idiom | CUDA Graphs (``GenerateFrame``) | ``quark.lazy()`` → ``MTLCommandBuffer`` accumulation, ``sync=False`` for overlap | **Pick:** ``quark.lazy()`` semantics → ``VkCommandBuffer`` accumulation; same trigger points (output read / explicit sync / op count threshold). The ``sync=False`` overlap requires a second compute queue + a ``VkSemaphore`` between commit and the next denoise; defer unless the §3.7 v1 perf number demands it |
| VAE backend | torch + custom CUDA kernels (or torch-native) | CoreML on the Apple Neural Engine via ``quark.taehv`` | **Open question.** Three candidates: <br>1. **OpenVINO** TAEHV export — Intel-native, runs on the GPU EU or NPU on Meteor Lake+. <br>2. **torch-on-XPU** — same TAEHV state-explicit decoder we already export, target ``intel_extension_for_pytorch`` or upstream torch XPU. <br>3. **Same SPIR-V kernels as the DiT** — write a TAEHV port in the kernel framework. Most work; least vendor lock-in. Recommend **OpenVINO** for v1 (matches the "vendor-native VAE" pattern of CoreML on Apple) |
| Weight pinning / parameter-on-device idiom | ``torch.nn.Module.to(cuda)`` + ``torch.cuda.graph()`` | ``_pin_params_to_device`` wraps numpy weights as ``QuarkTensor``s with persistent Metal buffer storage | Mirror Metal: pin numpy carriers as ``QuarkTensor``s with persistent Vulkan buffers (``VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT``) at startup |

**Shared API:** the ``Engine`` base class methods
(``gen_frame``, ``append_frame``, ``submit_frame``,
``next_pixels``, ``flush_pixels``, ``reset``, ``get_state``,
``load_state``, ``set_prompt``) all bind to subclass
implementations. ``EngineIntel`` follows the subclass pattern
``EngineMetal`` already established — no new public surface.

**``Engine.__new__`` factory dispatch.** Currently:

```python
if cls is Engine:
    if _IS_METAL:
        cls = EngineMetal
    else:
        cls = EngineCUDA
return object.__new__(cls)
```

This collapses "not Metal" into "CUDA", which fails on a Linux
box with an Intel GPU and no NVIDIA card. Pre-3.4: extend the
dispatch to probe device families:

```python
if cls is Engine:
    family = _detect_default_family()  # CUDA → METAL → INTEL_GPU → ROCM → CPU
    cls = {
        DeviceFamily.CUDA: EngineCUDA,
        DeviceFamily.METAL: EngineMetal,
        DeviceFamily.INTEL_GPU: EngineIntel,
    }[family]
return object.__new__(cls)
```

The ``_detect_default_family`` probe lives in ``device.py`` and
short-circuits in priority order: CUDA driver loadable → Metal
on darwin → Vulkan device with ``VK_KHR_cooperative_matrix`` →
fall through. Allow override via env (``QUARK_FORCE_FAMILY=
intel_gpu``).

**Tests:** ``tests/engine/test_intel.py`` skipped on
non-Vulkan, mirroring the ``test_metal.py`` shape. End-to-end
``Engine(model_uri, ...).gen_frame()`` smoke gated on Intel CI.

### §3.5 Subgroup width policy

Three options for v1, in ascending complexity:

1. **Pin SIMD32 via ``reqd_sub_group_size(32)``** for every
   kernel. Single variant per kernel; simplest pipeline.
   Optimal on Arc Battlemage (XMX SIMD16 with `m16n16` coopmat);
   suboptimal on Xe-LP iGPU (native SIMD8) and some shapes on
   Alchemist where SIMD16 wins. Some drivers refuse SIMD32 on
   high-register-pressure kernels.
2. **Driver-chosen, single-variant.** Ship one kernel; let
   Vulkan pick the subgroup width. Loses the ``caps.subgroup_
   width`` plumbing's optimisation surface — every shuffle
   reduction has to legalize at runtime, every kernel that
   asserted ``s.D % subgroup_width == 0`` validates against
   the wrong number.
3. **Per-width variants.** Build the kernel three times (W=8,
   16, 32), pick at dispatch time based on the device's
   reported preferred subgroup size. Most work; matches what
   modern shader compilers ship for vendor variance.

**v1 picks (1).** Pin SIMD32 with
``OpExecutionModeId LocalSize 32 1 1`` plus
``RequiredSubgroupSize 32`` (from
``VK_EXT_subgroup_size_control``). On rejection (driver returns
``VK_ERROR_*`` for a specific kernel), fall through to (2) for
that kernel only — i.e., re-emit without the ``RequiredSub
groupSize`` decoration and let the driver pick. Track the
fallback in caps so consumers (autotune cache) know.

The "per-width variants" path becomes v3 territory — only
worth it after the v1 perf number proves SIMD32 leaves enough
on the table to justify the build-time + cache-key cost.

### §3.6 NAX-equivalent flash-attention on SPIR-V

The plan said v1 ships without flash-attention on SPIR-V. With
NAX in tree, that decision is reinforced — Apple's
production attention path is hand-written, not framework-emitted,
because the framework's ``FragForEachOp`` / per-fragment-
coordinate model can't fuse the inline-Q-RoPE pre-pass into the
GEMM1 body without lane-layout-specific code.

**v1 (no flash-attention on SPIR-V):**

* Reject every kernel that uses ``FragReduceOp`` or
  ``FragForEachOp`` in ``is_valid_for(caps)`` for INTEL_GPU.
* The WaypointS ``owl_attn`` kernel — the autotuned framework
  variant in ``kernels/owl_attn/kernel.py`` — uses
  ``FragReduceOp`` for online softmax and is rejected. Today
  Apple uses NAX (``kernels/owl_attn/nax.py``) instead and the
  framework variant only runs on CUDA. SPIR-V will pattern-match
  the CUDA path until v3.
* Concretely: a Waypoint inference run on Intel GPU at v1
  fails with a clear diagnostic ("attention not yet
  available on SPIR-V; see PORTABILITY_PLAN §3.6"). v1 is
  smoke + non-attention kernel coverage only.

**v3 (real flash-attention on SPIR-V):**

Two engineering shapes worth consideration:

1. **Hand-written outside-framework module** mirroring
   ``kernels/owl_attn/nax.py``'s structure — emit
   ``OpCooperativeMatrixMulAddKHR`` / ``Load`` / ``Store``
   directly into a SPIR-V module, work around the
   driver-private lane mapping by doing the inline-Q-RoPE pre-
   pass through ``OpCooperativeMatrixLengthKHR`` element
   indexing (slower than the per-lane fp32 RoPE but
   coordinate-free). Mirror the segment loop + online-softmax
   shape; cooperative-matrix scope is `Subgroup` so the
   per-row max/sum reductions go through
   ``OpGroupNonUniformAdd``.
2. **Restructured framework path** — make ``FragReduceOp`` /
   ``FragForEachOp`` lower to a coord-free idiom that emits an
   explicit per-element softmax via ``OpCooperative
   MatrixLengthKHR`` and an inter-element ``OpGroupNonUniform
   Add`` for the reduce. Slower than option (1) but reusable
   for AMD HIP later.

Pick at v3 time after a microbench head-to-head on real Intel
hardware; option (1) is the safe path for v3.0, option (2) is
the better long-term shape if perf is acceptable.

### §3.7 Rollout

Adjusted from the original plan to reflect the kernel-cohort
gating from §3.6:

* **v1** — smoke pass on Intel Arc A770/A380 + one Xe iGPU
  (Arc Graphics in Meteor Lake or Lunar Lake). Kernel coverage:
  the elementwise / normalisation set (``silu``, ``ada_rmsnorm``,
  ``rmsnorm``, ``head_rmsnorm``, ``ada_gate_residual``,
  ``value_residual_packed``), the ``GemmKernel`` at one autotuned
  shape, ``patchify`` / ``unpatchify``. Plus the §3.4
  ``EngineIntel`` wired up enough for ``Engine(model_uri,
  load_weights=False)`` to construct without exception (no
  inference). Attention rejected.
* **v2** — full ``GemmKernel`` autotune space lit on Intel
  hardware, ``KVCacheUpdateKernel``, ``MoE*`` kernels. Real
  inference still gated on attention.
* **v3** — flash-attention path (§3.6 option 1 or 2). Real
  inference at the same LFPS thresholds as the Apple/CUDA
  benches.
* **v3.1** — TAEHV VAE on Intel via OpenVINO (or whichever
  §3.4 picks). Closes the loop; ``EngineIntel.gen_frame``
  produces real pixels.

---

## Risk / open questions (revised)

* ~~**Cooperative-matrix shape gap (highest priority).**~~
  **Resolved.** Battlemage exposes m=8/n=16/k=16 (bf16 +
  f16, both with f32 and same-dtype acc). Four shapes
  registered in ``ir/mma_registry``. Tile multiples shift to
  ``BM % 8, BN % 16, BK % 16``; the existing Apple/PTX
  autotune cache won't hit on first run, fresh autotune
  campaign on Intel hardware scheduled for §3.7 v1. Captured
  data + the four MmaConfig entries are in tree as of
  ``spirv-integration``.
* ⚠ **f16 / bf16 atomic-add unavailable on Battlemage.**
  Concrete confirmation of the prior "atomic float
  availability" risk: ``shaderBufferFloat16AtomicAdd = NO``
  AND ``shaderShared… = NO`` on Mesa 26.0.3 / Panther Lake.
  The Phase 2 ``bf16x2 → 2× scalar f16 atomic add``
  legalization **does not run on Intel** — the scalar f16
  atomic add is also unavailable. Two paths for §3.2:
  1. New legalization tier: scatter to f32 partials buffer +
     reduce kernel at sync points. Extra dispatch + extra
     device memory; acceptable for the rare split-k bf16
     path. Track as a §3.2-blocker for kernels with
     ``split_k > 1`` on bf16.
  2. Reject those kernels on Intel via ``is_valid_for(caps)``
     for v1; restore via path (1) in v2.
  Recommend (2) for v1 — split-k is rare on the Waypoint
  hot-shape set (the autotune cache picks split_k=1 for
  every shape on M5 Max).
* ⚠ **f32 smem atomic add also unavailable** on Battlemage
  (``shaderSharedFloat32AtomicAdd = NO``). Less urgent than
  bf16 — the kernels using f32 smem atomic are TBD-future
  reductions, none today. Mark v2.
* ⚠ **Push-constant cap is 256 bytes** on Battlemage (vs
  CUDA's 4096, Metal's effectively-unlimited setBytes).
  Quark's typical kernel-param pack is well under 256, but
  audit each kernel's ``param_struct_size`` during the §3.1
  driver impl. Anything over 256 spills to a UBO + extra
  descriptor — costs one more binding slot, no perf hit on
  steady-state.
* **``OpExtInst GLSL.std.450`` ULP slack.** Vulkan's
  transcendentals carry 4 ULP of slack vs PTX's
  ``ex2.approx.f32``. The rmsnorm / softmax cos_sim threshold
  on the v1 kernels may regress. Validate during §3.7 v1; fall
  back to a polynomial-range-reduction emit in legalize if
  needed. Track as a §3.7-blocker, not a §3.2-blocker.
* **``RequiredSubgroupSize`` rejection** under high register
  pressure → §3.5 fall-through to driver-chosen for that
  kernel only. Track in caps; surface in autotune cache key
  so a kernel that needed the fallback doesn't get cached
  results from a "pinned" run.
* **Driver maturity**. Intel's Linux drivers (Mesa ANV / Intel
  proprietary) have known reliability issues on non-standard
  workloads. Budget ≥1 week of driver-bug chasing during §3.2;
  the ``LoadMatrixOp`` / ``StoreMatrixOp`` /
  ``MmaOp`` cluster is the most likely epicentre.
* **VAE backend choice in §3.4** (OpenVINO vs torch-on-XPU vs
  same-SPIR-V) is unblocked by everything else but should not
  be deferred — the v1 ``EngineIntel`` smoke needs *some*
  decoder path, even if it's "decode on CPU as a placeholder".
  Land OpenVINO TAEHV export as a parallel workstream during
  §3.1 / §3.2.
* **``Engine.__new__`` dispatch refactor.** The current Apple-
  vs-CUDA branch is binary; extending it for Intel before
  §3.4 lands risks misdispatching on the existing CUDA / Metal
  paths. Land the refactor under a feature flag (``QUARK_
  ENGINE_PROBE_FAMILY=1``) with default-off behaviour
  preserved, switch the default once §3.4 is stable.
* **Async-copy legalization byte-equivalence on MSL.** Already
  in the original plan; restated because it's still un-validated
  and the SPIR-V ``AsyncCopyOp`` rewrite path will be the first
  production user. Validate the rewrite produces byte-identical
  output to the existing MSL inline path on a pipelined kernel
  before relying on it for SPIR-V.

## Effort estimate (revised)

The 1-month original estimate undersold §3.4 (didn't exist) and
oversold §3.2 (Phase 2 cleared more than expected). Calendar is
fresh from 2026-05-07.

| Sub-phase | Work | Calendar |
|---|---|---|
| 3.1 Runtime decision + cooperative-matrix shape spike + skeleton ``_spv_dispatch`` | 2 days | 1–2 |
| 3.1 ``SpvDriver`` + caps probe + smoke (compile + dispatch ``OpAdd``) | 3 days | 3–6 |
| 3.2 Lowerer scaffold + ``_SpvCtx`` + builder/assembler decision + 8 simplest visitors | 3 days | 7–10 |
| 3.2 Cooperative matrix path (``LoadMatrixOp`` / ``StoreMatrixOp`` / ``MmaOp``) on the spiked shape | 3 days | 11–13 |
| 3.2 Remaining visitors (control flow, smem, lane id, reduces, math intrinsics) | 4 days | 14–18 |
| 3.3 ``FragApplyOp`` lowering + ``coord_analysis`` walk + ``is_valid_for`` hook | 2 days | 19–20 |
| 3.4 ``EngineIntel`` skeleton + ``Engine.__new__`` dispatch refactor | 2 days | 21–22 |
| 3.4 OpenVINO TAEHV export (parallel; can run during 3.1–3.3 by another engineer) | 4 days | parallel |
| 3.5 ``RequiredSubgroupSize`` policy + autotune-cache integration | 1 day | 23 |
| 3.6 v1 attention rejection + diagnostic (no flash-attention work yet) | 0.5 day | 24 |
| 3.7 v1 rollout — kernel cohort, golden tests, smoke runs | 5 days | 25–29 |
| 3.7 v2 rollout — full GEMM autotune, KV cache, MoE | 5 days | 30–34 |
| 3.7 v3 rollout — flash-attention port (option 1: hand-written) | 7 days | 35–41 |

**SPIR-V v1 (no inference; smoke + non-attention kernels):
~5 weeks.** Add 2 weeks to v2, +1.5 weeks to v3 if option 1 is
picked. Plan for 1.5× on each chunk; the §3.1 spike is the
single biggest schedule-risk item — if cooperative-matrix
shape gap is structural, the Phase 1.4 mnemonic table grows
by another 1–2 weeks of autotune work.

Single-engineer estimate. Two engineers: §3.4 (OpenVINO export)
runs in parallel from day 1; saves ~4 days off the v3.1 wall.

## Sequencing constraint (revised)

* **§3.1 spike (cooperative-matrix shape probe) before any
  lowerer code.** The shape gap drives whether §3.2 is a port
  of the existing autotune cache or a parallel autotune
  campaign on Intel.
* **§3.4 ``Engine.__new__`` dispatch refactor before any v1
  smoke.** Currently the ``Engine`` factory will pick
  ``EngineCUDA`` on a Linux-Intel box and crash; the dispatch
  fix is small but blocks every consumer-facing test.
* **Phase 1.2 + 1.4 already landed.** No new sequencing
  constraint from Phase 1.

## Out of scope for this plan (revised)

* HIP backend for AMD RDNA / CDNA. The §3.1 / §3.2 work makes
  HIP strictly easier (the cooperative-matrix path generalises;
  the legalization pass already handles RDNA's missing
  primitives), but the actual port is its own plan.
* WebGPU / WGSL backend. WebGPU's subgroup extension shipped
  later than ``VK_KHR_cooperative_matrix`` and is still
  driver-variable; revisit when subgroup ops + matrix ops
  ship in WebGPU 1.1.
* CPU AMX backend. Different programming model entirely; would
  need its own plan.
* **TAEHV port to SPIR-V kernels.** The v3.1 plan defaults to
  OpenVINO. Writing TAEHV in the kernel framework is appealing
  for vendor independence but has poor cost / benefit at v1 —
  the VAE is ~10% of the per-frame budget on Apple, and Intel-
  native OpenVINO will likely match that. Revisit if OpenVINO
  proves a deployment headache (extra runtime dependency,
  model-format conversion churn).
