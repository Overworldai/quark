# popcorn — consolidated brief

A guided tour of what popcorn is, how a kernel flows through it, and
what it takes to bring up a new backend. Read this first when getting
oriented.

## What popcorn is

popcorn is a GPU kernel compiler. Authors describe a kernel once as
typed SSA IR, the compiler lowers it to a target's native source form
(today: PTX for NVIDIA, MSL for Apple Metal), hands that text to the
vendor runtime for JIT compile, and the resulting GPU function plugs
back into the host framework (PyTorch on NVIDIA, MLX on Apple).

Four load-bearing ideas hold the system together:

1. **One IR, many targets.** Kernels contain no framework- or
   hardware-specific code. They emit typed SSA ops over
   backend-neutral tensor abstractions; the concrete target is picked
   up at import time from the host platform.
2. **A small authoring surface** — a free-function namespace — wraps
   the IR builder so kernel bodies read like ordinary imperative code
   (`add`, `mul`, `for_range`, `if_`, `barrier`). Higher-level
   composition primitives (tile-load, staged-pipeline, MMA-body,
   epilogue) sit above that.
3. **A single polymorphic tensor type** hides the host framework from
   everything that isn't a kernel body — reference implementations,
   correctness oracles, baselines, test harnesses. There is exactly
   one file in the tree that contains framework-specific branches.
4. **Kernels self-register.** Each kernel lives in its own folder with
   a canonical file layout; a class decorator registers it at import
   time and every tool (bench, fuzz, autotune, the framework-level
   entry points) discovers it through a shared registry.

## The IR in one page

- **SSA, strongly typed, structured control flow.** Every op
  produces fresh `Value`s; there is no mutation. Control flow is
  expressed with `ForLoopOp`, `IfRegionOp`, `WhileLoopOp` whose
  bodies are nested `Region`s. Labels and branches do not exist at
  the IR level — the lowerers emit those.
- **`DType`** covers `F32` `F16` `BF16` `E4M3` `E5M2` `U32` `S32`
  `U16` `S16` `U8` `S8`, the bit-bucket types `B8` `B16` `B32`
  (untyped carriers used for fragment packing), and `PRED` (1-bit
  predicate). `DType` is simultaneously a string tag, an IR type,
  and a bridge to the host framework's dtype.
- **Tensor abstractions** — backend-neutral:
  - `GlobalTensor` — a gmem parameter with shape, strides, dtype.
    Subscript sugar for scalar loads/stores, plus a `tile(...)` view
    that produces another `GlobalTensor` pointing at a rectangular
    region.
  - `SharedRegion` — a shared-memory / threadgroup allocation, typed,
    with an explicit `Lifetime` annotation (`KERNEL`, `AUTO`,
    `IN_REGION(R)`, `BEFORE_REGION(R)`, `AFTER_REGION(R)`). Fluent
    methods cover gmem→smem tile copies, per-warp and per-lane views,
    and stage slots for double-buffered pipelines.
  - `FragTensor` — a per-lane register carrier for one matrix-unit
    tile. Typically produced by a matrix-load op and consumed by the
    MMA op.
  - `RegisterTile` — a logical (rows × cols) view over one or more
    `FragTensor`s that hides the per-backend lane layout behind a
    tile-level API (`map`, `reduce_along_cols`, `convert`,
    `for_each`).
- **Op catalogue** — arithmetic and logic, `fma`, `min`/`max`,
  comparisons, `select`, casts (including packed casts for fp8),
  bitcasts, math intrinsics (both precise and approximate variants),
  vector-register builds / extracts, scalar and vector loads / stores,
  asynchronous copy (commit / wait), atomic read-modify-write,
  barriers, thread-identity reads, cross-lane shuffles, warp-wide
  reductions and broadcasts, matrix-unit load / store / MMA, and four
  fragment primitives that stay in register (`FragApply`,
  `FragReduce`, `FragConvert`, `FragForEach`).
- **Matrix-unit shapes** are registered as typed descriptors. Each
  descriptor carries the tile shape (M × N × K), the per-register
  `(row, col)` offsets, and hardware gates (minimum compute
  capability, minimum Apple chip generation). Device capabilities
  are populated at probe time from the filtered set of shapes the
  current device supports; the autotuner picks a per-site shape from
  that set.
- **Validation** runs before lowering and catches SSA dominance
  violations, region terminator mismatches, out-of-bounds constant
  indices, rank / index-type mismatches, and unregistered MMA shape
  references. It also emits perf warnings (bank conflicts, unpadded
  strides, unvectorized stores) that are off by default and gated by
  an environment flag.

## The authoring layer

The IR builder is plumbing; kernel bodies don't call it directly.
Instead they use a free-function namespace that reads the active
builder from a context variable:

```
pop.add(a, b)                  # a + b as an IR op
pop.for_range(0, n, 1)         # context manager; Python ints auto-lifted
pop.if_(pred)                  # context manager
pop.barrier("block")           # sync across a workgroup
pop.yield_(carry)              # terminator; auto-flattens a named carry
pop.store_acc(dst, acc, ...)   # tile-level epilogue with optional activation
```

A composition layer above that provides `Accumulators`, `SmemTile`,
`SmemPlan`, `Stage`, `Carry`, `MmaBody`, and `PipelineBody`. A typical
GEMM-shaped kernel opens with `s, c, g = self.spec, self.config,
self.g`; allocates an accumulator; asks for a double-buffered pair of
shared-memory tiles; hands the producer / consumer / carry to a
pipeline runner; and finishes with a tile-level epilogue. The kernel
body is roughly a dozen lines of orchestration sitting on top of the
IR primitives.

## The compilation pipeline

```
    @kernel-decorated class
          │
          │  kernel.emit()
          ▼
       ir.Module          SSA, typed, structured control flow
          │
          │  validate_module(...)
          ▼
   (correctness + perf warnings)
          │
   ┌──────┴──────┐
   ▼             ▼
 PTX            MSL              one lowerer per target
 lowerer        lowerer
   │             │
   │             │
   ▼             ▼
 LoweredKernel  LoweredMslKernel  source text + launch metadata
   │             │
   │             │
   ▼             ▼
 CUDA driver    Metal driver      vendor runtime owns JIT
   │             │
   ▼             ▼
 compiled fn    compiled fn
   └──────┬──────┘
          ▼
    launcher.launch(buffers=[...])
```

Dispatch is decided once at import time. Nothing in the user-facing
surface branches on the backend at runtime.

The **launcher** owns one device and one driver per process. It
compiles kernels on demand (caching on the triple `(kernel_cls, spec,
config)`) and dispatches launches through the driver. Its `compile`
method runs `kernel.emit()`, asks the target's lowerer to produce an
artifact, hands the artifact to the driver, and wraps the result in a
`CompiledKernel`. `CompiledKernel.launch(buffers=[...])` validates
dtype and contiguity against the parameter spec, extracts device
pointers (CUDA) or threads through typed array objects (Metal), and
issues the launch.

An **autotune cache** sits in front of `compile`: a three-level
lookup (in-memory → per-user on-disk → bundled defaults checked in to
the repo), with a bounded online search on a miss. `compile(spec)`
with no config consults the cache; explicit configs bypass it. The
search is backend-agnostic — it calls the same `launch` path and times
with the backend's native synchronization primitives.

## How lowerers work

A lowerer is a visitor pass that walks an `ir.Module` and produces
two things: the target's native source text, plus enough metadata for
the driver to launch it (kernel name, shared-memory byte count,
per-backend extras such as input / output parameter names or atomic
bindings). Both current lowerers share the same skeleton.

### The contract

Each lowerer exposes:

1. A **frozen artifact dataclass** — source text plus launch
   metadata. `str(artifact)` returns the source so callers that only
   want the text can treat it as a string.
2. A **lowerer class** with:
   - a constructor that takes target-specific knobs (e.g. the PTX
     lowerer takes an `sm` number and auto-selects a matching PTX
     version; the MSL lowerer takes device capabilities and a flag
     governing shared-memory aliasing);
   - a `lower_module(module)` entry point that validates the module,
     picks its single function, and returns the artifact;
   - a per-function `lower_function(fn, module)` that owns the walk.

### The walk

Per-function lowering allocates a scratch `FnCtx` — a name / register
allocator, a buffer for emitted source lines, bookkeeping dictionaries
keyed on `Value.id`, a stack of enclosing structured ops (for
`YieldOp` resolution), shared-memory tracking, and a back-reference to
the enclosing module for MMA shape lookups. The walk itself is a
tabular dispatch:

```
def visit(op, ctx):
    handler = DISPATCH[type(op)]
    handler(lowerer, op, ctx)
```

The dispatch table is the single source of truth for which IR ops a
backend supports. Missing an entry produces a clear
`NotImplementedError` at lowering time — never silent misbehaviour.

### Per-op handlers

Each handler reads `op.operands`, `op.attrs`, `op.results`, allocates
names / registers for the result values, and appends lines to the
output buffer. Representative patterns:

- **Arithmetic** — lookup tables map the IR op kind to target syntax
  (e.g. an `ArithOp` with `kind="add"` and an `F32` result becomes
  `add.f32` on PTX and `float x = a + b;` on MSL). Integer multiplies
  take the `.lo` form on PTX, FMAs take the rounding-mode form, and
  bitwise ops run in the bit-typed class regardless of the declared
  integer signedness.
- **Comparisons and predicates** — PTX returns a predicate register
  via `setp.*`; MSL returns a `bool`. `Select` over predicates is
  specifically rejected on PTX (no `selp.pred`) and callers are steered
  to `and_` / `or_` on the IR level.
- **Conversions** — scalar casts are straightforward except for fp8
  destinations. PTX exposes only the packed two-at-a-time fp8 cvt, so
  the lowerer rejects scalar-to-fp8 `ConvertOp` and routes those
  callers through `PackedConvertOp`.
- **Structured control flow** — a `ForLoopOp` becomes a label / branch
  pair on PTX and C-style braces on MSL. Loop-carried values are
  handled by copying each body-end value into the carried register
  just before the back-edge. An `IfRegionOp` lowers to a predicated
  branch around its `then_` / `else_` regions with carried-value
  coalescing identical in shape to the loop case.
- **Shared memory** — a backend-neutral pass runs first (see below),
  assigning every `SmemAllocOp` to a pool slot with an offset. The
  per-backend handler consults the plan and emits one physical
  declaration per slot. Disjoint-lifetime allocations alias
  automatically.
- **Matrix-unit and fragments** — `LoadMatrixOp` lowers to a synchronous
  ldmatrix variant on PTX and a simdgroup load on MSL. `MmaOp` keys on
  the registered shape and emits the matching `mma.sync` variant or
  `simdgroup_multiply_accumulate`. The fragment primitives
  (`FragApply`, `FragReduce`, `FragConvert`, `FragForEach`) walk over
  per-lane slots entirely in register — no shared-memory round-trip —
  and each primitive takes a short lambda that describes its
  per-element or per-reduction body.

### The shared-memory layout pass

A backend-neutral pre-pass runs ahead of any lowerer. It builds a
half-open op-index interval per `SmemAllocOp` from its `Lifetime`
annotation, greedy-colors the intervals onto physical slots, and
hands each lowerer a plan that maps alloc id → slot id → offset +
size. Two allocations whose lifetimes don't overlap share a slot;
each slot's size is the max of the allocations placed in it. The
lowerers never run their own offset counter; they consume the plan.
The coloring is deterministic given the IR, so repeated lowerings of
the same module produce byte-identical output.

### PTX-specific points

- The emitted artifact is a full `.visible .entry` module: header
  (`.version`, `.target`, `.address_size`), optional dynamic
  shared-memory declaration, a `.reg` block produced by the register
  allocator, the instruction stream, and `ret`.
- One virtual register per SSA value. The register class is picked
  from the dtype: `.b32` covers F32 / S32 / U32 / B32, `.pred` covers
  predicates, and `.b16` / `.b8` / `.b64` cover the rest.
- The PTX version is auto-selected from the target SM. Blackwell
  requires `≥ 8.7`, Hopper `≥ 8.0`, Ada `≥ 8.4` (for fp8), Ampere
  `≥ 7.0`.
- Packed conversions are the canonical path for fp8 and fp16x2-style
  sources. Byte ordering of the packed destination is fixed by the
  ISA and the lowerer matches it; callers pass `(lo, hi)` meaning
  "lo goes at the +0 byte, hi at the +1 byte".

### MSL-specific points

- The emitted artifact is a **body string**, not a full function.
  The Metal runtime wraps it with the `[[kernel]] void ...` prefix,
  parameter bindings, and built-in variable declarations.
- Every `threadgroup` declaration has to sit at function-entry
  scope — Metal rejects them inside nested scopes. The lowerer hoists
  each one to the top of the body at emission time. Constants are
  hoisted similarly to avoid C-scoping issues where a constant
  CSE'd across an outer and inner region gets trapped inside the
  inner braces.
- Atomics are limited to `atomic_float`, `atomic_uint32`,
  `atomic_int32` on all shipping Apple GPUs through the current
  generation. There are no packed or sub-4-byte atomics.

## Adding a new backend

The rest of this doc is about what it takes to stand up a lowerer for
a new target — a different accelerator architecture, a new IR / source
form (e.g. a portable intermediate representation like SPIR-V, or a
C++-style shading language), or an alternate runtime for an existing
architecture. The design is deliberately staged so the new target
reuses every backend-neutral piece already in the tree: the IR and
validator, the authoring surface, the matrix-shape registry and
autotuner, the shared-memory coloring pass, the launcher cache, the
correctness oracle, and the fuzz / bench harnesses. The target-specific
work is concentrated in two modules: a lowerer and a driver.

### 1. The artifact

Define a frozen dataclass that holds the source text plus exactly the
metadata your runtime needs at launch time:

```
@dataclass(frozen=True)
class LoweredXxxKernel:
    source: str
    kernel_name: str
    smem_bytes: int
    # plus whatever else your runtime needs: entry symbol name,
    # parameter-binding table, resource-slot map, atomic-output flags,
    # specialisation-constant values, template arguments, ...
```

Keep it frozen — it flows through a cache keyed on `(kernel, spec,
config)` and anything downstream should be immutable. Implement
`__str__` returning the source so `str(artifact)` gives back the text.

### 2. The per-function scratch state

```
@dataclass
class _XxxCtx:
    module: Module | None = None
    lines: list[str] = field(default_factory=list)
    names: NameAlloc = field(default_factory=NameAlloc)
    indent: int = 1
    smem_bytes: int = 0
    smem_allocs: dict[int, tuple[name, offset, size]] = field(default_factory=dict)
    op_stack: list[Op] = field(default_factory=list)
```

The fields that always belong here: a sink for emitted lines, an
identifier allocator keyed on `Value.id` (so two handlers touching the
same value produce the same name), per-region shared-memory
bookkeeping, and a stack of open structured ops so `YieldOp` can find
the enclosing `ForLoopOp` / `IfRegionOp` without needing back-pointers
on the op base class.

### 3. The lowerer class

```
class XxxLowerer:
    def __init__(self, device_caps=None, **target_knobs): ...

    def lower_module(self, module):
        validate_module(module)
        fn = module.functions[0]                # single-function kernels
        return self.lower_function(fn, module)

    def lower_function(self, fn, module):
        ctx = _XxxCtx(module=module)
        ctx.smem_plan = compute_smem_layout(fn, enable_aliasing=True)
        self._emit_param_decls(fn, ctx)
        self._emit_smem_allocs(fn, ctx)
        self._walk(fn.body.ops, ctx)
        return LoweredXxxKernel(
            source="\n".join(ctx.lines),
            kernel_name=fn.name,
            smem_bytes=ctx.smem_plan.total_bytes,
        )

    def _walk(self, ops, ctx):
        for op in ops:
            self._visit(op, ctx)

    def _visit(self, op, ctx):
        handler = _DISPATCH.get(type(op))
        if handler is None:
            raise NotImplementedError(f"XxxLowerer: no handler for {type(op).__name__}")
        handler(self, op, ctx)
```

Two points worth flagging. First: **always** run the shared-memory
layout pass — it's backend-neutral, gives you lifetime-based aliasing
for free, and its output is what the `SmemAllocOp` handler consumes.
Second: single-function modules are the only shape supported today;
multi-function support is deferred. Raise on any module with more than
one function.

### 4. Per-op handlers

The dispatch table is the scope of the work. Around forty IR op
classes today, each handler 5–15 lines of 1:1 IR-to-target syntax.
The recommended bring-up order is:

| Tier | Ops | What you get |
|---|---|---|
| 1 | `ConstOp`, `ArithOp`, `CmpOp`, `SelectOp`, `ConvertOp`, `BitcastOp`, thread-identity family, `BarrierOp`, `ForLoopOp`, `IfRegionOp`, `YieldOp`, `LoadOp`, `StoreOp`, `VecLoadOp`, `VecStoreOp`, `SmemAllocOp` | a scalar / block-parallel kernel can run end-to-end, including structured control flow and shared memory |
| 2 | `AsyncCopyOp` / `AsyncCopyCommitOp` / `AsyncCopyWaitOp`, `AtomicRmwOp`, `ShuffleOp`, `SubgroupReduceOp`, `SubgroupBroadcastOp`, vector build / extract, `SplitB32Op` / `MergeB32Op`, `PackedConvertOp`, `UnpackedConvertOp` | tile-level kernels with warp-cooperative loads, cross-lane reductions, and fp8 / packed paths |
| 3 | `LoadMatrixOp`, `StoreMatrixOp`, `MmaOp`, `FragApplyOp`, `FragReduceOp`, `FragConvertOp`, `FragForEachOp` | matrix-unit kernels (GEMM, attention, MoE) with in-register fragment primitives |

Tier 3 is the hardest part of the bring-up because it maps directly to
the target's matrix-unit ISA. The MSL lowerer is the cleaner reference
here — its simdgroup-matrix path is concise and covers the same
conceptual ground as NVIDIA's ldmatrix / mma.sync machinery. Before
attempting tier 3, work through a small GEMM in tier 1 + 2 and confirm
the scalar path is correct under the fuzz harness.

Structural guidance for the handlers:

- **Lookup tables, not `if / elif` ladders.** Both existing backends
  translate `ArithOp.kind`, `MathOp.kind`, `ShuffleOp.kind`,
  `AtomicRmwOp.op`, and the reduction / compare kinds through short
  dict tables. This keeps each handler to the minimum logic that's
  actually branching on something other than the kind string.
- **No hidden state.** Handlers read from `ctx` and write to
  `ctx.lines`. They should not keep module-level mutable state — two
  concurrent `lower_module` calls on different modules have to be
  independent.
- **Fail loudly on unsupported combinations.** If your target can't
  represent a particular op-kind / dtype combination (e.g. no
  scalar-to-fp8 conversion), raise with a message pointing at the IR
  alternative the caller should use. Never silently approximate.

### 5. Register the matrix-unit shapes your target supports

Every `MmaConfig` descriptor carries hardware gates — minimum PTX
compute capability, minimum Apple chip generation. Add a gate for
your target to each shape your backend can lower, and extend the
"shapes for this chip" filter so device-capability probing picks them
up. The autotuner then enumerates `<site>_shape` values from the
filtered set automatically; no per-kernel change required.

Most tile shapes are roughly conjugate across architectures (`M × N
× K` with some per-register layout), so the gate is usually a pure
boolean. If your target's native tile is an exotic shape not already
in the registry, add a new descriptor with its row / col offsets
spelled out; the MMA handler in the lowerer then keys on that new
descriptor's name.

### 6. The driver

A driver is a thin adapter between the lowerer's artifact and the
vendor runtime. Define:

- A frozen `XxxCompiledModule` dataclass holding whatever handle the
  runtime returns (module handle, function pointer, native
  dispatch object, …).
- An `XxxDriver` class with:
  - `__init__(device=None)` — lazy, no vendor imports at module load
    so non-target processes don't pay the cost.
  - `probe(index) -> DeviceCaps` — fills in the capability fields the
    rest of the system consults: warp / subgroup size, max threads
    per block, max shared memory per block, register budget, supported
    dtypes, whether async copy is supported, whether the target has an
    fp8 matrix unit, the filtered matrix-shape set, the supported
    atomic dtypes, and a chip-generation tag.
  - `compile(lowered, entry_name, smem_bytes) -> XxxCompiledModule` —
    takes the source text, drives the vendor runtime's JIT, returns
    the handle.
  - `launch(...)` — either pointer-based (raw device pointers plus
    packed scalar bytes; writes land in place) or array-based (typed
    array objects in, typed array objects out, runtime allocates
    outputs). Both shapes are present in the existing drivers; pick
    whichever matches the vendor runtime.
  - Stream / queue helpers as needed: current-stream retrieval,
    explicit sync, optional graph-capture hooks.

Everything that touches the vendor runtime lives in this module.
Nothing else in the tree should import the vendor SDK. This
discipline keeps lowerer bring-up independent of runtime plumbing —
you can unit-test the lowerer on any machine, then bring up the
driver separately on target hardware.

### 7. Wire the launcher

Three edits in the device / launcher layer:

1. Add a `DeviceFamily` variant for the new target. Populate its
   probe path in the device module (attribute reads from the vendor
   runtime, filled into `DeviceCaps`).
2. Extend the "driver for family" function so it returns your driver
   class for the new family. Keep the import lazy — non-target
   processes should never load the new vendor SDK.
3. Extend the "lower for this device" dispatch in the launcher so it
   constructs your lowerer with the right target knobs for the probed
   device.

Nothing else in the launcher changes. Compile caching, autotune
integration, the correctness threshold table, the functional-layer
plumbing, and the bench / fuzz harnesses are all device-family
oblivious — they call through the launcher and let it decide.

### 8. Framework surface (optional)

If the new target has a host framework binding you'd like to
participate in — a tensor library on the host, a graph-capture
mechanism, a custom-op registry — extend the polymorphic tensor type
to route creation / arithmetic / cast / reduction primitives through
it. Reference implementations and baselines then run on the new
device automatically. If there is no framework path, the existing
surface is unaffected: the new target only matters inside the
launcher's compile-and-launch boundary.

### 9. Tests and validation

Two layers of testing catch different bugs:

- **Unit tests on emitted source.** Lower a minimal module and assert
  the artifact's shape: header, body, terminator. These are fast,
  device-free, and run in pre-commit. They catch regressions in
  op-kind → source mappings and in the register / name allocator.
- **Correctness fuzz on real kernels.** The oracle is cosine
  similarity between the kernel's output and a backend-agnostic
  reference implementation (thresholds come from a dtype ×
  accumulator table; NaN or Inf in the output is a hard fail). The
  fuzz harness sweeps problem shapes, dtypes, and configs. Once scalar
  loads / stores / structured control flow work, start with a simple
  block-reduction kernel; add shared memory and MMA support
  incrementally, re-running the fuzz at each step.

### 10. Autotune

Nothing target-specific to do. The autotune cache and search strategies
are backend-agnostic. They call the launcher's compile-and-time hook,
which times via the backend's own synchronisation primitives (CUDA
events on NVIDIA, `mx.eval` + host clock on Metal, whatever the new
target exposes). Once `probe` reports correct warp size, smem budget,
and matrix shapes, the autotuner produces valid candidate grids
automatically.

## What's fixed versus what the target owns

To a first approximation, the IR and everything it references are
fixed. The types, the op set, the validator, the matrix-shape
registry, the authoring surface, the block composition primitives, the
shared-memory coloring pass, the launcher, the autotuner, the
correctness oracle, and the fuzz and bench harnesses are all
backend-neutral. They should not require changes to support a new
target.

What a new target owns: the visitor pass that walks an `ir.Module`
and produces target source; the dataclass that packages the source
with launch metadata; the driver that drives the vendor runtime's
compile and launch; the device-capability probe; the hardware gates on
the matrix-shape descriptors the backend supports; and the
`DeviceFamily` variant plus the two small launcher switch points that
select the new lowerer and driver.

The operating question when bringing up a new backend is usually:
**for each IR op, what is the 1:1 target syntax?** Answering that, op
by op, against the existing dispatch-table shape, is the entirety of
the codegen work.
