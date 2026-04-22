# IR

SSA, typed, structured control flow. Every kernel's `build()` body
emits ops into a `Module` via the `Builder`.

```python
from quark.ir import Builder, DType, BufferType

b = Builder("my_module")
fn = b.begin_function("kern")
x_ptr = b.param("X", BufferType(DType.BF16))
# ... emit ops ...
b.end_function()
module = b.module
```

## Types

### DType

`F32` `BF16` `F16` `E4M3` `E5M2` `U32` `S32` `U16` `S16` `U8` `S8`
`B8` `B16` `B32` `PRED`.

`B*` are untyped bit buckets (used for fragment carriers). `PRED` is
a 1-bit predicate.

### Value

SSA handle. `value.dtype`, `value.width` (1 = scalar, >1 = packed
vector register). Immutable — every op produces fresh Values.

### Value operator overloads

Value objects support arithmetic operators that emit the corresponding
Builder op against the active builder:

```python
c = a + b          # b.add(a, b)
d = a * 3          # b.mul(a, b.const(...))
p = a > b          # b.cmp("gt", a, b)    — returns PRED
q = a // b         # b.div(a, b)          — integer
```

Overloads require an active Builder (set when inside a kernel
`build()` body). Python's `/` maps to `__truediv__`; Value only
implements `__floordiv__`, so use `//`.

### Tensor types

Three backend-neutral tensor abstractions in `quark.ir.tensor`:

| Type | Memory | Creation |
|---|---|---|
| `GlobalTensor` | gmem | via `TensorDecl` in kernel manifest, or `bld.param()` + wrap |
| `SharedRegion` | smem | `bld.smem_alloc(name, dtype, shape, pad=..., lifetime=...)` |
| `FragTensor` | MMA registers | `bld.load_matrix(...)` or `bld.frag_*` ops |
| `RegisterTile` | MMA registers (multi-tile) | `bld.register_tile_*` or wraps FragTensors |

### GlobalTensor

Gmem param. Shape, stride (row-major default), dtype. Subscript
sugar for loads / stores:

```python
# Inside build():
A = self.g.A                   # from TENSORS manifest
v = A[row, col]                # single-element load
A[row, col] = value            # single-element store

# Tile view — produces another GlobalTensor pointing at the tile
tile = A.tile(row=m_base, col=0, shape=(BM, K))
```

### SharedRegion

Smem region. See the full API in `quark.ir.tensor`:

```python
# Allocation — goes through Builder
q_region = bld.smem_alloc(
    "Q_smem", DType.BF16, (BlockQRows, Dh),
    pad=8,                      # row-padding (bank conflict avoidance)
    lifetime=Lifetime.auto(),   # or .in_region(), .before_region(), etc.
)

# Subscript — single element / slice
q_region[row, col] = value
v = q_region[row, col]

# Staged — for double-buffered KV pipelines
stage = q_region.stage(k & 1)           # picks a stage slot
stage[row, col] = value

# Warp view — per-warp slice
warp_q = q_region.warp_view(rows=NCW * MTiles * 16, warp_id=warp_id)

# Warp + lane view — sets dyn_offset for PTX and warp_dyn_offset for MSL
b_lane = b_region.warp_lane_view(
    rows=BK,
    warp_id=warp_id,
    lane_col_step=lane_col_step,        # gid * row_stride + tig * step
    # or lane_offset=lane_id * frag_elems for B-shuffled layouts
)

# Unified gmem → smem tile copy (replaces manual cp.async / scalar load)
q_region.copy_from(
    g.Q.tile(row=q_base, col=0, shape=(BlockQRows, Dh)),
    tid=tid, n_threads=NumWarps * 32,
    async_load=True,                    # cp.async on PTX, sync vec_load on MSL
    cast=compute_ir_dtype,              # optional bf16 → e4m3 on load
)
```

`Lifetime` kinds (from `quark.ir.lifetime`):

| Kind | Semantics |
|---|---|
| `KERNEL` | alive for the whole kernel |
| `AUTO` | inferred from first-use / last-use |
| `IN_REGION(region)` | alive only inside a Region (for-loop / if-region body) |
| `BEFORE_REGION(region)` | alive up to (but not entering) a region |
| `AFTER_REGION(region)` | alive from region exit onwards |

The `smem_layout` pass (`quark/lower/smem_layout.py`) builds an
interval graph from lifetimes and colors regions onto smem slots.
Lifetime-disjoint regions alias automatically — no manual
`aliasable=True` pool.

### FragTensor

Per-lane MMA register carrier. One FragTensor per MMA tile. Usually
produced by `load_matrix` (ldmatrix on PTX, simdgroup_load on MSL)
or by fragment primitive ops. Rarely manipulated directly — use
`RegisterTile`.

### RegisterTile

Logical (rows, cols) view over one or more FragTensors. Hides the
per-backend lane layout. Produced from `Accumulators.to_tile()` or
directly via `bld.register_tile_*`.

```python
s_tile: RegisterTile = ...              # (BlockQRows, KvTile) bf16

# Element-wise
s_tile = s_tile.map(lambda x: b.mul(x, scale_c))

# Cross-lane reduction
row_max = s_tile.reduce_along_cols(
    kind="max", cd_offsets=mma_cfg.shape.cd_offsets,
)                                        # list[Value], one per row-class

# Layout / dtype conversion
p_tile = s_tile.convert(
    dst_layout=FragLayout.A_FRAG,
    dst_dtype=DType.BF16,
)

# Side-effect walk (used by epilogues)
tile.for_each(
    lambda elem, row, col: g.Out.__setitem__((row, col), elem),
    cd_offsets=mma_cfg.shape.cd_offsets,
)
```

## Builder

The sole IR construction interface. `build()` bodies don't call
`Builder` methods directly — instead they go through
`quark.lang as qk`, which exposes every Builder op as a
free function that reads the active builder from a contextvar:

```python
import quark.lang as qk

c = qk.add(a, b)                           # == bld.add(a, b)
with qk.for_range(lo, hi, step) as (i,):   # == bld.for_loop
    ...
with qk.if_(pred):
    ...
qk.barrier("block")
```

`Builder.begin_function()` publishes the active builder; inside any
`@kernel.build()` body `qk.*` just works. For tests or scripts that
emit ops outside a kernel, wrap in `with qk.kernel_scope(bld):`.
Module setup (`begin_function`, `param`, `register_shape`) stays on
the `Builder` — those aren't part of the kernel-authoring surface.

The DSL free-function helpers (`tid`, `barrier`, `c`, `block_base`,
…) resolve the active `BlockContext`. See
[ARCHITECTURE.md](ARCHITECTURE.md#authoring-surface--quarklang).

### Arithmetic / logic

`add sub mul fma min max div rem shl shr and_ or_ xor neg abs`

### Comparison / select

`cmp(kind, a, b)` — `kind ∈ {"lt","le","eq","ne","gt","ge"}` → PRED.

`select(pred, t, f)` — ternary.

### Conversion

`convert(v, dst, rounding="rn")` — elementwise cast.

`packed_convert(...)` — multi-element packed cast (e.g. bf16x2 → fp8x2
in one reg).

`unpacked_convert(...)` — fp8 fragment → bf16 fragment through an
intermediate b16 load. Used by owl_attn's fp8 KV cache path.

`bitcast(v, dst)` — reinterpret bits, no rounding.

### Math intrinsics

`rcp_approx rsqrt_approx ex2_approx sqrt exp2 log2 tanh`.

Approximate variants lower to `approx` on PTX / `fast::*` on MSL.
Use them in perf-critical paths (attention softmax chains ex2 + rcp).

### Constants

`const(DType.F32, 3.14)` — deduplicated. Same dtype + value collapses
to one const Value.

Inside `build()`, prefer the DSL helper `c(value)` which auto-infers
dtype from the Python literal and caches per `BlockContext`:

```python
scale = c(0.125)                    # F32 by default
mask = c(-1, dtype=DType.S32)
```

### Vector register ops

`vec_build([a, b, c, d])`, `vec_extract(v, index)` — work on packed
Value registers (width > 1). Fragment-scale extracts are discouraged
at the kernel level; use `RegisterTile.map` / `for_each` instead.

### Memory

| Op | Purpose |
|---|---|
| `load(tensor, *idx)` | scalar load |
| `store(tensor, val, *idx)` | scalar store |
| `vec_load(tensor, *idx, width, dtype=...)` | packed gmem/smem load |
| `vec_store(tensor, vec, *idx)` | packed store |
| `async_copy(dst_smem, src_gmem, *, elem_bytes, count)` | cp.async |
| `async_commit()` / `async_wait(n)` | cp.async pipeline |
| `atomic_rmw("add", dst, idx, value)` | gmem atomic |
| `smem_alloc(name, dtype, shape, pad=, lifetime=)` | SharedRegion |

Prefer the `SharedRegion` / `GlobalTensor` fluent methods
(`copy_from`, subscript, `.tile`, `.warp_view`) over raw
Builder calls — they encapsulate the common patterns.

### MMA

```python
bld.register_shape(MmaShape.m16n8k16_bf16)
A_frag = bld.load_matrix(A_smem, "m16n8k16_bf16", which="a", ...)
B_frag = bld.load_matrix(B_smem, "m16n8k16_bf16", which="b", ...)
C = bld.mma("m16n8k16_bf16", A_frag, B_frag, C_prev)   # fma
```

Shape registry lives in `quark.ir.mma_registry`. Each `MmaConfig`
carries the `MmaShape` descriptor, per-register `(row, col)` offsets,
and `min_ptx_cc` / `min_metal_gen` chip gates. `shapes_for_chip(gen)`
filters the list to the set a given `ChipGeneration` supports — this
is what populates `DeviceCaps.matmul_shapes` at probe time.

Currently registered shapes:

| Shape | A × B → acc | Min PTX cc | Min Metal gen |
|---|---|---|---|
| `m8n8k8_bf16` | bf16 × bf16 → f32 | — (Metal-only) | M3 |
| `m16n8k8_bf16` | bf16 × bf16 → f32 | sm_80 (Ampere) | — |
| `m16n8k16_bf16` | bf16 × bf16 → f32 | sm_80 | — |
| `m16n8k16_f16` | f16 × f16 → f32 | sm_75 (Turing) | — |
| `m16n8k16_e4m3` | e4m3 × e4m3 → f32 | sm_120 (Blackwell) | — |
| `m16n8k16_e5m2` | e5m2 × e5m2 → f32 | sm_120 | — |
| `m16n8k32_e4m3` | e4m3 × e4m3 → f32 | sm_89 (Ada) | — |
| `m16n8k32_e5m2` | e5m2 × e5m2 → f32 | sm_89 | — |
| `m16n8k16_bf16_e4m3` | bf16 × e4m3 → f32 | sm_120 | — |

The autotuner picks a per-MMA-site shape from `DeviceCaps.matmul_shapes`
(see `Kernel.tune_space_resolved` and the kernel's `mma_sites()`
classmethod). Adding a new MMA means adding one `MmaConfig` to
`ir/mma_registry.py` with its chip gates; everything else
(`lookup_mma`, device caps, autotune enumeration) picks it up.

### Fragment primitives

Four ops that work with FragTensors and stay in registers (no smem
round-trip):

```python
# FragApplyOp — per-element body region
out = bld.frag_apply(
    "m16n8k16_bf16", in_frag,
    body=lambda elem: bld.mul(elem, scale),
    cd_offsets=...,
)

# FragReduceOp — cross-lane reduction
row_scalars = bld.frag_reduce(
    "m16n8k16_bf16", in_frag,
    kind="max", axis="row", cd_offsets=...,
)

# FragConvertOp — layout / dtype change
a_frag = bld.frag_convert(
    "m16n8k16_bf16", acc_frag,
    src_layout="acc", dst_layout="a_frag",
    src_dtype=DType.F32, dst_dtype=DType.BF16,
    cd_offsets=...,
)

# FragForEachOp — side effects (no result)
bld.frag_for_each(
    "m16n8k16_bf16", acc_frag,
    fn=lambda elem, row, col: ...,
    cd_offsets=...,
)
```

In practice block authors use the `RegisterTile` methods (`.map`,
`.reduce_along_cols`, `.convert`, `.for_each`), which wrap these ops.

### Thread identity

| Method | Returns |
|---|---|
| `thread_idx("x"\|"y"\|"z")` | `tid` |
| `block_idx("x"\|"y"\|"z")` | `bid` |
| `block_dim("x"\|"y"\|"z")` | block size |
| `grid_dim("x"\|"y"\|"z")` | grid size |
| `lane_id()` | 0..31 |
| `subgroup_id()` | warp id within block (`tid / 32`) |
| `group_id()` | quad id (`lane / 4`) — used by MMA frag layout |
| `thread_id_in_group()` | lane in quad (`lane & 3`) |

Inside `build()`, prefer the lazy DSL helpers (`tid`, `gid`, `tig`,
`warp_id`) which cache into `BlockContext`.

### Structured control flow

```python
# For loop
with bld.for_loop(lo=0, hi=K, step=BK, carried=[acc_init]) as [acc]:
    # body
    new_acc = ...
    bld.yield_(new_acc)
loop_results = bld.last_results()       # final carried values

# If
with bld.if_(pred, carried=[v_init]) as [v]:
    with bld.then_():
        bld.yield_(v_then)
    with bld.else_():
        bld.yield_(v_else)
```

### Cross-lane shuffles

| Op | Semantics |
|---|---|
| `shuffle("xor", v, mask)` | butterfly shuffle |
| `shuffle("up", v, delta)` | lane ± delta |
| `shuffle("down", v, delta)` | |
| `shuffle("idx", v, src_lane)` | broadcast from specific lane |
| `subgroup_reduce("add"\|"max"\|"min", v)` | full-warp reduction |
| `subgroup_broadcast(v, lane)` | broadcast from specific lane |

### Barriers

`barrier("block")` — block-wide sync. Must be unconditional across
all threads; don't place inside a divergent `if_`.

## Validation

`validate_module(module)` runs before lowering. Catches:

- SSA violations (use-before-def across regions)
- Region terminator mismatch (ForLoopOp body missing YieldOp with
  matching shapes)
- Tensor rank / index type mismatches
- Unregistered `MmaShape` references
- Const-index OOB against declared `SharedRegion.shape`
- Perf warnings (bank conflicts, unvectorized stores, unpadded strides)
  emitted via `PerfWarning`

Perf warnings are off by default; enable with `QUARK_ENABLE_PERF_WARNINGS=1`.

## See also

- [BLOCKS.md](BLOCKS.md) — higher-level composition (L0 / L1 / L2)
- [BACKEND.md](BACKEND.md) — PT polymorphic tensor API
