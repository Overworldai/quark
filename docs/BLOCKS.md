# Blocks

Blocks are how kernels compose. A kernel's `build()` body declares
smem, instantiates blocks, and drives the pipeline; the decorator
wires the BlockContext and finalizes the IR. Most of the scaffolding
is implicit — `build()` reads like the mathematical steps of the
kernel, with active context resolved via ContextVar lookups.

## Layers

| Layer | Location | Scope |
|---|---|---|
| **DSL** | `blocks/dsl/` | Framework: `Block`, `BlockContext`, `KernelContext`, `Accumulators`, `SmemTile` / `SmemTileSpec`, `Stage`, `Carry`, `TensorDecl`, `C`, free-function helpers (`c`, `tid`, `barrier`, `block_base`, …) |
| **L0** | `blocks/l0/` | One emit function = one hardware pattern. Single MMA tile, single cp.async line, single smem base calc. No loops, no barriers. Kernel authors rarely touch these directly — they live behind the L1 / DSL helpers. |
| **L1** | `blocks/l1/` | Leaf Block classes. Currently just `MmaBody` — the triple-nested MMA tile loop with multiple call shapes. Single-use helpers that used to live here (`IndexCache`, `WorkListLoad`, `QRegisterLoad`, `GatheredTileLoad`) are now free functions under `quark.lang`. |
| **L2** | `blocks/l2/` | Composition: `run_pipeline` + `PipelineBody` closure body, `SmemPlan.paired` / `.staged_pairs`. Owns the `for_loop`, barriers, carried accumulators. |
| **L3** | `kernels/*/kernel.py` | The kernel. Constructs L1 / L2, calls `PipelineBody(...).run(...)`. |

Rule of thumb:

- Pattern used in 2+ kernels **or** >15 lines with clear single
  responsibility → L1 Block **or** `qk.*` free function (prefer the
  free function unless the pattern has durable state).
- Multi-L1 orchestration with carried state + barriers → L2 Block.
- Everything else — inline in `build()`.

Don't put reusable patterns in kernel folders. If two kernels need
it, it belongs in L1 / L2 / `quark.lang`.

## DSL primitives

```python
from quark.blocks import (
    # Core
    Block, BlockContext, KernelContext, SetupBlock,
    # Value / const / tensor / smem helpers
    C, c, TensorDecl,
    # Typed data containers
    Accumulators, SmemTile, SmemTileSpec, Stage, Carry,
    # L1 blocks
    MmaBody,
    # L2 blocks
    run_pipeline, PipelineBody, IterCtx, SmemPlan,
    # Free-function helpers (resolve active BlockContext)
    tid, lane_id, warp_id, gid, tig, barrier, block_base, block_idx,
)
```

Kernels also import `quark.lang as qk` for IR ops (`qk.mul`,
`qk.for_range`, `qk.barrier`), epilogue helpers (`qk.store_acc`,
`qk.atomic_store_acc`, `qk.silu`, `qk.cast`), and the memory
helpers that replaced the L1 dataclass wrappers (`qk.work_list_load`,
`qk.index_cache`, `qk.q_register_load`). See
[ARCHITECTURE.md](ARCHITECTURE.md#authoring-surface--quarklang) and
[IR.md](IR.md).

### Retired classes (use the free-function replacement)

| Old L1/L2 class | Use instead |
|---|---|
| `TileLoad` | `smem_tile.load_from(gmem, row=, col=, cast=)` |
| `GatheredTileLoad` | `smem_tile.gather_from(gmem, index=, col=, cast=)` |
| `ScatterStore` / `SiluCastStore` | `qk.store_acc(dst, acc, row=, col=, cast=, activation=)` |
| `AtomicScatterAdd` | `qk.atomic_store_acc(dst, acc, col=, index=, weight=, op="add")` |
| `NormalizeAndStore` (attn) | `qk.store_acc(..., per_warp=True, row_scale=l_vals)` |
| `WorkListLoad` | `grp_start, expert = qk.work_list_load(work_list, work_idx)` |
| `IndexCache` | `smem_region = qk.index_cache(name, gmem, count=, base=)` |
| `QRegisterLoad` | `q_frags = qk.q_register_load(g_q, smem=, q_row=, warp_id=, MT=, Dh=, kv_pad=)` |
| `Gemm1QK` / `Gemm2PV` (attn) | `MmaBody(a=q_frags, b=stage.k, acc=s_init)` |
| `OnlineSoftmax.emit_with_state(ctx, ...)` | `softmax(s_acc_vals=, o_vals=, m_vals=, l_vals=)` (the class is still callable) |

### `C(value, dtype=None)` / `c(value)`

Constant **spec**. Type inferred from Python literal: `int → U32`,
negative int → `S32`, `float → F32`. Override: `C(3, DType.S32)`.

Wrap `C` in the `c(...)` free function inside `build()` to materialize
as an IR Value via the active BlockContext cache (same constant →
same Value, deduplicated):

```python
scale = c(0.125)                # F32
mask = c(-1, dtype=DType.S32)   # forced type
```

Most arithmetic accepts Python ints / floats directly now — you
rarely need `c(...)` unless you're building an expression that
requires an explicit `Value`. In particular `qk.for_range(0, n, 1)`
auto-lifts the Python ints.

### `TensorDecl`

Entry in the kernel `TENSORS` manifest. Shape and dtype are callables
of `(spec, config)` so they can depend on tuning knobs:

```python
TensorDecl(
    "W_in",
    dtype=lambda s, c: s.b_dtype,
    shape=lambda s, c: (
        s.n_experts * s.H,
        (s.D // c.BK) * (c.BK + c.b_pad) if c.b_shuffle and c.b_pad > 0 else s.D,
    ),
    role="in",                    # or "out" / "scratch"
)
```

### `Accumulators(MT, NT, width=4, dtype=F32, fill_value=0.0)`

Declarative MT × NT grid of accumulator registers. Used as
`PipelineBody.carry`; `run_pipeline` populates `acc.results` with the
loop-final Values so epilogue helpers can take just the Accumulators
and pull both layout + values.

```python
acc = Accumulators.from_mma(mma_cfg, BM=c.BM, BN=c.BN, n_warps=c.n_warps)
mma = MmaBody(acc=acc)            # shape + K_inner inferred at call time
PipelineBody(stages=stages, produce=produce, consume=mma, carry=acc).run(
    n_iters=K_outer, n_stages=c.n_stages,
)
# After .run: acc.results is populated.
qk.store_acc(g.Out, acc, row=m_base, col=n_base + warp_off, cast=DType.BF16)
```

`Accumulators.from_mma(mma_cfg, *, BM, BN, n_warps)` derives MT / NT /
width from an `MmaConfig` and the tile shape — the common case for
GEMM-shaped kernels. Hand-constructing via `Accumulators(MT=, NT=,
width=, dtype=)` still works for non-GEMM shapes.

`acc.init()` emits the zero-initialized vec Values; `acc.count()` is
`MT*NT`. The Builder is resolved from the active kernel scope — no
`bld` arg.

### `SmemTile(name, dtype, shape, pad=0, lane_col_step=2)` / `SmemTile.spec(...)`

Declarative smem allocation. Auto-emits into the active `KernelContext`
on construction (no explicit `.emit(bld)` needed).

```python
# Inside build():
q_tile = SmemTile("Q_smem", DType.BF16, (BlockQRows, Dh), pad=8)
# q_tile.smem is the SharedRegion.
# q_tile.lane is the per-lane view (MMA frag loads).
# q_tile.load_from(g_q, row=..., col=..., cast=...) — cooperative gmem→smem.
# q_tile.gather_from(g_x, index=toks, col=k_col, cast=...) — indirect rows.
```

`SmemTile.spec(dtype, shape, pad=, lane_col_step=)` returns a frozen
`SmemTileSpec` (no name, no allocation yet). Use it with
`Stage.staged(n, **specs)` to template per-stage tiles:

```python
stages = Stage.staged(
    c.n_stages,
    k=SmemTile.spec(s.b_dtype, (KvTile, Dh), pad=c.KvPad, lane_col_step=lcs),
    vt=SmemTile.spec(s.b_dtype, (Dh, KvTile), pad=c.KvPad, lane_col_step=lcs),
)
# → [Stage(k=K_s0, vt=Vt_s0), Stage(k=K_s1, vt=Vt_s1)]
```

### `Stage(**tiles)` / `Stage.staged(n, **specs)`

Attribute-addressable per-iteration container. Replaces the
`dict[str, SmemTile]` stages pattern — inside `consume` / `produce`
you get `ictx.stage.k`, `ictx.stage.vt` with proper attribute access.
Subscript (`stage["k"]`) still works for back-compat.

### `Carry(**slots)`

Named multi-subset loop carry with attribute access. Replaces manual
flatten / unflatten of a `(o_vals, m_vals, l_vals)` tuple inside the
consume body. Accepted slot spec forms (mix freely in one Carry):

| Spec | Meaning |
|---|---|
| `Accumulators` | emits `MT*NT` vec init Values |
| `(count, init_value, dtype)` | `count` copies of `qk.const(dtype, init_value)` |
| `list[Value]` / tuple | pre-built; no init callback |
| Single `Value` | one-slot scalar |

```python
carry = Carry(
    o=o_acc,                              # Accumulators
    m=(n_ml, -1e30, DType.F32),           # scalar array
    l=(n_ml,  0.0,  DType.F32),
)

def consume(ictx):
    carry = ictx.carry                    # a Carry instance, slots rebound
    s_vals = mma1(a=q_frags, b=ictx.stage.k, acc=s_acc.init())
    o, m, l, p = softmax(s_acc_vals=s_vals, o_vals=carry.o, m_vals=carry.m, l_vals=carry.l)
    carry.o = mma2(a=p, b=ictx.stage.vt, acc=o)
    carry.m = m
    carry.l = l
    return carry                          # run_pipeline flattens under the hood

final = PipelineBody(stages=..., produce=..., consume=consume, carry=carry).run(...)
# `final` is the same Carry rebound to loop-end values — read final.l / final.o directly.
```

`carry.bound_to(flat)` returns a fresh Carry with the same spec but
values seeded from a flat tuple — used for nested pipelines where an
outer loop's carry feeds the inner's init (see `owl_attn`).

`qk.yield_(carry)` auto-flattens a Carry argument — inside a
hand-rolled `for_range`, the idiom is `qk.yield_(carry)` rather than
`qk.yield_(*carry.flatten())`.

### `BlockContext` (via `ctx` param or free-function helpers)

Thread-identity cache + builder handle.

| Attribute | Semantics |
|---|---|
| `ctx.bld` | the active `Builder` |
| `ctx.tid` | `thread_idx("x")`, lazily |
| `ctx.gid` | `group_id()` — `lane_id >> 2` — for MMA frag layout |
| `ctx.tig` | `thread_id_in_group()` — `lane_id & 3` |
| `ctx.tig_x2` | `tig * 2` |
| `ctx.lane_id`, `ctx.warp_id` | |
| `ctx.n_threads` | block size |
| `ctx.mma_cfg` | current `MmaConfig` |
| `ctx.const(dtype, value)` | cached IR const materialization |
| `ctx.c(value, dtype=None)` | auto-typed const shorthand |

Free-function helpers (`tid()`, `barrier("block")`, `block_base("x", BN)`,
`warp_id()`, `lane_id()`) resolve the active BlockContext. Inside
`build()` they're the idiomatic choice — no `ctx.` prefix clutter.

### `KernelContext`

Top-level builder. Normally the `@kernel` decorator constructs this
for you; `self.make_ctx(mma_cfg)` materializes the BlockContext and
publishes it for the free-function helpers. Kernel authors typically
never touch `KernelContext` directly — the decorator handles it and
binds `self.bctx`, `self.m_base`, `self.n_base`, `self.g` before
`build()` runs.

## L0 — hardware-pattern emit functions

Each L0 function emits exactly one IR pattern. No loops, no barriers,
no state.

| Function | Purpose |
|---|---|
| `emit_smem_base(b, smem, lane_col_step)` | per-lane SharedRegion view for MMA frag loads (`gid * row_stride + tig * step`) |
| `emit_mma_tile(b, a_smem_lane, b_smem_lane, shape_id, …, m_tile, n_tile, kk, acc_in)` | single MMA: load A/B frag + fma |
| `emit_cooperative_load(b, …)` | scalar tile load fallback |
| `emit_tile_load(b, dst_smem, src_gmem, rows, cols, …)` | cp.async 16-byte line-per-thread (with scalar fallback) |
| `emit_gathered_tile_load(b, dst_smem, src_gmem, index_smem, …)` | cp.async gather (index-sourced row addresses) |
| `emit_shuffled_bfrag_load(ctx, b_lane, …)` | vectorized shuffled-B fragment load |
| `emit_index_cache(b, gmem, smem, count, …)` | cooperative 1D smem cache of a gmem array |
| `emit_silu_cast_epilogue(b, acc_results, g_out, …)` / `emit_atomic_scatter_epilogue(b, …)` | legacy epilogue emitters (use `qk.store_acc` / `qk.atomic_store_acc` instead) |

Most kernel authors never call these directly — the DSL methods
(`SmemTile.load_from`, `.gather_from`) and the `qk.*` free functions
wrap them. Reach for L0 when assembling a novel L1 pattern.

## L1 — leaf Block classes

| Block | One-line |
|---|---|
| `MmaBody(shape=None, acc=None, BK=None, K_inner=None, MT=None, NT=None, n_offset=0, b_shuffled=False)` | Triple-nested MMA tile loop. `shape` defaults to `active_bctx().mma_cfg`; `K_inner` is inferred from the B tile's shape[1] at call time; MT/NT come from `acc` if supplied. Three call forms: `mma(ictx)`, `mma(stage, carry)`, or `mma(a=, b=, acc=)`. |

### MmaBody — three call forms

```python
mma = MmaBody(acc=acc)             # minimal — everything derivable at call time

# 1. GEMM-style: consume takes the SmemPlan stage + carry from IterCtx.
body = PipelineBody(stages=stages, produce=produce, consume=mma, carry=acc)
body.run(n_iters=K_outer, n_stages=c.n_stages)

# 2. Explicit (stage, carry) form — rare, only when carry isn't ictx.carry.
new_carry = mma(stage_plan, o_vals)

# 3. Triton-style tl.dot form — pass operands directly.
# ``a`` / ``b`` each accept a SmemTile (smem-backed load_matrix) or
# pre-loaded register fragments (list[list[Value]] indexed [mt][inner_k]).
s_vals = mma1(a=q_frags, b=stage.k, acc=s_acc.init())
carry.o = mma2(a=p_frags, b=stage.vt, acc=o_vals)
```

Only `MmaBody` remains as an L1 Block. The former L1/L2 dataclass
wrappers (`IndexCache`, `WorkListLoad`, `QRegisterLoad`,
`GatheredTileLoad`) are now free functions under `quark.lang` —
each was a 4-50 line emit body that didn't benefit from a class.

### Epilogues (L1-ish, live in `quark.lang`)

Always frag → smem → vec_store for block-sized tiles; never scalar
direct to gmem. The `qk.store_acc` fast path stages through smem
and issues cooperative vec_stores. Use the attention-specific
`per_warp=True` / `row_scale=` knobs for per-warp output slices +
`O /= l` normalization.

```python
# Dense block store (GEMM-style).
qk.store_acc(g.Out, acc, row=m_base, col=n_base + warp_off, cast=DType.BF16)

# MoE with silu fusion.
qk.store_acc(g.h, acc, row=token_row, col=n_base,
              cast=DType.BF16, activation="silu", stage_in_smem=True)

# MoE outproj — atomic scatter-add with row weights.
qk.atomic_store_acc(g.Out, acc, col=n_base, index=toks, weight=weights)

# Attention — O /= l + per-warp staging. Staging smem is auto-allocated
# (AUTO lifetime; smem-layout pass aliases it over dead upstream regions).
qk.store_acc(g_out, o_acc, row=out_row_warp, col=0,
              cast=DType.BF16, per_warp=True, row_scale=final.l)
```

## L2 — composition blocks

| Block | Purpose |
|---|---|
| `PipelineBody(stages, produce, consume, carry, epilogue?, consume_tail?)` + `.run(n_iters, n_stages)` | closure-body pipelined loop. Single code path for `n_stages ∈ {1, 2}`. `bctx` defaults to `active_bctx()`. |
| `run_pipeline(*, n_iters, body, n_stages)` | lower-level entry, same thing as `body.run(...)`. |
| `IterCtx(iter_idx, stage, stage_idx, carry, bctx, is_tail)` | per-iteration context passed to `produce` / `consume`. `carry` is a `Carry` instance when one was declared, else a flat `tuple[Value, ...]`. |
| `SmemPlan.paired(name, dtype, *, a_shape, b_shape, a_pad, b_pad, mma_cfg, n_warps, b_shuffled)` | paired A/B smem with MMA-aware per-lane B view; one SmemPlan per pipeline stage. |
| `SmemPlan.staged_pairs(..., n_stages=)` | returns a list of `n_stages` SmemPlans in one call. |

### Pipeline semantics at `n_stages=1`

1. `body.produce(ictx)` — prefetch into `ictx.stage`
2. `qk.async_commit()` / `qk.async_wait(0)` — iff produce emitted any `cp.async` (detected via `Builder.async_emissions` counter)
3. `barrier("block")`
4. `body.consume(ictx)` → new carry
5. `barrier("block")`
6. `qk.yield_(*new_carry)`

### `n_stages=2` — software-pipelined double buffer

Prologue prefetches iters 0, 1 into stages 0, 1; steady state loops
`half_iters - 1` times computing a pair and prefetching the next
pair; epilogue drains the final pair. `consume_tail` optionally
overrides the last two iterations (attn uses this to skip the
online-softmax rescale that would otherwise run against past
end-of-K).

### Runtime `n_iters`

`run_pipeline` accepts a runtime `Value` for `n_iters` at
`n_stages=1` (owl_attn's inner loop runs a segment-variable chunk
count); `n_stages=2` requires a compile-time `int` because the
prologue / epilogue need the half-iter count statically.

### Return value

`PipelineBody(...).run(...)` returns the final carry:

* `Carry` instance when the pipeline was declared with a `Carry`
  (the Carry is rebound to loop-end values — read `final.o`,
  `final.l`, etc. directly).
* flat `tuple[Value, ...]` otherwise.

`_stash_results_on_carry` writes each Accumulators slot's `.results`
automatically so `qk.store_acc(g.Out, acc, …)` works without the
kernel having to unpack.

## Inside-kernel patterns (not extracted to blocks)

- **`RegisterTile` fluent ops** — `.map(fn)` / `.reduce_along_cols` /
  `.convert` / `.for_each` for stay-in-register fragment transforms.
  Fragment primitives are IR ops (`FragApplyOp`, `FragReduceOp`,
  `FragConvertOp`, `FragForEachOp`) — the `RegisterTile` helpers just
  expose them fluently. See [IR.md](IR.md).
- **`SharedRegion` subscript / `warp_view` / `copy_from`** — unified
  surface for smem R/W without ad-hoc `view(dyn_offset=…)` math. See
  [IR.md](IR.md).
- **`qk.min` / `qk.max(x, dim=)` / `qk.sum(x, dim=)`** — dim-kwarg
  sugar over `frag_reduce`; routes via `active_bctx().mma_cfg`.
