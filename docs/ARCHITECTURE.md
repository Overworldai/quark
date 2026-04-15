# Architecture

## Pipeline

```
@kernel class MyKernel         ─┐
  TENSORS manifest              │  Kernel.emit() → IR Module
  build() body                 ─┘
                                       │
                                       ▼
                               validate_module (structural + perf warnings)
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                                             ▼
         PtxLowerer                                      MslLowerer
    (src/popcorn/lower/ptx)                       (src/popcorn/lower/msl)
                │                                             │
                ▼                                             ▼
          PTX text                                       MSL source
                │                                             │
                ▼                                             ▼
        CudaDriver.compile                           MlxDriver.compile
     (ctypes → libcuda)                           (mlx.fast.metal_kernel)
                │                                             │
                ▼                                             ▼
          GPU binary                                    Metal function
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                          CompiledKernel.launch(buffers=[...])
```

Backend dispatch is decided at **import time** by `sys.platform`
(`popcorn.backend.IS_METAL`). Apple → MSL / MLX. Everything else →
PTX / CUDA. No runtime branching in user code.

## IR

Static-single-assignment, strongly typed, structured control flow.

- **Values** — `Value(dtype, width)` — SSA; every op produces fresh
  Values. No mutation.
- **Structured control flow** — `ForLoopOp`, `IfRegionOp`,
  `WhileLoopOp`. Bodies are `Region`s. Never labels + bra at the IR
  level; lowerers emit those.
- **Tensor types** (`popcorn.ir.tensor`):
  - `GlobalTensor` — gmem param. Shape, stride, dtype.
  - `SharedRegion` — smem allocation. Typed, with `Lifetime`.
  - `FragTensor` — MMA register layout. Per-lane slots.
  - `RegisterTile` — logical (rows, cols) view over one or more
    FragTensors. The typed abstraction over lane layout. See
    [IR.md](IR.md).
- **DType** — `F32` `BF16` `F16` `E4M3` `E5M2` `U32` `S32` `PRED`
  `B16` `B32` `B8`. Unified str + IR + backend bridge:
  `DType("bf16")` from a string, `DType.BF16.backend` gives the
  `mx.bfloat16` / `torch.bfloat16` matching the active backend.
- **Fragment primitives**:
  - `FragApplyOp` — per-element transform with a body region.
  - `FragReduceOp` — cross-lane row/col reduction (max/min/add/mul).
  - `FragConvertOp` — layout/dtype conversion (e.g. ACC f32 → A_FRAG bf16).
  - `FragForEachOp` — side-effect walk (store-per-element epilogues).

### Authoring surface — `popcorn.lang`

Kernels author IR via the `popcorn.lang` namespace (not `Builder`
methods directly). Every `Builder` op is exposed as a free function
that reads the active builder from `_ACTIVE_BUILDER`:

```python
import popcorn.lang as pop

pop.mul(a, b)                 # == bld.mul(a, b); or just a * b (Value op overloading)
pop.for_range(0, n, 1)        # context manager; Python ints auto-lifted to U32
pop.if_(pred)                 # context manager
pop.const(DType.F32, 1.0)
pop.barrier("block")
pop.yield_(carry)             # auto-flattens a Carry; else pass Values explicitly
```

Higher-level authoring helpers in `popcorn.lang` compose the DSL
primitives:

```python
# Epilogues.
pop.store_acc(dst, acc, row=, col=, cast=, activation=)
pop.atomic_store_acc(dst, acc, col=, index=, weight=)
pop.silu(values) / pop.cast(values, dtype)

# Memory.
grp_start, expert = pop.work_list_load(work_list, work_idx)
toks_smem         = pop.index_cache("toks", g.token_ids, count=c.BM, base=grp_start)
q_frags           = pop.q_register_load(g_q, smem=q_tile, q_row=..., warp_id=..., MT=..., Dh=..., kv_pad=...)
```

`Builder.begin_function()` publishes the active builder; inside any
`@kernel`'s `build()` body `pop.*` just works. For tests and scripts
that emit ops outside a kernel, wrap in `with pop.kernel_scope(bld):`.

Module setup (`begin_function`, `param`, `register_shape`) stays on
the `Builder` — those aren't part of the kernel-authoring surface.

See [IR.md](IR.md) for the full op list.

## Registry

`@kernel("name", spec=..., config=..., problems=..., baselines=...,
reference=...)` registers a Kernel subclass in the global dict.
`kernels/__init__.py` imports every subfolder so the decorator fires
at package load. **Adding a kernel = adding a folder.** Tools
(bench, fuzz, autotune) discover it via `all_kernels()`.

Required overrides (validated by the decorator):

| Hook | Purpose |
|---|---|
| `problems()` | Bench/fuzz problem list (or pass `problems=` to the decorator) |
| `tune_space()` | Autotune search space |
| `param_spec()` | Parameter layout (default: derive from `emit()`) |
| `TENSORS` | `list[TensorDecl]` — the gmem buffer manifest |
| `build()` | IR body — the decorator wraps it in `emit()` |

Optional overrides (hookable via decorator kwargs):

| Hook | Decorator kwarg | Default |
|---|---|---|
| `problems()` | `problems=fn` or `problems=[...]` | raises |
| `baselines()` | `baselines=fn(kernel, tensors)` | empty list |
| `reference()` | `reference=fn(kernel, *tensors)` | `NotImplementedError` |
| `from_problem()` | — | `cls(SPEC_CLS(**problem), CONFIG_CLS.default_for(spec))` |
| `correctness_threshold()` | set `CORRECTNESS_THRESHOLD: float` on the class | dtype table |

## Kernel layout

```
src/popcorn/kernels/my_kernel/
    __init__.py        triggers @kernel registration
    spec.py            frozen KernelSpec dataclass (M, N, K, dtypes, ...)
    config.py          frozen KernelConfig with default_for(spec)
    kernel.py          @kernel class + build() body
    reference.py       fn(kernel, *tensors) -> out
    problems.py        fn() -> list[Problem]
    baselines.py       fn(kernel, tensors) -> list[Baseline]
```

See [ADDING_A_KERNEL.md](ADDING_A_KERNEL.md) for the full template.

## Launcher

```python
launcher = Launcher(device=current_device())
compiled = launcher.compile(MyKernel, spec, config)
compiled.launch(buffers=[A, B, Out])
```

- `Launcher` owns the device handle + autotune cache.
- `compile()`: `emit()` → `validate_module` → lower → driver. Cached.
- `ParamSpec` splits IR params into buffers (pointer + dtype + align)
  and scalars. `.launch(buffers=...)` extracts `data_ptr()` on torch,
  routes through MLX kernel args on Metal.
- `prepare_launch_tensors(tensors)` — optional per-kernel hook for
  pre-launch transforms (GEMM uses this to cache weight shuffles).

## Correctness metric

Cosine similarity only. Threshold from a dtype × accumulator table
(`popcorn/correctness.py:_THRESHOLD_TABLE`). Override per-kernel by
setting `CORRECTNESS_THRESHOLD: float` on the class.

`check_correctness(out, ref, out_dtype)` accepts raw torch **or** mlx
tensors — internally routes through `PT.cosine_sim`. No detach-to-cpu
ritual at call sites.

NaN / Inf in output → hard fail regardless of cos_sim.

## Platform abstraction (PT)

`popcorn.backend.PT` is the polymorphic tensor surface. All tensor
creation / arithmetic / reductions / casts / reshapes / FFI go
through `PT.*`. One file (`backend.py`) holds every
torch-vs-mlx branch.

```python
from popcorn.backend import PT, IS_METAL

A = PT.randn(M, K)                      # mlx on Metal, torch on CUDA
A = PT.astype(A, PT.bfloat16)
C = PT.matmul(A, PT.transpose(B))
cs = PT.cosine_sim(C, ref)              # float in [-1, 1]

@PT.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def graph(X, Y): ...                     # torch.compile on CUDA, identity on Metal
```

See [BACKEND.md](BACKEND.md) for the full surface + how to add a
third backend.

## Autotune

`tune_space()` / `tune_space_resolved(spec, device)` declare a
cartesian product of knob values. `AutotuneCache` (one per `Launcher`,
lives in `src/popcorn/autotune.py`) implements a three-level lookup:

```
hot dict → ~/.cache/popcorn/<hash>.json → configs/<kernel>_<problem>.json
```

On a miss the cache runs an inline **search** (blocking, on the calling
thread) and persists the winner to disk. Two depths:

- **`"fast"`** (default) — up to 16 seed-first candidates, one timing
  pass. Seeded from existing on-disk configs for this (kernel, device)
  pair so new shapes start from known-good neighbours.
- **`"full"`** — full genetic search (population × generations, same
  warm seeding). Triggered by `POPCORN_MAX_AUTOTUNE=1`, the
  `with popcorn.max_autotune():` context manager, or the per-op
  `.autotune()` warmup API (`pcf.gemm.autotune(A, B)`).

`tools/autotune.py` is a thin CLI wrapper over the same
`genetic_search()` primitive, adding PTX-dump-on-compile-error and
saving results to the bundled `configs/` dir.

See [FUNCTIONAL.md](FUNCTIONAL.md) for the warmup API and
[BENCHMARKING.md](BENCHMARKING.md) for tags, timing, and config layout.
