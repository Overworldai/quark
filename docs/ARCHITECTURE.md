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
    (src/quark/lower/ptx)                       (src/quark/lower/msl)
                │                                             │
                ▼                                             ▼
          PTX text                                       MSL source
                │                                             │
                ▼                                             ▼
        CudaDriver.compile                          MetalDriver.compile
     (ctypes → libcuda)                       (metal-cpp + nanobind shim)
                │                                             │
                ▼                                             ▼
          GPU binary                                    Metal function
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                          CompiledKernel.launch(buffers=[...])
```

Backend dispatch is decided at **import time** by `sys.platform`
(`quark.backend.IS_METAL`). Apple → MSL via the metal-cpp +
nanobind driver. Everything else → PTX / CUDA via the libcuda
ctypes binding. No runtime branching in user code.

## IR

Static-single-assignment, strongly typed, structured control flow.

- **Values** — `Value(dtype, width)` — SSA; every op produces fresh
  Values. No mutation.
- **Structured control flow** — `ForLoopOp`, `IfRegionOp`,
  `WhileLoopOp`. Bodies are `Region`s. Never labels + bra at the IR
  level; lowerers emit those.
- **Tensor types** (`quark.ir.tensor`):
  - `GlobalTensor` — gmem param. Shape, stride, dtype.
  - `SharedRegion` — smem allocation. Typed, with `Lifetime`.
  - `FragTensor` — MMA register layout. Per-lane slots.
  - `RegisterTile` — logical (rows, cols) view over one or more
    FragTensors. The typed abstraction over lane layout. See
    [IR.md](IR.md).
- **DType** — `F32` `BF16` `F16` `E4M3` `E5M2` `U32` `S32` `PRED`
  `B16` `B32` `B8`. Unified str + IR + backend bridge:
  `DType("bf16")` from a string, `DType.BF16.backend` gives the
  `mx.bfloat16` / `torch.bfloat16` matching the active backend (the
  torch resolution is baseline-only; the runtime path maps `DType` to
  the string keys `QuarkTensor` uses — `"bf16"`, `"e4m3"`, etc.).
- **Fragment primitives**:
  - `FragApplyOp` — per-element transform with a body region.
  - `FragReduceOp` — cross-lane row/col reduction (max/min/add/mul).
  - `FragConvertOp` — layout/dtype conversion (e.g. ACC f32 → A_FRAG bf16).
  - `FragForEachOp` — side-effect walk (store-per-element epilogues).

### Authoring surface — `quark.lang`

Kernels author IR via the `quark.lang` namespace (not `Builder`
methods directly). Every `Builder` op is exposed as a free function
that reads the active builder from `_ACTIVE_BUILDER`:

```python
import quark.lang as qk

qk.mul(a, b)                 # == bld.mul(a, b); or just a * b (Value op overloading)
qk.for_range(0, n, 1)        # context manager; Python ints auto-lifted to U32
qk.if_(pred)                 # context manager
qk.const(DType.F32, 1.0)
qk.barrier("block")
qk.yield_(carry)             # auto-flattens a Carry; else pass Values explicitly
```

Higher-level authoring helpers in `quark.lang` compose the DSL
primitives:

```python
# Epilogues.
qk.store_acc(dst, acc, row=, col=, cast=, activation=)
qk.atomic_store_acc(dst, acc, col=, index=, weight=)
qk.silu(values) / qk.cast(values, dtype)

# Memory.
grp_start, expert = qk.work_list_load(work_list, work_idx)
toks_smem         = qk.index_cache("toks", g.token_ids, count=c.BM, base=grp_start)
q_frags           = qk.q_register_load(g_q, smem=q_tile, q_row=..., warp_id=..., MT=..., Dh=..., kv_pad=...)
```

`Builder.begin_function()` publishes the active builder; inside any
`@kernel`'s `build()` body `qk.*` just works. For tests and scripts
that emit ops outside a kernel, wrap in `with qk.kernel_scope(bld):`.

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
src/quark/kernels/my_kernel/
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
  and scalars. `.launch(buffers=...)` extracts `data_ptr()` on
  `QuarkTensor` (or a `torch.Tensor` for the baseline reference
  path) and routes through ``_metal_dispatch.queue_launch`` on Metal
  / direct ``cuLaunchKernel`` on CUDA.
- `prepare_launch_tensors(tensors)` — optional per-kernel hook for
  pre-launch transforms (GEMM uses this to cache weight shuffles).

## Correctness metric

Cosine similarity only. Threshold from a dtype × accumulator table
(`quark/correctness.py:_THRESHOLD_TABLE`). Override per-kernel by
setting `CORRECTNESS_THRESHOLD: float` on the class.

`check_correctness(out, ref, out_dtype)` accepts `QuarkTensor` or
torch tensors — internally routes through `PT.cosine_sim`.
No detach-to-cpu ritual at call sites.

NaN / Inf in output → hard fail regardless of cos_sim.

## Tensors — runtime vs. reference

Two tensor types, two audiences:

- **Runtime path** — `QuarkTensor` (`quark.runtime.tensor`).
  Owns a `cuMemAlloc`'d device pointer, supports shape / stride /
  offset metadata, arithmetic / slicing / reshape / permute via
  on-device PTX utility kernels (`runtime/kernels.py`). This is
  what `quark.nn.Module` parameters hold and what `qf.*` calls
  consume on CUDA. **Zero torch dependency.**
- **Reference / baseline path** — `quark.backend.PT`. Thin torch
  wrapper used exclusively by `reference.py` / `baselines.py` in
  each kernel folder and by tests that need framework-native
  semantics (e.g. a `flex_attention` baseline). Torch is pulled in
  as an optional dev dep for this leg.

```python
# Runtime
from quark.runtime.tensor import QuarkTensor
A = QuarkTensor.randn(M, K, dtype="bf16")
C = qf.gemm(A, B)                     # QuarkTensor in, QuarkTensor out

# Reference (test / autotune correctness gate) — pure numpy
import numpy as np

A_ref = np.random.randn(M, K).astype(np.float32)
C_ref = A_ref @ B_ref.T
```

On Metal the same `QuarkTensor` runtime path applies — there's no
MLX or `mx.array` interop on the inference path; quark drives Metal
directly through the `_metal_dispatch` C++ extension. References
are plain numpy (the per-kernel `reference.py` files use
`quark.runtime.npconv.to_f32_numpy` to upcast device tensors);
see [WEIGHTS.md](WEIGHTS.md) for how parameters reach device via
the `QuarkTensor` path.

## Autotune

`tune_space()` / `tune_space_resolved(spec, device)` declare a
cartesian product of knob values. `AutotuneCache` (one per `Launcher`,
lives in `src/quark/autotune.py`) implements a two-level lookup:

```
hot dict → ~/.cache/quark/<hash>.json
```

On a miss the cache runs an inline **search** (blocking, on the calling
thread) and persists the winner to disk. Two depths:

- **`"fast"`** (default) — up to 16 seed-first candidates, one timing
  pass. Seeded from existing on-disk configs for this (kernel, device)
  pair so new shapes start from known-good neighbours.
- **`"full"`** — full genetic search (population × generations, same
  warm seeding). Triggered by `QUARK_MAX_AUTOTUNE=1`, the
  `with quark.max_autotune():` context manager, or the per-op
  `.autotune()` warmup API (`qf.gemm.autotune(A, B)`).
