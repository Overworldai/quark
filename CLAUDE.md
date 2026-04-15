# popcorn

GPU kernel compiler. Typed IR → PTX (CUDA) / MSL (Metal) → GPU binary.
Torch + MLX are the only backends; every other op goes through
`popcorn.backend.PT`.

## Orientation

```python
import popcorn.lang as pop                       # pop.add/mul/..., pop.store_acc, pop.silu, pop.cast,
                                                 # pop.work_list_load, pop.index_cache, pop.q_register_load
from popcorn.backend import PT
from popcorn.ir import Builder, DType            # DType is str+IR+backend bridge (DType.BF16.backend → mx/torch dtype)
from popcorn.blocks import (
    Accumulators, SmemTile, SmemTileSpec, Stage, Carry, TensorDecl,
    MmaBody, SmemPlan,                           # MmaBody(acc=...), SmemPlan.staged_pairs
    PipelineBody, IterCtx, run_pipeline,         # PipelineBody(...).run(n_iters=..., n_stages=...)
)
from popcorn.kernels.base import Kernel, MmaSite
from popcorn.kernels.decorator import kernel
```

Kernel `build()` bodies open with `s, c, g = self.spec, self.config,
self.g`; `self.bctx` / `self.m_base` / `self.n_base` are auto-bound
by the decorator. Tile loads go through
`smem_tile.load_from(gmem_tensor, row=, col=, cast=)` (or
`smem_tile.gather_from(gmem, index=, col=)` for indirect loads).
Epilogues use `pop.store_acc(dst, acc, row=, col=, cast=, activation=)`
or `pop.atomic_store_acc(...)`. Stages are built via
`Stage.staged(n, **SmemTileSpec(...))`; loop carry is a `Carry(**slots)`
with attribute access (`ictx.carry.o`, `carry.l`, …).

Most of the MMA boilerplate is now implicit: `MmaBody(acc=acc)` infers
`shape` from the active BlockContext's `mma_cfg` and `K_inner` from
the B tile's shape at `__call__` time; `PipelineBody(...).run(n_iters,
n_stages)` replaces the separate `body = PipelineBody(...)` +
`run_pipeline(body, ...)` pair. Accumulator lists come from
`Accumulators(MT=..., NT=..., width=...).init()` (no Builder arg).

## Docs

| File | Contents |
|---|---|
| [README.md](README.md) | Install, first run, doc index |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, IR, lowerers, registry, launcher |
| [docs/ADDING_A_KERNEL.md](docs/ADDING_A_KERNEL.md) | Kernel folder template + contract |
| [docs/IR.md](docs/IR.md) | Builder API, tensor types, fragment primitives, RegisterTile |
| [docs/BLOCKS.md](docs/BLOCKS.md) | L0 / L1 / L2 block surface, DSL primitives |
| [docs/BACKEND.md](docs/BACKEND.md) | `PT` polymorphic tensor, Metal vs CUDA dispatch |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | Timing, tags, autotune cache |
| [docs/FUNCTIONAL.md](docs/FUNCTIONAL.md) | `popcorn.functional` torch/MLX entry point |
| [docs/TESTING.md](docs/TESTING.md) | Test structure, correctness metric |
| [docs/DEBUGGING.md](docs/DEBUGGING.md) | Symptom → recipe |
| [docs/CONVENTIONS.md](docs/CONVENTIONS.md) | Naming, style, forbidden patterns |

## Commands

```bash
make setup                             # onboarding (.venv + install + hooks)
make test                              # unit + smoke (<1s)
make fuzz [KERNEL=name]                # correctness sweep
make fuzz TAG=smoke
make bench [KERNEL=name]               # perf vs baselines
make bench TAG=metal-dense             # production model path, Metal
make bench TAG=cuda-moe                # production model path, CUDA fp8+shuffle
make autotune [KERNEL=name]            # save best configs
make dump-ptx KERNEL=name              # print PTX
```

## Layout

```
src/popcorn/
    backend.py            PT (polymorphic tensor) — every torch-vs-mlx branch
    correctness.py        cos_sim check (accepts raw PT tensors)
    autotune.py           3-level cache (hot / disk / bundled)
    device.py             Device, DeviceCaps, ChipGeneration (typed per-chip key)
    lang/                 popcorn.lang — free-function authoring surface
        __init__.py       pop.add / pop.mul / pop.for_range / pop.kernel_scope / …
                          (for_range auto-lifts Python ints; yield_ auto-flattens a Carry)
        epilogue.py       pop.store_acc / pop.atomic_store_acc / pop.silu / pop.cast
        memory.py         pop.work_list_load / pop.index_cache / pop.q_register_load
    ir/                   Builder, Value, Op, Tensor, Module, validator
        types.py          DType (str+IR+backend), MemSpace, ValueShape, ScalarType
        frag_tile.py      FragLayout + RegisterTile + per-backend lane maps
        lifetime.py       SharedRegion lifetime kinds
        mma_registry.py   MmaConfig descriptors + shapes_for_chip(gen)
                          (single source of truth; replaces gemm/mma_shapes)
    blocks/
        dsl/              blocks/dsl — DSL framework, split by concern
            __init__.py        re-exports + module-level helpers (c, barrier, tid, …)
            context.py         ContextVars + active_kctx / active_bctx
            tensors.py         TensorDecl + C + _to_value
            accumulators.py    Accumulators (.init / .from_mma)
            carry.py           Stage + Carry (attribute-access loop carry)
            smem_tile.py       SmemTile + SmemTileSpec (.load_from / .gather_from / .spec)
            block_context.py   BlockContext + Block + SetupBlock
            kernel_context.py  KernelContext
        l0/               atomic emit functions (tile_loader, gathered_tile_loader,
                          index_cache, epilogue, async_copy, mma_tile, smem_base,
                          shuffled_bfrag_load, cooperative_load)
        l1/               leaf Block classes (MmaBody — callable as mma(ictx) or
                          mma(a=, b=, acc=) Triton-style)
        l2/               composition: run_pipeline + PipelineBody (with .run(n_iters=,
                          n_stages=) method), SmemPlan.paired / .staged_pairs
    lower/
        ptx/              PtxLowerer → PTX text
        msl/              MslLowerer → MSL source
        smem_layout.py    Lifetime-aware smem coloring / aliasing pass
    kernels/
        base.py           Kernel, KernelSpec, KernelConfig, MmaSite, Problem, Baseline
        decorator.py      @kernel(…, problems=, baselines=, reference=)
        registry.py       @register + get / all_kernels / names
        gemm/             attn/ owl_attn/ kv_cache_update/ moe_inproj/ moe_outproj/
    functional/           popcorn.functional — torch/MLX callables wrapping the
                          registered kernels (torch.library.custom_op + fake fns
                          on CUDA; eager dispatch on MLX); offline shuffle_b_for_*
                          helpers for b_shuffle=True fast paths
    launcher/             Launcher, CompiledKernel, ParamSpec
    drivers/              cuda.py (ctypes), mlx.py
    runtime/              libcuda ctypes binding
    weight_shuffle.py     offline B-preshuffle for fp8+shuffle GEMM fast path

tests/                    ir/ lower/ launcher/ kernels/ (smoke auto-discovers)
tools/                    bench.py fuzz.py autotune.py profile_metal.py
configs/                  autotune JSON (gitignored)
```
