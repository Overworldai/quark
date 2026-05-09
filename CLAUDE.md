# quark

GPU kernel compiler. Typed IR → PTX (CUDA) / MSL (Metal) /
SPIR-V (Intel Vulkan) → GPU binary. Runtime inference path uses
`QuarkTensor` (ctypes → libcuda) on CUDA and `mx.array` (MLX) on
Metal; the SPIR-V backend dispatches via Vulkan compute through
the `_spv_dispatch` C extension and accepts numpy ndarrays
end-to-end (QuarkTensor-on-SPV pool is pending). No torch
dependency in the hot path. `quark.backend.PT` still exists as
the polymorphic wrapper for kernel `reference.py` / `baselines.
py` (torch optional dev extra).

Backends: see `docs/PORTABILITY_PLAN.md` for the SPIR-V backend
roadmap. v1 (smoke pass on Battlemage) and v3 (real flash-
attention on SPIR-V) are both landed on the `spirv-integration`
branch — every framework kernel that has a smoke fixture
(rmsnorm, gemm, attn, owl_attn, moe_*, value_residual, …)
runs correctly on Intel Battlemage at the default config.

## Orientation

```python
import quark.lang as qk                       # qk.add/mul/..., qk.store_acc, qk.silu, qk.cast,
                                                 # qk.work_list_load, qk.index_cache, qk.q_register_load
from quark.backend import PT
from quark.ir import Builder, DType            # DType is str+IR+backend bridge (DType.BF16.backend → mx/torch dtype)
from quark.blocks import (
    Accumulators, SmemTile, SmemTileSpec, Stage, Carry, TensorDecl,
    MmaBody, SmemPlan,                           # MmaBody(acc=...), SmemPlan.staged_pairs
    PipelineBody, IterCtx, run_pipeline,         # PipelineBody(...).run(n_iters=..., n_stages=...)
)
from quark.kernels.base import Kernel, MmaSite
from quark.kernels.decorator import kernel
```

Kernel `build()` bodies open with `s, c, g = self.spec, self.config,
self.g`; `self.bctx` / `self.m_base` / `self.n_base` are auto-bound
by the decorator. Tile loads go through
`smem_tile.load_from(gmem_tensor, row=, col=, cast=)` (or
`smem_tile.gather_from(gmem, index=, col=)` for indirect loads).
Epilogues use `qk.store_acc(dst, acc, row=, col=, cast=, activation=)`
or `qk.atomic_store_acc(...)`. Stages are built via
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
| [docs/WEIGHTS.md](docs/WEIGHTS.md) | Safetensors loader, HF Hub, `nn.Module` state dict |
| [docs/WAYPOINT_15.md](docs/WAYPOINT_15.md) | Waypoint-1.5 model + 5090 / 4090 benchmarks |
| [docs/FUNCTIONAL.md](docs/FUNCTIONAL.md) | `quark.functional` call surface (QuarkTensor / MLX) |
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
src/quark/
    backend.py            PT (polymorphic tensor) — every torch-vs-mlx branch
    correctness.py        cos_sim check (accepts raw PT tensors)
    autotune.py           3-level cache (hot / disk / bundled)
    device.py             Device, DeviceCaps, ChipGeneration (typed per-chip key)
    lang/                 quark.lang — free-function authoring surface
        __init__.py       qk.add / qk.mul / qk.for_range / qk.kernel_scope / …
                          (for_range auto-lifts Python ints; yield_ auto-flattens a Carry)
        epilogue.py       qk.store_acc / qk.atomic_store_acc / qk.silu / qk.cast
        memory.py         qk.work_list_load / qk.index_cache / qk.q_register_load
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
    functional/           quark.functional — per-kernel callables; dispatch
                          by input type (QuarkTensor → CUDA, mx.array → MLX).
                          Offline shuffle_b_for_* helpers for b_shuffle=True
                          fast paths
    nn/                   inference-only Module / Parameter / ModuleList;
                          io.py = pure-python safetensors loader (mmap + pinned
                          DMA); layers.py = Linear / MLP / OwlAttn / KVCacheUpdate / …
    models/               waypoint_15.py — 24-layer DiT reference model
    runtime/              QuarkTensor (tensor.py), libcuda ctypes binding
                          (cuda.py), on-device PTX utility kernels (kernels.py)
    launcher/             Launcher, CompiledKernel, ParamSpec
    drivers/              cuda.py (ctypes), mlx.py
    weight_shuffle.py     offline B-preshuffle for fp8+shuffle GEMM fast path

tests/                    ir/ lower/ launcher/ kernels/ (smoke auto-discovers)
tools/                    bench.py fuzz.py autotune.py profile_metal.py
configs/                  autotune JSON (gitignored)
```
