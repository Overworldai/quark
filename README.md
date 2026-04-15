<p align="center">
  <img src="assets/popcorn.jpg" alt="popcorn" width="360"/>
</p>

<h1 align="center">popcorn</h1>

<p align="center">
  GPU kernel compiler. Typed IR → PTX (CUDA) / MSL (Metal) → GPU binary.
</p>

---

**Zero hardware-specific runtime deps.** Torch + MLX are the only
backends. `libcuda` is driven directly via ctypes (`runtime/cuda.py`);
no cupy, no pycuda, no triton. Kernels are written once and lower to
both PTX (NVIDIA) and MSL (Apple).

## Install

```bash
git clone https://github.com/carsonpo/popcorn
cd popcorn
make setup              # creates .venv, installs, wires pre-commit hook
```

`make setup` requires `uv`. The `.venv` is pinned to Python 3.13.

## Run

```bash
make test               # unit + smoke (<1s; auto-skips CUDA on Apple)
make fuzz               # correctness sweep across every kernel
make bench              # perf vs backend-fast baselines
make autotune KERNEL=gemm
```

Tag-filtered sweeps match production model paths:

```bash
make bench TAG=metal-dense    # full dense FFN path on Metal
make bench TAG=metal-moe      # full MoE FFN path on Metal
make bench TAG=cuda-dense     # same on CUDA with fp8+shuffle
make bench TAG=cuda-moe
```

## Write a kernel

```python
# src/popcorn/kernels/my_kernel/kernel.py

@kernel(
    "my_kernel",
    spec=MySpec, config=MyConfig,
    problems=my_problems,        # bench / fuzz problem list
    baselines=my_baselines,      # reference implementations to beat
    reference=my_reference,      # correctness oracle
)
class MyKernel(Kernel):
    TENSORS: ClassVar = [
        TensorDecl("A", dtype=..., shape=...),
        TensorDecl("B", dtype=..., shape=...),
        TensorDecl("Out", dtype=..., shape=..., role="out"),
    ]

    def build(self) -> None:
        # Compose L1/L2 blocks; IR emission is implicit
        ...
```

Add the folder, the registry picks it up. Zero edits elsewhere. See
[docs/ADDING_A_KERNEL.md](docs/ADDING_A_KERNEL.md).

## Call a kernel from torch or MLX

```python
import popcorn.functional as pcf

C = pcf.gemm(A, B)                      # torch.Tensor or mx.array; dispatched by type
y = pcf.attention(Q, K, V_t, B=..., n_kv_heads=..., gqa_ratio=..., seq_len=..., kv_len=...)
```

Every production kernel is a registered `torch.library.custom_op`
with a fake fn, so `torch.compile(fullgraph=True)` traces through
cleanly. Full surface in [docs/FUNCTIONAL.md](docs/FUNCTIONAL.md).

## Examples

Walk-through from vector-add → flash attention, one file per step,
each building on the previous:

| Step | File | Introduces |
|---|---|---|
| 1 | [examples/01-vector-add.md](examples/01-vector-add.md) | `@kernel`, `TENSORS`, `build()`, scalar load/store |
| 2 | [examples/02-block-reduction.md](examples/02-block-reduction.md) | `SmemTile`, cooperative load, barriers |
| 3 | [examples/03-softmax.md](examples/03-softmax.md) | Row-wise reductions, two-pass compute, `pop.store_acc` |
| 4 | [examples/04-gemm-basic.md](examples/04-gemm-basic.md) | `SmemPlan`, `Accumulators`, `MmaBody`, `PipelineBody.run` |
| 5 | [examples/05-gemm-pipelined.md](examples/05-gemm-pipelined.md) | `n_stages=2` double buffer, cp.async |
| 6 | [examples/06-flash-attention.md](examples/06-flash-attention.md) | `Carry`, online softmax, `pop.q_register_load`, `pop.store_acc(per_warp=, row_scale=)` |

## Docs

| File | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, IR, lowerers, registry, launcher |
| [docs/ADDING_A_KERNEL.md](docs/ADDING_A_KERNEL.md) | New kernel walkthrough with templates |
| [docs/FUNCTIONAL.md](docs/FUNCTIONAL.md) | `popcorn.functional` torch/MLX entry point |
| [docs/IR.md](docs/IR.md) | Builder API, tensor types, fragment primitives, RegisterTile |
| [docs/BLOCKS.md](docs/BLOCKS.md) | L0 / L1 / L2 block surface, DSL primitives |
| [docs/BACKEND.md](docs/BACKEND.md) | Polymorphic tensor (PT), Metal vs CUDA dispatch |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | Timing protocol, tags, autotune cache |
| [docs/TESTING.md](docs/TESTING.md) | Test structure, correctness metric, fuzz |
| [docs/DEBUGGING.md](docs/DEBUGGING.md) | Symptom → recipe |
| [docs/CONVENTIONS.md](docs/CONVENTIONS.md) | Naming, style, forbidden patterns |
