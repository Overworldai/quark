<p align="center">
  <img src="assets/quark.png" alt="quark" width="360"/>
</p>

<h1 align="center">quark</h1>

<p align="center">
  GPU kernel compiler. Typed IR → PTX (CUDA) / MSL (Metal) → GPU binary.
</p>

---

**Zero hardware-specific runtime deps.** `libcuda` is driven directly
via ctypes (`runtime/cuda.py`); on Metal a metal-cpp + nanobind
shim (`drivers/_metal_dispatch`) drives the GPU directly — no torch,
no cupy, no pycuda, no triton, no MLX in the inference path. Device
tensors live in `QuarkTensor` (`runtime/tensor.py`) on both backends,
weights load from `.safetensors` without numpy or torch. Kernels are
written once and lower to both PTX (NVIDIA) and MSL (Apple). Torch
is an optional dev dep used by reference implementations and the
autotune correctness gate only.

## Install

```bash
git clone https://github.com/carsonpo/quark
cd quark
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
# src/quark/kernels/my_kernel/kernel.py

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

## Call a kernel

```python
import quark.functional as pcf
from quark.runtime.tensor import QuarkTensor

A = QuarkTensor.randn(M, K, dtype="bf16")
B = QuarkTensor.randn(N, K, dtype="bf16")
C = pcf.gemm(A, B)
y = pcf.attention(Q, K, V_t, B=..., n_kv_heads=..., gqa_ratio=..., seq_len=..., kv_len=...)
```

`QuarkTensor` is the unified device-tensor type across both backends;
the active backend is decided by the device the tensor lives on
(CUDA via the ctypes driver on Linux/Windows, Metal via metal-cpp +
nanobind on Apple Silicon). Full surface in
[docs/FUNCTIONAL.md](docs/FUNCTIONAL.md).

## Build a model

`quark.nn` is a PyTorch-shaped, inference-only module layer. No
autograd, no optimizer — just `nn.Module` / `nn.Parameter` holding
`QuarkTensor`s, with `state_dict()` / `load_state_dict()` and a
`forward()` convention. Every leaf module lowers to a `pcf.*` kernel.

```python
import quark.nn as nn
from quark.nn.io import load_safetensors, load_from_hub

sd = load_from_hub("Overworld/Waypoint-1.5-1B")   # or load_safetensors("model.safetensors")
model = MyModel(cfg)
model.load_state_dict(sd)
out = model(x)
```

The safetensors loader is pure Python — mmap + `cuMemHostRegister` →
pinned DMA straight to device, no host-side numpy / torch staging.
See [docs/WEIGHTS.md](docs/WEIGHTS.md).

## Waypoint-1.5 reference model

`quark.models.waypoint_15` is a 24-layer DiT built entirely from
`pcf.*` kernels. End-to-end benchmark on a 5090: **14.5 ms / NFE ≈
55 pixel FPS** at 32×16 = 512 tokens/frame, vs **26.4 ms / NFE** for
the bf16 torch `world_engine` baseline. See
[docs/WAYPOINT_15.md](docs/WAYPOINT_15.md).

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
| [docs/FUNCTIONAL.md](docs/FUNCTIONAL.md) | `quark.functional` call surface (QuarkTensor) |
| [docs/WEIGHTS.md](docs/WEIGHTS.md) | Safetensors loader, HF Hub path, `nn.Module` state dicts |
| [docs/WAYPOINT_15.md](docs/WAYPOINT_15.md) | Waypoint-1.5 reference model + 5090 / 4090 benchmarks |
| [docs/IR.md](docs/IR.md) | Builder API, tensor types, fragment primitives, RegisterTile |
| [docs/BLOCKS.md](docs/BLOCKS.md) | L0 / L1 / L2 block surface, DSL primitives |
| [docs/BACKEND.md](docs/BACKEND.md) | Polymorphic tensor (PT), Metal vs CUDA dispatch |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | Timing protocol, tags, autotune cache |
| [docs/TESTING.md](docs/TESTING.md) | Test structure, correctness metric, fuzz |
| [docs/DEBUGGING.md](docs/DEBUGGING.md) | Symptom → recipe |
| [docs/CONVENTIONS.md](docs/CONVENTIONS.md) | Naming, style, forbidden patterns |
