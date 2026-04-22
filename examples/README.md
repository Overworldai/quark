# examples — from vector add to flash attention

A progression that walks you through quark's abstractions, one step
at a time. Each example is a minimal, working kernel; each file
introduces one or two new ideas on top of the last.

| Step | File | Introduces |
|---|---|---|
| 1 | [01-vector-add.md](01-vector-add.md) | `@kernel`, `TENSORS`, `build()`, scalar load/store |
| 2 | [02-block-reduction.md](02-block-reduction.md) | `SmemTile`, cooperative load, barriers, subgroup reductions |
| 3 | [03-softmax.md](03-softmax.md) | Two-pass compute, `SmemTile.load_from`, `pop.for_range` |
| 4 | [04-gemm-basic.md](04-gemm-basic.md) | `SmemPlan`, `Accumulators`, `MmaBody`, `PipelineBody.run` |
| 5 | [05-gemm-pipelined.md](05-gemm-pipelined.md) | `n_stages=2` double buffer, `produce` / `consume`, cp.async |
| 6 | [06-flash-attention.md](06-flash-attention.md) | `Carry`, online softmax, `pop.q_register_load`, `MmaBody(a=, b=)` Triton-style |

The examples aren't in the `@kernel` registry (they're teaching
material); the full registered kernels live in
[`src/quark/kernels/`](../src/quark/kernels/) and use the same
patterns. Once you've read through the examples, look at
`kernels/gemm/kernel.py` — it's almost exactly example 5 with a
production-grade config / baselines / reference attached.

Each example has the same shape:

1. **Goal** — one-sentence problem statement
2. **New ideas** — the abstractions introduced
3. **Spec / Config** — problem definition + tuning knobs
4. **build()** — the kernel body
5. **Recap** — what got added to the toolkit

Reference material (deeper docs) is linked from each page.
