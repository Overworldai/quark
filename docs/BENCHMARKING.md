# Benchmarking

## Commands

```bash
make bench                              # all kernels, all problems
make bench KERNEL=gemm                  # one kernel
make bench KERNEL=gemm PROBLEM=owl_360p # one kernel + named problem
make bench TAG=metal-dense              # tag-filtered sweep
make bench BENCH_MS=100                 # longer timing budget
```

`KERNEL=`, `PROBLEM=`, `TAG=` compose — `KERNEL=gemm TAG=metal` runs
every metal-tagged gemm problem.

## Tags

Tag a problem in its `problems.py` entry; bench filters by exact
tag membership.

### Canonical scheme

| Tag | Meaning |
|---|---|
| `smoke` | CI-fast regression gate; at least one per kernel |
| `metal` | Metal production path (bf16 across the board) |
| `metal-dense` | Metal + dense-FFN model path |
| `metal-moe` | Metal + MoE-FFN model path |
| `cuda` | CUDA production path (bf16 A, e4m3 B preshuffled, e4m3 compute, bf16 out) |
| `cuda-dense` | CUDA + dense-FFN |
| `cuda-moe` | CUDA + MoE-FFN |
| `production` | Superset of real model workloads across platforms |

### Model → tag mapping

Production model: 32 Q heads × 16 KV heads × Dh 64. Per layer:

| Role | Shape | Dense | MoE |
|---|---|---|---|
| QKV proj | M × 2048 → 4096 | ✓ | ✓ |
| owl_attn | seg-sparse flash attn, 17 frames | ✓ | ✓ |
| kv_cache_update | K-RoPE + ring write | ✓ | ✓ |
| attn output proj | M × 2048 → 2048 | ✓ | ✓ |
| FFN up/gate | M × 2048 → 8192 | ✓ | |
| FFN down | M × 8192 → 2048 | ✓ | |
| moe_inproj | M·topk × 2048 → 2048 | | ✓ |
| moe_outproj | M·topk × 2048 → 2048 | | ✓ |

`make bench TAG=metal-dense` fires the exact per-layer kernels the
dense model runs on Apple silicon. `TAG=metal-moe` swaps FFN for MoE.

## Timing protocol

Three-phase async-queued (no sync between individual launches):

1. **Probe** — run+sync until ≥1ms, estimate per-iter runtime.
2. **Warmup** — async-queue `WARMUP_MS / per_iter` iters, sync once.
3. **Bench** — async-queue `BENCH_MS / per_iter` iters between two
   events / `mx.eval` boundaries, sync, report `elapsed / n_iters`.

Output: microseconds per launch. TFLOPS = `flops() / us / 1e6`.

Default budgets: `WARMUP_MS=10`, `BENCH_MS=50`. Override on the CLI.

Timing is backend-agnostic — it goes through
`quark.backend.time_callable`.

## Config resolution

Launch order:

1. Tuned JSON at `configs/{KERNEL}_{problem.name}.json`.
2. Fall back to `cls.CONFIG_CLS.default_for(spec)` (the `Kernel.from_problem` path).
3. `QUARK_FORCE_DEFAULT_CONFIG=1` skips the JSON lookup.

Bench output marks each row `[tuned]` or `[default]`.

## Autotune

```bash
make autotune                                  # every kernel × every problem
make autotune KERNEL=gemm
make autotune KERNEL=gemm PROBLEM=owl_360p
make autotune TAG=production
make autotune --no-save                        # dry run
```

Genetic search with parallel-thread compile over
`cls.tune_space_resolved(spec, device)`:

1. Start from `cls.tune_space()`, then for every `MmaSite` the
   kernel declares inject one `<site>_shape` knob with legal values
   drawn from `device.caps.matmul_shapes` intersected with the site's
   dtype axes. Kernels with no MMA sites get their `tune_space()`
   through unchanged.
2. Pin fields that aren't in the resolved space to `default_for(spec)`
   (or to `Problem.config_overrides` if set).
3. Enumerate candidates:
   - **small space** (cartesian product ≤ `max(1024, pop_size*8)`):
     exhaustive walk + `is_valid_for(device.caps)` filter.
   - **large space** (adding per-site shape knobs blows this up
     fast): random-sample combos until `pop_size*8` valid ones are
     collected or `pop_size*32` attempts exhausted. Prints the
     reduction ratio for visibility.
4. Search:
   - **dense** (n_valid ≤ pop_size): one exhaustive generation.
   - **genetic**: tournament selection, uniform crossover, Gaussian
     index mutation; `--gens` generations with `--early-stop` on no
     improvement.
5. Per-generation: parallel-thread compile (PTX JIT releases the GIL
   on CUDA; Metal compiles serially but fast) → correctness check →
   time → record.
6. Save the winner to `configs/{KERNEL}_{problem.name}.json`.

Re-autotune after any kernel-emit refactor. The old JSON is likely
stale; perf comparisons vs. stale configs lie.

CLI knobs: `--pop 64 --gens 16 --mutate-prob 0.2 --early-stop 3
--seed 0 --max-workers N`.

`QUARK_CONFIGS_DIR` overrides the save path (default
`<repo>/configs/`).

Saved JSON:

```json
{
  "kernel": "gemm",
  "problem": {"M": 1024, "N": 1024, "K": 1024, "a_dtype": "bf16", ...},
  "problem_name": "1k_bf16",
  "config": {
    "BM": 64, "BN": 64, "BK": 16,
    "n_warps": 4, "n_stages": 2,
    "a_pad": 8, "b_pad": 0,
    "main_shape": "m16n8k16_bf16"
  },
  "runtime_us": 12.34
}
```

`main_shape` (and any other `<site>_shape` fields) identifies the
MMA descriptor the autotuner picked for that call site. An empty
string means the kernel resolved to its compute-dtype default.

## Output

Per-kernel table with columns:

```
┌────────────────┬─────────┬────────────┬──────────┬────────────┬────────────────────────────────┐
│ Problem        │ Config  │  Time (μs) │   TFLOPS │    cos_sim │             Status             │
├────────────────┼─────────┼────────────┼──────────┼────────────┼────────────────────────────────┤
│ owl_360p       │  tuned  │      784.1 │    10.65 │   0.99342  │ ok                             │
```

- `Time (μs)` — average per-launch time.
- `TFLOPS` — `flops / time`.
- `cos_sim` — single-shot correctness check against `reference()`.
- `Status` — `ok` / `ERR:<type>: <message>` / `slow` warnings.

Baselines appear as additional rows labelled with the baseline name
(`mx.fast.sdpa[bf16]`, `torch._scaled_mm[e4m3]`, etc.).
