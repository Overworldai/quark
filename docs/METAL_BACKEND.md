# Metal Backend — Status, Architecture, and Roadmap

This doc consolidates what was previously split across `MLX_REMOVAL_PLAN.md`,
`METAL_DISPATCH_ARCHITECTURE.md`, and `METAL_KERNEL_OPTIMIZATION_PLAN.md`. It
covers the implementation that's landed, the design behind it, and the
remaining perf work.

---

## Current Status

### MLX Removal — Complete

`drivers/mlx.py` deleted, `pyproject.toml` no longer depends on `mlx`.
Replaced with:

- `drivers/metal.py` + `_metal_dispatch.cpp` + `_metal_cpp/` — a
  metal-cpp + nanobind native driver
- `quark.lazy()` context manager in
  [`quark/__init__.py`](../src/quark/__init__.py) — lazy queue dispatch
  with auto-commit. Supports `sync=False` for fire-and-forget commits
  whose output is consumed only by later GPU work.
- `runtime/tensor.QuarkTensor` — the unified GPU tensor type used by
  both Metal and CUDA backends (replaces the earlier `MetalArray`
  numpy subclass that was retired in commit `730a822`).
- `drivers/metal_harness.py` — signature/binding-plan generator

### Backend Dispatch Infrastructure

- **`@kernel` decorator** ([decorator.py](../src/quark/kernels/decorator.py)):
  kernels can define `build_metal` / `build_cuda` / etc. that override the
  default `build`. Decorator picks based on `current_device().caps.family`.
  ~10 LOC change, scales cleanly.
- **`Kernel.bind_caps(caps)`** ([base.py](../src/quark/kernels/base.py)):
  explicit caps pinning for tests / autotune.
- **`self.caps` accessor** inside any `build*` method — kernels can branch
  finer-grained (e.g. NAX availability) at IR build time.

### Lowerer Vectorization (MSL)

[visitors.py](../src/quark/lower/msl/visitors.py):

- `vec_load` / `vec_store` same-dtype path emits packed `T2/T3/T4`
  reinterpret_cast (was N scalar reads).
- `async_copy` synchronous fallback emits register-routed `T4`
  reinterpret_cast pairs (was N scalar elementwise).
- `for_loop` / `yield` extended to support multi-component vec carries
  (per-component locals + per-component yield assignments) — required
  for NAX accumulators flowing through K-loops.

[lower.py](../src/quark/lower/msl/lower.py):

- `uses_nax` ctx flag with conditional
  `<MetalPerformancePrimitives/...>` header injection.
- NAX preamble (lane Coord + descriptor + op handle) hoisted to
  function-entry scope.

### NAX MMA Codegen

[mma.py](../src/quark/lower/msl/mma.py) and
[mma_registry.py](../src/quark/ir/mma_registry.py):

- **`m16n32k16_nax_bf16`** shape registered, gated on
  `min_metal_gen=METAL_M5`, `metal="nax:1:2:1"` payload.
- `_visit_mma_nax` — emits `_nax_op.run(ct_a, ct_b, ct_c)` with
  cooperative-tensor allocation / copy / drain.
- `_visit_load_matrix_nax` — per-lane scalar reads with BaseNAXFrag
  layout (`fm = (qid&4) | ((lane>>1)&3)`,
  `fn = ((qid&2) | (lane&1)) * 4`).
- `_visit_store_matrix_nax` — mirror per-lane writes.
- TM × TN generalized: any `(a_regs=8, b_regs=16, c_regs=16)` NAX shape
  works.

### Kernel Rewrites — All on the IR Now

| kernel | what changed |
|---|---|
| **silu** | rewritten with `vec_load`/`vec_store`; x-axis grid for parity with `fast_silu` |
| **rmsnorm** | `build_metal` — warp-per-row register-stash + multi-sg cross-warp reduction (smem `tg_sums[n_warps]` + simd_sum broadcast) |
| **ada_rmsnorm** | `build_metal` — same multi-sg pattern with `(1+scale)·y + bias` epilogue |
| **ada_gate_residual** | `build_metal` — flat warp-per-row `vec_load → fma → vec_store` |
| **gemm** | `build_metal` — NAX single-warp + multi-warp paths (12 tile configs), inner-K unroll, load front-loading, NAX-aware `is_valid` |

### Functional Layer Wired to IR

[`pcf.silu`](../src/quark/functional/silu.py),
[`pcf.rmsnorm`](../src/quark/functional/rmsnorm.py),
[`pcf.ada_rmsnorm`](../src/quark/functional/ada_rmsnorm.py),
[`pcf.ada_gate_residual`](../src/quark/functional/ada_gate_residual.py)
all route through `queue_launch_ir`
([_dispatch.py](../src/quark/functional/_dispatch.py)) — pull pipeline
+ binding plan from the launcher-compiled IR module, dispatch via
`_md.queue_launch`, return `QuarkTensor`. Bypasses the per-call
`alloc_from_decl` overhead. `pcf.gemm` does the equivalent inline in
`_try_nax_gemm`, also through `launcher().compile(GemmKernel, ...)`.

### Override Layer Retired

Deleted: `fast_silu.py`, `fast_rmsnorm.py`, `fast_ada_rmsnorm.py`,
`fast_ada_gate_residual.py`, `fast_sigmoid.py`, `nax_gemm.py` — IR is
at parity for all of these. NAX GEMM now goes through
`GemmKernel.build_metal` via `launcher().compile`.

### Goldens Regenerated

9 `.msl` golden tests updated for the new packed `T4` reinterpret_cast
emission ([tests/lower/golden/msl/](../tests/lower/golden/msl/)).

---

## Performance

### Per-kernel microbench (M5 Max, bf16)

| kernel | IR | hand-written (deleted) | parity |
|---|---|---|---|
| silu (1M) | 246 µs | 200 µs | ~80% (silu math vec gap) |
| rmsnorm (4096×2048) | 411 µs | 399 µs | 97% |
| ada_rmsnorm | 417 µs | 403 µs | 97% |
| ada_gate_residual | 667 µs | 655 µs | 98% |
| gemm 2048³ | **26.36 TF** | 26.31 TF | **101% (slightly beats)** |

### Waypoint-1.5 forward pass (24 layers, 360p)

| | median |
|---|---|
| **quark lazy** | **10.69 ms** (1.46× MLX) |
| quark eager | 11.15 ms |
| MLX baseline | 15.56 ms |

**This is an honest number.** The bench runs real flash attention
(`pcf.owl_attn` via the IR-emitted NAX kernel at
`kernels/owl_attn/nax.py`; the standalone MSL kernel in
`drivers/nax_attn.py` was retired in commit `32771c0`) with
BK=128 KV chunks, online softmax, segment-based KV iteration,
and GQA support. `kv_cache_update` is excluded (cache primed once
at warmup, frozen for timed passes — same as MLX's bench pattern).

GPU time decomposition (from profiling):

| Component | Time | Share |
|---|---|---|
| GEMM (4 per layer × 24) | ~6.3 ms | 59% |
| Attention (24 calls, BK=128) | ~1.7 ms | 16% |
| Norms/silu/gate_residual | ~0.5 ms | 5% |
| Python dispatch + eval overhead | ~2.2 ms | 20% |

The forward is GPU-bound: GEMMs run at NAX hardware throughput
(~16 TF/s). Python dispatch overlaps with GPU execution via
auto-commit at 50 ops. Zero `to_numpy` pipeline breaks in the
timed region.

### Why the 3.21× gap to MLX exists

The fc1 GEMM (M=128, N=8192, K=2048) is the single biggest contributor.
At that shape:

| | TFLOPS | µs |
|---|---|---|
| IR NAX (any tile config 64×128 to 128×256) | 6.18 | 695 |
| **MLX** | **16.49** | **260** |

The IR matched the (now-deleted) hand-written kernel — both at the **same**
~6 TFLOPS structural ceiling. We confirmed via 12-config tile sweep
(BM=64/128, BN=128/256, BK=16/32/64) that this isn't a tile-config
problem. It's also not bandwidth-bound (BW lower bound is ~100 µs).
The kernel is **memory-load-latency stall-bound**: direct gmem reads,
no overlap between load and MMA.

MLX uses **smem-staged double-buffered async loads**: tile k+1 is
prefetched into the alternate smem buffer in parallel with the MMA on
tile k, hiding load latency behind compute. That's the 2.6×.

### Root cause of the MLX gap: memcpy overhead in benchmark

The ~2.7× gap between our IR NAX GEMM (~6 TF) and MLX (~16.5 TF)
at fc1 shape (128×8192×2048) was a **benchmark artifact**. The bench
passed raw numpy arrays which triggered a 33.5 MB `memcpy` into
Metal-owned buffers **on every call** (input cache is cleared after
each `eval_queue`). MLX's inputs are already in Metal buffers (zero
copy at dispatch time).

With `LazyBuffer` handles (data already in Metal-owned pool buffers):

| | fc1 (128×8192×2048) |
|---|---|
| IR NAX (numpy inputs, memcpy) | 693 µs / 6.20 TF |
| **IR NAX (LazyBuffer, zero-copy)** | **267 µs / 16.07 TF** |
| MLX native | 260 µs / 16.49 TF |

**At parity.** The kernel itself was always as fast as MLX. The entire
"6 TF ceiling" was memcpy overhead that obscured the real kernel perf.

The production path (`pcf.gemm` with `LazyBuffer` inputs from the
world_engine forward) doesn't hit this because weights and activations
flow through the lazy dispatch queue as `LazyBuffer` handles.

### What was done right (not wasted)

Several improvements landed during the investigation that ARE genuine
wins regardless of the bench bug:

- **Vec4 reinterpret_cast loads/stores** in NAX LoadMatrix/StoreMatrix
  — explicit contiguous access (verified identical perf to MLX's
  ``BaseNAXFrag::load`` with ``Int<1>{}``; no external dependency).
- **In-place accumulator drain** — MMA writes ct_c results back into
  C operand's names; yield copies eliminated (875 → 747 MSL lines).
- **`setFastMathEnabled(false)`** — matches MLX's compile options.
- **`dispatchThreadgroups`** when grid is exact — matches MLX.
- **`ResourceHazardTrackingModeUntracked`** — matches MLX's buffer
  allocation, avoids per-dispatch hazard-tracking overhead.
- **silu `exp_approx`** — `metal::fast::exp(-x)` instead of
  `metal::fast::exp2(-x * log2e)` saves one mul per element
  (246 → 199 µs).

### Experiments that confirmed no difference (for reference)

All tested at the kernel level (not the bench level), confirmed to
produce identical TFLOPS when the memcpy artifact is removed:

- smem staging (single and double-buffered) — adds barrier cost,
  no benefit since Apple's L2 handles data reuse for direct loads.
- Runtime inner-K loop vs Python-unroll — identical kernel perf.
- `float[]` arrays / `vec<float,8>` / scalar accumulators — Apple's
  compiler generates the same code for all three.
- BK=128/256 with MLX templates — larger body doesn't help (the
  "perf wall" was memcpy, not body-size optimization).
- 16 simdgroups (WM=4 WN=4) — no improvement over 8.
- Pre-compiled metallib vs runtime `newLibraryWithSource` — identical
  perf (confirmed with `xcrun metal -O2`).
- Swizzled block indexing (swizzle_log=2) — no measurable difference
  at these shapes.

### Test status

**439/441** pass. 2 deselected pre-existing failures (`make_tensors`
AttributeError on `owl_attn` / `moe` / `kv_cache` smoke tests —
unrelated to this work).

### Files diff summary (since `7e20101 test: mlx removal spike`)

- **modified**: 78 files, +2698/−2684 lines (most of the diff is
  reorganization — new IR/lowerer code replacing deleted MLX code)
- **deleted**: 5 `fast_*.py` overrides + 5 `spike/` files +
  `drivers/mlx.py`
- **new**: `drivers/metal.py`, `_metal_dispatch.cpp`, `_metal_cpp/`,
  `metal_harness.py`, `setup.py`, this doc

---

## Architecture

### Why we removed MLX

Quark's Metal path used MLX for exactly one thing: turning an MSL
source string into an executable compute kernel via `mx.fast.metal_kernel`.
Everything else (the lowerer, the launcher, the kernel library) was
already ours. MLX was effectively a 700 MB wheel to call
`newLibraryWithSource:` — an Apple-framework method that ships with
every Mac.

The CUDA path went through the same cleanup: PyTorch was removed from
the lowering / dispatch path by binding directly to `libcuda.so`.
Removing MLX from the Metal side mirrors that.

Specific blockers in MLX that motivated the switch:

1. **`mx.fast.metal_kernel` cannot include MPP headers.** MLX prepends
   `metal::utils()` to user source, opening a namespace scope; MPP
   header inclusion inside a non-global scope fails with `namespaces
   can only be defined in global or namespace scope`. We need MPP for
   NAX `matmul2d`.
2. **`mx.fast.metal_kernel` outputs are forced fresh allocations.**
   `array::make_arrays(...)` allocates a new buffer every call.
   `kv_cache_update` (writes into existing `K_cache`/`Vt_cache`) and
   any graph-capture flow (needs stable buffer addresses) require
   pre-allocated outputs.

Both are MLX-internal limitations. Neither requires Xcode to fix from
our side — Apple's `newLibraryWithSource:` runtime compile API ships
with every Metal-capable Mac. Confirmed empirically on a machine
without `/Library/Developer/CommandLineTools/`.

### Lazy queue dispatch (the key dispatch-side architecture)

Every `_md.queue_launch` does the **minimum work needed to return a
buffer pointer** and queues the actual encoding:

1. Allocate output `MTLBuffer` from the pool (so a numpy / `LazyBuffer`
   view can point at its unified-memory storage).
2. Record a `QueuedOp` in the C-side queue — pipeline state pointer,
   input/output buffer references, binding plan, grid/threadgroup,
   smem.
3. Return `(handle, ptr)` to Python.

**No `setBuffer` / `dispatchThreads` happens at queue time.**

`_md.eval_queue` (or auto-commit on threshold) walks the queue in a
single C++ tight loop, encodes everything, commits, and waits **once**
at the end. CPU/GPU overlap: while the GPU executes batch N, the CPU
encodes batch N+1.

This mirrors MLX's `eval_impl` pattern. We don't replicate MLX's full
lazy graph (graph nodes stay in Python, costing ~3-6 µs per launch in
arg marshalling) — but the FFI cost matches: ≤ 1 µs gap at small N,
zero at N ≥ 256K.

### Backend dispatch (the kernel-side architecture)

`@kernel` decorator's generated `emit()` picks `build_<family>` over
`build` based on `current_device().caps.family`. CUDA paths stay
untouched; Metal-specific kernels add a `build_metal` method that
shares TENSORS / spec / config / tune_space with the default build but
emits Metal-flavored IR (e.g. register-stash instead of cp.async
pipeline, NAX MMA shape selection).

### NAX MMA codegen pattern

Apple's MetalPerformancePrimitives (MPP) expose a per-simdgroup
hardware matmul accelerator on M5+. Following MLX's `steel/gemm/nax.h`
pattern (which `GemmKernel.build_metal` emits):

1. `matmul2d_descriptor(M, N, K, ...)` — compile-time descriptor
   defining one MMA's shape (16×32×16 for our registered NAX shape).
2. `mpp::tensor_ops::matmul2d<desc, execution_simdgroup> gemm_op` —
   per-kernel handle.
3. Cooperative tensors `ct_a`/`ct_b`/`ct_c` populated from per-lane
   `vec<T, N>` register fragments.
4. `gemm_op.run(ct_a, ct_b, ct_c)` — dispatches the MMA on NAX
   hardware.
5. Per-lane fragment loading uses the `BaseNAXFrag` lane→(fm, fn)
   coordinate map.

The IR emits this pattern via three visitors:

- `_visit_mma_nax` — operand stitch + run + drain
- `_visit_load_matrix_nax` — per-lane scalar reads (gmem or smem)
- `_visit_store_matrix_nax` — per-lane scalar writes

Plus a one-time per-function preamble emitting the descriptor, op
handle, and Coord setup. Header injection of MPP happens via a
`uses_nax` ctx flag.

**Critically: Apple's MPP runs on the legacy
`MTLComputeCommandEncoder` — no `MTL4ComputeCommandEncoder`,
`MTLTensor`, or `setTensor:atIndex:` needed.** This means we don't
have to migrate the dispatch path to MTL4; just compile at
`MTLLanguageVersion4_0` when `supportsFamily_(MTLGPUFamilyMetal4)`.

---

## Remaining Work

### High-value (perf)

1. **GEMM NAX at MLX parity** ✅ — kernel matches MLX at fc1
   (~16 TF on both); remaining gap is dispatch overhead.
2. **Zero-copy weight pinning** ✅ — `bench_waypoint_forward.py`
   now pins all weights through `QuarkTensor.from_numpy` (Metal-owned
   pool buffers); GEMM inputs go zero-copy via `metal_handle`.
3. **End-to-end forward at 1.57× MLX** ✅ — 10.18 ms vs MLX 15.58 ms
   on Waypoint-1.5 360p forward (24 layers, 128 tokens, bf16).
4. **`owl_attn` Metal path** — the bench currently substitutes
   attention with a GEMM stand-in. See "Medium-value" below for the
   actual blocker — investigation revealed it's *two layers deep*
   (IR validators reject NAX offsets, AND `OnlineSoftmaxBlock`'s
   GEMM2 K-step formula assumes `shape_k >= shape_n`). Closing it
   needs ~2 days of focused work on `online_softmax_block.py`.
   The stand-in is *heavier* (one 128×2048×2048 GEMM/layer ≈ 128
   GFLOP) than the real attention (Q[128, 64] × Kᵀ[64, 2176] +
   softmax + P × V[2176, 64] ≈ 36 GFLOP + softmax), so the reported
   forward time is an upper bound for the real forward.
5. **silu `exp_approx`** ✅ — done (246 → 199 µs).

### Medium-value

3. **Bias / activation / split-K** wiring inside the NAX `build_metal`.
   Currently falls through to the simdgroup_matrix path when those
   are set. Means the NAX path doesn't get fused-bias / fused-silu
   epilogue, costing one extra kernel pass per layer.
4. **`pcf.gemm` IR-only fast path** ✅ — done. `_try_nax_gemm` now
   compiles `GemmKernel` through `launcher().compile` and dispatches
   via `_md.queue_launch` directly (the same lazy path as
   `queue_launch_ir`). `drivers/nax_gemm.py` deleted; the IR
   `GemmKernel.build_metal` is the single source for NAX GEMM MSL.
5. **`owl_attn` Metal NAX path** — see the dedicated section below.

### Low-priority / known limitations

- **`is_valid_for(caps)` smem check** rejects NAX configs with
  BK > 16 on the simdgroup_matrix accounting basis, even though
  `build_metal` doesn't allocate smem. Needs a NAX-aware smem
  estimator.
- **MOE kernels (`moe_inproj`, `moe_outproj`, `owl_attn`)** still
  emit pre-existing `make_tensors` AttributeError under
  `tests/functional/test_dispatch.py` — pre-existing, unrelated
  to this work.

---

## Plan: `owl_attn` Metal NAX path

**Goal**: real `pcf.owl_attn` running on Metal NAX, replacing the bench
stand-in (single 128×2048×2048 GEMM/layer). Closes the asterisk on the
1.61× MLX forward number.

### Background — the architectural mismatch

The autotune correctly enumerates `m16n32k16_nax_bf16`, but
`is_valid_for(caps)` → `emit()` fails with:

```
ValueError: LoadMatrixOp: reg_offsets has 8 entries,
            fragment width is 16
```

Root cause: the registry's NAX shape declares `b_regs=16, c_regs=16`
(two 16×16 fragments fused for N=32) while `b_offsets`/`cd_offsets` have
8 entries — they're a per-fragment template, not used by the NAX MSL
visitor (it has its own coordinate map). `MmaBody`/`OnlineSoftmaxBlock`
pass `reg_offsets=cfg.*_offsets` baked in for the PTX path, and the
validator rejects the count mismatch.

That's the **surface** error. Investigation revealed a deeper layer:
even with the validator fix (drop `reg_offsets` for NAX shapes,
~30 lines), `OnlineSoftmaxBlock`'s GEMM2 K-step formula
`nk_per_kstep = max(shape_k // shape_n, 1)` assumes `shape_k >= shape_n`.
For NAX (k=16, n=32) it produces `GEMM2_K_STEPS = 1` while
`MmaBody._dot` iterates `KvTile / shape_k = 2` inner-K steps —
`IndexError` on `p_frags[m][inner_k]`. Generalizing this requires
reworking the P-fragment construction so one S-accumulator produces
*multiple* A-fragments (the inverse of the current "many S-tiles →
one A-frag" `frag_convert` pattern).

### Why retrofitting `OnlineSoftmaxBlock` is the wrong path

`OnlineSoftmaxBlock` and `MmaBody` are designed around per-fragment
reasoning where C-fragments and A-fragments have **different per-lane
layouts** — true on PTX, where converting C → A means packing multiple
C-fragments into one A-fragment via `frag_convert`. NAX doesn't have
that mismatch:

- `BaseNAXFrag` (16×16, 8 per-lane elements, kElemRows=2,
  kElemCols=4) is the unit of NAX work.
- C-fragments and A-fragments share the **same per-lane layout**.
  Only dtype differs (acc f32 vs compute bf16/f16).

In MLX's `scatter_sdpa.metal` the `S` accumulator is reused directly
as an A-fragment for the next MMA — no convert step:

```cpp
// S = Q @ K^T (Stile is NAXTile<float, 1, 2>)
NAXFrag_t::mma(
    Stile.frag_at(iq, ik), Stile.frag_at(iq, ik+1),  // C: 2 frags = 16×32
    Qtile.frag_at(0, 0),                              // A: 1 frag  = 16×16
    Ktile.frag_at(0, 0), Ktile.frag_at(1, 0));        // B: 2 frags = 32×16

// O += S_exp @ V — S used directly as the A-fragment
NAXFrag_t::mma(
    Otile.frag_at(iq, id), Otile.frag_at(iq, id+1),
    Stile.frag_at(iq, ik),                            // ← S frag = A frag
    Vtile.frag_at(0, 0), Vtile.frag_at(0, 1));
```

Generalizing `OnlineSoftmaxBlock` to handle this would mean two
disjoint code paths with different per-fragment vs per-tile reasoning.
Cleaner: skip those abstractions entirely and emit IR directly,
modelled on the world_engine reference.

### Reference implementation

[`world_engine/src/mlx_metal/ext/kernels/scatter_sdpa.metal`](../../world_engine/src/mlx_metal/ext/kernels/scatter_sdpa.metal)
(~200 LOC, single simdgroup BQ=32 BK=32 BD=64 variant). Built directly
on MLX's `NAXTile<T, TileRows, TileCols>` / `NAXFrag` from
[`mlx/.../steel/attn/nax.h`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/nax.h).

### Phase 1 — proof of correctness (~1 day)

Goal: `pcf.owl_attn` runs on Metal NAX with cosine ≥ 0.97 vs the
existing simdgroup_matrix `build()` reference, single configuration.

- Add `OwlAttnKernel.build_metal()` that gates on:
  - `mma_cfg.shape_id == "m16n32k16_nax_bf16"`
  - `s.Dh == 64`, `c.KvTile == 32`, `c.NCW == 1`, `c.MTiles == 1`
  - Falls through to `self.build()` for any other config
- Single-simdgroup, BQ=16 (one warp owns 16 Q rows × full Dh=64 output)
- Single-segment (full attention over `[0, num_buckets * tpf)`), no
  ring/tail iteration yet
- Q comes in pre-RoPE'd (move RoPE outside or skip for v1; the spec
  has `cos`/`sin` tensors for inline RoPE, but they're optional in
  the kernel body)
- **No** online softmax — single-pass standard softmax over the full
  KV range (correctness baseline; perf intentionally suboptimal for
  long KV)
- Validate against `tests/functional/test_owl_attn.py` (or write one
  if missing) using the existing problems list

IR primitives used (all already present):

| Op | Use |
|---|---|
| `bld.load_matrix("a"\|"b", "m16n32k16_nax_bf16", reg_offsets=None)` | Load Q (16×16), K, V (32×16 pairs) |
| `bld.mma("m16n32k16_nax_bf16", a, b, c)` | 16×32×16 NAX MMA (Q@K^T and S@V) |
| `bld.frag_apply("m16n32k16_nax_bf16", body)` | per-lane scalar transforms (scale, exp, mul) |
| `bld.frag_reduce(..., axis="row", cd_offsets=cfg.cd_offsets)` | row-class max/sum (returns 2 row-class values) |
| `bld.frag_convert(...)` | f32 → bf16 cast for S → A reuse |
| `bld.store_matrix("m16n32k16_nax_bf16")` | Write O |

The one uncertain piece is the C → A dtype cast for `S_exp @ V`.
World_engine sidesteps it by feeding f32 `S` directly (Apple's NAX
accepts mixed `(A=f32, B=bf16, C=f32)` descriptors). Two options:

- **(simpler)** Insert a `frag_convert(f32 → bf16)` over the S-tile
  before GEMM2. One extra elementwise pass, negligible cost.
- **(faster)** Register a mixed-dtype shape `m16n32k16_nax_f32xbf16`
  with `a_dtype=F32, b_dtype=BF16, acc_dtype=F32`. Add the matching
  matmul2d_descriptor to the NAX visitor. No cast at all.

Default to the cast option for v1; revisit if profiling shows it as
a hotspot.

### Phase 2 — feature parity (~1 day)

Goal: real `pcf.owl_attn` end-to-end with all the production knobs.

- **Online softmax**: replace single-pass with the standard flash
  recurrence — per KvTile chunk:
  1. `frag_apply(scale)` on S
  2. `frag_reduce(max, axis="row")` → `new_max`
  3. `frag_apply(exp_sub, selectors=new_max)` on S → P
  4. `frag_apply(rescale)` on O accumulator using
     `factor = exp2(old_max - new_max)`
  5. `frag_reduce(sum, axis="row")` accumulating into `sum_score`
  6. GEMM2: `O += P @ V`

  All ops already exist; the structure mirrors `OnlineSoftmaxBlock`'s
  Metal path but emits inline rather than going through the abstraction.

- **Segment iteration** — Python-unrolled across `max_segments`,
  reading `(start, length)` pairs from the segments tensor. Inside
  each segment, an inner KvTile loop. Matches the per-segment dispatch
  in `owl_attn`'s spec.

- **Multi-simdgroup variant** — NCW > 1 partitions Q rows across warps;
  MTiles > 1 stacks Q sub-tiles per warp. Per-warp `frag_apply` /
  `frag_reduce` already isolate per-lane state, so the partition is
  natural.

- **Q-RoPE inline** — read `cos`/`sin` tables, emit fp32 ortho-RoPE
  on Q in registers before the main loop. Uses `frag_apply` with
  selectors keyed off `(h, w, frame_t)`. Optional for v1; can be
  pre-computed by the caller.

### Phase 3 — wire and bench (~half day)

- `OwlAttnKernel.is_valid()` already passes for NAX shapes (the
  `mma.shape.k == 8` filter only catches `m8n8k8_bf16`); nothing to
  change there.
- Update `tests/bench_waypoint_forward.py` to use real `pcf.owl_attn`
  instead of the GEMM stand-in on the Metal path.
- Re-run the bench, update the perf table in this doc.

### Risk register

| Risk | Mitigation |
|---|---|
| `frag_convert(f32 → bf16)` not yet wired for NAX | Probably already works (per-lane elementwise cast); fall back to mixed-dtype shape registration if not |
| Multi-simdgroup partition needs work | Start single-warp; multi-warp is incremental |
| Spec compatibility (`cos`/`sin` inline RoPE) | Pre-RoPE Q outside the kernel for v1; inline as an enhancement |
| NAX `c_regs=16` interactions with our shared epilogue (`store_acc`) | Use direct `bld.store_matrix(...)` instead of routing through `store_acc` for v1 |
| `OwlAttnSpec.compute_dtype` set to e4m3 | Reject in `is_valid_for(caps)` for the NAX path; fall through to the existing simdgroup_matrix `build()` |

### Phase 4 — optimization experiments (post-completion)

Once Phase 3 lands and the bench is honest, the following are worth
tuning:

- **Online softmax variants.** Try the alternatives MLX experimented
  with:
  - **Two-pass** (current plan): max → exp → sum → rescale, two
    `frag_reduce` calls per chunk. Standard flash attention.
  - **Single-pass with deferred scaling.** Bake `scale * log2e` into
    Q during smem staging (matches the world_engine `seq_staged`
    variant). Eliminates the per-chunk `frag_apply(scale)` —
    saves one elementwise pass per KvTile chunk (~256 ops at our
    shape).
  - **Block-skipping** when an entire KV block's max is below the
    running threshold (cheap predicate, big win for long KV with
    sparse attention masks).
- **Q staging in threadgroup memory.** World_engine's BK=64 variant
  stages Q once into TG memory and reads from there across all KV
  chunks, instead of reloading from gmem. ~4 KB smem for BQ=32 BD=64
  half. Worth it when KV is long enough that Q gmem reads dominate.
- **K/V cooperative load.** World_engine reads K/V directly from
  gmem (relying on Apple's L2). Try `cp.async`-style TG staging
  for K/V — likely a wash on small KvTiles, may help at KvTile≥128.
- **Mixed-dtype matmul descriptor.** Skip the `frag_convert(S → A)`
  by registering `m16n32k16_nax_f32xbf16` (A=f32, B=bf16, C=f32) in
  the registry and the NAX visitor. Apple's NAX hardware supports
  this directly. One fewer pass per KvTile chunk.
- **Inline ortho-RoPE.** Move RoPE from `pcf.rmsnorm`-adjacent code
  into the attention kernel itself, applied to Q post-load and to K
  on the fly during MMA inputs (matches the world_engine
  `seq_staged_impl` design, which keeps Q in TG with pre-applied
  RoPE).
- **Persistent O accumulators** — one threadgroup processes multiple
  Q tiles back-to-back, amortizing kernel launch + Q-RoPE setup.
  Requires a different grid shape; meaningful when Q rows are short
  relative to KV.
- **Custom M=32 NAXFrag.** World_engine's `nax_m32.h` defines a
  32×32×16 fused descriptor (one matmul2d call producing a 32×32
  output via two stacked 16-row halves). Doubles the M throughput
  per MMA. Would need a new registry shape `m32n32k16_nax_bf16` and
  visitor support.

### Phase 5 — kernel-side cleanup (post-completion)

- **Generalize to `attn`**. The existing `attn` kernel (non-segmented)
  shares the same architecture; once `owl_attn`'s NAX path is proven,
  factor the common bits into a reusable `nax_attn_body` helper that
  both kernels call. Don't do this preemptively — wait until two
  callers exist.
- **Decommission `OnlineSoftmaxBlock` for Metal**. Once both attn
  kernels have NAX `build_metal` paths, the Metal `mma_shape_id is
  not None` branch in `online_softmax_block.py` becomes dead. Delete
  it; the PTX path stays for CUDA.
- **Add NAX-aware smem estimator** so `is_valid_for(caps)` doesn't
  reject NAX configs with BK > 16 on simdgroup_matrix accounting.
  Currently masked by the kernel's own `is_valid` filter; will
  matter once a kernel emits NAX without the corresponding manual
  filter.
- **Consider mixed-dtype NAX shapes in the registry** as a general
  pattern. attn's GEMM2 isn't the only place where C-output of one
  MMA is the A-input of another. Registering the cross-product of
  `(a_dtype, b_dtype, acc_dtype)` for NAX shapes lets kernels skip
  cast passes.

### What NOT to do

- ~~Generalize `OnlineSoftmaxBlock`~~ — wrong abstraction for NAX
  (covered above).
- ~~Change the registry's `c_regs` to 8~~ — would break the existing
  GEMM `build_metal` which depends on 16 slots per lane.
- ~~Hand-written MSL kernel under `drivers/`~~ — we just consolidated
  `nax_gemm.py` away in favor of the IR `build_metal` pattern.
  Stay in the IR.

---

## Design notes carried forward

### Eager dispatch is forfeited deliberately

Unlike MLX, we don't build a lazy graph in C++. Graph nodes stay in
Python; this costs ~3-6 µs per launch (arg marshalling) that MLX
avoids. The trade is simplicity: no cross-stream dependency tracking,
no fences, no residency sets. Single-stream per process. For the
workloads we target, the per-launch overhead is irreducibly bounded
by Python.

### Unified memory, no copies

Apple Silicon: any malloc'd pointer becomes an `MTLBuffer` via
`newBufferWithBytesNoCopy:`. Inputs from numpy arrays flow through
that path; outputs live in pool-allocated `MTLBuffer`s exposed as
numpy views. Zero host↔GPU copies, same as MLX and Torch MPS.

### Scalars via `setBytes:`, not buffer-backed

`setBytes:length:atIndex:` is the native path for inline values < 4 KB;
saves an allocation per scalar.

### Pipeline cache is mandatory

`device.newLibraryWithSource:options:error:` is ~134 ms per call.
Compiled pipelines are keyed on `(hash(full_source), entry_name)` and
held for the process lifetime.

### The "all-MLX or all-quark" finding (historical)

Earlier in the migration we tested a hybrid: some ops via MLX
(`gemm`, `rmsnorm`, etc.), others via our IR (`kv_cache_update`,
`owl_attn`, anything needing in-place writes or NAX). Result: **48.9 ms
forward** — slower than either pure path. Causes:

1. `MetalArray` ↔ `mx.array` conversion costs ~9 µs per tensor at
   every crossing (`MetalArray` was the predecessor to today's
   `QuarkTensor`; both numpy-backed wrappers around a Metal buffer,
   but neither bridged zero-copy to MLX's `mx.array`).
2. MLX's lazy graph fragments at every crossing.
3. Fragmented graphs can't pipeline across kernel boundaries.

The conclusion ("all-MLX or all-quark per forward") drove the decision
to commit to the all-quark path even at the 3.2× perf cost, with
double-buffered NAX as the path forward to close it.

---

## Repo layout (Metal backend)

```
src/quark/
├── drivers/
│   ├── metal.py              # MetalDriver: probe, compile, launch
│   ├── _metal_dispatch.cpp   # nanobind C extension (queue + eval)
│   ├── _metal_cpp/           # metal-cpp headers (Apple)
│   └── metal_harness.py      # signature/binding-plan generator
├── lower/msl/
│   ├── lower.py              # MslLowerer + LoweredMslKernel + ctx flags
│   ├── visitors.py           # vec_load/store/async_copy + for_loop/yield
│   ├── mma.py                # simdgroup_matrix + NAX MMA visitors
│   └── ...
├── kernels/
│   ├── gemm/kernel.py        # NAX build_metal (single + multi-warp)
│   ├── rmsnorm/kernel.py     # build_metal: register-stash + multi-sg
│   ├── ada_rmsnorm/kernel.py # build_metal
│   ├── ada_gate_residual/kernel.py
│   ├── silu/kernel.py        # vec_load/vec_store rewrite
│   └── decorator.py          # @kernel — build_<family> dispatch
├── functional/
│   ├── _dispatch.py          # queue_launch_ir + queue_strided_copy
│   ├── silu.py / rmsnorm.py / ada_*  # IR via queue_launch_ir
│   └── gemm.py               # _try_nax_gemm via launcher().compile
├── ir/mma_registry.py        # m16n32k16_nax_bf16 registered
├── runtime/tensor.py         # QuarkTensor + _MetalStorage
└── __init__.py               # quark.lazy() context manager
```
