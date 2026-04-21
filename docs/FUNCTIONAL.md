# `popcorn.functional` — call surface

Every popcorn production kernel is exposed as a callable that accepts
backend-native tensors. Dispatch is decided per-call from the input
type:

- `PopcornTensor` (from `popcorn.runtime.tensor`) → CUDA path.
- `mx.array` (MLX) → Metal path.

No torch runtime dep on either leg. Torch is only pulled in if a test
baseline or reference implementation reaches for it (optional dev
extra).

```python
import popcorn.functional as pcf
from popcorn.runtime.tensor import PopcornTensor

A = PopcornTensor.randn(M, K, dtype="bf16")
B = PopcornTensor.randn(N, K, dtype="bf16")

C   = pcf.gemm(A, B)                                           # matmul
out = pcf.attention(Q, K, V_t, B=..., n_kv_heads=..., ...)     # flash attention
out = pcf.owl_attn(Q, K_cache, Vt_cache, segments, n_segments, ...)
pcf.kv_cache_update(K, V, frame_t, frozen, Vt_cache, segs, ns, K_cache, ...)
Y   = pcf.moe_inproj(X, W_in, token_ids, work_list, n_experts=...)
Y   = pcf.moe_outproj(h_in, W_out, token_ids, sw, wl, M=..., n_experts=...)
```

## How tensors become a spec

Each kernel class has a `spec_from_tensors(*inputs, **scalar_kwargs)`
classmethod. Tensor shapes supply every dim they uniquely determine
(e.g. `M`, `K` from `A.shape`); dims that can't be recovered from a
flat layout (`B`, `n_kv_heads`, `seq_len`, …) come in as kwargs.

| Op | Positional tensors | Required kwargs |
|---|---|---|
| `gemm(A, B)` | A, B | — (dtype kwargs are optional) |
| `attention(Q, K, V_t)` | Q, K, V_t | `B`, `n_kv_heads`, `gqa_ratio`, `seq_len`, `kv_len` |
| `owl_attn(Q, K_cache, Vt_cache, segments, n_segments)` | 5 tensors | `B`, `n_kv_heads`, `gqa_ratio`, `H_spatial`, `W_spatial`, `num_buckets`, `pinned_dilation` |
| `kv_cache_update(K, V, frame_t, frozen, Vt_cache, segments, n_segments, K_cache)` | 8 tensors | `B`, `n_kv_heads`, `H_spatial`, `W_spatial`, `num_buckets`, `pinned_dilation` |
| `moe_inproj(X, W_in, token_ids, work_list)` | 4 tensors | `n_experts` |
| `moe_outproj(h_in, W_out, token_ids, slot_weights, work_list)` | 5 tensors | `M`, `n_experts` |

All `spec_from_tensors` methods validate shape consistency and raise
`ValueError` on mismatch — the error names the offending tensor so
you can track down the call site.

## Config resolution and autotuning

Every functional call resolves its config through a single path —
`AutotuneCache.lookup_or_search` — with no separate "functional layer"
resolution. The three lookup levels are:

1. **Hot** — in-memory dict; sub-microsecond.
2. **Disk** — `~/.cache/popcorn/<hash>.json`; loaded and validated once,
   then promoted to hot.
3. **Bundled** — `configs/<kernel>_<problem>.json` checked in to the repo.

On a **miss**, the cache runs an inline search on the calling thread
(blocking). Two search depths are available:

| Depth | When used | How |
|---|---|---|
| `"fast"` | default | Up to 16 seed-first candidates, one timing pass |
| `"full"` | explicit | Full genetic search (pop × gens), same warm seeding |

**Controlling the depth:**

```python
# 1. env var — forces "full" for the entire process
POPCORN_MAX_AUTOTUNE=1 python serve.py

# 2. context manager — scoped to one block
import popcorn
with popcorn.max_autotune():
    pcf.gemm(A, B)   # cache miss → full genetic search

# 3. per-op .autotune() warmup API — always "full", recommended at server startup
pcf.gemm.autotune(A_example, B_example)   # blocks ~10–30 s, then cached forever
```

**Server-startup pattern:**

```python
import popcorn.functional as pcf

# Tune once at startup — blocks while the genetic search runs.
# On subsequent restarts the disk cache is hit immediately.
pcf.gemm.autotune(A_example, B_example)
pcf.gemm.autotune(A_example, B_large)

for batch in dataloader:
    out = pcf.gemm(batch, W)   # hot-dict hit every call
```

Winner configs are persisted to `~/.cache/popcorn/` automatically.
Run `make autotune KERNEL=<name>` to populate the bundled `configs/` dir
for shipping pre-tuned configs with the package.

## Weight shuffle — offline

`b_shuffle=True` kernels read a permuted B layout. The shuffle must
happen once offline (at model-load time) and the per-step call uses
the pre-shuffled buffer:

```python
from popcorn.functional import shuffle_b_for_gemm

B_shuf = shuffle_b_for_gemm(A, B)                             # call once
C      = pcf.gemm(A, B_shuf, b_shuffled=True)                 # per step
```

`shuffle_b_for_moe_inproj(X, W_in, n_experts=..., top_k=...)` and
`shuffle_b_for_moe_outproj(h_in, W_out, M=..., n_experts=...)` work
the same way. The helpers resolve the same config `pcf.*` would, so
the shuffle layout always matches the kernel's fragment loader.
`nn.Linear` exposes the same shuffle via `.prepare(...)` at module
init — recommended for `nn.Module`-based models.

## PopcornTensor path (CUDA)

Allocations, reductions, slicing, reshape, permute, and arithmetic
all live on `PopcornTensor` itself — no torch needed:

```python
from popcorn.runtime.tensor import PopcornTensor

A = PopcornTensor.randn(M, K, dtype="bf16")
B = PopcornTensor.zeros(N, K, dtype="bf16")
C = PopcornTensor.zeros(M, N, dtype="bf16")
pcf.gemm(A, B, out=C)                     # in-place write into C
C = C.reshape(M // 2, 2, N)
```

All views share storage (refcount on the underlying `cuMemAlloc`
block). Allocations are pooled / stream-ordered via
`cuMemAllocAsync` on the default stream; see
[docs/ARCHITECTURE.md](ARCHITECTURE.md).

Persistent output buffers — the `out=` kwarg on every wrapper —
avoid the per-call `cuMemAllocAsync` that `auto_alloc` would
otherwise pay:

```python
C = PopcornTensor.zeros(M, N, dtype="bf16")
for batch in loader:
    pcf.gemm(batch, W, out=C)             # writes into C, no fresh allocation
```

## Using under MLX

No compile wrapping today. `pcf.*` on `mx.array` inputs calls into
the kernel eagerly and returns a fresh `mx.array` (since MLX arrays
are immutable — kernels that logically write in place still return
new buffers). `mx.compile` wrapping of `pcf.*` callables is a
follow-up.

## CUDA graph capture

Every launch path is capture-safe: `data_ptr()` extraction, no
`.item()` host syncs, no allocator-allocated scratch during the
launch. `nn.Module` forward passes over `PopcornTensor` inputs
capture cleanly — see `scripts/generate.py` for the end-to-end
Waypoint-1.5 capture+replay loop.

## See also

- [docs/WEIGHTS.md](WEIGHTS.md) — loading safetensors into an `nn.Module`
- [docs/ADDING_A_KERNEL.md](ADDING_A_KERNEL.md) — how to add a kernel
  and wire up its functional wrapper
- [docs/BACKEND.md](BACKEND.md) — `PT` polymorphic tensor surface
  (baseline / reference path, not the runtime)
