# `popcorn.functional` — torch / MLX entry point

Every popcorn production kernel is exposed as a framework-native
callable. Pass `torch.Tensor`s (CUDA) or `mx.array`s (Metal); dispatch
is decided per-call from the input type.

```python
import popcorn.functional as pcf

C = pcf.gemm(A, B)                                            # matmul
out = pcf.attention(Q, K, V_t, B=..., n_kv_heads=..., ...)    # flash attention
out = pcf.owl_attn(Q, K_cache, Vt_cache, cos, sin, segs, ns, B=..., ...)
K_cache, Vt_cache, segs, ns = pcf.kv_cache_update(
    K, V, cos, sin, frame_t, Vt_cache, segs, ns, K_cache, ...)
Y = pcf.moe_inproj(X, W_in, token_ids, work_list, n_experts=...)
Y = pcf.moe_outproj(h_in, W_out, token_ids, sw, wl, M=..., n_experts=...)
```

On CUDA every op is registered via `torch.library.custom_op`, so
`torch.compile(fullgraph=True)` traces through cleanly. Inside a
compiled region call the raw op directly — `torch.ops.popcorn.gemm(...)` —
or keep using `pcf.gemm`; Dynamo will pick it up either way.

## How tensors become a spec

Each kernel class has a `spec_from_tensors(*inputs, **scalar_kwargs)`
classmethod. Tensor shapes supply every dim they uniquely determine
(e.g. `M`, `K` from `A.shape`); dims that can't be recovered from a
flat layout (`B`, `n_kv_heads`, `seq_len`, …) come in as kwargs.

| Op | Positional tensors | Required kwargs |
|---|---|---|
| `gemm(A, B)` | A, B | — (dtype kwargs are optional) |
| `attention(Q, K, V_t)` | Q, K, V_t | `B`, `n_kv_heads`, `gqa_ratio`, `seq_len`, `kv_len` |
| `owl_attn(Q, K_cache, Vt_cache, cos, sin, segments, n_segments)` | 7 tensors | `B`, `n_kv_heads`, `gqa_ratio`, `H_spatial`, `W_spatial`, `num_buckets`, `pinned_dilation` |
| `kv_cache_update(K, V, cos, sin, frame_t, Vt_cache, segments, n_segments, K_cache)` | 9 tensors | `B`, `n_kv_heads`, `H_spatial`, `W_spatial`, `num_buckets`, `pinned_dilation` |
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

`b_shuffle=True` kernels read a permuted B layout. Inside
`torch.compile` we want the shuffle to be constant-folded out of the
graph, not re-run per call. `popcorn.functional.shuffle` exposes
offline helpers:

```python
from popcorn.functional import shuffle_b_for_gemm

B_shuf = shuffle_b_for_gemm(A, B)                             # call once
C = pcf.gemm(A, B_shuf, b_shuffled=True)                       # per step
```

`shuffle_b_for_moe_inproj(X, W_in, n_experts=..., top_k=...)` and
`shuffle_b_for_moe_outproj(h_in, W_out, M=..., n_experts=...)` work
the same way. The helpers resolve the same config `pcf.*` would, so
the shuffle layout always matches the kernel's fragment loader.

## Autograd

Every op registers an autograd that raises `NotImplementedError` with
a clear message. The surface is inference-only. Training-mode kernels
(attention backward, GEMM grad) are tracked as a future proposal.

## Using under `torch.compile`

```python
@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def step(A, B, token_ids, work_list):
    Y = pcf.moe_inproj(A, B, token_ids, work_list, n_experts=16)
    return Y * 2
```

Verified in `tests/functional/test_compile.py` (CUDA-only). Fake fns
evaluate the kernel's `TensorDecl.shape(spec, config)` callable at
trace time, so Dynamo sees correct shapes / dtypes without running
the kernel.

`kv_cache_update` is marked `mutates_args=("Vt_cache", "segments",
"n_segments", "K_cache")` so Dynamo's alias analysis treats it as an
in-place write. Its fake fn returns aliases of the input buffers.

## Using under MLX

No compile wrapping today. `pcf.*` on `mx.array` inputs calls into
the kernel eagerly and returns a fresh `mx.array` (since MLX arrays
are immutable — kernels that logically write in place still return
new buffers). `mx.compile` wrapping of `pcf.*` callables is a
follow-up.

## CUDA graph capture

Every launch path is designed to be capture-safe: `data_ptr()`
extraction, no `.item()` host syncs, no allocator-allocated scratch
during the launch. The one remaining allocation is per-launch
scalar-arg packing. CUDA-graph regression
tests land in `tests/functional/test_cuda_graph.py` when that work lands.

## Schema stability

Registered schemas are a public contract. Adding kwargs with defaults
is fine; removing or reordering is a breaking change. CI snapshots
every `popcorn::*` schema in `tests/functional/test_schema.py` (not
yet landed) and fails on accidental change.

## See also

- [docs/ADDING_A_KERNEL.md](ADDING_A_KERNEL.md) — how to add a kernel
  and wire up its functional wrapper
- [docs/BACKEND.md](BACKEND.md) — `PT` polymorphic tensor surface
