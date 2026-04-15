# Backend (PT polymorphic tensor)

One file — `src/popcorn/backend.py` — owns every torch-vs-mlx branch.
Everything else in the repo imports `PT` and calls `PT.whatever(...)`.

```python
from popcorn.backend import PT, IS_METAL
```

- `IS_METAL` — platform dispatch flag. Decided at import time from
  `sys.platform == "darwin"`. No runtime branching.
- `PT` — class holding every polymorphic tensor op as a `@staticmethod`.

## Rule

**Zero torch/mlx-specific calls outside of `backend.py`, kernel
baselines that deliberately measure a backend-fast library
(`torch.compile` + `flex_attention`, `mx.fast.scaled_dot_product_attention`),
and the lowerer drivers (`drivers/cuda.py`, `drivers/mlx.py`).**

If a caller finds itself reaching for `import torch` or
`import mlx.core`, the right move is to add the primitive to `PT`.

## Dtype constants

```python
PT.float32   PT.float16   PT.bfloat16
PT.int32     PT.int64     PT.int16    PT.int8
PT.uint32    PT.uint16    PT.uint8
```

These alias to the active backend's type. On Metal they're
`mx.float32` etc.; on CUDA `torch.float32`.

String ↔ dtype bridges:

```python
PT.dtype_from_str("bf16")          # → PT.bfloat16
PT.dtype_to_str(PT.float32)        # → "f32"
```

## Creation

| Op | Notes |
|---|---|
| `PT.randn(*shape, dtype=None)` | default backend device |
| `PT.zeros(*shape, dtype=None)` | |
| `PT.ones(*shape, dtype=None)` | |
| `PT.arange(*args, dtype=None)` | torch-like args (`[start,] stop [,step]`) |
| `PT.linspace(start, stop, n, dtype=None)` | |
| `PT.tensor(data, dtype=None)` | from Python list / np.array |
| `PT.zeros_like(x)` | |

## Arithmetic / shape

| Op | Notes |
|---|---|
| `PT.matmul(a, b)` | |
| `PT.transpose(x, dim0=-1, dim1=-2)` | |
| `PT.broadcast_to(x, shape)` | |
| `PT.repeat_interleave(x, n, dim=-1)` | |
| `PT.cat(arrays, dim=-1)` | |
| `PT.astype(x, dtype)` | |
| `PT.set_slice(dst, *, axis, start, stop, src)` | backend-agnostic `dst[start:stop] = src` |
| `PT.where(cond, a, b)` | |
| `PT.index_copy(dst, dim, index, src)` | in-place on torch, out-of-place on mlx — always use the return value |

## Math

| Op | |
|---|---|
| `PT.exp` `PT.cos` `PT.sin` | |

## Reductions / stats

| Op | Return |
|---|---|
| `PT.cosine_sim(a, b)` | Python `float` ∈ [-1, 1] |
| `PT.abs_diff_stats(a, b)` | `(mean_abs, max_abs)` as floats |
| `PT.has_nan(x)` | `bool` |
| `PT.all_finite(x)` | `bool` |
| `PT.first_non_finite_index(x)` | `int | None` |
| `PT.numel(x)` | `int` |

## Fast-path attention

```python
y = PT.attention(q, k, v, *, mask=None, scale=None)
```

Dispatches to:
- `mx.fast.scaled_dot_product_attention` on Metal (flash-attn)
- `F.scaled_dot_product_attention` on CUDA (flash / memory-efficient)

Shapes: `q [B, Hq, Lq, D]`, `k/v [B, Hkv, Lkv, D]`. GQA (Hq ≠ Hkv)
is handled by the backend. `mask` is additive; broadcasts to
`[B, Hq, Lq, Lkv]`.

## torch.compile polyfill

```python
@PT.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def step(K, V, cos, sin): ...
```

On CUDA this is literally `torch.compile`. On Metal it's identity
(mlx has no compile layer) — the decorator + kwargs pass through
unchanged. Write one baseline, get both backend treatments for free.

Bare use (no args) also works: `PT.compile(fn)` → `torch.compile(fn)`
on CUDA, `fn` on Metal.

## Runtime

| Op | |
|---|---|
| `PT.synchronize()` | block until all queued work completes |
| `PT.device_name()` | human-readable device string |

## MLX driver notes

`drivers/mlx.py` passes `init_value=0` to `mx.fast.metal_kernel`, so
every output buffer MLX allocates on the caller's behalf starts
zero-filled. This matters for kernels that read-modify-write their
output (atomic-scatter-add in `moe_outproj`; any partial-write
scatter target). Without it the accumulator reads garbage on the
first launch and correctness collapses silently.

Torch-allocated outputs still need explicit zeroing at the call
site (the kernel harness does this in `tools/bench.py` via
`PT.zero_(output_buf)` before the timed region).

## FFI (use rarely)

| Op | When |
|---|---|
| `PT.to_cpu_numpy(x)` | when you need the actual host-side array (`make_tensors` index-cache construction) |
| `PT.to_torch_cpu(x)` | legacy — `check_correctness` no longer needs this |
| `PT.tensor_type()` | `isinstance(x, PT.tensor_type())` for backend-agnostic type checks |

## Adding a PT primitive

Pattern — branch on `IS_METAL` or `PT._is_mx(x)` once, lazy-import,
return the native result:

```python
@staticmethod
def new_op(x, *, k: int):
    if PT._is_mx(x):
        import mlx.core as mx

        return mx.do_thing(x, k=k)
    import torch

    return torch.do_thing(x, k=k)
```

- Lazy-import inside the method so Metal-only hosts never touch
  torch import paths (and vice versa).
- Prefer `PT._is_mx(x)` (per-tensor dispatch) for ops where a user
  might pass a host scalar alongside a device tensor. Use
  `IS_METAL` (module-level flag) when the op is platform-wide
  (e.g. `synchronize`).
- Return whatever the backend returns — don't try to homogenize
  tensor wrappers.

Add a one-line entry in [this doc](#creation) under the right section.

## Adding a third backend

Every PT method is a `if backend_a: ... elif backend_b: ... else:`
block. A new backend requires:

1. Decide the module-level platform flag (`IS_METAL` is ours; a new
   backend might add `IS_ROCM` or `IS_OPENCL`).
2. Add dtype aliases to the top of `PT` (the class body is a cascade
   of `if / elif` that defines them).
3. Walk the op table and add the new branch to each.
4. Add a driver under `src/popcorn/drivers/` and a lowerer under
   `src/popcorn/lower/` targeting the backend ISA.
5. Wire dispatch in `Launcher.compile`.

For reference the CUDA path is ~200 lines (driver) + ~800 lines
(PTX lowerer + visitors); the Metal path is ~180 lines (driver) +
~1200 lines (MSL lowerer + visitors).
