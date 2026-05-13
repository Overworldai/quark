# Phase 2 design — int8 KV cache quantization

Status: design only; not implemented.

## What we need

OwlAttn's int8 compute path (Phase 3) needs to consume:
- `K_cache_s8` — same shape as existing `K_cache` (`B * n_kv_heads * capacity, Dh`), s8 instead of bf16
- `Vt_cache_s8` — same shape as existing `Vt_cache` (`B * n_kv_heads * Dh, capacity`), s8
- `K_scales` — per-token f32 scales (`B * n_kv_heads * capacity`)
- `V_scales` — per-token f32 scales (`B * n_kv_heads * capacity`)

`kv_cache_update` writes the existing bf16 caches every frame.
Phase 2 needs to also produce the s8 + scale tensors.

## Two implementation paths

### A. Fork `kv_cache_update` (substantial, in-place)

Modify the 766-line existing kernel to ALSO write s8 outputs +
scales. Adds ~4 tensors to TENSORS, which breaks all existing
callers (binding-order change), and adds substantial body logic
for the per-token absmax reduction.

**Cost:** ~1 day for the kernel changes + ~half day to update
every kv_cache_update call site. **High disruption** — touches
production code paths.

### B. New `kv_cache_quantize` kernel (additive, clean)

A separate kernel that runs AFTER `kv_cache_update` (extra
dispatch per layer per frame). Consumes the bf16 `K_cache` /
`Vt_cache` outputs, produces the s8 + scales tensors. Zero
disruption to existing code.

**Cost:** ~1 day for the new kernel module. Adds 1 dispatch
per layer per frame; small relative to attention compute time.

**This is the recommended path.** Mirrors how we built
GemmIntKernel as a parallel module rather than mutating GemmKernel.

## Algorithm

Per-token signed-symmetric quant — same as the validated path in
`/tmp/run_int8_gemm_scales.py`:

```python
# For each token (one Dh-vector):
abs_max = max(|x[token, :]|)          # f32 across Dh elements
scale   = abs_max / 127               # f32
q       = round(x / scale).clip(-127, 127).astype(s8)
# dequant ≈ q * scale; cos_sim > 0.99 vs original bf16
```

Validated in `/tmp/run_int8_gemm_scales.py` at cos > 0.9999.

## Per-tensor specifics

### K_cache → K_cache_s8 + K_scales

`K_cache` shape `(B * n_kv_heads * capacity, Dh)`, row-major.
Each row is one token's Dh-vector for one (batch, head).

Per token: reduce-max across the row's Dh=64 elements, derive
scale, quantize entire row. The reduce is intra-row, intra-warp.
With SIMD32 + Dh=64: 2 elements per lane → 1 abs + 1 max +
subgroupMax = 1 cross-lane reduce.

Output:
- `K_cache_s8`: same shape, s8 dtype
- `K_scales`: shape `(B * n_kv_heads * capacity,)`, f32 dtype,
  one scale per row.

WG layout: 1 token per WG (32 threads = 1 SIMD32 wave). Grid =
`(B * n_kv_heads * capacity, 1, 1)`. Trivial dispatch.

### Vt_cache → Vt_cache_s8 + V_scales

`Vt_cache` shape `(B * n_kv_heads * Dh, capacity)`, transposed.
Each *column* is one token's Dh-vector for one (batch, head).
Per-token reduce is now across COLUMNS — Dh rows × 1 col.

Two implementation choices:

**B.1 Same-layout quant (Vt_cache_s8 stays transposed)**

Each thread reduces a column. With 32 lanes per warp covering
`capacity` tokens at chunked stride. Cross-row reduction
required.

Per WG: process N tokens in parallel. Each lane handles a column.
Loop over Dh rows, accumulating abs-max per lane. Quantize at
the end.

Smem usage minimal (no cross-thread reduction needed if each
lane owns one column). Just per-lane state.

**B.2 Detranspose during quant (Vt → V_cache_s8 row-major)**

While quantizing, write s8 output in row-major (token-rows × Dh-cols)
layout. This makes V_cache_s8 consistent with K_cache_s8 layout
for the OwlAttn int8 path.

Detranspose adds more cross-thread coordination but matches
the natural row-major access pattern for attention. Slightly
more complex kernel, but cleaner downstream consumption.

**Recommended: B.1.** Keep V transposed in s8 to match the
existing OwlAttn V access pattern. Detranspose is a separate
optimization decision (and would be wasted work if OwlAttn int8
ends up loading V in transposed mode anyway).

## Module structure

```
src/quark/kernels/kv_cache_quantize/
├── __init__.py
├── spec.py        — KVCacheQuantizeSpec: (B, n_kv_heads, capacity, Dh)
├── config.py      — minimal, no tunables for first cut
├── kernel.py      — body
├── reference.py   — numpy reference (per-token absmax → scale → quant)
├── problems.py    — production shape (Waypoint-1.5-1B 360p)
└── baselines.py   — empty
```

TENSORS:
```
[K_cache_bf16  — input,  shape (B*n_kv_heads*capacity, Dh)]
[Vt_cache_bf16 — input,  shape (B*n_kv_heads*Dh, capacity)]
[K_cache_s8    — out,    shape (B*n_kv_heads*capacity, Dh)]
[Vt_cache_s8   — out,    shape (B*n_kv_heads*Dh, capacity)]
[K_scales      — out,    shape (B*n_kv_heads*capacity,)]
[V_scales      — out,    shape (B*n_kv_heads*capacity,)]
```

## Validation plan

1. **GLSL probe** at small shape: validate per-token absmax +
   quant against numpy reference. cos_sim of dequantized values
   vs original > 0.999.
2. **Production kernel** at Waypoint shape (`capacity=8704`,
   `n_kv_heads=16`, `Dh=64`): same correctness target.
3. **End-to-end OwlAttn int8** (Phase 3): take quantized cache +
   scales as inputs; produce attention output. Compare cos_sim
   vs bf16 reference at saturated KV.

## Estimated effort

- Phase 2.0 GLSL probe (algorithm validation): ~1 hour
- Phase 2.1 production kernel module: ~half day
- Phase 2.2 correctness validation at multiple shapes: ~1 hour

Total Phase 2: **~half day** if no surprises. Lower risk than
expected because the algorithm is already validated end-to-end
in `/tmp/run_int8_gemm_scales.py`.
