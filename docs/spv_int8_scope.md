# int8 m8n16k32 SPV path — scope

Target: 2× theoretical bf16 throughput by using Battlemage's
`m8n16k32_s8_s32` cooperative matrix instead of `m8n16k16_bf16_f32`.

## Background

Battlemage advertises (via `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`):
- `m8n16k16` for bf16/f16 inputs (k=16 inner dim)
- **`m8n16k32` for s8/u8 inputs** (k=32, doubled inner dim)

Same MMA hardware, but int8 packs 2× the operands per cycle. **One
`m8n16k32` s8 MMA does 8192 ops/cycle** vs **4096 ops/cycle** for
m8n16k16 bf16. Theoretical 2× throughput; real-world after
overlap shrinkage typically 1.3–1.5× LFPS.

After the llama.cpp comparison: our 1.2 TFLOPS bf16 attention is at
the Vulkan ceiling on Battlemage. Int8 would lift that ceiling to
~2.4 TFLOPS — still ~2% of peak but the only architecture-level
lever inside the Vulkan stack.

## Reality check FIRST (Phase 0, half-day)

Before any multi-day kernel work, **microbench an int8 GEMM at
m8n16k32 to verify the 2× throughput is actually achievable** on
Mesa anv. Three things this would prove:

1. Mesa anv emits XMX for s8 m8n16k32 (not falling back to scalar)
2. The 2× theoretical actually materializes on Battlemage
3. The s8 MMA doesn't hit a different Vulkan ceiling (e.g. the
   driver does something stupid with int loads)

If int8 GEMM doesn't show 2× over bf16 GEMM at matched problem
shape, kill the path before writing OwlAttn s8.

## Phases

### Phase 0 — Verify throughput unlock (half day)

1. **Register `m8n16k32_intel_s8_s32`** in `src/quark/ir/mma_registry.py`.
   Mirror the four existing Intel shapes. Fields:
   - `m=8, n=16, k=32`
   - `a_dtype=DType.S8`, `b_dtype=DType.S8`, `acc_dtype=DType.S32`
   - `a_regs = 8` (m=8 × k=32 / 32 lanes = 8 s8/lane)
   - `b_regs = 16` (n=16 × k=32 / 32 lanes = 16 s8/lane)
   - `c_regs = 4` (m=8 × n=16 / 32 lanes = 4 s32/lane)
   - `lane_col_step = 4` (s8 in b32 carrier = 4 elements per b32 reg)
2. **Patch `_visit_mma` for the int8 signed-operand flag**. Add
   `MatrixASignedComponents | MatrixBSignedComponents |
   MatrixCSignedComponents | MatrixResultSignedComponents`
   (mask 0x1E) to `OpCooperativeMatrixMulAddKHR` when A/B dtype
   is signed int. Without this Mesa rejects mixed-sign or
   reinterprets as unsigned.
3. **Add Int8 capability** ensure in `_ensure_coopmat_caps` for
   s8 paths. SPIR-V `Int8` capability + `Int8CooperativeMatrixKHR`.
4. **Write a microbench** `/tmp/bench_int8_gemm.py`: int8 GEMM at
   M=N=K=128 (or so) at both shapes, compare wall time. Target:
   s8 should be ~50% of bf16's wall time.

**Decision gate:** if Phase 0 shows < 1.5× speedup, kill the int8
plan. Restart from "accept Vulkan ceiling and ship".

### Phase 1 — int8 dispatch from a real kernel (1 day)

Pick the GEMM kernel for first integration (simpler than OwlAttn):

1. Extend `GemmKernel` to handle `compute_dtype=DType.S8` —
   Q/K cast on load (smem path), scale plumbing.
2. Per-row scale infrastructure: scale tensors for A and B.
3. Epilogue: int32 → f32 with scale multiply before output cast.
4. Validate: int8 GEMM cos_sim > 0.99 vs bf16 reference at one shape.

### Phase 2 — Quantization scheme for KV cache (1 day)

1. Modify `kv_cache_update` to also write int8 versions of K and V
   plus a per-token scale tensor. K_cache stays bf16 for backward
   compat; add `K_cache_s8` and `K_scales` as new outputs.
2. Per-token scale = `max(abs(K[t, :])) / 127`. fp32 stored.
3. Tests: verify int8 cache round-trip cos_sim > 0.99 vs bf16.

### Phase 3 — OwlAttn int8 compute path (2 days)

The hard one. Reference: SageAttention paper, MLX int8 KV cache
pattern (currently lost from repo but the design is well-known).

```
Q_bf16 → quantize-on-load → Q_s8 + Q_scale (per Q row)
K_s8 cache + K_scale  (from kv_cache_update)
V_s8 cache + V_scale

s_acc_s32 = Q_s8 @ K_s8^T            (m8n16k32 MMA)
s_acc_f32 = s_acc_s32 * Q_scale[m] * K_scale[n]
softmax(s_acc_f32) → p_f32, m, l
p_s8 + p_scale = quantize(p_f32)     (per Q row)
o_acc_s32 += p_s8 @ V_s8^T           (m8n16k32 MMA)
o_acc_f32 = o_acc_s32 * p_scale[m] * V_scale[n]
output_bf16 = cast(o_acc_f32 / l)
```

Per-row scales work fine for our shapes; per-block (SageAttention-2)
would be more accurate but adds complexity. Start with per-row.

The k=32 inner dim is the win: instead of 4 k-tiles per Dh=64 chunk
(at k=16 bf16), we do 2 k-tiles (at k=32 s8). Halves the chain
length for QK and AV.

### Phase 4 — Validation (half day)

1. OwlAttn int8 cos_sim vs bf16 reference: target > 0.99 at
   saturated KV (capacity=8704).
2. End-to-end Waypoint frame at int8: latent MAE vs bf16 reference
   target < 5% (similar to MLX's measured 0.9%).
3. Benchmark: target ≥ 1.5× LFPS improvement at 360p saturation.

### Phase 5 — Other kernels (1 day)

GEMM (in/out projection) is the second-biggest GPU consumer after
OwlAttn. Wire int8 compute for the in_proj, out_proj sites in
Waypoint blocks. Smaller win individually but compounds.

## Total estimate

5 days end-to-end if Phase 0 passes. **Phase 0 alone is half a day
and is the go/no-go gate.** If Phase 0 fails (no throughput unlock
on this Mesa anv) the entire plan is dead and we accept the
Vulkan ceiling.

## Risks

1. **Mesa anv may not actually use XMX for s8 m8n16k32 in practice**
   — the API exposure ≠ the JIT actually emitting DPAS. Phase 0
   tests this.
2. **Accuracy could be a problem at long context** — int8 quant
   error compounds across the 8704-token KV. SageAttention reports
   acceptable accuracy at typical contexts; we'd need to validate
   at our specific saturated shape.
3. **Per-row scales may be too coarse** for attention output
   quality. If cos_sim < 0.99, fall back to per-block scales
   (4–8 cols per scale) — adds complexity.
4. **Quantize-on-load latency may eat the throughput gain** —
   we'd be doing the per-row quant work in the kernel itself,
   not for free. Phase 0 microbench won't catch this; Phase 4
   benchmark would.

## Files touched (estimate)

- `src/quark/ir/mma_registry.py` (new shape entry)
- `src/quark/lower/spv/lower.py` (s8 capability, signed-operand flag)
- `src/quark/kernels/gemm/kernel.py` + `spec.py` (int8 path)
- `src/quark/kernels/kv_cache_update/kernel.py` + `spec.py` (s8 outputs)
- `src/quark/kernels/owl_attn/kernel.py` + `spec.py` (int8 compute path)
- `src/quark/kernels/owl_attn/config.py` (compute_dtype = S8 option)
- New: `src/quark/kernels/owl_attn/_int8_quant.py` (per-row quant helpers)

## Hard constraint

77 SPV regression tests must stay green at each phase boundary.
