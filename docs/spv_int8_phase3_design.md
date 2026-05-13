# Phase 3 design — OwlAttn int8 compute path

Status: design doc + skeleton; full kernel body lands in subsequent
iterations.

## Goal

Run OwlAttn's QK + softmax + AV pipeline using int8 m8n16k32 MMA
instead of bf16 m8n16k16. Theoretical 2× throughput from doubled
inner-K. Realistic LFPS gain after overlap shrinkage: 1.3–1.5×.

## What feeds in

From the int8 KV quantization (Phase 2):
- `K_cache_s8`  shape `(B * n_kv_heads * capacity, Dh)`
- `K_scales`    shape `(B * n_kv_heads * capacity,)` — one f32 per token
- `Vt_cache_s8` shape `(B * n_kv_heads * Dh, capacity)`
- `V_scales`    shape `(B * n_kv_heads * capacity,)` — one f32 per token

From the host:
- `Q` bf16  shape `(B * tpf, qkv_dim)` (packed QKV layout)
- `segments`, `n_segments`, `frame_t` — same as OwlAttn

## What comes out

- `output` bf16  shape `(B * tpf, n_q_heads * Dh)`

## Algorithm

### Per KV iter (= one KvTile chunk of seq_k)

```
1. Load K_s8 chunk to smem
   - shape: (KvTile, Dh) s8, where KvTile = 32 = mma_k for int8
2. Load Vt_s8 chunk to smem
   - shape: (Dh, KvTile) s8 (transposed)
3. Load K_scales[token range] and V_scales[token range] (KvTile f32 each)
4. mma1: s_s32 += Q_s8 @ K_s8^T
   - m8n16k32 s8/s8/s32 MMA
   - s_s32 has shape (BlockQRows, KvTile) s32 per warp
5. Convert s_s32 to f32 with scale:
   s_f32[m, n] = s_s32[m, n] * Q_scale[m] * K_scale[n]
   - Q_scale is per-Q-row (computed once at kernel entry)
   - K_scale is per-KV-token (KvTile entries per iter)
6. Apply causal mask / softmax_scale (1/sqrt(Dh))
7. Online softmax update (max + exp + l):
   m_new = max(m, max(s_f32, axis=col))
   p_f32 = exp(s_f32 - m_new)
   l_new = exp(m_old - m_new) * l_old + sum(p_f32, axis=col)
   o_acc *= exp(m_old - m_new)
8. Quantize p_f32 to s8 with per-row P_scale:
   P_scale[m] = max(|p_f32[m, :]|) / 127
   p_s8 = round(p_f32 / P_scale[m])
9. mma2: o_s32 += p_s8 @ Vt_s8^T
   - m8n16k32 s8/s8/s32 MMA (k inside is KvTile, n is Dh)
   - Wait: Vt is (Dh, KvTile) → for the AV multiply we need
     (KvTile, Dh) as B operand. With k=32 (Vt's KvTile dim) and
     n=16 (Dh chunk), need 4 sub-MMAs for Dh=64
10. Convert o_s32 → f32 with scales: o_f32 += o_s32 * P_scale[m] * V_scale[n]
    Wait: V_scale is per-TOKEN not per-Dh-COL. Need to think about
    the scale axes more carefully.
```

### Scale axes (critical):

For QK = Q × K^T (output [m_rows, n_tokens]):
- Q_scale is along m (one per Q row)
- K_scale is along n (one per K token)
- s_f32[m, n] = s_s32[m, n] * Q_scale[m] * K_scale[n] ✓

For AV = P × V (output [m_rows, dh_cols]):
- P_scale is along m (one per Q row)
- V_scale is along k (Vt's KvTile axis = one per KV token)
- BUT the MMA reduces over k, so V_scale needs to be folded into
  P or V before the multiply.

This is the central design question. Two options:

**Option A** — V_scale per-K-token, P_scale per-Q-row:
- We can multiply P by V_scale[k] for each k before MMA.
- But the MMA reduces over k, so per-k scaling means
  o[m, n] = sum_k(P[m, k] * V_scale[k] * V[k, n])
  = sum_k(P_scaled[m, k] * V[k, n])  where P_scaled[m, k] = P[m, k] * V_scale[k]
- This is doable: scale P by V_scale per-column before quant to s8.

**Option B** — per-Dh scales for V instead of per-token:
- Reshape: V quantization scale per (head, Dh-col) instead of per (head, token).
- Then o_f32[m, n] = o_s32[m, n] * P_scale[m] * V_scale[n]
- Same axis pattern as QK.
- BUT changes Phase 2 quant: V scales become [n_heads × Dh] not
  [n_heads × num_tokens]. Different reduction axis.

**Decision:** Stick with **Option A**. Per-token V scales (matches
existing Phase 2 Vt-quant). Fold V_scale into P_scaled at quant time:
```
P_scaled[m, k] = P[m, k] * V_scale[k]  // before P → s8 quant
P_scaled_max[m] = max(|P_scaled[m, :]|)
P_scale[m] = P_scaled_max[m] / 127
P_s8[m, k] = round(P_scaled[m, k] / P_scale[m])
// MMA: o_s32 = P_s8 @ V_s8
// Dequant: o_f32 = o_s32 * P_scale[m]  (V already folded into P)
```

This works because V_scale doesn't appear in the dequant step —
it's already baked into the P_s8 values. Each (m, k) of P had its
own scaling, summed by the MMA. The dequant only multiplies by
P_scale[m].

## Module structure

```
src/quark/kernels/owl_attn_int8/
├── __init__.py
├── spec.py        # OwlAttnIntSpec — same shape as OwlAttnSpec
├── config.py      # OwlAttnIntConfig — fixed KvTile=32 (mma_k)
├── kernel.py      # build() body
├── reference.py   # numpy reference (int8 quant via Phase 2 algo + bf16 ref attn)
├── problems.py    # Waypoint-1.5-1B 360p shape
├── baselines.py   # empty
```

### TENSORS

```
[Q,         shape (B*tpf, qkv_dim) bf16]   # packed Q
[K_s8,      shape (B*n_kv_heads*capacity, Dh) s8]
[K_scales,  shape (B*n_kv_heads*capacity,) f32]
[Vt_s8,     shape (B*n_kv_heads*Dh, capacity) s8]
[V_scales,  shape (B*n_kv_heads*capacity,) f32]
[segments,  shape (B*max_segments*2,) s32]
[n_segments,shape (B,) s32]
[frame_t,   shape (1,) s32]
[output,    shape (B*tpf, n_q_heads*Dh) bf16, role="out"]
```

## Implementation phases

### 3.1 Skeleton (this iteration)

Create the module with stub `build()` that just raises
NotImplementedError. Spec + config + reference + TENSORS shapes
correct. Imports and registration work.

### 3.2 Q-quantize-on-load + QK MMA

Implement Q quantization (per-row absmax → scale → s8) inside the
kernel. Then s8 QK MMA producing s32 accumulator. Skip the
softmax + AV for now; just dump s_s32 (or its dequantized version)
as the "output" for correctness validation against an int8 reference.

### 3.3 Online softmax + AV

Add the softmax (max + exp + rescale) and the AV MMA.

### 3.4 End-to-end validation

cos_sim > 0.99 vs bf16 OwlAttn at saturated KV. If lower, iterate
on scale design (per-block vs per-row, etc.).

### 3.5 Benchmark

LFPS at 360p saturation. Compare to bf16 OwlAttn baseline.
Target: ≥1.5× LFPS.

## Estimated effort

- 3.1 skeleton: ~30 min
- 3.2 QK with int8: ~half day
- 3.3 softmax + AV: ~half day
- 3.4 validation + scale tuning: ~half day
- 3.5 bench: ~1 hour

Total: **~1.5-2 days** of focused work.

## Risks

1. **Numerical accuracy**: per-row scales might be too coarse,
   especially for attention softmax which has heavy-tailed
   distributions. SageAttention papers suggest 99%+ cos_sim is
   achievable; we'll see at our specific shape.

2. **Quantize-on-load latency**: doing Q quant inside the kernel
   adds work. Should still net-positive but the gain ratio may be
   smaller than the 2× theoretical.

3. **Mesa anv int8 MMA + smem layout**: validated up to GemmIntKernel
   sizes. OwlAttn has different sizes (KvTile=32 inside a longer K
   loop). Should work but might surface new corner cases.
