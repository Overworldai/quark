# Vulkan capability probe

Self-contained ~180-line C program that queries every Vulkan capability
the SPIR-V backend's lowerer / driver / mma_registry branches on, on
every physical device the host exposes. PORTABILITY_PLAN §3.1's first
action.

## Build and run

```sh
cc -O2 -Wall -o probe_coopmat probe_coopmat.c -lvulkan
./probe_coopmat
```

Requires `libvulkan-dev`, `libvulkan1`, and a Vulkan ICD (Mesa / Intel
proprietary). Output goes to stdout; persist alongside this file as
`probe_output.<chip>.txt` for the record.

## What it dumps

For each `VkPhysicalDevice`:

* **Identity** — `deviceName`, `vendorID`, `deviceID`, API version
* **Compute limits** — `maxComputeWorkGroupInvocations`,
  `maxComputeSharedMemorySize`, `maxPushConstantsSize`,
  `maxComputeWorkGroupSize`
* **Subgroup** — `subgroupSize`, supported shader stages, supported
  ops bitfield (basic / vote / arithmetic / ballot / shuffle /
  shuffleRel / clustered / quad)
* **BFloat16 features** (`VK_KHR_shader_bfloat16`) —
  `shaderBFloat16Type`, `shaderBFloat16DotProduct`,
  `shaderBFloat16CooperativeMatrix`
* **Atomic float** (`VK_EXT_shader_atomic_float[2]`) — buffer + smem
  variants of f32 add, f16 add, f16 min/max
* **Cooperative-matrix shapes** (`VK_KHR_cooperative_matrix`) — the
  `(M, N, K, AType, BType, CType, ResultType, scope, sat)` tuples

Property2 / Features2 chained queries are loaded via
`vkGetInstanceProcAddr` explicitly — some loader / Mesa combos drop
chained sType structures when the application advertises a Vulkan API
version below 1.3 through the static dispatcher. The probe sets
`apiVersion = VK_API_VERSION_1_3` in `VkApplicationInfo` and uses the
loader-resolved function pointers to sidestep that footgun.

## Captured results

### Intel Panther Lake — Battlemage iGPU (vendor=0x8086, device=0xb080, Mesa 26.0.3)

```
limits.maxComputeWorkGroupInvocations = 1024
limits.maxComputeSharedMemorySize     = 49152  (48 KiB)
limits.maxPushConstantsSize           = 256
limits.maxComputeWorkGroupSize        = (1024, 1024, 1024)

subgroup.subgroupSize        = 32
subgroup.supportedStages     = Compute Vertex Fragment Geometry TessCtrl TessEval
subgroup.supportedOperations = basic vote arithmetic ballot shuffle shuffleRel clustered quad

bf16.shaderBFloat16Type              = yes
bf16.shaderBFloat16DotProduct        = yes
bf16.shaderBFloat16CooperativeMatrix = yes

atomic.f32Add(buf)   = yes    f32Add(smem) = NO
atomic.f16Add(buf)   = NO     f16Add(smem) = NO
atomic.f16Min(buf)   = yes    f16Min(smem) = yes

coopmat: 6 shapes  (all `subgroup` scope, no saturating-accumulation)
   M=8  N=16  K=16    f16/f16  →  f16
   M=8  N=16  K=16    f16/f16  →  f32
   M=8  N=16  K=16    bf16/bf16 → bf16
   M=8  N=16  K=16    bf16/bf16 → f32
   M=8  N=16  K=32    s8/s8    →  s32
   M=8  N=16  K=32    u8/u8    →  u32
```

(Full output in `probe_output.battlemage_ptl.txt`.)

### llvmpipe (CPU rasteriser fallback)

`subgroupSize=8`, 4 coopmat shapes all `8×8×8`. f32 atomic add available
in both buffer + smem (CPU has no actual atomic constraints). No
`bfloat16` or `bfloat16CooperativeMatrix`. Useful as an integration-
test fallback when Intel hardware is not available, **not** as a perf
target.

## Implications for §3.1 / §3.2 / §3.5

### Confirmed green
- **bf16 path intact.** `shaderBFloat16Type / DotProduct /
  CooperativeMatrix` all true. `bf16/bf16 → bf16` and
  `bf16/bf16 → f32` cooperative-matrix shapes both exposed; the quark
  "compute_dtype=bf16, acc=f32" idiom maps without rewrite.
- **Subgroup size 32.** Same as Apple, same as our existing kernel
  assumptions. Pin via `RequiredSubgroupSize 32` per §3.5; no
  per-width-variant build needed for v1.
- **All subgroup ops needed are native.** `arithmetic`, `shuffle`,
  `shuffleRel` — covers `OpGroupNonUniformAdd`, `…Shuffle`,
  `…ShuffleXor` (the butterfly reductions and lane swaps the
  kernels use).
- **48 KiB threadgroup memory.** Matches CUDA's static-smem ceiling.
  Kernels that target the static-smem CUDA path (most of them) port
  without smem rework.
- **Buffer-side f32 atomic add** native. No legalization needed for
  the GEMM split-k atomic-store path on the buffer side.

### Newly confirmed risks (PLAN §3.1's "atomic float availability"
risk now concrete)

- **`shaderBufferFloat16AtomicAdd = NO`** and `shaderShared… = NO`.
  The Phase 2 `bf16x2 atomic add → 2× scalar f16 atomic add`
  legalization **will not run on Battlemage** — the scalar f16 atomic
  add isn't exposed either. Two paths:
  1. Add a second-tier legalization that scatters bf16 partial sums
     to a separate f32 buffer, then runs a small reduce kernel on
     that buffer at synchronisation points. Cost: extra dispatch +
     extra device memory; acceptable if the kernel is split-k-rare.
  2. Skip split-k for kernels that need bf16-acc atomic on Intel,
     accept the perf hit. Reasonable for v1.
  Track as a §3.2 blocker for any kernel currently using `split_k>1`
  on bf16.
- **`shaderSharedFloat32AtomicAdd = NO`** also missing. f32 smem
  atomic add isn't available either. Kernels that use f32 atomic
  reductions in shared memory (rare today, but the plan flagged
  this as a future need) need a smem→buffer escape. Less urgent
  than f16 atomic; mark as v2.
- **Push constants capped at 256 bytes.** Smaller than CUDA (4096)
  and Metal (effectively unlimited via setBytes). Quark kernels
  carry their param pack inline; verify each kernel's
  `param_struct_size` ≤ 256 during §3.1 driver impl. Anything over
  spills to a UBO + extra descriptor.

### Per-fragment shape & tile multiples

- **Per-fragment shape is `m=8, n=16, k=16`** for fp16/bf16 (and
  `k=32` for s8/u8). Apple NAX uses `m=16, n=32, k=16`; PTX uses
  `m=16, n=8, k=16` (transposed convention). Add four entries to
  `quark.ir.mma_registry`:
  - `m8n16k16_f16_f16_subgroup`
  - `m8n16k16_f16_f32_subgroup`
  - `m8n16k16_bf16_bf16_subgroup`
  - `m8n16k16_bf16_f32_subgroup`
  Plus optional s8/u8 variants for the int8 KV path
  (`m8n16k32_s8_s32_subgroup`, `m8n16k32_u8_u32_subgroup`).
- **Tile multiples shift.** GemmKernel configs require `BM % 8 ==
  0`, `BN % 16 == 0`, `BK % 16 == 0` (or `K % 32 == 0` for the int8
  shapes). Existing Apple multiples (`BM % 16, BN % 32`) won't
  validate without a fresh autotune campaign on Intel hardware.
- **Subgroup scope only.** Workgroup-scope coopmat isn't exposed;
  the SPIR-V emit must set `OpExecutionMode SubgroupSize 32`
  consistently with `subgroupSize` (already pinned per §3.5).
- **No FP8.** No `e4m3` / `e5m2` cooperative_matrix on Battlemage.
  The fp8 KV-cache + fp8 attention paths from CUDA stay CUDA-only
  on v1. (FP8 was already CUDA-only on Apple; nothing new lost.)

### What still needs probing (deferred to v2 / v3 hardware)

Re-run on every distinct target chip and append above:

- Lunar Lake / Meteor Lake (smaller iGPU, Xe-LPG)
- Arc A-series (Alchemist) and B-series (Battlemage discrete)
- AMD RDNA3+ (when HIP/SPIR-V port lands)

Different chips will have different subgroup sizes (Meteor Lake reports
SIMD8/16/32 driver-chosen) and may not advertise the same coopmat
tuples. The `SubgroupSize` pin and the autotune cache need per-chip
data.
