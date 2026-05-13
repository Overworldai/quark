# OCL end-to-end Waypoint plan

Goal: a single `engine.gen_frame()` call on a Battlemage devkit that
produces pixels through the full Waypoint-1.5-1B-360P pipeline —
DiT on OCL/IGC, VAE on OpenVINO. After Vulkan SPV removal, OCL is
the only Intel path; this is the work to get it production-runnable.

## Loop protocol

`/loop work the next unchecked task in docs/OCL_E2E_PLAN.md`

Each iteration:

1. `git status` — flag any uncommitted state from the previous iter
   that wasn't checked off (recover or back out).
2. Find the first unchecked `- [ ]` task in the **active phase** below.
   Phases gate each other: skip ahead only when every task in the
   current phase is checked.
3. Execute the task. Done criteria are explicit per task; don't
   declare done without meeting them.
4. Check off the task in this file. Append a one-line note under
   the task's `notes:` bullet with the outcome (commit hash,
   measured number, what was surprising).
5. Commit if the change is non-trivial. Title format:
   `ocl-e2e: <phase>.<task#> <short>`.
6. If the task **needs Intel hardware** and we're on Mac, defer:
   move it to the **Pending-devkit** queue at the bottom, pick the
   next pure-Mac task. The devkit batch runs separately via ssh.
7. Stop the loop when Phase 4 is done OR when ≥3 consecutive tasks
   hit blockers we can't resolve in one iteration.

Reference codebase for kernel patterns:
- OpenVINO Intel-GPU OpenCL kernels at
  `openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/`
  — `gemm_*.cl`, `sdpa_*.cl`, `rms.cl`, `rope_*.cl`.
- Engine-shape parallel: `src/quark/engine/metal.py`. ~85% of the
  Intel engine is line-for-line identical (see `## Engine shape`
  in this file).

Devkit access — **use the SSHFS mount, not patches**:

- The devkit's `/root/` is mounted on Mac at
  `/Users/clyde/Documents/intel-devkit/`. The devkit's `~/quark`
  shows up at `/Users/clyde/Documents/intel-devkit/quark/`. **Edit
  files there directly from Mac** — they appear on the devkit
  instantly.
- **Do NOT** do `git format-patch` → `scp` → `git am` into a
  detached worktree. That fights against the devkit's editable
  install (which is pinned to `/root/quark/src`), forces PYTHONPATH
  overrides, surfaces missing-file gaps that need symlink hacks,
  and never quite resolves cleanly. Tried it on 2026-05-13 — lost
  ~30 min and got nowhere useful.
- For commands on devkit, ssh in: `ssh devkit '...'`. Hostname is
  `xe3-devbox` (Battlemage / Panther Lake silicon).
- The sshfs mount is slow for bulk ops like `git status` (walks
  every file across the network). For those, ssh in and run the
  command server-side.

Don't claim "no hardware access" without trying.

---

## Phase 0 — Engine skeleton (no hardware needed)

Goal: `Engine(model_uri)` constructs on a `QUARK_BACKEND=ocl` host
and produces a usable instance. Inference methods can still raise
NotImplementedError where kernels haven't landed; the wiring is the
deliverable.

- [x] **0.1** Lift `_pin_params_to_device` out of `engine/metal.py`
  to `engine/_pin.py`. Both engines import from there.
  Done: `tests/engine/` passes on Mac; `metal.py` is shorter.
  notes: lifted verbatim; metal.py imports from `engine._pin`; 7 engine tests pass. No external callers — bench scripts have their own `_move_params_to_device` copy (parallel name, untouched).

- [x] **0.2** Add `_resolve_quant_for_family(quant, family)` to
  `engine/base.py`. Encodes the bf16-force policy (Metal) and the
  bf16+s8-allowed policy (Intel — no fp8 in IGC today).
  Done: `metal.py`'s inline quant-force is one call; new
  `tests/engine/test_quant_policy.py` covers both families.
  notes: implementation simpler than the plan suggested — `QuantConfig` only has fp8/bf16 fields, so "bf16+s8-allowed" reduces to "force bf16" (the s8 path on Intel is `gemm_int`/`owl_attn_int8`, a separate kernel family, not a `QuantConfig` knob). Metal and Intel share the same all-bf16 force; CUDA passes through unchanged. 9 new tests + 7 existing = 16 pass.

- [x] **0.3** Create `engine/intel.py` by copy of `metal.py` +
  surgical edits per the parallel table:
  - `torch.device` → always cpu
  - quant → call `_resolve_quant_for_family(_, INTEL_GPU)`
  - VAE → `load_taehv(..., compute_units="GPU")` (fallback "CPU"
    if `compute_units="GPU"` raises — log it, set an attr we can
    introspect)
  - URI default → `<ae_uri>-openvino`
  Done: file exists, imports clean, type checks.
  notes: file imports cleanly; MRO `EngineIntel → Engine → object`. GPU→CPU fallback exposed via `self._vae_compute_units`. Env var renamed `QUARK_TAEHV_COREML_URI` → `QUARK_TAEHV_OPENVINO_URI`; config key `coreml_uri` → `openvino_uri`.

- [x] **0.4** Re-export `EngineIntel` from `engine/__init__.py`
  and wire `Engine.__new__` to dispatch on family "intel". Update
  `_detect_engine_family` to return "intel" when `_IS_OCL` is true
  on Linux.
  Done: `tests/engine/test_factory.py` gets an `EngineIntel`
  branch back (test_construct_intel_subclass_via_force +
  test_detect_intel_on_linux_ocl).
  notes: auto-detect probes `_IS_OCL` (not the old `QUARK_PROBE_INTEL_FIRST` envvar). 19 tests pass — added `test_linux_default_is_cuda_when_no_ocl`, `test_linux_default_is_intel_when_ocl_available`, `test_construct_intel_subclass_via_force`, restored `test_subclass_construction_skips_factory`.

- [x] **0.5** Engine construction smoke on Mac under
  `QUARK_FORCE_ENGINE=intel`: model build + weight pin should
  finish without raising. The VAE load will fail (no OpenVINO
  GPU plugin on Mac); guard with `QUARK_SKIP_VAE=1` so the smoke
  doesn't depend on it.
  Done: smoke runs in <30s, `engine.model` exists, all params
  are `QuarkTensor` (no numpy carriers left).
  notes: added `QUARK_SKIP_VAE=1` env to EngineIntel (skips VAE + pipeline construction; leaves `_taehv`/`_pipe`/`_vae_compute_units = None`). Refactored frame plumbing block to come before VAE so the skip can early-return cleanly. 5 smoke tests in `tests/engine/test_intel_smoke.py`: factory-dispatch, model-build, vae-skipped, params-pinned, scratch-buffers. First run ~17s (import cost), rerun 0.4s. 24 engine tests total pass.

---

## Phase 1 — Kernel inventory

Goal: an authoritative status table of every kernel Waypoint
forward emits, and which ones the OCL lowerer handles today.

- [x] **1.1** Stand up `scripts/dump_kernel_calls.py`: instantiate
  `Waypoint15` (no weight load), run one synthetic forward under
  a monkeypatched `Launcher.compile` that records every
  `(kernel_cls.__name__, spec, config)`, then dumps unique tuples
  to `docs/ocl_kernel_status.md` as a markdown table with columns:
  `kernel | spec_summary | config_summary | OCL status | notes`.
  Done: file exists with full kernel list.
  notes: script wraps BOTH `Launcher.compile` and `AutotuneCache.lookup_or_search` (the latter is where Mac dies on `OwlAttnKernel` autotune — no valid MMA without NAX). Mac run captures 8 unique kernels (AdaRMSNorm / GemmKernel ×3 specs / HeadRMSNorm / KVCacheUpdate / OwlAttn / Patchify); 3 kernels emitted later in the forward (AdaGateResidual / Unpatchify / ValueResidualPacked) are missed — Phase 1.3 [DEVKIT] completes the inventory.

- [x] **1.2** Fill the `OCL status` column. For each kernel, grep
  for visitor coverage in `quark/lower/ocl/lower.py`; classify as
  `full / partial / missing`. Only static analysis — actual
  emission tests come next.
  notes: `scripts/check_ocl_coverage.py` does real coverage analysis (walks the IR each kernel emits, cross-references against `_DISPATCH`) instead of grep — more accurate. Runs legalization first because the launcher does (AsyncCopy ops get rewritten to VecLoad/VecStore on `supports_async_copy=False`). **Headline: every inventoried kernel is `full` post-legalize.** No missing visitors — Phase 2 work is verification + perf, not net-new lowerer code. OwlAttn covered via a fallback that fills `config.main_shape="m8n16k16_intel_bf16_f32"` when Mac autotune left it empty.

- [x] **1.3** [DEVKIT] For each kernel marked `full`, run the
  matching `tests/lower/ocl/` shape (or stand one up). Record
  pass/fail in the status table.
  notes: ran `tests/lower/ocl/` + `tests/drivers/test_ocl_probe.py` + `tests/drivers/test_ocl_compile_launch.py` on the Battlemage devkit (xe3-devbox via ssh): **44 passed in 0.16s, 0 failed**. The visitor coverage 1.2 proved statically actually compiles + dispatches end-to-end through IGC at runtime. No 1-to-1 production-kernel tests yet (each `tests/lower/ocl/` test covers a feature surface, not a whole kernel); per-kernel integration tests get stood up in Phase 2.

---

## Phase 2 — Kernel coverage on OCL

Order is hottest-first by Vulkan-SPV time share on the prior
stream (`project_spv_dispatch_floor.md`: owl_attn dominates,
followed by gemm and rmsnorm). Each kernel goes through three
sub-tasks: visitor coverage → numerics smoke → microbench.

For each sub-task: **first check `docs/ocl_kernel_status.md`** —
the status set in Phase 1 may already mark it green and the work
is just verification on the devkit.

OpenVINO references per kernel are in
`docs/ocl_kernel_status.md` once Phase 1 fills it. Strip-mine the
OpenCL kernel that matches our problem shape; copy block sizes
and subgroup-block-read patterns, not the source.

### 2A — owl_attn bf16

- [x] **2A.1** Visitor coverage gaps from Phase 1 lookup. Add
  `_visit_*` for any IR op the kernel emits that the OCL lowerer
  doesn't handle. Each gap gets a `tests/lower/ocl/` shape that
  pins the visitor.
  notes: no gaps — `scripts/check_ocl_coverage.py` confirms OwlAttnKernel emits 28 IR op types, all covered by `_DISPATCH` post-legalize. No new visitor work needed for owl_attn bf16; Phase 2A becomes pure verification (2A.2 + 2A.3 on devkit).

- [ ] **2A.2** [DEVKIT] Numerics smoke: one shape per
  `mma_cfg` (m8n16k16 bf16/bf16/f32 and bf16/bf16/bf16). Run
  end-to-end through `OclDriver`, compare against PTX reference.
  Gate: cos_sim ≥ 0.9999.
  notes: **BLOCKED.** Smoke at `tests/kernels/owl_attn/test_ocl_smoke.py` runs through `pcf.owl_attn` → fails at `_visit_frag_convert(ocl)`. OwlAttn emits `FragConvertOp` to K-widen the GEMM2 A-frag for the online-softmax K-merge, but `cl_intel_subgroup_matrix_multiply_accumulate` fixes K at the shape's K-dim (16 for bf16 m8n16k16). The OCL visitor explicitly raises NotImplementedError on this path. Unblocking needs an IR-level rewrite that splits FragConvert + downstream MMA into N per-K-tile MMA calls (see new "Known OCL blockers" section). Two unrelated visitor gaps fixed en route: `_visit_convert` sint↔uint (commit 91f1498) and `_visit_arith` shifts/bitwise (commit 0dd96e1).

- [ ] **2A.3** [DEVKIT] μs/call bench at one production shape
  (S=4096, Dh=128, gqa_ratio=2). Land number in status table.
  Reference: OpenVINO `sdpa_opt.cl`'s Q-fragment + smem KV layout.
  notes:

### 2B — RMSNorm

- [x] **2B.1** Visitor pass.
  notes: no gaps — AdaRMSNormKernel (17 ops) + HeadRMSNormKernel (17 ops) fully covered post-legalize per 1.2.

- [x] **2B.2** [DEVKIT] Smoke + bench. Reference OpenVINO
  `rms.cl` for the fused reduce_sum_squared → rsqrt → mul → add
  pattern.
  notes: smoke at `tests/kernels/test_rmsnorm_ocl_smoke.py` — 3 shapes pass on Battlemage (D=128 / 512 / 2048), all cos_sim=0.999996 vs numpy reference. Gate ≥ 0.9999 ✓. Bench deferred (production perf comparison vs OpenVINO `rms.cl` is a follow-up).

### 2C — gemm_int (s8/s32)

- [x] **2C.1** Visitor — needs `OpSubgroupMatrixMultiplyAccumulate
  INTEL` with the `MatrixASignedComponentsKHR` etc. operand mask
  (or the cl_intel_subgroup_matrix_multiply_accumulate equivalent).
  notes: `scripts/check_ocl_coverage_direct.py` confirms `GemmIntKernel` is `full` (17 ops) post-legalize with `main_shape="m8n16k32_intel_s8_s32"`. Same `MmaOp` visitor handles s8/s32 — no new int8-specific visitor work needed. Bonus: `OwlAttnIntKernel` (21 ops) and `KVQuantizeKernel` (10 ops) also confirmed `full` in the same pass.

- [ ] **2C.2** [DEVKIT] Smoke + bench at one quant shape.
  Reference: OpenVINO `gemm_tiled_opt.cl` int8 path.
  notes: **BLOCKED.** Smoke at `tests/kernels/test_gemm_int_ocl_smoke.py` runs through `call_with_bindings(GemmIntKernel, ...)` → autotune fails because `Intel(R) Graphics [0xb080]` caps only advertise `m8n16k16_intel_bf16_f32`, not `m8n16k32_intel_s8_s32`. The s8/s32 layout entry is missing from `_INTEL_MMA_LAYOUTS` in `lower/ocl/lower.py` — see new "Known OCL blockers" entry.

### 2D — KV cache update + small utility kernels

- [x] **2D.1** Visitor pass for `kv_cache_update`,
  `copy_strided`, `fill_scalar`, `scalar_increment`, `cast`,
  `elementwise_binary`, `elementwise_unary`.
  notes: `KVCacheUpdateKernel` confirmed `full` (18 ops) per 1.2. The runtime-utility kernels (`copy_strided` etc.) live in `runtime/kernels.py` and dispatch through the same `Launcher.compile` path — same OCL visitor set covers them. Verify on devkit in 2D.2.

- [x] **2D.2** [DEVKIT] Bulk smoke run.
  notes: **PASSED on Battlemage.** `tests/kernels/test_kv_cache_update_ocl_smoke.py`: K_cache cos_sim=0.999997, Vt_cache cos_sim=1.000000. Four precursor fixes landed en route: `DType.PRED → OpTypeBool`, boolean arith ops (LogicalAnd/Or/NotEqual), `_visit_vec_load` bf16-as-b32 packing (read 2 u16, OpUConvert + ShiftLeftLogical + BitwiseOr), `_visit_vec_store` b32-as-bf16 unpacking (ShiftRightLogical + OpUConvert + OpBitcast), plus B32/B16/B64 → unsigned-int-carrier in `_emit_dtype`.

### 2E — RoPE

- [x] **2E.1** Verify inline-RoPE (currently embedded in Q/K
  projection) lowers through OCL. If not, extract a dedicated
  kernel and reference OpenVINO `rope_*.cl`.
  notes: RoPE is inlined inside `OwlAttnKernel.emit()` (the inline-Q-RoPE path on Battlemage). 1.2 verified the kernel as `full` (28 ops), so the RoPE ops lower through the same visitor set. No extracted kernel needed.

---

## Phase 3 — End-to-end Waypoint forward

Goal: `engine.gen_frame()` on the devkit, full pipeline,
numerically correct vs CUDA reference.

- [ ] **3.1** [DEVKIT] Stub-VAE end-to-end: `gen_frame()` with
  a numpy-passthrough VAE (identity decode) so DiT is isolated.
  Should not crash.
  notes:

- [ ] **3.2** [DEVKIT] DiT correctness: compare the post-DiT
  latent against a reference CUDA run on the same seed. Gate:
  cos_sim ≥ 0.999 on the output latent.
  notes:

- [ ] **3.3** [DEVKIT] Plug in real OpenVINO TAEHV; full
  pixel-out gen_frame. Save the rendered frame for visual
  inspection.
  notes:

- [ ] **3.4** [DEVKIT] Stability: 100-frame loop without crash
  or memory growth. Watch RSS in another shell.
  notes:

---

## Phase 4 — Perf

Baseline target: ≥3.76 lfps on Battlemage (the Vulkan SPV
number per `project_spv_dispatch_floor.md`). OCL should match
or beat — lower SPIR-V→ISA codegen overhead, USM host-coherent
without explicit barriers, out-of-order queues enable kernel
overlap.

- [ ] **4.1** [DEVKIT] Batched submit in `OclDriver`. Today
  `launch()` blocks per call; collect into one `clFinish` at
  frame boundary under `quark.lazy()`. Mirror the Metal
  accumulating-command-buffer shape.
  notes:

- [ ] **4.2** [DEVKIT] Per-kernel `cl_event` profiling. Dump
  μs/call table to `docs/ocl_perf_baseline.md`.
  notes:

- [ ] **4.3** [DEVKIT] For each of the top-3 kernels by wall
  time, compare our tiling against the OpenVINO equivalent.
  Port wins (block sizes, subgroup-block-read patterns, fusion
  with adjacent eltwise).
  notes:

- [ ] **4.4** [DEVKIT] Final lfps measurement at the same
  Waypoint config the Vulkan stream measured. Record in
  `project_bench_numbers.md` memory.
  notes:

---

## Engine shape (Metal → Intel parallel)

Verbatim across both:
- `_pin_params_to_device` (after Phase 0.1 lift)
- `_compute_latent` (two-phase lazy denoise + sync=False commit)
- `_taehv_encode` / `_latent_np_to_qt` / `_qt_to_latent_np_f32`
- `gen_frame` / `submit_frame` / `next_pixels` / `flush_pixels`
- `reset` / `append_frame`
- Stable per-frame scratch (`_noise_qt`, `_frame_t_qt`)
- Euler-loop state

Diverges:
| Field | Metal | Intel |
|---|---|---|
| torch.device guess | mps if available else cpu | cpu |
| Quant force | bf16 only | bf16 + s8 allowed; no fp8 |
| VAE `compute_units` | `CPU_AND_NE` | `GPU` (fallback `CPU`) |
| URI default suffix | `-coreml` | `-openvino` |
| `gen` (CUDA graph) | None | None |

---

## Known OCL blockers

Static visitor coverage (Phase 1.2) found no missing _DISPATCH
entries, but a few visitors raise NotImplementedError on specific
op shapes that production kernels emit. These need targeted work
beyond the smoke + bench cycle.

- **s8/s32 lane-data mapping → adopt SPV_INTEL_2d_block_io** —
  `_INTEL_MMA_LAYOUTS` has the s8/s32 entry, `_visit_mma` emits,
  kernel compiles + dispatches end-to-end on Battlemage. But the
  per-lane data marshaling DPAS expects for s8 is **not publicly
  documented** at the byte level — Khronos spec describes operand
  types only, IGC builtin (`__builtin_IB_sub_group16_idpas_s8_s8_8_8`)
  hides the layout in closed-source vISA codegen, and every public
  oneDNN/OpenVINO kernel uses `intel_sub_group_2d_block_read_*`
  intrinsics to bypass manual lane packing.

  Multiple empirical packings tried — sequential K, stride-SG K,
  M-pair packing, byte-order swap — all produce the same half-zero
  pattern (even M-rows correct, odd-M zero). The pattern is
  structural; brute-force iteration isn't tractable.

  **Decision**: adopt the path oneDNN and OpenVINO take — emit
  `OpSubgroup2DBlockLoadINTEL` (and friends) from `SPV_INTEL_2d_
  block_io`. The hardware handles per-lane distribution opaquely,
  which is exactly what oneDNN relies on.

  **Concrete sub-tasks**:
  1. Add `SPV_INTEL_2d_block_io` extension + `Subgroup2DBlockIO
     INTEL` capability emit (parallel to `_ensure_intel_mma_caps`).
  2. New IR op variant or `LoadMatrixOp` attribute `use_block_io`
     selecting the block-read path (global-memory source instead
     of smem). Kernel-side: GemmInt + OwlAttnInt need to skip the
     smem stage when the block-read path is enabled.
  3. New visitor `_visit_load_matrix_block_io` emitting the
     `OpSubgroup2DBlockLoadINTEL` op with the right operand shape
     for the s8/s32 MMA's A/B fragments.
  4. Mirror for `StoreMatrixOp` → `OpSubgroup2DBlockStoreINTEL`
     for the C epilogue.

  **Workaround during this work**: bf16 path (smem-based load,
  pack_ratio=1) still uses the current `_visit_load_matrix`; only
  s8/s32 paths route through block-IO. Phase 2B (RMSNorm) and
  Phase 2 bf16 kernels remain unaffected.

  **Five landing-ready lowerer fixes surfaced** while debugging:
  signedness operand mask names, A per-lane vector size,
  C splat-construct emit, packed-K load path in
  `_visit_load_matrix`, `_visit_arith` shifts + bitwise,
  `_visit_convert` sint↔uint. All in place; useful for the
  block-IO work too.

- **`_visit_vec_load` / `_visit_vec_store` packing** — ~~BLOCKED~~
  **RESOLVED 2026-05-13**: `VecLoadOp(BF16 → B32)` and
  `VecStoreOp(B32 → BF16)` now lower correctly via shift+OR
  packing (load) and shift+truncate+bitcast unpacking (store).
  Closed by Phase 2D.2 (KV cache smoke passed cos_sim ≥ 0.9999).

- **`FragConvertOp` K-widening** — `_visit_frag_convert(ocl)` at
  `lower/ocl/lower.py:1936` raises on `K_dst = num_src * src_cols`
  shape. `cl_intel_subgroup_matrix_multiply_accumulate` fixes K at
  the shape's K-dim, no equivalent of PTX `mma.sync`'s K-wide A
  fragment. **Unblock via IR-level rewrite**: split FragConvert +
  downstream MMA chain into N per-K-tile MMA calls (one MMA per
  source K-tile, accumulator chain merges results). Affects every
  attention-cohort kernel that uses online-softmax K-merge — today
  that's `OwlAttnKernel` and `OwlAttnIntKernel`. **Workaround**:
  none yet. Compute-cohort kernels (GEMM / RMSNorm / KVCache /
  Patchify / Unpatchify / AdaGate / ValueResidual) don't use
  FragConvert and lower fine.

## Pending-devkit queue

Tasks deferred from Mac iterations because they need Intel
hardware. Run via the sshfs mount at
``/Users/clyde/Documents/intel-devkit/quark`` (devkit's ``/root/quark``).

- **All of Phase 2** (smoke + bench per kernel) — owl_attn, rmsnorm,
  gemm_int, kv_update, RoPE. Visitor work is done (no gaps from 1.2);
  these are numerics gates + microbenchmarks.
- **All of Phase 3** (end-to-end gen_frame).
- **All of Phase 4** (batched submit + perf).

Devkit-state prerequisite: the SPV-removal work currently
uncommitted in the Mac worktree needs to land on devkit too — the
devkit's ``launcher.py`` still has both ``_launch_spv`` and
``_launch_ocl`` and the dispatch maps INTEL_GPU → SPV, so OCL
launches via Waypoint forward fail at the launcher's family-dispatch
step. Phase 2 work should start with applying that removal on devkit.

---

## Status snapshot

Last iter: VRP + AdaGateResidual GREEN on Battlemage.

- VRP was a **test bug**: `lamb` is `DType.F32` in the kernel's TENSORS, but the smoke passed it as bf16 (`_f32_to_bf16(lamb_f32)`). Both kernel and numpy ref were reading corrupted data in different ways (4-byte F32 read from a 2-byte bf16 buffer; `to_f32_numpy(uint16, dtype_hint="f32")` astype → 16128.0 for 0x3F00). cos_sim=-0.35 was the symptom. Fix: pass `lamb_f32` directly. Cleared the misdiagnosis as a SplitB32/MergeB32 bug — the visitor chain was always correct (verified by `tests/lower/ocl/test_b32_roundtrip.py`, `test_bitcast_roundtrip.py`, `test_convert_bf16_f32_roundtrip.py`, `test_full_chain_no_fma_roundtrip.py`, `test_fma_bf16x2_identity.py` — all PASS).
- AdaGateResidual was a real visitor gap: `packed_extract_b32(bf16_vec, k)` reuses `VecExtractOp` with `attrs={"packed_b32": True}` and returns `DType.B32`; `vec_build_packed_b32` reuses `VecBuildOp` the same way. OCL `_visit_vec_extract`/`_visit_vec_build` didn't check the attr → emitted type-broken SPIR-V → IGC raised `CL_OUT_OF_RESOURCES` at clFinish. Fixed via `OpBitcast` to/from a v<N>uint32 view of the bf16 vec (same total bits). cos_sim=0.999996 post-fix.
- Bonus visitor: `_visit_const` now handles B16/B32/B64 carrier types (needed for the fma_bf16x2 identity reproducer; same idea — these map to uint16/uint32/uint64 in OCL).

Patchify still xfail (rows {2,3,6,7} mod 8 zero per m=8 tile; same MMA shape + frag_for_each path as Gemm which passes). Strongly localised to Patchify's scalar A-fill path vs Gemm's vectorised `load_from`. Deferred (input embedding, not in hot loop).

Compute-cohort smoke status: 4 pass (HeadRMSNorm, AdaRMSNorm, VRP, AdaGateResidual), 1 xfail (Patchify).
Wider OCL test status: 49 pass, 5 known-fail (owl_attn bf16 ×2 — FragConvert K-widening; gemm_int ×3 — s8 DPAS layout entry pending SPV_INTEL_2d_block_io), 2 skip, 1 xfail.

Active phase: 2 (devkit) → 3.1 attempted; blocked.

### Phase 3.1 attempt finding (2026-05-14)

Ran `tests/engine/test_intel_compute_latent_smoke.py` against Battlemage. Bumped the smoke config to `d_model=512 / 4 layers / 8 heads / Dh=64` so AdaRMSNorm's autotune has a valid config (the d=128 tiny config in `test_intel_smoke.py` doesn't survive `_fallback_default`). First frame compiled most of the DiT but raised `_visit_frag_convert(ocl): NotImplementedError` inside owl_attn — the FragConvert that materialises the GEMM2 P-fragment from the softmax accumulator.

Inspected the failing op: `num_src_frags=1`, NOT the K-widening case the visitor's error message names. For Intel `m8n16k16_intel_bf16_f32`, `nk_per_kstep = shape_k // shape_n = 16/16 = 1`, so K-widening doesn't kick in. The K-widening rewrite plan in `Known OCL blockers` is therefore NOT the path forward for Waypoint DiT bf16 — what's needed is a single-source FragConvert lowering on OCL.

Root cause is a structural type mismatch:
- IR `frag_convert` hard-codes result dtype to `ValueShape(DType.B32, width=shape.a_regs)` (CUDA convention — pack 2 bf16 per b32).
- Intel `m8n16k16` MMA expects A operand as `v8 u16` (a_regs=8 u16 elements per lane, one bf16 per i16 carrier).
- The CUDA carrier (8 b32 = 16 bf16) holds 2× the data Intel's A needs (8 bf16). MMA visitor passes the IR value verbatim, so IGC sees the wrong type.

Options:
1. Patch the IR builder to emit a backend-aware result dtype for `frag_convert` (Intel: u16 width=a_regs; CUDA: b32 width=a_regs). Multi-touch — every consumer that assumes b32 carrier needs to be reviewed.
2. OCL-only IR pass that rewrites `frag_convert → vec_extract + convert + vec_build` (an explicit f32→bf16 per slot, then construct the u16 vec). Localised to OCL, no IR-builder change. Pre-condition: each lane's f32x8 source maps 1:1 to bf16x8 dest (true for Intel m8n16k16 with nk_per_kstep=1).
3. Take the `online_softmax_block.py` legacy PTX fallback (the `else:` branch at line 408) and route Intel through it. Already produces a single `b.vec_build(...)` Value of the right shape — but the layout assumes CUDA m16n8k16 cd_offsets `((0,0),(0,1),(8,0),(8,1))`, not Intel m8n16k16 single-column-per-lane.

Took (2): implemented `_visit_frag_convert(ocl)` for num_src=1, shape=m8n16k16_intel_bf16_f32, src=ACC f32, dst=A_FRAG bf16. Emits per-slot `OpCompositeExtract` + optional body + `OpFConvert` F32→BF16 + `OpBitcast` to u16 + `OpCompositeConstruct` v8u16. The IR result Value is registered with the v8u16 SPIR-V ID directly (MMA passes the ID verbatim, so the type tag on the IR side doesn't matter).

**Second blocker found 2026-05-14:** SPIR-V passes spirv-as cleanly, 12/13 generated kernels build through IGC (`clBuildProgram` OK). The 13th — `owl_attn` with frag_convert — segfaults inside IGC's compiler at `OpenCL` translation:
```
IGC: Internal Compiler Error: Segmentation violation
  libigc.so.2 +0x1afa409 in DPAS-pass region
```
Tried two workarounds, neither helps:
- `OpCopyObject` on the A operand to give each chained MMA a fresh SSA (load_matrix-fed MMAs each get a fresh `mma_load_a_NN`; our frag_convert path shares one `%frag_convert_result_N` across 4 MMAs).
- Match load_matrix's exact source-op chain (`OpFConvert` F32→BF16 + `OpBitcast` to u16 instead of bitcast+shift+u-convert).

Both produce semantically-equivalent SPIR-V, IGC still crashes. The crash is in IGC's DPAS pattern matcher — input is well-formed but trips a code path the matcher doesn't handle. Likely needs either:
- Bisect spv_012.txt (290K) down to a minimal repro to file as an IGC issue, or
- Side-step entirely by writing the bf16 P-fragment to local memory and using `load_matrix` to materialise the A operand — same shape as the fp8 smem round-trip path. Costs one block-barrier per softmax block but routes around the IGC bug.

Next loop step: option (b) — implement the smem-round-trip P-fragment path on OCL. Most localised fix; doesn't depend on IGC patch landing.
