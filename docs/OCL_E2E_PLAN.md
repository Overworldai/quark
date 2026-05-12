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

Devkit access: SSHFS + `ssh devkit` per `quark/docs/DEVKIT.md`.
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

- [ ] **0.3** Create `engine/intel.py` by copy of `metal.py` +
  surgical edits per the parallel table:
  - `torch.device` → always cpu
  - quant → call `_resolve_quant_for_family(_, INTEL_GPU)`
  - VAE → `load_taehv(..., compute_units="GPU")` (fallback "CPU"
    if `compute_units="GPU"` raises — log it, set an attr we can
    introspect)
  - URI default → `<ae_uri>-openvino`
  Done: file exists, imports clean, type checks.
  notes:

- [ ] **0.4** Re-export `EngineIntel` from `engine/__init__.py`
  and wire `Engine.__new__` to dispatch on family "intel". Update
  `_detect_engine_family` to return "intel" when `_IS_OCL` is true
  on Linux.
  Done: `tests/engine/test_factory.py` gets an `EngineIntel`
  branch back (test_construct_intel_subclass_via_force +
  test_detect_intel_on_linux_ocl).
  notes:

- [ ] **0.5** Engine construction smoke on Mac under
  `QUARK_FORCE_ENGINE=intel`: model build + weight pin should
  finish without raising. The VAE load will fail (no OpenVINO
  GPU plugin on Mac); guard with `QUARK_SKIP_VAE=1` so the smoke
  doesn't depend on it.
  Done: smoke runs in <30s, `engine.model` exists, all params
  are `QuarkTensor` (no numpy carriers left).
  notes:

---

## Phase 1 — Kernel inventory

Goal: an authoritative status table of every kernel Waypoint
forward emits, and which ones the OCL lowerer handles today.

- [ ] **1.1** Stand up `scripts/dump_kernel_calls.py`: instantiate
  `Waypoint15` (no weight load), run one synthetic forward under
  a monkeypatched `Launcher.compile` that records every
  `(kernel_cls.__name__, spec, config)`, then dumps unique tuples
  to `docs/ocl_kernel_status.md` as a markdown table with columns:
  `kernel | spec_summary | config_summary | OCL status | notes`.
  Done: file exists with full kernel list.
  notes:

- [ ] **1.2** Fill the `OCL status` column. For each kernel, grep
  for visitor coverage in `quark/lower/ocl/lower.py`; classify as
  `full / partial / missing`. Only static analysis — actual
  emission tests come next.
  notes:

- [ ] **1.3** [DEVKIT] For each kernel marked `full`, run the
  matching `tests/lower/ocl/` shape (or stand one up). Record
  pass/fail in the status table.
  notes:

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

- [ ] **2A.1** Visitor coverage gaps from Phase 1 lookup. Add
  `_visit_*` for any IR op the kernel emits that the OCL lowerer
  doesn't handle. Each gap gets a `tests/lower/ocl/` shape that
  pins the visitor.
  notes:

- [ ] **2A.2** [DEVKIT] Numerics smoke: one shape per
  `mma_cfg` (m8n16k16 bf16/bf16/f32 and bf16/bf16/bf16). Run
  end-to-end through `OclDriver`, compare against PTX reference.
  Gate: cos_sim ≥ 0.9999.
  notes:

- [ ] **2A.3** [DEVKIT] μs/call bench at one production shape
  (S=4096, Dh=128, gqa_ratio=2). Land number in status table.
  Reference: OpenVINO `sdpa_opt.cl`'s Q-fragment + smem KV layout.
  notes:

### 2B — RMSNorm

- [ ] **2B.1** Visitor pass.
  notes:

- [ ] **2B.2** [DEVKIT] Smoke + bench. Reference OpenVINO
  `rms.cl` for the fused reduce_sum_squared → rsqrt → mul → add
  pattern.
  notes:

### 2C — gemm_int (s8/s32)

- [ ] **2C.1** Visitor — needs `OpSubgroupMatrixMultiplyAccumulate
  INTEL` with the `MatrixASignedComponentsKHR` etc. operand mask
  (or the cl_intel_subgroup_matrix_multiply_accumulate equivalent).
  notes:

- [ ] **2C.2** [DEVKIT] Smoke + bench at one quant shape.
  Reference: OpenVINO `gemm_tiled_opt.cl` int8 path.
  notes:

### 2D — KV cache update + small utility kernels

- [ ] **2D.1** Visitor pass for `kv_cache_update`,
  `copy_strided`, `fill_scalar`, `scalar_increment`, `cast`,
  `elementwise_binary`, `elementwise_unary`.
  notes:

- [ ] **2D.2** [DEVKIT] Bulk smoke run.
  notes:

### 2E — RoPE

- [ ] **2E.1** Verify inline-RoPE (currently embedded in Q/K
  projection) lowers through OCL. If not, extract a dedicated
  kernel and reference OpenVINO `rope_*.cl`.
  notes:

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

## Pending-devkit queue

Tasks deferred from Mac iterations because they need Intel
hardware. Run as one ssh batch when convenient.

(empty)

---

## Status snapshot

Last loop iteration: 0.2 done — `_resolve_quant_for_family` added; Metal force-bf16 now one call. QuantConfig has no s8 field, so Intel uses the same all-bf16 force.
Active phase: 0.
Next task: 0.3.
