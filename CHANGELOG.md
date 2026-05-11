# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Initial public release

First public release of quark — a typed-IR GPU kernel compiler that
lowers a single kernel description to PTX (NVIDIA) and MSL (Apple
Metal). Includes a runtime layer (`QuarkTensor`, ctypes-driven
backends with no torch dep on the inference path), a kernel registry,
an autotuning launcher, an `nn.Module`-shaped model layer, and a
standalone Waypoint-1.5 inference engine.

### Features

- **One IR, two backends.** Kernels are written once in
  `quark.lang` / `quark.blocks` and lower to PTX or MSL based on the
  active device. Adding a backend is a `@register_lowerer(family)`
  registration plus a per-op visitor table.
- **Pluggable lowerer registry + caps-driven legalization pass.**
  Vendor-neutral `subgroup_width`, per-backend MMA mnemonic tables,
  and a target-agnostic legalization pass that expands ops the
  active backend can't do natively.
- **`QuarkTensor` device-tensor type.** Single polymorphic tensor
  that wraps `libcuda` ctypes on Linux/Windows and a metal-cpp +
  nanobind shim on Apple Silicon. No torch / cupy / pycuda / triton /
  MLX in the inference hot path.
- **Pure-Python safetensors loader.** `mmap` + `cuMemHostRegister` →
  pinned DMA straight to device, no host-side numpy / torch staging.
- **`quark.functional` (`qf.*`).** Free-function call surface for
  every production kernel — `qf.gemm`, `qf.attention`, `qf.rmsnorm`,
  `qf.silu`, etc. Each call autotunes on first use and caches.
- **`quark.nn`.** PyTorch-shaped, inference-only module layer
  holding `QuarkTensor`s. `state_dict()` / `load_state_dict()`
  parity, `forward()` convention, every leaf lowering to a `qf.*`
  kernel.
- **Waypoint-1.5 reference model.** 24-layer DiT in
  `quark.models.waypoint_15`, plus `quark.Engine` — a standalone
  inference wrapper matching the legacy `world_engine.WorldEngine`
  surface.
- **TAEHV VAE on the Apple Neural Engine.** `quark.taehv` runs the
  Waypoint-1.5 VAE encoder/decoder via CoreML on the ANE, fully
  torch-free at runtime; the VAE artifacts are downloaded from
  Hugging Face on first use.
- **Production-shape autotune for both backends.** PTX m16n8k16
  bf16 / fp8 (sm_80+), per-shape NAX m16n32k16 bf16 (Apple M5+),
  with cuBLAS as a candidate alongside the in-tree GEMM kernel.
- **Kernel coverage.** GEMM (incl. fused gate/residual + silu
  epilogues), AdaRMSNorm / HeadRMSNorm / RMSNorm, OwlAttn (segment-
  sparse flash attention with inline ortho-RoPE), KVCacheUpdate,
  Patchify / Unpatchify, AdaGateResidual, EulerStep, ValueResidual,
  ControllerInputEmbedding, MLP / MLPFusion, MoE (router / inproj /
  outproj / reduce, three routing modes), SiLU, Elementwise,
  QuantizeE4M3, Randn, CopyStrided, Increment.
