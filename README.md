<p align="center">
  <img src="assets/quark.png" alt="quark" width="360"/>
</p>

<h1 align="center">quark</h1>

<p align="center">
  GPU kernel compiler and runtime for efficient <a href="https://github.com/Wayfarer-Labs/world_engine">Waypoint Model</a> inference
</p>

---

Kernels are written as typed SSA IR and lowered to PTX on NVIDIA
(via `libcuda` ctypes) or MSL on Apple Silicon (via a metal-cpp +
nanobind shim). The inference path has no torch, MLX, cupy, or
pycuda dependency: device memory is owned by `QuarkTensor`, weights
load from `.safetensors` via mmap + pinned DMA, and dispatch goes
straight through the driver. Torch is an optional dev dep, used by
per-kernel correctness oracles.

The framework generalises, but Waypoint-1.5 is the model wired
end-to-end today.

## Waypoint-1.5

24-layer DiT, autoregressive video. Pixels in, pixels out:

```python
import quark

engine = quark.Engine("Overworld/Waypoint-1.5-1B")   # bf16 default; quant="fp8" on sm_89+
engine.append_frame(seed_uint8)                       # [4, H, W, 3] uint8
for _ in range(n_frames):
    rgb = engine.gen_frame(ctrl=ctrl_input)           # [4, H, W, 3] uint8
```

`quant="bf16"` if you don't have fp8. Per-component control on
`Waypoint15Config.quant` (`linear` / `kv_cache` / `attn_compute` /
`moe`, each `"fp8"` or `"bf16"`).

### Architecture

| | |
|---|---|
| d_model | 2048 |
| layers | 24 |
| heads / KV-heads | 32 / 16 (GQA ratio 2) |
| MLP expansion | ×4 (8192) |
| patchify | 2×2 |
| scheduler | 5 NFE/frame (4 denoise + 1 commit) |
| temporal compression (VAE) | 4× |

Attention is segment-sparse. Local window 16, global window 128,
dilation 8, period 4. Layers `{3, 7, 11, 15, 19, 23}` run global; the
rest run local. Both share `qf.owl_attn` — segment-sparse flash
attention with inline ortho-RoPE — and vary only on `(num_buckets,
pinned_dilation)`.

VAE is **TAEHV** (Ollin Boer Bohan,
[madebyollin/taehv](https://github.com/madebyollin/taehv), MIT). On
CUDA it runs through torch. On Apple Silicon it runs on the ANE via
CoreML — no torch at inference time.

### Preset geometry

| preset | pixel input | VAE latent | token grid | tokens/frame |
|---|---|---|---|---|
| 360p | 640 × 360 | 32 × 16 | 16 × 8 | 128 |
| 720p | 1280 × 720 | 64 × 32 | 32 × 16 | 512 |

### Performance (RTX 4090)

Median over 40–60 `gen_frame` calls, post-warmup, in-CUDA-graph.
Numbers cover the full DiT forward (attention + GEMM + norm + KV
update + epilogues). VAE decode runs after the loop and isn't counted. Requires sm_80+.

| preset | quant | per-NFE | latent FPS | pixel FPS¹ |
|---|---|---:|---:|---:|
| 360p | bf16 | 5.6 ms | 36 | 144 |
| 360p | fp8  | 4.0 ms | 50 | 199 |
| 720p | bf16 | 14.7 ms | 14 | 55 |
| 720p | fp8  | 10.1 ms | 20 | 79 |

¹ Pixel FPS = latent FPS × 4× temporal compression. One `gen_frame`
runs all 5 NFE and produces 4 subframes.

## Install

```bash
uv add git+https://github.com/Overworldai/quark.git
```

For development:

```bash
git clone https://github.com/Overworldai/quark
cd quark
make setup     # .venv (Python 3.13), deps, pre-commit
```

macOS compiles the `_metal_dispatch` extension (metal-cpp + nanobind).
Linux / Windows skip it.

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for prerequisites,
quantization knobs, and common pitfalls.

## Kernel surface

~25 kernels in `src/quark/kernels/`: GEMM (with optional fused gate +
residual and silu epilogues), segment-sparse flash attention
(`owl_attn`), KV cache update with inline RoPE, AdaRMSNorm /
HeadRMSNorm / RMSNorm, Patchify / Unpatchify, AdaGateResidual,
EulerStep, ValueResidual, ControllerInputEmbedding, MoE (router /
inproj / outproj / reduce, three routing modes), plus SiLU,
Elementwise, QuantizeE4M3, Randn.

Each kernel has a python reference, a fuzz/bench problem set, and one
or more baselines. Calling it:

```python
import quark.functional as qf
from quark.runtime.tensor import QuarkTensor

A = QuarkTensor.randn(M, K, dtype="bf16")
B = QuarkTensor.randn(N, K, dtype="bf16")
C = qf.gemm(A, B)
```

Backend follows the tensor's device — CUDA on Linux / Windows, Metal
on Apple Silicon. Same call, different lowering.

Pipeline (IR → validate → lower → driver → registry / launcher):
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Acknowledgements

- **TAEHV** — *Tiny AutoEncoder for Hunyuan Video*, Ollin Boer Bohan
  ([madebyollin/taehv](https://github.com/madebyollin/taehv), MIT).
  Vendored at commit `7dc60ec` as `src/quark/models/taehv.py`; MIT
  notice preserved in the file header.

## License

[GPL-3.0-or-later](LICENSE). Any work distributing a derivative of
quark must be GPL-3.0 too.
