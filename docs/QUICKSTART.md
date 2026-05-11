# Quickstart

Get from a fresh checkout to a running Waypoint-1.5 inference loop.
Targets one of:

- **Linux + NVIDIA** (RTX 30/40/50-series, sm_80+ recommended for fp8)
- **macOS** Apple Silicon, M5 or later

For the kernel framework internals (IR, lowerers, autotune, registry),
see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Install

### Direct git pin only

```bash
uv add git+https://github.com/Overworldai/quark.git
```

### Development clone

```bash
git clone https://github.com/Overworldai/quark
cd quark
make setup
```

`make setup` creates `.venv/` pinned to Python 3.13, installs the
package in editable mode with the `[dev]` extras, and wires up the
pre-commit hook (`ruff format`, `ruff check`, `ty check`, pytest
smoke).

### Prerequisites

| platform | needs |
|---|---|
| Linux + NVIDIA | a recent NVIDIA driver (≥ 550 series for fp8), and `libcuda.so` on the loader path (standard for any CUDA-capable host) |
| macOS Apple Silicon | Xcode Command Line Tools (`xcode-select --install`) for the C++ extension build; an M5 or later for NAX MMA paths |
| both | Python ≥ 3.11, `uv` if using `make setup` |

The macOS install also compiles a small metal-cpp + nanobind extension
(`_metal_dispatch`) on first install — this is the Metal driver. On
Linux/Windows the extension is gated off; install is pure Python.

### Verify

```python
import quark
from quark.device import current_device

print(current_device().caps.name)        # e.g. "NVIDIA GeForce RTX 4090"
print(current_device().caps.family)      # DeviceFamily.CUDA or .METAL
```

---

## 2. First inference run

```python
import quark
from PIL import Image
import numpy as np

# 1. Load the model. ``Engine`` is the high-level wrapper around the
#    Waypoint-1.5 DiT — it owns the VAE, the KV cache, and the
#    per-frame denoise loop. First call downloads weights from HF and
#    runs the autotune search (~10-30 s cold cache; later runs are
#    millisecond-fast).
engine = quark.Engine("Overworld/Waypoint-1.5-1B")     # default: fp8 on sm_89+

# 2. Seed the model with the first frame. Input is a uint8 RGB stack
#    of 4 temporally-adjacent frames (Waypoint generates 4-frame
#    chunks at a time via TAEHV's 4× temporal compression).
seed = np.asarray(Image.open("seed.jpg").convert("RGB").resize((1280, 720)))
seed_stack = np.broadcast_to(seed, (4, 720, 1280, 3)).copy()    # [4, H, W, 3]
engine.append_frame(seed_stack)

# 3. Generate. Each ``gen_frame`` call produces another 4-frame
#    pixel chunk conditioned on the controller input.
ctrl = quark.CtrlInput(button={32}, mouse=(0.1, 0.0))   # 32 = SPACE; mouse dx/dy
frame_stack = engine.gen_frame(ctrl=ctrl)              # [4, H, W, 3] uint8

# 4. Each call advances the model's internal frame counter by one
#    latent frame (= 4 video subframes). Repeat with whatever
#    controller sequence you want.
for _ in range(60):
    frame_stack = engine.gen_frame(ctrl=quark.CtrlInput())
```

### Choosing a model

| model URI | preset | params | notes |
|---|---|---|---|
| `Overworld/Waypoint-1.5-1B` | 720p | 1 B | default; 512 tokens/frame |
| `Overworld/Waypoint-1.5-1B-360P` | 360p | 1 B | smaller compute, 128 tokens/frame |
| `Overworld-Models/WP1.5-4B-BetaH` | 720p | 4 B | larger model |

---

## 3. Quantization

The default `quark.Engine(...)` uses end-to-end fp8 on sm_89+ NVIDIA
GPUs (4090, 5090, H100, B100…). Drop to bf16 with `quant="bf16"` if
you hit precision issues or your hardware lacks fp8 MMA support:

```python
engine = quark.Engine("Overworld/Waypoint-1.5-1B", quant="bf16")
```

Per-component control via `QuantConfig` (e.g. keep MoE experts in
bf16 while everything else is fp8):

```python
from quark.models.waypoint_15 import QuantConfig

engine = quark.Engine(
    "Overworld-Models/WP1.5-4B-BetaH",
    quant=QuantConfig(linear="fp8", kv_cache="fp8",
                      attn_compute="fp8", moe="bf16"),
)
```

Apple Silicon forces all-bf16 (Metal has no native fp8 in MSL).

---

## 4. Common runtime knobs

| knob | location | effect |
|---|---|---|
| `quant=` | `Engine(...)` ctor | `"fp8"` / `"bf16"` / `QuantConfig(...)` |
| `taehv_cache_dir=` | `Engine(...)` ctor (Apple) | where to mirror the CoreML VAE artifacts; defaults to `$QUARK_TAEHV_CACHE` then `~/.cache/quark/taehv` |
| `QUARK_DISABLE_CUBLAS=1` | env | force the PTX GEMM path instead of cuBLAS |
| `QUARK_DISABLE_AUTOTUNE=1` | env | skip the autotune search; use bundled configs |
| `QUARK_FORCE_CUBLAS=1` | env | shortcut cuBLAS-eligible specs past the autotuner (default on) |
| `engine.gen_frame(return_img=False)` | call site | skip the VAE decode (returns `None`); useful for DiT-only benchmarks |

The autotune cache lives at `~/.cache/quark/`. Clearing it forces a
re-search on next run. Cached configs persist across reboots and across
quark versions (keyed on the spec hash + device cap fingerprint).

---

## 5. Driving the DiT directly

Skip the `Engine` wrapper when you want to inspect the model layer or
plug into your own VAE / scheduler:

```python
from quark.models.waypoint_15 import Waypoint15

model = Waypoint15.from_pretrained("Overworld/Waypoint-1.5-1B")
model.prepare(shuffle=True)        # bake fp8 quantization + weight shuffle
                                    # for the b_shuffle=True GEMM fast path

# Forward pass — single denoise step. ``sigma_idx`` indexes into
# ``cfg.scheduler_sigmas``; ``frame_t`` is the per-batch frame counter
# the KV cache uses to address the ring buffer.
out = model(x, sigma_idx=0, frame_t=ft)
```

`x` is a `QuarkTensor` shaped `[1, channels * H * W]` (flat-2D
latent). The 5-step denoise + commit loop in `Engine.gen_frame` is
implemented in `quark.models.waypoint_15.GenerateFrame`; the source
file is short — readable end-to-end.

---

## 6. Common pitfalls

- **CUDA out of memory on first run.** Autotune materialises every
  candidate kernel during the search, which can briefly spike memory
  usage. Either run `QUARK_DISABLE_AUTOTUNE=1` once to warm the
  bundled defaults, or wait for the search to finish then retry on a
  fresh process.

- **First frame is slow.** The first `gen_frame` call captures a CUDA
  graph; subsequent calls replay it. Steady-state per-frame time
  drops by 5-10× after the first.

- **`pip install` fails on macOS with "Metal.framework not found".**
  Make sure `xcode-select --install` ran and `xcrun -sdk macosx
  --show-sdk-path` returns a valid path.

- **Numerical wedge on fp8.** Fp8 needs sm_89+ for `m16n8k32_e4m3`
  MMA. On sm_80 (A100), pass `quant="bf16"`. If the engine raises
  `QuantUnsupportedError`, that's the signal.

---

## 7. Where next

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the kernel framework
  internals: IR, pipeline, lowerers, registry, autotune, launcher,
  tensors.
- **`examples/`** — a six-step walkthrough from vector-add through
  flash attention, one folder per step. Each example is a real
  kernel that ships in the registry.
- **`src/quark/kernels/`** — every production kernel lives here, one
  folder per kernel with the same six-file layout (`kernel.py`,
  `spec.py`, `config.py`, `reference.py`, `problems.py`,
  `baselines.py`).
- **`src/quark/models/waypoint_15.py`** — the full DiT model
  definition (~600 lines, readable end-to-end).
