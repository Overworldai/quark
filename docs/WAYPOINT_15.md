# Waypoint-1.5

A 24-layer DiT for autoregressive video generation, implemented end to
end as a `quark.nn.Module` with every leaf lowering to a `pcf.*`
kernel. Target weights: `Overworld/Waypoint-1.5-1B` on HF Hub.

The high-level entry point is `quark.Engine` — append seed frames,
generate new frames, ship pixels in / pixels out:

```python
import quark

engine = quark.Engine("Overworld/Waypoint-1.5-1B")        # default fp8
# engine = quark.Engine("Overworld/Waypoint-1.5-1B", quant="bf16")   # safe fallback
# engine = quark.Engine("Overworld/Waypoint-1.5-1B",
#                      quant=quark.models.waypoint_15.QuantConfig(moe="bf16"))

engine.append_frame(seed_frame_uint8)                     # seed
for _ in range(n_frames):
    rgb = engine.gen_frame(ctrl=ctrl_input)               # uint8 [T,H,W,3]
```

Lower-level (skip the VAE wrapper, drive the DiT directly):

```python
from quark.models.waypoint_15 import Waypoint15, Waypoint15Config

model = Waypoint15.from_pretrained("Overworld/Waypoint-1.5-1B")
model.prepare(shuffle=True)        # fp8 defaults to cfg.quant.linear
out = model(x, sigma_idx=0, frame_t=0)
```

Per-component quantization is on `Waypoint15Config.quant` (a
`QuantConfig` with `linear` / `kv_cache` / `attn_compute` / `moe`,
each `"fp8"` or `"bf16"`). This replaces the previous
`QUARK_NO_FP8` / `QUARK_MOE_NO_FP8` env-var knobs and the
`use_fp8` / `use_f16` config booleans.

`scripts/generate.py` is the end-to-end driver: downloads weights,
loads through `nn.io`, optionally captures CUDA graphs, replays per
frame, and writes a latent `.npy` (decoded to mp4 by
`scripts/decode_latents.py`).

```bash
python scripts/generate.py --preset 720p --n-frames 60 --output demo.npy \
                           --b-shuffle --fp8
```

## Architecture

| | |
|---|---|
| d_model | 2048 |
| layers | 24 |
| heads / kv-heads | 32 / 16 (GQA ratio 2) |
| MLP expansion | ×4 (8192) |
| patchify | 2×2 |
| scheduler | 5 NFE/frame — 4 denoise + 1 commit |

Per-layer attention pattern: local window 16, global window 128,
dilation 8, period 4, offset −1 → layers {3, 7, 11, 15, 19, 23} run
global attention; the rest run local. Both paths go through the same
`owl_attn` kernel (segment-sparse flash attention) with different
`(num_buckets, pinned_dilation)` per layer.

## Preset geometry

| preset | pixel input | VAE latent | grid | tokens/frame |
|---|---|---|---|---|
| 360p | 640×360 | 32×16 | 16×8 | 128 |
| 720p | 1280×720 | 64×32 | 32×16 | 512 |

## Benchmarks

### RTX 5090 — 720p

```
  128 latent frames, 512 tokens/frame
  5 NFE/frame (4 denoise + 1 commit)
  NFE:  14.5 ms (68.9/s)
  LFPS: 13.8 latent frames/s
  FPS:  55.2 pixel frames/s (4× temporal upsample)
  total: 9.28 s
```

### RTX 4090 — 720p

```
  64 latent frames, 512 tokens/frame
  5 NFE/frame (4 denoise + 1 commit)
  NFE:  16.4 ms (60.9/s)
  LFPS: 12.2 latent frames/s
  FPS:  48.7 pixel frames/s (4× temporal upsample)
  total: 5.25 s
```

### World-engine baseline (RTX 5090, DiT-only, bf16, quant=None)

```
  64 latent frames, 512 tokens/frame
  5 NFE/frame (4 denoise + 1 commit)
  NFE:  26.4 ms (37.9/s)
  LFPS: 7.6 latent frames/s
  FPS:  30.3 pixel frames/s (4× temporal upsample)
  total: 8.44 s
```

Same model definition, same weights, same NFE schedule. quark on
the same GPU delivers **1.8× NFE throughput and 1.8× LFPS** over the
bf16 `world_engine` torch reference, driven by fp8+shuffle GEMMs
end-to-end and the fused segment-sparse `owl_attn` kernel.

Reproduce with `scripts/bench_world_engine.py` for the baseline and
`scripts/generate.py --profile` for a per-module breakdown of the
quark path.

## What the numbers do NOT include

- **VAE decode** (`TAEHV`) — torch-only, runs after the DiT loop in
  `scripts/generate.py` just to produce an mp4. Excluded from the
  NFE/LFPS numbers above.
- **First-frame warmup** — autotune search on cold caches takes
  ~10–30 s. Subsequent runs read `~/.cache/quark/*.json` and
  start in milliseconds.
- **Safetensors load** — timed separately in generate.py output. On
  a warm page cache with pinned-DMA path: ~0.3 s for the 1 B
  checkpoint.

## Tuning surface

`model.prepare(shuffle=True, fp8=True)` does three things:

1. Recursively calls `prepare()` on every sub-module. `nn.Linear`
   uses this to pre-shuffle its weight into the `b_shuffle=True`
   GEMM layout — amortizes the permutation across startup instead
   of paying it every forward.
2. Quantizes eligible `nn.Linear` weights to e4m3 and caches the
   per-tile scales.
3. Captures a CUDA graph per `sigma_idx` in the denoise schedule,
   so the per-frame loop becomes a `graph.replay()` instead of a
   fresh launch-list assembly.

See [WEIGHTS.md](WEIGHTS.md) for the loader path and
[FUNCTIONAL.md](FUNCTIONAL.md) for the kernel call surface.

## Source

- [src/quark/models/waypoint_15.py](../src/quark/models/waypoint_15.py) — model + config + generate-one-frame
- [src/quark/nn/layers.py](../src/quark/nn/layers.py) — leaf `nn.Module`s
- [scripts/generate.py](../scripts/generate.py) — end-to-end video generator
- [scripts/bench_world_engine.py](../scripts/bench_world_engine.py) — world-engine torch baseline
