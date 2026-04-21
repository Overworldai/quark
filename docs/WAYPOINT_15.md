# Waypoint-1.5

A 24-layer DiT for autoregressive video generation, implemented end to
end as a `popcorn.nn.Module` with every leaf lowering to a `pcf.*`
kernel. Target weights: `Overworld/Waypoint-1.5-1B` on HF Hub.

```python
from popcorn.models.waypoint_15 import Waypoint15, Waypoint15Config
from popcorn.nn.io import load_from_hub

cfg   = Waypoint15Config()                          # 720p defaults
model = Waypoint15(cfg)
sd    = load_from_hub("Overworld/Waypoint-1.5-1B", dtype="bf16")
model.load_state_dict(sd)
model.prepare(shuffle=True, fp8=True)
out = model(x, sigma_idx=0, frame_t=0)
```

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

Same model definition, same weights, same NFE schedule. popcorn on
the same GPU delivers **1.8× NFE throughput and 1.8× LFPS** over the
bf16 `world_engine` torch reference, driven by fp8+shuffle GEMMs
end-to-end and the fused segment-sparse `owl_attn` kernel.

Reproduce with `scripts/bench_world_engine.py` for the baseline and
`scripts/generate.py --profile` for a per-module breakdown of the
popcorn path.

## What the numbers do NOT include

- **VAE decode** (`TAEHV`) — torch-only, runs after the DiT loop in
  `scripts/generate.py` just to produce an mp4. Excluded from the
  NFE/LFPS numbers above.
- **First-frame warmup** — autotune search on cold caches takes
  ~10–30 s. Subsequent runs read `~/.cache/popcorn/*.json` and
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

- [src/popcorn/models/waypoint_15.py](../src/popcorn/models/waypoint_15.py) — model + config + generate-one-frame
- [src/popcorn/nn/layers.py](../src/popcorn/nn/layers.py) — leaf `nn.Module`s
- [scripts/generate.py](../scripts/generate.py) — end-to-end video generator
- [scripts/bench_world_engine.py](../scripts/bench_world_engine.py) — world-engine torch baseline
