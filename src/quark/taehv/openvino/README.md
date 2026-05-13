# OpenVINO TAEHV backend

Numpy-only runtime for the [TAEHV](https://github.com/madebyollin/taehv)
video autoencoder on Intel iGPU / dGPU / Arc / Battlemage / NPU /
CPU, mirroring the
[`quark.taehv.coreml`](../coreml/) backend's API.

## Quick start

### One-time export

```sh
python -m quark.taehv.openvino.export \
    --ae-repo Overworld-Models/taehv1_5 \
    --latent-height 16 --latent-width 32 \
    --cache-dir ./openvino_cache
```

Produces `<cache-dir>/<latH>x<latW>/{encoder,decoder}.{xml,bin}`
(~23 MB per resolution; weights are fp16-compressed). Requires the
`[export]` extras (torch + upstream `taehv`).

### Runtime

```python
from quark.taehv import load_taehv, PipelinedDecoder

# Backend auto-selected: ``-coreml`` on darwin, ``-openvino`` elsewhere.
# Or pick explicitly with ``backend=`` kwarg / URI suffix.
ae = load_taehv(
    "./openvino_cache",     # local dir or HF repo
    latent_height=16, latent_width=32,
    compute_units="GPU",    # GPU / CPU / NPU / AUTO
)
pipe = PipelinedDecoder(ae)

for fi, latent in enumerate(latents):
    pipe.submit(latent)
    img = pipe.next()
    if img is not None:
        save(img)
img = pipe.flush()
```

## Env-var knobs

| variable | values | default | effect |
|---|---|---|---|
| `QUARK_TAEHV_BACKEND` | `coreml` / `openvino` | platform default | force backend |
| `QUARK_TAEHV_OPENVINO_DIR` | path | unset | local export-dir shortcut |
| `QUARK_TAEHV_OPENVINO_URI` | HF repo id | unset | HF-hosted IR |
| `QUARK_TAEHV_OPENVINO_DEVICE` | `GPU` / `CPU` / `NPU` / `AUTO` | `GPU` | OpenVINO target |
| `QUARK_TAEHV_OPENVINO_THREADS` | int | 4 (CPU) | INFERENCE_NUM_THREADS cap |
| `QUARK_TAEHV_OPENVINO_PRECISION` | `f32` / `bf16` / `f16` | unset (`f32` on CPU) | INFERENCE_PRECISION_HINT |
| `QUARK_TAEHV_OPENVINO_STREAMS` | int | 1 | NUM_STREAMS (1 for latency) |
| `QUARK_TAEHV_OPENVINO_CACHE` | path | `~/.cache/quark/taehv-openvino` | local IR mirror root |

## Device fallback

Requesting `GPU` / `NPU` on a host that doesn't expose the device
emits a `RuntimeWarning` and falls back to CPU with a device-specific
install hint:

- **GPU on Linux** needs `intel-level-zero-gpu`, `level-zero`,
  `intel-opencl-icd` (apt).
- **NPU on Lunar/Meteor/Panther Lake** needs
  `intel-driver-compiler-npu`, `intel-level-zero-npu`.

The OpenVINO wheel ships the plugins themselves; only the system
loader libraries are missing on a fresh Ubuntu.

## Performance notes (Battlemage / Intel Core Ultra X7 358H)

For the TAEHV decoder at 360p (latent 16×32, 4-pixel-frames-per-call):

| precision hint | AE collect (16-frame warmup) |
|---|---|
| `f32` (default) | **32 ms** |
| `bf16` | 31 ms (marginal) |
| `f16` | 45 ms (regression) |

`NUM_STREAMS=1` is the optimum for `PipelinedDecoder` (single
in-flight depth); `NUM_STREAMS=2` regresses 78% by splitting CPU
cores across stalled streams. `PERFORMANCE_HINT=LATENCY` is set
automatically.

When the AE runs alongside a heavy GPU workload (DiT denoising),
pipeline overlap drops the AE wall contribution to **5.6 ms / frame**
once the worker thread settles — effectively free.

## Numerical correctness

The IR is validated against the upstream PyTorch reference in
`tests/test_taehv_openvino.py::test_pytorch_reference_parity`:

- Encoder: **cos=1.000000 exact**, max\|diff\|=0.0000
- Decoder frames: **cos=1.000000**, max\|diff\|≤0.0014
- Decoder state outputs: **cos=1.000000**, max\|diff\|≤0.015

(The small state-output diff is the fp16-weight storage rounding
envelope.) The full suite runs in ~4 s and self-skips when torch or
upstream `taehv` aren't installed.

## File layout

```
src/quark/taehv/openvino/
├── README.md          ← this file
├── __init__.py        ← ``load()`` public entry point
├── fetch.py           ← local-dir + HF snapshot resolution
├── runtime.py         ← OpenVINOTAEHV (encode/decode/reset)
├── pipeline.py        ← PipelinedDecoder (1-deep concurrent overlap)
└── export.py          ← one-time PyTorch → OpenVINO IR converter
```

The PyTorch trace classes (`EncoderStatic`, `DecoderExplicitState`)
are shared with the CoreML exporter via
[`../_pt_traces.py`](../_pt_traces.py).
