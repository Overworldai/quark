# Loading weights

`quark.nn.io` loads `.safetensors` files directly into
`QuarkTensor` without numpy, torch, or mlx in the middle. The loader
mmap's the file, page-locks the data region via `cuMemHostRegister`,
and DMAs each tensor straight to device at pinned-memory speeds
(~6× the pageable path).

```python
from quark.nn.io import load_safetensors, load_from_hub

sd = load_safetensors("model.safetensors")        # local file
sd = load_from_hub("Overworld/Waypoint-1.5-1B")   # HF Hub — uses huggingface_hub
```

Both return `dict[str, QuarkTensor]`. Hand that to an
`nn.Module.load_state_dict(sd)` and the module's parameters are
populated in-place.

## The full inference path

```python
import quark.nn as nn
from quark.nn.io import load_from_hub
from quark.models.waypoint_15 import Waypoint15, Waypoint15Config

cfg   = Waypoint15Config()
model = Waypoint15(cfg)
sd    = load_from_hub("Overworld/Waypoint-1.5-1B")
model.load_state_dict(sd)
model.prepare()          # weight shuffles, graph capture, etc.
out = model(x, sigma_idx=0, frame_t=0)
```

## `nn.Parameter` / `nn.Module`

PyTorch-shaped, inference-only. Parameters hold `QuarkTensor`s on
both backends (CUDA and Metal). No autograd, no optimizer.

```python
import quark.nn as nn
from quark.nn.module import _randn, _zeros

class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        self.weight = nn.Parameter(_randn(d_out, d_in, dtype="bf16"))
        self.bias   = nn.Parameter(_zeros(d_out, dtype="bf16"))

    def forward(self, x):
        return pcf.gemm(x, self.weight.data, bias=self.bias.data)
```

`_randn` / `_zeros` / `_tensor` in `nn.module` are the
allocation helpers — they emit `QuarkTensor` on both backends.
Parameters are discovered by attribute name; nested
`nn.Module` and `nn.ModuleList` children are walked recursively by
`state_dict()` / `load_state_dict()` / `parameters()`.

## Key remapping

Upstream safetensors dumps often use keys that don't match your
module tree verbatim (`transformer.h.0.mlp.fc1.weight` vs
`blocks.0.mlp.fc1.weight`). The loader doesn't remap — do that
step in Python before passing to `load_state_dict`:

```python
sd = load_from_hub("org/repo")
sd = {_rename(k): v for k, v in sd.items()}
model.load_state_dict(sd, strict=True)
```

`strict=True` (the default) fails loud on missing / unexpected keys
so rename bugs surface immediately.

## Dtype coercion at load

```python
sd = load_safetensors("model.safetensors", dtype="bf16")
```

Casts every tensor to `dtype` after load. Useful when the file
ships in f32 and you want the device copy in bf16 without the
round-trip through a larger host buffer.

If the model contains any `nn.Linear` with `out_dtype="f16"`,
`load_state_dict` automatically casts *all* remaining bf16
parameters to f16 after loading — see `module.py:222`. This
keeps mixed-precision models coherent (half-precision activations
everywhere) without the caller hand-casting each parameter.

## Partial loads

```python
sd = load_safetensors("model.safetensors", names={"noise_emb.weight", "noise_emb.bias"})
```

`names` restricts the load to an allowlist — useful for loading
just the noise-conditioner weights in a rollout test without
materializing the whole model on device.

## Transfer path details

1. Parse the 8-byte header length + JSON header.
2. If the file fits under ~25% of available host RAM, read fully
   into a `bytearray` to avoid on-demand page faults during DMA.
   Otherwise mmap the file.
3. `cuMemHostRegister` the entire data region once — pins it for
   DMA without per-tensor registration overhead.
4. For each tensor: `QuarkTensor.from_bytes(view, shape, dtype)`
   issues a `cuMemcpyHtoD` straight from the pinned region.
5. `cuMemHostUnregister` at the end — weights live on device,
   not in host RAM.

The loader returns as soon as every `cuMemcpyHtoD` has been
enqueued. The caller typically hits `PT.synchronize()` (or the
first subsequent `pcf.*` call) before touching the tensors.

## See also

- [src/quark/nn/io.py](../src/quark/nn/io.py) — the loader
- [src/quark/nn/module.py](../src/quark/nn/module.py) —
  `Module` / `Parameter` / `state_dict` / `load_state_dict`
- [docs/WAYPOINT_15.md](WAYPOINT_15.md) — end-to-end worked example
