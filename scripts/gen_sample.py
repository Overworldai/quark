"""Generate a sample video from a Waypoint-1.5 model and report FPS.

Usage:
    python scripts/gen_sample.py <model-uri>

e.g.
    python scripts/gen_sample.py Overworld/Waypoint-1.5-1B-360P
    python scripts/gen_sample.py Overworld-Models/WP1.5-4B-MoE-Base

Each ``gen_frame`` call decodes to ``_TAEHV_SUBFRAMES`` video
subframes via TAEHV. On macOS the decode runs on the ANE
``PipelinedDecoder`` worker thread (overlapping with caller-side
work between frames) — wired implicitly inside ``engine.gen_frame``.
On CUDA the call runs synchronously on the main stream. Same loop
on both backends.

Frames are buffered in memory and written to ``out.mp4`` after the
timed loop so the encode I/O doesn't pollute the FPS measurement.
"""

import sys
import time
import urllib.request

import cv2
import imageio.v3 as iio
import numpy as np
import torch

from quark import CtrlInput, Engine
from quark.models.waypoint_15 import QuantConfig

_OUTPUT_PATH = "out.mp4"
_OUTPUT_FPS = 60  # video container fps
_TAEHV_SUBFRAMES = 4  # one ``gen_frame`` call → 4 video subframes
_WARMUP_FRAMES = 3  # untimed; lets autotune + ANE pipeline warm
model_name = sys.argv[1]
engine = Engine(model_name, quant=QuantConfig())

# A representative controller sequence: a few mouse-and-WASD inputs
# followed by sustained directional held-keys, then idle frames.
controller_sequence = [
    CtrlInput(mouse=[0.2, 0.2]), CtrlInput(button={32}), CtrlInput(), CtrlInput(), CtrlInput(),
    CtrlInput(button={1}), CtrlInput(), CtrlInput(), CtrlInput(button={1, 32}),
    CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(),
] * 4
controller_sequence += [CtrlInput()] * 8
controller_sequence += (
    [CtrlInput(button={32})] * 10
    + [CtrlInput(button={65})] * 10
    + [CtrlInput(button={68})] * 10
    + [CtrlInput(button={83})] * 10
)
controller_sequence += [CtrlInput()] * 10

# Seed frame from the Biome repo's sample seeds.
SEED_URL = "https://raw.githubusercontent.com/Overworldai/Biome/14343a6/seeds/sunken_city_depths.jpg"
seed_frame = cv2.imdecode(
    np.frombuffer(urllib.request.urlopen(SEED_URL).read(), np.uint8), cv2.IMREAD_COLOR
)
size = (1280, 720) if '360P' not in model_name else (640, 360)
seed_frame = cv2.resize(seed_frame, size)
seed_frame_x4 = torch.from_numpy(np.repeat(seed_frame[None], 4, axis=0))

engine.append_frame(seed_frame_x4)

# Warmup so autotune + the ANE worker thread are hot before timing.
for _ in range(_WARMUP_FRAMES):
    engine.gen_frame(CtrlInput())

n_calls = len(controller_sequence)
all_pixels: list = []

t0 = time.perf_counter()
for ctrl in controller_sequence:
    all_pixels.append(engine.gen_frame(ctrl))
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

n_subframes = n_calls * _TAEHV_SUBFRAMES
print(
    f"[gen_sample] {n_calls} latent frames in {elapsed:.2f}s "
    f"({n_calls / elapsed:.2f} LFPS, {n_subframes / elapsed:.2f} subframe-FPS)"
)
print(f"[gen_sample] {1000 * elapsed / n_calls:.2f} ms / latent frame")

# Encode after the timed loop so I/O doesn't pollute the FPS measurement.
with iio.imopen(_OUTPUT_PATH, "w", plugin="pyav") as out:
    out.write(seed_frame_x4, fps=_OUTPUT_FPS, codec="libx264")
    for pixels in all_pixels:
        out.write(pixels.cpu().numpy(), fps=_OUTPUT_FPS, codec="libx264")
print(f"[gen_sample] wrote {_OUTPUT_PATH}")
