#!/usr/bin/env python3
"""Encode video/image frames into latent .npy for generate.py --seed-latent.

    python scripts/encode_frame.py input.png --preset 360p
    python scripts/encode_frame.py input.mp4 --preset 720p
    python scripts/encode_frame.py input.jpg --preset 360p --output seed.npy

Reads an image or video, resizes to 16:9 target resolution, tiles
frames 4× for the VAE's temporal compression, and encodes through
the streaming TAEHV. The AE internally resizes 16:9 → 2:1 before
encoding.

Resolution presets:
    360p: 640×360 input → AE encodes at 512×256 → 128 tokens/frame
    720p: 1280×720 input → AE encodes at 1024×512 → 512 tokens/frame

Temporal: 4 pixel frames → 1 latent frame. For a single image,
the frame is tiled 4× before encoding.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
import torch

torch.backends.cudnn.enabled = False

TEMPORAL_COMPRESS = 4

# 16:9 input resolution — the AE internally resizes to 2:1 for encoding.
PRESETS = {
    "360p": {"pixel_h": 360, "pixel_w": 640},
    "720p": {"pixel_h": 720, "pixel_w": 1280},
}


def read_frames(path: str, n_frames: int | None = None) -> list[np.ndarray]:
    """Read frames from image or video. Returns list of [H, W, 3] uint8 RGB."""
    img = cv2.imread(path)
    if img is not None:
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if n_frames is not None and len(frames) >= n_frames:
            break
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {path}")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Encode video/image to VAE latent .npy")
    parser.add_argument("input", help="path to image (.png/.jpg) or video (.mp4)")
    parser.add_argument("--preset", choices=["360p", "720p"], default="360p",
                        help="resolution preset")
    parser.add_argument("--n-frames", type=int, default=None,
                        help="max pixel frames to read from video (default: all)")
    parser.add_argument("--output", default=None, help="output .npy path")
    parser.add_argument("--ae-repo", default="Overworld-Models/taehv1_5")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    p = PRESETS[args.preset]
    pixel_h, pixel_w = p["pixel_h"], p["pixel_w"]
    output = args.output or args.input.rsplit(".", 1)[0] + "_latent.npy"

    print(f"preset: {args.preset} — {pixel_w}x{pixel_h} pixels")

    # ── Read input frames ──
    print(f"reading {args.input} …")
    raw_frames = read_frames(args.input, n_frames=args.n_frames)
    print(f"  {len(raw_frames)} frame(s), {raw_frames[0].shape[1]}x{raw_frames[0].shape[0]}")

    # ── Resize to target resolution ──
    resized = []
    for frame in raw_frames:
        r = cv2.resize(frame, (pixel_w, pixel_h), interpolation=cv2.INTER_AREA)
        resized.append(r)

    # ── Tile for 4× temporal compression ──
    # The VAE encoder compresses 4 pixel frames → 1 latent frame.
    # For a single image, tile it 4× so we get 1 latent frame.
    # For N frames, pad up to the next multiple of 4.
    n_raw = len(resized)
    n_padded = TEMPORAL_COMPRESS * ((n_raw + TEMPORAL_COMPRESS - 1) // TEMPORAL_COMPRESS)
    while len(resized) < n_padded:
        resized.append(resized[-1])  # repeat last frame

    n_latent = n_padded // TEMPORAL_COMPRESS
    print(f"  {n_raw} input → {n_padded} tiled (4×) → {n_latent} latent frame(s)")

    # ── Stack into [T, H, W, 3] uint8 for the encoder ──
    pixel_batch = np.stack(resized, axis=0)  # [T, H, W, 3] uint8
    pixel_tensor = torch.from_numpy(pixel_batch)  # keep as uint8, encoder handles conversion

    # ── Encode ──
    sys.path.insert(0, "/workspace/world_engine/src")
    try:
        from ae import ChunkedStreamingTAEHV
    except ImportError:
        sys.path.insert(0, "/Users/work/world_engine/src")
        from ae import ChunkedStreamingTAEHV

    print(f"loading AE from {args.ae_repo} …")
    ae = ChunkedStreamingTAEHV.from_pretrained(args.ae_repo, device=args.device)

    print(f"encoding {n_padded} pixel frames → {n_latent} latent frame(s) …")
    t0 = time.perf_counter()
    with torch.inference_mode():
        latent = ae.encode(pixel_tensor)
    elapsed = time.perf_counter() - t0

    latent_np = latent.float().cpu().numpy()
    print(f"  encoded in {elapsed:.1f}s, latent shape: {latent_np.shape}")

    # ── Save ──
    np.save(output, latent_np)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
