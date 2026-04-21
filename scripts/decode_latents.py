#!/usr/bin/env python3
"""Decode latent .npy to video using streaming TAEHV AE.

    python scripts/decode_latents.py out.npy
    python scripts/decode_latents.py out.npy --output video.mp4 --fps 15

Reads a [T, C, H, W] f32 latent array saved by generate.py and decodes
through the streaming TAEHV autoencoder. The AE has 4× temporal
compression, so T latent frames → T*4 pixel frames.

TAEHV handles permute + uint8 cast internally — output is
[T*4, H, W, 3] uint8 ready for cv2.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

torch.backends.cudnn.enabled = False

TEMPORAL_COMPRESS = 4


def load_decoder(ae_repo: str, device: str = "cuda"):
    sys.path.insert(0, "/workspace/world_engine/src")
    from ae import ChunkedStreamingTAEHV

    ae = ChunkedStreamingTAEHV.from_pretrained(ae_repo, device=device)

    @torch.inference_mode()
    def decode(latent_np: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(latent_np).to(device=device, dtype=torch.bfloat16)
        decoded = ae.decode(t)  # [T*4, H, W, 3] uint8
        return decoded.cpu().numpy()

    return decode


def main():
    parser = argparse.ArgumentParser(description="Decode latent .npy to video")
    parser.add_argument("latent_path", help="path to latent .npy file [T, C, H, W]")
    parser.add_argument("--output", default=None, help="output .mp4 path (default: <input>.mp4)")
    parser.add_argument("--fps", type=int, default=60, help="output video fps (default: 60, matching training data)")
    parser.add_argument("--ae-repo", default="Overworld-Models/taehv1_5")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = args.output or args.latent_path.replace(".npy", ".mp4")

    print(f"loading latents from {args.latent_path} …")
    latents = np.load(args.latent_path)  # [T, C, H, W]
    if latents.ndim == 3:
        latents = latents[np.newaxis]
    T_latent = latents.shape[0]
    T_pixel = T_latent * TEMPORAL_COMPRESS
    print(f"  {T_latent} latent frames → {T_pixel} pixel frames ({TEMPORAL_COMPRESS}× temporal)")
    print(f"  latent shape: {latents.shape[1:]}")

    print(f"loading AE decoder from {args.ae_repo} …")
    decode = load_decoder(args.ae_repo, device=args.device)

    print(f"decoding {T_latent} latent frames …")
    t0 = time.perf_counter()
    pixel_chunks = []
    for i in range(T_latent):
        # Decode one latent frame at a time — the streaming decoder
        # maintains internal temporal state across calls.
        chunk = decode(latents[i:i+1])  # [1, C, h, w] → [4, H, W, 3] uint8
        pixel_chunks.append(chunk)
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  decoded {i+1}/{T_latent}")
    all_pixels = np.concatenate(pixel_chunks, axis=0)
    elapsed = time.perf_counter() - t0
    print(f"  {all_pixels.shape[0]} pixel frames in {elapsed:.1f}s ({all_pixels.shape[0] / elapsed:.1f} fps)")

    import cv2

    h, w = all_pixels.shape[1], all_pixels.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, args.fps, (w, h))
    for frame in all_pixels:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"saved {output} ({all_pixels.shape[0]} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
