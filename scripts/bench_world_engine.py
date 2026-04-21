#!/usr/bin/env python3
"""Benchmark world_engine's bf16 baseline with the same NFE / LFPS / FPS
numbers that ``scripts/generate.py`` prints for popcorn.

    python scripts/bench_world_engine.py
    python scripts/bench_world_engine.py --preset 360p --n-frames 128
    python scripts/bench_world_engine.py --quant intw8a8   # quantized baseline

world_engine's ``gen_frame`` = one ``_denoise_pass`` (``len(sigmas)-1``
steps) plus one ``_cache_pass`` (commit) — matches popcorn's
``n_denoise + 1`` NFE accounting exactly.

Pass ``--return-img`` to include the VAE decode in the timing; by
default we run DiT-only (``return_img=False``) which matches the
``dit_only=True`` configuration in world_engine's
``examples/benchmark.py`` and is what popcorn's generate.py measures.
"""

from __future__ import annotations

import argparse
import random
import sys
import time


def _setup_paths():
    for p in ("/Users/work/world_engine/src", "/workspace/world_engine/src"):
        try:
            import os

            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        except Exception:
            pass


def _build_ctrl_sequence(n_frames: int):
    """Mirror the ``--ctrl demo`` sequence in scripts/generate.py."""
    from world_engine import CtrlInput

    seq = [
        CtrlInput(mouse=(0.2, 0.2)), CtrlInput(button={32}), CtrlInput(), CtrlInput(), CtrlInput(),
        CtrlInput(button={1}), CtrlInput(), CtrlInput(), CtrlInput(button={1, 32}),
        CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(),
    ] * 4
    seq += [CtrlInput()] * 8
    seq += (
        [CtrlInput(button={32})] * 10
        + [CtrlInput(button={65})] * 10
        + [CtrlInput(button={68})] * 10
        + [CtrlInput(button={83})] * 10
    )
    seq += [CtrlInput()] * 10
    # Extend if the caller asked for more frames than the demo covers.
    while len(seq) < n_frames:
        seq += [
            CtrlInput(
                button=set(random.sample(range(1, 65), random.randint(0, 4))),
                mouse=(random.random() * 0.4 - 0.2, random.random() * 0.4 - 0.2),
                scroll_wheel=random.choice((-1, 0, 1)),
            )
            for _ in range(16)
        ]
    return seq[:n_frames]


def main():
    parser = argparse.ArgumentParser(description="Bench world_engine vs popcorn generate.py")
    parser.add_argument("--repo", default="Overworld/Waypoint-1.5-1B")
    parser.add_argument("--preset", choices=["360p", "720p"], default="720p")
    parser.add_argument("--n-frames", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--quant",
        default=None,
        choices=[None, "intw8a8", "fp8w8a8", "nvfp4"],
        help="Quantization; default None = pure bf16 baseline.",
    )
    parser.add_argument("--return-img", action="store_true",
                        help="Include VAE decode in timing (default: DiT only).")
    parser.add_argument("--load-weights", action="store_true",
                        help="Actually load safetensors (default: random init, same as world_engine benchmark.py).")
    args = parser.parse_args()

    _setup_paths()

    import torch

    # Match popcorn's environment — cudnn is a known perf/correctness footgun
    # on the CUDA box per user's standing guidance.
    torch.backends.cudnn.enabled = False

    from world_engine import CtrlInput, WorldEngine

    repo_suffix = "-360P" if args.preset == "360p" else ""
    repo = args.repo + repo_suffix

    print(f"loading {repo} (quant={args.quant}, bf16) …")
    t0 = time.perf_counter()
    engine = WorldEngine(
        repo,
        quant=args.quant,
        device="cuda",
        dtype=torch.bfloat16,
        load_weights=args.load_weights,
    )
    torch.cuda.synchronize()
    print(f"  engine ready ({time.perf_counter() - t0:.1f}s)")

    cfg = engine.model_cfg
    pH, pW = cfg.patch
    tpf = int(cfg.height) * int(cfg.width)
    sigmas = list(cfg.scheduler_sigmas)
    n_denoise = len(sigmas) - 1
    temporal_compress = int(cfg.temporal_compression) if cfg.temporal_compression else 1

    print(f"preset: {args.preset} — {tpf} tokens/frame")
    print(f"  latent grid: {cfg.height}x{cfg.width}, {args.n_frames} frames")
    print(f"  model: {cfg.n_layers} layers, d_model={cfg.d_model}")
    print(f"  scheduler: {n_denoise} denoise + 1 commit = {n_denoise + 1} NFE")

    # ── Warmup: triggers torch.compile + CUDA graphs in world_engine ──
    print(f"warming up ({args.warmup} frames — compiling kernels) …")
    t_warm = time.perf_counter()
    for _ in range(args.warmup):
        engine.gen_frame(return_img=args.return_img)
    torch.cuda.synchronize()
    print(f"  warmup done ({time.perf_counter() - t_warm:.1f}s)")

    # Reset so the measured run starts from a clean KV cache — mirrors
    # world_engine's benchmark.py setup() pattern.
    engine.reset()
    # One extra frame post-reset, same as examples/benchmark.py's setup().
    engine.gen_frame(return_img=args.return_img)
    torch.cuda.synchronize()

    ctrl_sequence = _build_ctrl_sequence(args.n_frames)

    # ── Generate frames ──
    print(f"\ngenerating {args.n_frames} frames ({n_denoise} denoise + 1 commit each) …")
    torch.cuda.synchronize()
    t_gen_start = time.perf_counter()
    for ctrl in ctrl_sequence:
        engine.gen_frame(ctrl=ctrl, return_img=args.return_img)
    torch.cuda.synchronize()
    gen_elapsed = time.perf_counter() - t_gen_start

    # ── Summary (same layout as scripts/generate.py) ──
    nfe = n_denoise + 1
    n = args.n_frames
    lfps = n / gen_elapsed
    mode = "DiT+VAE" if args.return_img else "DiT only"
    print(f"\n{'='*50}")
    print(f"  world_engine baseline ({mode}, bf16, quant={args.quant})")
    print(f"  {n} latent frames, {tpf} tokens/frame")
    print(f"  {nfe} NFE/frame ({n_denoise} denoise + 1 commit)")
    print(f"  NFE:  {gen_elapsed / n / nfe * 1000:.1f} ms ({lfps * nfe:.1f}/s)")
    print(f"  LFPS: {lfps:.1f} latent frames/s")
    print(f"  FPS:  {lfps * temporal_compress:.1f} pixel frames/s ({temporal_compress}x temporal)")
    print(f"  total: {gen_elapsed:.2f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
