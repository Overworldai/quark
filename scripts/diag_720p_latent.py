#!/usr/bin/env python3
"""Dump a single 720p latent + analyse spatial periodicity.

Quick diagnostic for Bug 2: at 720p the DiT produces malformed
latents with strong 4× horizontal periodicity in pixel space. We
want to know:

  1. Does the periodicity show up in the latent itself, or only after
     TAEHV decode? (i.e. is it the DiT's fault or the AE's fault)
  2. What's the period in latent space — 4, 8, 16, 32 cols?
  3. Is it constant across channels / temporal frames?

Usage:
    uv run --offline python scripts/diag_720p_latent.py
    QUARK_DISABLE_NAX=1 uv run --offline python scripts/diag_720p_latent.py
    QUARK_DISABLE_AUTOTUNE=1 uv run --offline python scripts/diag_720p_latent.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_quark_world_engine as bq  # noqa: E402


def _build_model_and_run(preset: str, n_warmup: int, n_collect: int, seed: int):
    """Build a model fresh, run warmup, then capture ``n_collect`` latents.

    Returns ``[n_collect, C, H, W] f32`` numpy.
    """
    import quark
    from quark.models.waypoint_15 import CtrlInput, Waypoint15
    from quark.nn.io import load_from_hub

    cfg = bq._make_config(preset)
    ph, pw = cfg.patch
    H, W = cfg.height * ph, cfg.width * pw
    C = cfg.channels
    n_denoise = len(cfg.scheduler_sigmas) - 1

    repo_suffix = "-360P" if preset == "360p" else ""
    repo = "Overworld/Waypoint-1.5-1B" + repo_suffix
    print(f"loading {repo} …")
    raw_sd = load_from_hub(repo, dtype="bf16")
    sd = bq._remap_state_dict(raw_sd, cfg)
    raw_sd.clear()
    model = Waypoint15(cfg)
    model.load_state_dict(sd, strict=False)
    model.prepare()
    bq._move_params_to_device(model)
    bq._sync()

    ctrl_dev, ctrl_fill = bq._make_ctrl_input(model)
    if ctrl_fill is not None:
        ctrl_dev = ctrl_fill(CtrlInput())
    ctrl_emb = model.encode_ctrl(ctrl_dev)

    # Single-shot autotune warmup so kernels compile.
    autotune_x = bq._randn(1, C * H * W, dtype="bf16")
    autotune_ft = bq._zeros(1, dtype="s32")
    model(autotune_x, sigma_idx=0, frame_t=autotune_ft, ctrl_emb=ctrl_emb)
    model(autotune_x, sigma_idx=0, frame_t=autotune_ft, ctrl_emb=ctrl_emb)
    bq._sync()

    import quark.nn as nn

    sigmas = list(cfg.scheduler_sigmas)
    dsig_tensors = [
        bq._tensor([float(sigmas[si + 1] - sigmas[si])], dtype="f32") for si in range(n_denoise)
    ]
    euler_steps = nn.ModuleList([nn.EulerStep() for _ in range(n_denoise)])

    rng = np.random.default_rng(seed)

    def _gen_one(fi: int):
        x_f32 = rng.standard_normal((1, C * H * W)).astype(np.float32)
        x = bq._f32_to_carrier(x_f32, "bf16")
        ft = bq._tensor([fi], dtype="s32")
        with quark.lazy():
            cur = x
            for si in range(n_denoise):
                v = model(cur, sigma_idx=si, frame_t=ft, ctrl_emb=ctrl_emb, frozen=True)
                cur = euler_steps[si](cur, v, dsig_tensors[si])
            # Read latent BEFORE commit so we see the denoised prediction.
        bq._sync()
        latent_np = bq._np_to_f32(cur).reshape(1, C, H, W)
        with quark.lazy():
            model(cur, sigma_idx=n_denoise, frame_t=ft, ctrl_emb=ctrl_emb)
        bq._sync()
        return latent_np[0]

    # Warmup frames so KV cache is populated.
    for i in range(n_warmup):
        _gen_one(i)

    out = np.zeros((n_collect, C, H, W), dtype=np.float32)
    for i in range(n_collect):
        out[i] = _gen_one(n_warmup + i)
    return out


def _analyze(latents: np.ndarray, *, label: str) -> None:
    """Print stats + spatial periodicity of latents.

    ``latents``: ``[T, C, H, W]`` f32.
    """
    T, C, H, W = latents.shape
    print(f"\n=== {label} ===")
    print(f"  shape: T={T} C={C} H={H} W={W}")
    print(f"  finite: {np.isfinite(latents).mean():.4f}")
    print(f"  mean: {latents.mean():.4f}, std: {latents.std():.4f}")
    print(f"  min: {latents.min():.4f}, max: {latents.max():.4f}")

    # Periodicity check: split W into 2/4/8/16 horizontal blocks, compare
    # block means + block-vs-block correlation.
    for n_blk in (2, 4, 8, 16):
        if W % n_blk != 0:
            continue
        bw = W // n_blk
        # ``[T, C, H, n_blk, bw]`` → mean over (T, C, H, bw) per block
        blocks = latents.reshape(T, C, H, n_blk, bw)
        per_block_mean = blocks.mean(axis=(0, 1, 2, 4))
        # cross-block correlation: average over (T, C, H), correlation
        # between blocks 0 and i.
        b0 = blocks[..., 0, :].reshape(-1)
        corrs = []
        for i in range(1, n_blk):
            bi = blocks[..., i, :].reshape(-1)
            # Pearson correlation
            b0_z = (b0 - b0.mean()) / (b0.std() + 1e-9)
            bi_z = (bi - bi.mean()) / (bi.std() + 1e-9)
            corrs.append(float((b0_z * bi_z).mean()))
        print(f"  W split into {n_blk} blocks (each {bw} cols):")
        print(f"    per-block means: {per_block_mean.round(4).tolist()}")
        print(f"    corr(block_0, block_i): {[round(c, 3) for c in corrs]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["360p", "720p"], default="720p")
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-collect", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default=None,
                   help="save the captured latents to this .npy")
    p.add_argument("--knobs", default="",
                   help="comma list of env knobs reported in the label")
    args = p.parse_args()

    print(f"diag: preset={args.preset}, knobs={args.knobs!r}")
    print(f"  QUARK_DISABLE_NAX={os.environ.get('QUARK_DISABLE_NAX', '')!r}")
    print(f"  QUARK_DISABLE_AUTOTUNE={os.environ.get('QUARK_DISABLE_AUTOTUNE', '')!r}")
    print(f"  QUARK_DISABLE_CUBLAS={os.environ.get('QUARK_DISABLE_CUBLAS', '')!r}")

    t0 = time.perf_counter()
    latents = _build_model_and_run(args.preset, args.n_warmup, args.n_collect, args.seed)
    print(f"  wall: {time.perf_counter() - t0:.1f}s")

    label = f"{args.preset} {args.knobs}".strip()
    _analyze(latents, label=label)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save, latents)
        print(f"  saved {args.save}")


if __name__ == "__main__":
    main()
