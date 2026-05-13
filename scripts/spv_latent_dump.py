"""SPV correctness check: run Waypoint forward with fixed seed and dump
the output latent. Intent: cross-check with a Metal/CUDA reference run
of the same seed → same latent dump → cosine-similarity compare.

VAE encode/decode aren't ported to Intel SPV yet, so we can't ingest a
seed JPEG and emit pixel frames end-to-end. But the model's compute is
the interesting bit anyway — if SPV-emitted latent matches Metal /
CUDA latent on the same input, the kernel chain is correct.

Usage:
    PYTHONPATH=src python scripts/spv_latent_dump.py

Writes ``out_latent_spv.npy`` with shape (n_frames, channels, H, W).
"""

from __future__ import annotations

import os

os.environ.setdefault("QUARK_BACKEND", "spv")

import sys
import time

import numpy as np
import torch

import quark
from quark.models.waypoint_15 import Waypoint15, Waypoint15Config
from quark.runtime.tensor import QuarkTensor


def _seeded_rand_bf16(shape, seed: int) -> QuarkTensor:
    """Reproducible bf16 tensor from a numpy seed."""
    rng = np.random.default_rng(seed)
    f32 = rng.standard_normal(shape).astype(np.float32)
    # bf16 = top 16 bits of f32, packed into uint16 storage
    bits = f32.view(np.uint32) >> 16
    bf16_u16 = bits.astype(np.uint16)
    return QuarkTensor.from_numpy(bf16_u16, dtype="bf16").reshape(*shape)


def main(model_uri: str = "Overworld/Waypoint-1.5-1B-360P", seed: int = 42, out_path: str = "out_latent_spv.npy") -> None:
    print(f"loading {model_uri} ...")
    model = Waypoint15.from_pretrained(model_uri, dtype="bf16")
    cfg = model.cfg
    print(f"  d_model={cfg.d_model} n_layers={cfg.n_layers} channels={cfg.channels}")

    print("model.prepare() ...")
    model.prepare()

    H, W = cfg.height, cfg.width
    n_sigmas = len(cfg.scheduler_sigmas) - 1
    print(f"  H={H} W={W} n_sigmas={n_sigmas}")

    # Build a deterministic seed latent so output can be cross-checked
    # against another backend (Metal / CUDA) running with the same seed.
    latent = _seeded_rand_bf16((1, cfg.channels, H, W), seed=seed)
    ctrl_input = QuarkTensor.zeros(1, model.ctrl_emb._padded_in, dtype="bf16") if hasattr(model, "ctrl_emb") else None

    # Warm autotune + compile cache
    print("warmup ...")
    with quark.lazy():
        e = model.encode_ctrl(ctrl_input)
        _ = model(latent, sigma_idx=0, ctrl_emb=e, frozen=True)

    # Proper denoise loop: each sigma's output becomes next sigma's
    # input via Euler step. The last call uses ``frozen=False`` to
    # commit into the KV cache (the exact pattern ``GenerateFrame``
    # uses on CUDA).
    from quark import nn as qk_nn

    sigmas = cfg.scheduler_sigmas
    dsig_tensors = [
        QuarkTensor.from_numpy(np.array([float(sigmas[i + 1] - sigmas[i])], dtype=np.float32))
        for i in range(n_sigmas)
    ]
    euler_steps = [qk_nn.EulerStep() for _ in range(n_sigmas)]

    # Snapshot each stage's output to a numpy array EAGERLY (sync after
    # every stage). Otherwise the model's per-Linear output cache
    # silently aliases all stage results to the same buffer, and the
    # to_numpy() at the end reads only the LAST stage 4×.
    stage_outputs = []
    print(f"running {n_sigmas} denoise steps + commit ...")
    t0 = time.perf_counter()
    e = model.encode_ctrl(ctrl_input)
    x = latent
    for s in range(n_sigmas):
        frozen = s != n_sigmas - 1
        with quark.lazy():
            v = model(x, sigma_idx=s, ctrl_emb=e, frozen=frozen)
        # Force materialisation; copy out so the next stage's writes
        # don't clobber this snapshot.
        v_np = v.to_numpy()
        if v_np.dtype == np.uint16:
            v_np = (v_np.astype(np.uint32) << 16).view(np.float32)
        stage_outputs.append(v_np.copy())
        # Apply Euler step in a fresh lazy block — same pattern.
        with quark.lazy():
            x = euler_steps[s](x, v, dsig_tensors[s])
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  forward(s) done in {elapsed:.1f} ms")

    # stage_outputs is already a list of numpy arrays (snapshotted)
    for i, arr in enumerate(stage_outputs):
        print(f"  stage[{i}]: shape={arr.shape}  "
              f"finite={np.isfinite(arr).sum()}/{arr.size}  "
              f"min={np.nanmin(arr):.4f} max={np.nanmax(arr):.4f}  "
              f"mean={np.nanmean(arr):.4f}  std={np.nanstd(arr):.4f}")

    final = np.stack(stage_outputs)
    np.save(out_path, final)
    print(f"\nwrote {out_path} (shape={final.shape}, dtype={final.dtype})")
    print("seed:", seed)
    print(
        f"To compare on another backend, run this script with the same\n"
        f"seed and ``QUARK_BACKEND=cuda`` or run on Apple Metal, then:\n"
        f"  python -c \"import numpy as np; a=np.load('{out_path}'); "
        f"b=np.load('out_latent_ref.npy'); print('cos_sim:', "
        "(a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))\""
    )


if __name__ == "__main__":
    uri = sys.argv[1] if len(sys.argv) > 1 else "Overworld/Waypoint-1.5-1B-360P"
    main(uri)
