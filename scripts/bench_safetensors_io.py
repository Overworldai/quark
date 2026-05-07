#!/usr/bin/env python3
"""Benchmark ``quark.nn.io.load_safetensors`` against ``safetensors``.

    python scripts/bench_safetensors_io.py path/to/model.safetensors
    python scripts/bench_safetensors_io.py Overworld/Waypoint-1.5-1B

Reports wall time with a per-phase breakdown. Second run (post-cache)
usually matches the cold cache time once the OS page cache is warm;
compare the first run for cold-load perf.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time


def _resolve(path_or_repo: str) -> str:
    if os.path.isfile(path_or_repo):
        return path_or_repo
    if os.path.isdir(path_or_repo):
        return os.path.join(path_or_repo, "model.safetensors")
    import huggingface_hub

    local_dir = huggingface_hub.snapshot_download(
        path_or_repo, allow_patterns=["model.safetensors"]
    )
    return os.path.join(local_dir, "model.safetensors")


def _sync():
    from quark.runtime.cuda import CudaRuntime

    CudaRuntime.instance().stream_synchronize(0)


def _bench_quark(path: str) -> float:
    from quark.nn.io import load_safetensors

    t0 = time.perf_counter()
    sd = load_safetensors(path)
    _sync()
    elapsed = time.perf_counter() - t0
    n_tensors = len(sd)
    total_bytes = 0
    for t in sd.values():
        total_bytes += t.numel() * {
            "f32": 4,
            "f16": 2,
            "bf16": 2,
            "s32": 4,
            "s64": 8,
            "u8": 1,
            "s8": 1,
        }.get(t.dtype, 0)
    print("quark.nn.io.load_safetensors:")
    print(f"  {n_tensors} tensors, {total_bytes / 1e9:.2f} GB")
    print(f"  wall: {elapsed * 1000:.0f} ms")
    print(f"  effective rate: {total_bytes / elapsed / 1e9:.2f} GB/s")
    del sd
    return elapsed


def _bench_safetensors(path: str) -> float | None:
    try:
        from safetensors import safe_open
    except ImportError:
        print("(safetensors not installed — skipping reference benchmark)")
        return None
    import torch

    t0 = time.perf_counter()
    sd = {}
    with safe_open(path, framework="pt", device="cuda") as f:
        for name in f:
            sd[name] = f.get_tensor(name)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print("safetensors (ref, torch / pt device='cuda'):")
    print(f"  wall: {elapsed * 1000:.0f} ms")
    del sd
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to .safetensors file OR HF repo id")
    ap.add_argument(
        "--drop-cache",
        action="store_true",
        help="hint to drop the OS page cache between runs (needs root)",
    )
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    local = _resolve(args.path)
    size_gb = os.path.getsize(local) / 1e9
    print(f"file: {local}  ({size_gb:.2f} GB)\n")

    for i in range(args.runs):
        if i > 0 and args.drop_cache:
            with contextlib.suppress(Exception):
                os.system("sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null")
        print(f"── run {i + 1} ──")
        q = _bench_quark(local)
        s = _bench_safetensors(local)
        if s is not None:
            ratio = q / s
            if ratio > 1:
                print(f"  quark is {ratio:.2f}x slower than safetensors")
            else:
                print(f"  quark is {1 / ratio:.2f}x faster than safetensors")
        print()


if __name__ == "__main__":
    sys.exit(main() or 0)
