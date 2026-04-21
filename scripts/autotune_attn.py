#!/usr/bin/env python3
"""Autotune + benchmark owl_attn and kv_cache_update kernels.

    python scripts/autotune_attn.py
    python scripts/autotune_attn.py --preset 720p
    python scripts/autotune_attn.py --clear

Autotunes with full 16-bucket ring occupancy (steady-state), then
benchmarks each kernel and prints per-call timings.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_IS_METAL = sys.platform == "darwin"


def _bench_kernel(launcher, cls, spec, config, problem_params):
    """Compile + bench a kernel. Returns microseconds per call."""
    from popcorn.launcher.launcher import _time_callable

    kernel = cls(spec, config)
    ck = launcher.compile(cls, spec, config)
    tensors = cls.make_tensors(problem_params)
    launch_tensors = kernel.prepare_launch_tensors(tensors)
    buffers = [launch_tensors[b.name] for b in ck.param_spec.buffers]
    # Warmup.
    for _ in range(5):
        ck.launch(buffers=buffers)
    return _time_callable(lambda: ck.launch(buffers=buffers), warmup_ms=10.0, bench_ms=100.0)


def main():
    parser = argparse.ArgumentParser(description="Autotune + bench attention kernels")
    parser.add_argument("--preset", choices=["360p", "720p"], default="720p")
    parser.add_argument("--clear", action="store_true", help="delete cached configs before tuning")
    args = parser.parse_args()

    PRESETS = {
        "360p": {"H_spatial": 8, "W_spatial": 16},
        "720p": {"H_spatial": 16, "W_spatial": 32},
    }
    p = PRESETS[args.preset]
    H, W = p["H_spatial"], p["W_spatial"]
    tpf = H * W

    B = 1
    n_kv_heads = 16
    gqa_ratio = 2
    n_q_heads = n_kv_heads * gqa_ratio
    Dh = 64
    max_segments = 3

    patterns = [
        ("dense", 16, 1),
        ("dilated", 16, 8),
    ]

    if args.clear:
        import glob

        cache_dir = os.path.expanduser("~/.cache/popcorn")
        for f in glob.glob(os.path.join(cache_dir, "*.json")):
            os.remove(f)
            print(f"  deleted {f}")

    import popcorn
    from popcorn.functional._dispatch import launcher
    from popcorn.kernels import get
    from popcorn.kernels.kv_cache_update.spec import KVCacheUpdateSpec
    from popcorn.kernels.owl_attn.spec import OwlAttnSpec

    lnch = launcher()
    print(f"preset: {args.preset} ({H}x{W} = {tpf} tokens/frame)")
    print()

    results = []

    for pattern_name, num_buckets, pinned_dilation in patterns:
        print(f"=== {pattern_name} (num_buckets={num_buckets}, pd={pinned_dilation}) ===")
        capacity = num_buckets * tpf + tpf

        # ── owl_attn ──
        owl_cls = get("owl_attn")
        owl_spec = OwlAttnSpec(
            B=B,
            n_kv_heads=n_kv_heads,
            gqa_ratio=gqa_ratio,
            H_spatial=H,
            W_spatial=W,
            num_buckets=num_buckets,
            pinned_dilation=pinned_dilation,
            Dh=Dh,
            packed_qkv=True,
        )
        owl_problem = {
            "B": B, "n_kv_heads": n_kv_heads, "gqa_ratio": gqa_ratio,
            "H_spatial": H, "W_spatial": W,
            "num_buckets": num_buckets, "pinned_dilation": pinned_dilation,
            "Dh": Dh, "packed_qkv": True,
        }

        print(f"  owl_attn: autotuning …")
        t0 = time.perf_counter()
        with popcorn.max_autotune():
            owl_config = lnch._autotune.lookup_or_search(owl_cls, owl_spec)
        print(f"    config: {owl_config} ({time.perf_counter() - t0:.1f}s)")

        owl_us = _bench_kernel(lnch, owl_cls, owl_spec, owl_config, owl_problem)
        n_q = n_kv_heads * gqa_ratio
        owl_flops = 4 * B * n_q * tpf * capacity * Dh
        owl_tflops = owl_flops / (owl_us * 1e-6) / 1e12
        print(f"    runtime: {owl_us:.1f} us  ({owl_tflops:.2f} TFLOPS)")
        results.append((f"owl_attn/{pattern_name}", owl_us))

        # ── kv_cache_update ──
        kv_cls = get("kv_cache_update")
        kv_spec = KVCacheUpdateSpec(
            B=B,
            n_kv_heads=n_kv_heads,
            Dh=Dh,
            H_spatial=H,
            W_spatial=W,
            num_buckets=num_buckets,
            pinned_dilation=pinned_dilation,
            packed_qkv=True,
            n_q_heads=n_q_heads,
        )
        kv_problem = {
            "B": B, "n_kv_heads": n_kv_heads,
            "H_spatial": H, "W_spatial": W,
            "num_buckets": num_buckets, "pinned_dilation": pinned_dilation,
            "Dh": Dh, "packed_qkv": True, "n_q_heads": n_q_heads,
        }

        print(f"  kv_cache_update: autotuning …")
        t0 = time.perf_counter()
        with popcorn.max_autotune():
            kv_config = lnch._autotune.lookup_or_search(kv_cls, kv_spec)
        print(f"    config: {kv_config} ({time.perf_counter() - t0:.1f}s)")

        kv_us = _bench_kernel(lnch, kv_cls, kv_spec, kv_config, kv_problem)
        print(f"    runtime: {kv_us:.1f} us")
        results.append((f"kv_cache/{pattern_name}", kv_us))

        print()

    # ── Per-NFE estimate ──
    # 24 layers: 18 dense + 6 dilated (global_attn_period=4, offset=-1).
    dense_attn = next((us for name, us in results if name == "owl_attn/dense"), 0)
    dilated_attn = next((us for name, us in results if name == "owl_attn/dilated"), 0)
    dense_kv = next((us for name, us in results if name == "kv_cache/dense"), 0)
    dilated_kv = next((us for name, us in results if name == "kv_cache/dilated"), 0)

    attn_total = 18 * dense_attn + 6 * dilated_attn
    kv_total = 18 * dense_kv + 6 * dilated_kv
    combined = attn_total + kv_total

    print("Per-NFE estimate (24 layers: 18 dense + 6 dilated):")
    print(f"  owl_attn:        {attn_total / 1000:.2f} ms  (18×{dense_attn:.0f} + 6×{dilated_attn:.0f} us)")
    print(f"  kv_cache_update: {kv_total / 1000:.2f} ms  (18×{dense_kv:.0f} + 6×{dilated_kv:.0f} us)")
    print(f"  combined:        {combined / 1000:.2f} ms")


if __name__ == "__main__":
    main()
