#!/usr/bin/env python3
"""End-to-end sweep: per-shape NAX override for the mlp.fc2 GEMM at 720p.

Profile (M5 Max, ``profile_kernels_720p.py``) shows mlp.fc2+gate is
the dominant kernel at 720p — ``M=512 K=8192 N=2048`` accounts for
52% of the predicted forward time per frame. The universal config
``BM=64 BN=128 BK=64 n_warps=8 n_stages=2 a_pad=8 b_pad=16`` was
tuned on 360p (``M=128``); this sweep tests whether a per-shape
override targeted at the wider M=512 dim wins end-to-end at 720p.

Like ``sweep_qkv_proj.py``, runs each candidate inside a subprocess
with ONE per-shape entry patched into ``_NAX_PER_SHAPE`` and reports
the bench's saturated forward time. The candidate set focuses on
configs that should benefit from the larger M dimension: wider BM
(half as many threadgroups in M, better arithmetic intensity), and
larger BN with BK=64 (M=512 has 8 tile rows, plenty of work to
amortize wider tile cost).

Usage:
    uv run --offline python scripts/sweep_fc2_720p.py
"""
from __future__ import annotations

import json
import subprocess
import time

# Tile candidates for fc2 (M=512, K=8192, N=2048).
# Universal baseline: BM=64 BN=128 BK=64 n_warps=8 n_stages=2 a_pad=8 b_pad=16.
# Each candidate is checked against ``GemmKernel.is_valid``; invalid → skipped.
_CANDIDATES: list[dict | None] = [
    None,  # baseline (no override) — sanity
    # Bigger M tiles — fewer total threadgroups, better re-use of A.
    dict(BM=128, BN=128, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=16,
         label="BM=128"),
    dict(BM=128, BN=128, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8,
         label="BM=128 BK=32"),
    dict(BM=128, BN=64, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=16,
         label="BM=128 BN=64"),
    # Wider BN — fewer N tiles, K=8192 amortizes wider tile.
    dict(BM=64, BN=256, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=16,
         label="BN=256"),
    dict(BM=64, BN=256, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8,
         label="BN=256 BK=32"),
    # Bigger BM AND BN.
    dict(BM=128, BN=256, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8,
         label="BM=128 BN=256 BK=32"),
    dict(BM=128, BN=256, BK=16, n_warps=8, n_stages=2, a_pad=0, b_pad=0,
         label="BM=128 BN=256 BK=16"),
    # More warps — more parallelism per threadgroup.
    dict(BM=64, BN=128, BK=64, n_warps=16, n_stages=2, a_pad=8, b_pad=16,
         label="nw=16"),
    dict(BM=128, BN=128, BK=64, n_warps=16, n_stages=2, a_pad=8, b_pad=16,
         label="BM=128 nw=16"),
    # Triple buffer — K=8192 has plenty of K-iters to hide more prefetch latency.
    dict(BM=64, BN=128, BK=64, n_warps=8, n_stages=3, a_pad=8, b_pad=16,
         label="n_stages=3"),
]


def _run_bench(cfg: dict | None) -> tuple[float, float]:
    """Run the bench with ``cfg`` patched into the fc2 (512, 8192, 2048, None)
    per-shape slot. Returns (forward_initial_ms, forward_saturated_ms).
    """
    if cfg is None:
        patch = ""
    else:
        patch = f"""
import importlib
g = importlib.import_module('quark.functional.gemm')
from quark.kernels.gemm.config import GemmConfig
def _force():
    g._NAX_PER_SHAPE[(512, 8192, 2048, None)] = GemmConfig(
        BM={cfg['BM']}, BN={cfg['BN']}, BK={cfg['BK']},
        n_warps={cfg['n_warps']}, n_stages={cfg['n_stages']},
        a_pad={cfg['a_pad']}, b_pad={cfg['b_pad']},
        main_shape='m16n32k16_nax_bf16',
    )
_orig_build = g._build_per_shape_table
def _patched_build():
    _orig_build()
    _force()
g._build_per_shape_table = _patched_build
"""
    src = f"""
import sys, json, io, contextlib
sys.path.insert(0, 'scripts')
{patch}

# Bench: 200 timed frames at 720p so saturated stats land cleanly.
sys.argv = ['bench', '--n-frames', '200', '--warmup', '5', '--no-decode',
            '--preset', '720p']
import bench_quark_world_engine as bq
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try: bq.main()
    except SystemExit: pass
out = buf.getvalue()

forward_full = None
forward_init = None
forward_sat = None
for line in out.splitlines():
    if 'Forward:' in line and 'ms avg' in line:
        forward_full = float(line.split('Forward:')[1].split('ms')[0].strip())
    if 'initial' in line and 'median' in line and 'ms' in line:
        # "               initial 10: median  XX.X ms ..."
        try:
            forward_init = float(line.split('median')[1].split('ms')[0].strip())
        except Exception: pass
    if 'saturated' in line and 'median' in line and 'ms' in line:
        try:
            forward_sat = float(line.split('median')[1].split('ms')[0].strip())
        except Exception: pass
print(json.dumps({{
    'forward_full': forward_full,
    'forward_init': forward_init,
    'forward_sat': forward_sat,
}}))
"""
    res = subprocess.run(
        ["uv", "run", "--offline", "python", "-c", src],
        capture_output=True, text=True, timeout=900,
    )
    for line in res.stdout.splitlines():
        try:
            obj = json.loads(line)
            if "forward_full" in obj:
                init = obj.get("forward_init") or obj["forward_full"]
                sat = obj.get("forward_sat") or obj["forward_full"]
                if init is None or sat is None:
                    raise RuntimeError("init or sat None")
                return init, sat
        except (json.JSONDecodeError, ValueError):
            continue
    raise RuntimeError(
        f"bench didn't report Forward: line. stderr tail:\n{res.stderr[-400:]}\n"
        f"stdout tail:\n{res.stdout[-400:]}"
    )


def main():
    print(
        f"Sweeping {len(_CANDIDATES) - 1} fc2 overrides (M=512, K=8192, N=2048) "
        f"plus 1 baseline at 720p — bench runs 200 frames + 5 warmup, "
        f"~70 s wall each.\n"
    )
    print(f"{'config':<32s}  {'initial':>9s}  {'saturated':>11s}  vs sat baseline")
    print("-" * 78)
    sat_baseline = None
    results = []
    for cfg in _CANDIDATES:
        label = "baseline (no override)" if cfg is None else cfg["label"]
        t0 = time.perf_counter()
        try:
            init_ms, sat_ms = _run_bench(cfg)
        except Exception as e:
            print(f"  {label:<32s}  FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        elapsed = time.perf_counter() - t0
        if cfg is None:
            sat_baseline = sat_ms
        delta = "" if sat_baseline is None else f"  {sat_ms - sat_baseline:+6.2f} ms"
        print(f"  {label:<32s}  {init_ms:6.2f} ms  {sat_ms:8.2f} ms{delta}  ({elapsed:.1f}s)")
        results.append((label, init_ms, sat_ms, cfg))

    if sat_baseline is not None:
        wins = [(label, init, sat, c) for (label, init, sat, c) in results
                if c is not None and sat < sat_baseline - 0.2]
        wins.sort(key=lambda x: x[2])
        if wins:
            print(f"\nWinners (saturated forward < baseline - 0.2 ms):")
            for label, init, sat, cfg in wins:
                print(f"  {label}: init={init:.2f} sat={sat:.2f} ({sat - sat_baseline:+.2f} ms)")
                print(f"    cfg: {cfg}")
        else:
            print("\nNo override beats the universal config end-to-end at 720p fc2.")


if __name__ == "__main__":
    main()
