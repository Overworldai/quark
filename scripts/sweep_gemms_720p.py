#!/usr/bin/env python3
"""End-to-end sweep: per-shape NAX overrides for the four hot 720p GEMMs.

Profile (``profile_kernels_720p.py``, 2026-05-03 M5 Max) showed:
  mlp.fc2+gate     52.4% — M=512 K=8192 N=2048
  out_proj+gate    13.5% — M=512 K=2048 N=2048
  mlp.fc1+silu      7.9% — M=512 K=2048 N=8192
  qkv_proj          5.1% — M=512 K=2048 N=4096

The fc2 sweep (``sweep_fc2_720p.py``, 2026-05-03) found
``BM=128 BN=128 BK=32`` saves 12.0 ms on the saturated forward
(262.70 → 250.70). This sweep extends the same protocol to the
other three shapes.

Strategy: layered sweep. After fc2 wins, lock its config in and
search out_proj on top so we measure each addition's *marginal*
end-to-end win. Adding 4 distinct MSL pipelines could cause the
shader-cache pressure that doomed the original full per-shape
table — but the layered protocol catches that as a regression on
already-locked entries.

Each shape gets the same candidate set:
  - baseline (no override) sanity
  - BM=128 (wider M tile, half the threadgroups in M)
  - BM=128 BK=32, BK=32 (smaller BK with more pipeline iters —
    the surprise winner for fc2)
  - BN=256, BN=256 BK=32 (wider N tile)
  - BM=128 BN=256 BK=32 (both wider)
  - n_stages=3 (deeper pipeline, K=8192 has slack to amortize)
  - nw=16 (more warps per threadgroup)

Usage:
    uv run --offline python scripts/sweep_gemms_720p.py
    uv run --offline python scripts/sweep_gemms_720p.py --shape out_proj
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

# (M, K, N, activation) for each shape at 720p (tpf=512).
_SHAPES: dict[str, tuple[int, int, int, str | None]] = {
    "fc2":      (512, 8192, 2048, None),
    "out_proj": (512, 2048, 2048, None),
    "fc1":      (512, 2048, 8192, "silu"),
    "qkv_proj": (512, 2048, 4096, None),
}

# Locked winners from prior sweeps — applied to every candidate run on
# the *next* shape to measure marginal wins. Key: shape name; Value:
# the override dict (same field set as a candidate).
_LOCKED: dict[str, dict] = {
    "fc2": dict(BM=128, BN=128, BK=32, n_warps=8, n_stages=2,
                a_pad=8, b_pad=8),
    "out_proj": dict(BM=128, BN=128, BK=32, n_warps=8, n_stages=2,
                     a_pad=8, b_pad=8),
}

_CANDIDATES: list[dict | None] = [
    None,
    dict(BM=128, BN=128, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=16,
         label="BM=128"),
    dict(BM=128, BN=128, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8,
         label="BM=128 BK=32"),
    dict(BM=64, BN=128, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8,
         label="BK=32"),
    dict(BM=64, BN=256, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=16,
         label="BN=256"),
    dict(BM=64, BN=256, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8,
         label="BN=256 BK=32"),
    dict(BM=128, BN=256, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8,
         label="BM=128 BN=256 BK=32"),
    dict(BM=64, BN=128, BK=64, n_warps=8, n_stages=3, a_pad=8, b_pad=16,
         label="n_stages=3"),
    dict(BM=128, BN=128, BK=32, n_warps=16, n_stages=2, a_pad=8, b_pad=8,
         label="BM=128 BK=32 nw=16"),
]


def _patch_block(entries: list[tuple[tuple[int, int, int, str | None], dict]]) -> str:
    """Build a Python patch source that installs ``entries`` (key →
    GemmConfig) into ``_NAX_PER_SHAPE`` via the build-table override.
    """
    if not entries:
        return ""
    lines = [
        "import importlib",
        "g = importlib.import_module('quark.functional.gemm')",
        "from quark.kernels.gemm.config import GemmConfig",
        "def _force():",
    ]
    for (M, K, N, act), c in entries:
        act_repr = "None" if act is None else repr(act)
        lines.append(
            f"    g._NAX_PER_SHAPE[({M}, {K}, {N}, {act_repr})] = GemmConfig("
            f"BM={c['BM']}, BN={c['BN']}, BK={c['BK']}, "
            f"n_warps={c['n_warps']}, n_stages={c['n_stages']}, "
            f"a_pad={c['a_pad']}, b_pad={c['b_pad']}, "
            f"main_shape='m16n32k16_nax_bf16')"
        )
    lines += [
        "_orig_build = g._build_per_shape_table",
        "def _patched_build():",
        "    _orig_build()",
        "    _force()",
        "g._build_per_shape_table = _patched_build",
    ]
    return "\n".join(lines)


def _run_bench(entries: list[tuple[tuple[int, int, int, str | None], dict]]) -> tuple[float, float]:
    patch = _patch_block(entries)
    src = f"""
import sys, json, io, contextlib
sys.path.insert(0, 'scripts')
{patch}

sys.argv = ['bench', '--n-frames', '200', '--warmup', '5', '--no-decode',
            '--preset', '720p']
import bench_quark_world_engine as bq
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try: bq.main()
    except SystemExit: pass
out = buf.getvalue()

forward_init = None
forward_sat = None
for line in out.splitlines():
    if 'initial' in line and 'median' in line and 'ms' in line:
        try: forward_init = float(line.split('median')[1].split('ms')[0].strip())
        except Exception: pass
    if 'saturated' in line and 'median' in line and 'ms' in line:
        try: forward_sat = float(line.split('median')[1].split('ms')[0].strip())
        except Exception: pass
print(json.dumps({{'init': forward_init, 'sat': forward_sat}}))
"""
    res = subprocess.run(
        ["uv", "run", "--offline", "python", "-c", src],
        capture_output=True, text=True, timeout=900,
    )
    for line in res.stdout.splitlines():
        try:
            obj = json.loads(line)
            if obj.get("init") is not None and obj.get("sat") is not None:
                return float(obj["init"]), float(obj["sat"])
        except (json.JSONDecodeError, ValueError):
            continue
    raise RuntimeError(
        f"bench didn't report Forward stats. stderr tail:\n{res.stderr[-400:]}\n"
        f"stdout tail:\n{res.stdout[-400:]}"
    )


def _sweep_shape(shape_name: str, locked: dict[str, dict]) -> dict | None:
    M, K, N, act = _SHAPES[shape_name]
    locked_entries = [(_SHAPES[n], locked[n]) for n in locked]
    print(f"\n{'=' * 72}")
    locked_str = ", ".join(locked.keys()) or "none"
    print(f"  Sweep {shape_name} at 720p: M={M} K={K} N={N} activation={act}")
    print(f"  Locked: {locked_str}")
    print(f"{'=' * 72}")
    print(f"{'config':<32s}  {'initial':>9s}  {'saturated':>11s}  vs sat baseline")
    print("-" * 78)
    sat_baseline = None
    results = []
    for cfg in _CANDIDATES:
        label = "baseline (no override for shape)" if cfg is None else cfg["label"]
        entries = list(locked_entries)
        if cfg is not None:
            entries.append(((M, K, N, act), cfg))
        t0 = time.perf_counter()
        try:
            init_ms, sat_ms = _run_bench(entries)
        except Exception as e:
            print(f"  {label:<32s}  FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        elapsed = time.perf_counter() - t0
        if cfg is None:
            sat_baseline = sat_ms
        delta = "" if sat_baseline is None else f"  {sat_ms - sat_baseline:+6.2f} ms"
        print(f"  {label:<32s}  {init_ms:6.2f} ms  {sat_ms:8.2f} ms{delta}  ({elapsed:.1f}s)")
        results.append((label, init_ms, sat_ms, cfg))

    if sat_baseline is None:
        return None
    wins = [(l, i, s, c) for (l, i, s, c) in results
            if c is not None and s < sat_baseline - 0.2]
    if not wins:
        print(f"\n  No override beats baseline for {shape_name}.")
        return None
    wins.sort(key=lambda x: x[2])
    print(f"\n  Winners (sat < baseline - 0.2 ms):")
    for label, init, sat, cfg in wins:
        print(f"    {label}: init={init:.2f} sat={sat:.2f} ({sat - sat_baseline:+.2f} ms)")
    best = wins[0]
    print(f"\n  >> Picking best: {best[0]} ({best[2] - sat_baseline:+.2f} ms)")
    return best[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default="all",
                   help="one of out_proj, fc1, qkv_proj, all (default: all). "
                        "fc2 already swept — see sweep_fc2_720p.py.")
    args = p.parse_args()

    locked = dict(_LOCKED)
    targets = ["out_proj", "fc1", "qkv_proj"] if args.shape == "all" else [args.shape]

    for shape_name in targets:
        winner = _sweep_shape(shape_name, locked)
        if winner is not None:
            locked[shape_name] = {k: v for k, v in winner.items() if k != "label"}

    print(f"\n{'#' * 72}")
    print(f"  Final locked overrides for 720p:")
    for n, c in locked.items():
        print(f"    {n} {_SHAPES[n]}: {c}")
    print(f"{'#' * 72}")


if __name__ == "__main__":
    main()
