#!/usr/bin/env python3
"""End-to-end sweep over (WM, WN) for the NAX attention path.

The NAX attn kernel (``kernels/owl_attn/nax.py``) is a stand-alone
IR-emitted module that bypasses quark's standard autotune machinery,
so the (WM, WN) selection lives in a hand-coded per-shape table at
``functional.owl_attn._NAX_ATTN_PER_SHAPE``. This sweep is the
"manual autotune" that populates that table — it's the same protocol
``sweep_gemms_720p.py`` runs for the GEMM tile config, just over
the simdgroup-count axis instead of (BM, BN, BK).

Per-frame work for the kernel only depends on ``n_simdgroups =
WM * WN`` (each simdgroup processes one 16-row Q slice; the WM/WN
split is a layout artefact that doesn't affect the dispatch shape
on the segment-sparse attention pattern). So we only need to sweep
the product axis. WM=N WN=1 and WM=1 WN=N produce byte-identical
output and the same threadgroup geometry; sweeping both costs
nothing in time but proves the equivalence holds at the bench level.

Constraints from ``NaxAttnSpec.__post_init__``:
  * ``WM >= 1`` and ``WN >= 1``
  * ``tpf % (16 * WM * WN) == 0`` — at 720p tpf=512, max WM*WN=32

Usage:
    uv run --offline python scripts/sweep_attn_simdgroups.py
    uv run --offline python scripts/sweep_attn_simdgroups.py --preset 360p
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

# n_simdgroups candidates per preset. The kernel collapses (WM, WN)
# into ``n_simdgroups`` for layout purposes, so we only need to vary
# the product. Tested as WM=N WN=1 for simplicity; WM=1 WN=N and
# 2D splits like WM=2 WN=2 produce identical bench numbers (verified
# byte-equal output in tests/kernels/owl_attn/test_nax_multisimd.py).
_PRESET_CONFIGS = {
    # tpf=512 → max WM*WN = 32. Up to 16 covers the full range that
    # fits in Apple's 1024 thread / TG budget at 32 threads/simdgroup.
    "720p": [1, 2, 4, 8, 16],
    # tpf=128 → max WM*WN = 8. Past 4 forces TG-launches very low
    # (1 TG covers ALL 128 Q rows at WM=8) and likely under-utilises
    # the GPU's SM count.
    "360p": [1, 2, 4, 8],
}


def _run_bench(WM: int, WN: int, *, preset: str, n_frames: int = 200) -> tuple[float, float, float, float]:
    """Run the bench with explicit ``QUARK_NAX_ATTN_WM`` /
    ``QUARK_NAX_ATTN_WN`` and return (init_med_ms, sat_med_ms,
    init_lfps, sat_lfps).
    """
    src = f"""
import os
os.environ['QUARK_NAX_ATTN_WM'] = '{WM}'
os.environ['QUARK_NAX_ATTN_WN'] = '{WN}'
import sys, json, io, contextlib
sys.path.insert(0, 'scripts')
sys.argv = ['bench', '--preset', '{preset}', '--n-frames', '{n_frames}',
            '--warmup', '5', '--no-decode']
import bench_quark_world_engine as bq
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try: bq.main()
    except SystemExit: pass
out = buf.getvalue()
init_med, sat_med, init_lfps, sat_lfps = None, None, None, None
for line in out.splitlines():
    if 'initial' in line and 'median' in line and 'ms' in line:
        try:
            init_med = float(line.split('median')[1].split('ms')[0].strip())
            init_lfps = float(line.split('LFPS')[0].split('(')[-1].strip())
        except Exception: pass
    if 'saturated' in line and 'median' in line and 'ms' in line:
        try:
            sat_med = float(line.split('median')[1].split('ms')[0].strip())
            sat_lfps = float(line.split('LFPS')[0].split('(')[-1].strip())
        except Exception: pass
print(json.dumps({{'init': init_med, 'sat': sat_med,
                   'init_lfps': init_lfps, 'sat_lfps': sat_lfps}}))
"""
    res = subprocess.run(
        ["uv", "run", "--offline", "python", "-c", src],
        capture_output=True, text=True, timeout=900,
    )
    for line in res.stdout.splitlines():
        try:
            obj = json.loads(line)
            if obj.get("init") is not None and obj.get("sat") is not None:
                return (
                    float(obj["init"]),
                    float(obj["sat"]),
                    float(obj.get("init_lfps") or 0.0),
                    float(obj.get("sat_lfps") or 0.0),
                )
        except (json.JSONDecodeError, ValueError):
            continue
    raise RuntimeError(
        f"bench didn't report Forward stats. stderr tail:\n{res.stderr[-400:]}\n"
        f"stdout tail:\n{res.stdout[-400:]}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["360p", "720p"], default="720p")
    p.add_argument("--n-frames", type=int, default=200,
                   help="frames per bench run (default 200; saturated stats need >=178)")
    p.add_argument("--repeats", type=int, default=2,
                   help="repeat each candidate N times to median out thermal noise")
    args = p.parse_args()

    candidates = _PRESET_CONFIGS[args.preset]
    print(f"\n{'=' * 78}")
    print(f"  NAX attention sweep — preset {args.preset}, {args.n_frames} frames")
    print(f"  {len(candidates)} simdgroup-count candidates × {args.repeats} repeats each")
    print(f"  Each row is the median over repeats; baseline is n_simd=1.")
    print(f"{'=' * 78}")
    print(
        f"  {'(WM, WN)':<12s} {'n_simd':>6s}  "
        f"{'init med':>9s} {'sat med':>9s} {'init LFPS':>10s} {'sat LFPS':>9s} "
        f"{'vs n=1 sat':>12s}  {'wall':>6s}"
    )
    print("  " + "-" * 76)

    baseline_sat = None
    rows = []
    for n in candidates:
        WM, WN = n, 1
        sats: list[float] = []
        inits: list[float] = []
        sat_lfps_list: list[float] = []
        init_lfps_list: list[float] = []
        t0 = time.perf_counter()
        for r in range(args.repeats):
            try:
                init_ms, sat_ms, init_lfps, sat_lfps = _run_bench(
                    WM=WM, WN=WN, preset=args.preset, n_frames=args.n_frames,
                )
                inits.append(init_ms)
                sats.append(sat_ms)
                init_lfps_list.append(init_lfps)
                sat_lfps_list.append(sat_lfps)
            except Exception as e:
                print(f"  WM={WM} WN={WN}: REPEAT {r} FAILED: {type(e).__name__}: {str(e)[:50]}")
        elapsed = time.perf_counter() - t0
        if not sats:
            continue
        sats.sort(); inits.sort(); sat_lfps_list.sort(); init_lfps_list.sort()
        sat_med = sats[len(sats) // 2]
        init_med = inits[len(inits) // 2]
        init_lfps_med = init_lfps_list[len(init_lfps_list) // 2]
        sat_lfps_med = sat_lfps_list[len(sat_lfps_list) // 2]
        if n == 1:
            baseline_sat = sat_med
        delta = (sat_med - baseline_sat) if baseline_sat is not None else 0.0
        flag = ""
        if baseline_sat is not None and n != 1:
            if delta < -1.0:
                flag = "  WIN"
            elif delta > 1.0:
                flag = "  REGR"
        print(
            f"  ({WM},{WN:>2d})        {n:>6d}  "
            f"{init_med:6.1f} ms {sat_med:6.1f} ms "
            f"{init_lfps_med:7.2f}    {sat_lfps_med:6.2f}     "
            f"{delta:+6.2f} ms  ({elapsed:5.1f}s){flag}"
        )
        rows.append((n, WM, WN, init_med, sat_med, init_lfps_med, sat_lfps_med, delta))

    if not rows or baseline_sat is None:
        print("\nNo data — bench failed for all candidates?")
        return

    print(f"\n  Best by saturated forward (lower is better):")
    rows.sort(key=lambda r: r[4])  # sat_med
    for n, WM, WN, init_med, sat_med, init_lfps, sat_lfps, delta in rows[:3]:
        print(
            f"    n_simd={n:<3d} (WM={WM} WN={WN}): sat={sat_med:6.2f} ms "
            f"({sat_lfps:5.2f} LFPS), {delta:+6.2f} ms vs n_simd=1"
        )


if __name__ == "__main__":
    main()
