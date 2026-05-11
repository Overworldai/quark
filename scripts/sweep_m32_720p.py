#!/usr/bin/env python3
"""End-to-end sweep: try the new m32n32k16_nax_bf16 main_shape on each
720p hot GEMM, measuring saturated forward time.

Hypothesis (per the world_engine MLX investigation, 2026-04-14): m=32
NAX MMA halves the per-K-iter MMA dispatch count vs m=16, with a
microbench win on wide-N shapes (qkv_proj +3%, fc1 +8%, fc2 -6%,
out_proj -6%) and aggregate end-to-end ~+1.7% with int8 KV.

This sweep tests whether the same trade applies to quark's bf16 NAX
GEMM at 720p shapes (M=512). The IR plumbing was extended in
``src/quark/ir/mma_registry.py`` (m32n32k16 entry),
``src/quark/lower/msl/mma.py`` (shape-parametric _nax_frag_layout +
component counts), and ``src/quark/kernels/gemm/kernel.py``
(parametric M_FRAG / N_FRAG / shape_id through the multi-warp NAX
builder).

We layer the m=32 override on top of the already-locked 720p winners
(fc2 + out_proj BM=128 BK=32 from ``sweep_gemms_720p.py``) so the
delta we measure is m=16 vs m=32 main_shape at the same tile config.

Usage:
    uv run --offline python scripts/sweep_m32_720p.py
    uv run --offline python scripts/sweep_m32_720p.py --shape fc2
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

# Locked baseline winners from prior sweeps — m=16 main_shape.
_BASELINE_M16: dict[str, dict] = {
    "fc2":      dict(BM=128, BN=128, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8),
    "out_proj": dict(BM=128, BN=128, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8),
}

# Candidate tile configs for m=32 — per-fragment is now 32×32 so SM/SN
# must be multiples of 32. BM=64 BN=128 → SM=32 SN=32 (TM=1, TN=1) at
# n_warps=8 (WM=2, WN=4). BM=128 BN=128 → SM=64 SN=32 (TM=2, TN=1).
# BM=128 BN=256 → SM=64 SN=64 (TM=2, TN=2).
_M32_CANDIDATES = [
    dict(BM=64, BN=128, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=16,
         label="m32 BM=64 BN=128 BK=64"),
    dict(BM=128, BN=128, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8,
         label="m32 BM=128 BN=128 BK=32"),
    dict(BM=128, BN=128, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=16,
         label="m32 BM=128 BN=128 BK=64"),
    dict(BM=128, BN=256, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8,
         label="m32 BM=128 BN=256 BK=32"),
    dict(BM=128, BN=256, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=16,
         label="m32 BM=128 BN=256 BK=64"),
]


def _patch_block(entries: list[tuple[tuple[int, int, int, str | None], dict, str]]) -> str:
    """Build patch source. Each entry is (key, cfg_dict, main_shape)."""
    if not entries:
        return ""
    lines = [
        "import importlib",
        "g = importlib.import_module('quark.functional.gemm')",
        "from quark.kernels.gemm.config import GemmConfig",
        "def _force():",
    ]
    for (M, K, N, act), c, main_shape in entries:
        act_repr = "None" if act is None else repr(act)
        lines.append(
            f"    g._NAX_PER_SHAPE[({M}, {K}, {N}, {act_repr})] = GemmConfig("
            f"BM={c['BM']}, BN={c['BN']}, BK={c['BK']}, "
            f"n_warps={c['n_warps']}, n_stages={c['n_stages']}, "
            f"a_pad={c['a_pad']}, b_pad={c['b_pad']}, "
            f"main_shape={main_shape!r})"
        )
    lines += [
        "_orig_build = g._build_per_shape_table",
        "def _patched_build():",
        "    _orig_build()",
        "    _force()",
        "g._build_per_shape_table = _patched_build",
    ]
    return "\n".join(lines)


def _run_bench(entries) -> tuple[float, float]:
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


def _sweep_shape(shape_name: str) -> None:
    M, K, N, act = _SHAPES[shape_name]
    print(f"\n{'=' * 78}")
    print(f"  Sweep {shape_name} m=32 vs m=16 at 720p: M={M} K={K} N={N} act={act}")
    print(f"{'=' * 78}")
    print(f"{'config':<36s}  {'initial':>9s}  {'saturated':>11s}  vs sat baseline")
    print("-" * 84)

    # Build m=16 baseline (other 720p winners locked + this shape's
    # winner from the prior sweep, or universal fallback).
    locked_entries = []
    for n, c in _BASELINE_M16.items():
        if n == shape_name:
            continue
        locked_entries.append((_SHAPES[n], c, "m16n32k16_nax_bf16"))

    if shape_name in _BASELINE_M16:
        base_cfg = _BASELINE_M16[shape_name]
    else:
        # No prior winner — use the universal fallback config.
        base_cfg = dict(BM=64, BN=128, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=16)

    base_entries = locked_entries + [(_SHAPES[shape_name], base_cfg, "m16n32k16_nax_bf16")]
    t0 = time.perf_counter()
    init0, sat0 = _run_bench(base_entries)
    elapsed0 = time.perf_counter() - t0
    print(
        f"  {'m16 baseline (locked)':<36s}  {init0:6.2f} ms  {sat0:8.2f} ms"
        f"   +0.00 ms  ({elapsed0:.1f}s)"
    )

    for cfg in _M32_CANDIDATES:
        m32_entries = locked_entries + [(_SHAPES[shape_name], cfg, "m32n32k16_nax_bf16")]
        t0 = time.perf_counter()
        try:
            init_ms, sat_ms = _run_bench(m32_entries)
        except Exception as e:
            print(f"  {cfg['label']:<36s}  FAILED: {type(e).__name__}: {str(e)[:50]}")
            continue
        elapsed = time.perf_counter() - t0
        delta = sat_ms - sat0
        flag = "  WIN" if delta < -0.5 else "     " if abs(delta) < 0.5 else "  "
        print(
            f"  {cfg['label']:<36s}  {init_ms:6.2f} ms  {sat_ms:8.2f} ms  "
            f"{delta:+6.2f} ms  ({elapsed:.1f}s){flag}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default="all",
                   help="one of fc2, out_proj, fc1, qkv_proj, all (default: all)")
    args = p.parse_args()

    targets = list(_SHAPES.keys()) if args.shape == "all" else [args.shape]
    for shape_name in targets:
        _sweep_shape(shape_name)


if __name__ == "__main__":
    main()
