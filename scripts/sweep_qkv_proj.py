#!/usr/bin/env python3
"""End-to-end sweep: try one per-shape NAX config override for the
qkv_proj shape ``(M=128, K=2048, N=4096, activation=None)`` only,
keeping the universal fallback for every other shape.

Why we ran this experiment originally: an old batched-mode GEMM
microbench (multiple matmuls scheduled into one ``mx.eval``)
suggested MLX was 2.5× faster than quark on qkv_proj. That
methodology was wrong — see ``scripts/bench_gemm_compare.py``
for the corrected per-call comparison; in per-call mode (which
matches the world model's chained ops) qkv_proj is essentially
tied between engines (MLX 1.005×). The motivating ``2.5×`` claim
was a batched-mode artefact, not a kernel-throughput gap.

But the sweep itself is still valid as an end-to-end check that
adding ONE per-shape entry (for ANY reason) doesn't regress the
real bench. Adding a single per-shape override pulls in 1 extra
MSL pipeline (instead of 4 from a full per-shape table) so the
shader-cache pressure that killed earlier per-shape attempts
should be much smaller.

RESULT (2026-05-02, M5 Max, --no-decode bench):
    baseline: 59.10 ms
    Every one of 9 tile candidates regressed by +0.30 to +14.80 ms;
    the closest (BN=256 nw=16, +0.30 ms) was within noise. The
    universal config is at the local optimum for qkv_proj's
    end-to-end access pattern.

Usage:
    uv run --offline python scripts/sweep_qkv_proj.py
"""
from __future__ import annotations

import json
import subprocess
import time

# Tile candidates — wider BN (fewer threadgroups → better waves on
# the GPU when the M dim is small relative to N), and a few BK / pad
# variants. Each candidate is checked against ``GemmKernel.is_valid``
# inside the subprocess; invalid ones are skipped.
_CANDIDATES = [
    None,  # null override — keeps the universal config (sanity)
    dict(BM=64,  BN=256, BK=64, n_warps=8,  n_stages=2, a_pad=8, b_pad=16,
         label="BN=256"),
    dict(BM=64,  BN=256, BK=32, n_warps=8,  n_stages=2, a_pad=8, b_pad=8,
         label="BN=256 BK=32"),
    dict(BM=64,  BN=256, BK=16, n_warps=8,  n_stages=2, a_pad=0, b_pad=0,
         label="BN=256 BK=16"),
    dict(BM=128, BN=128, BK=64, n_warps=8,  n_stages=2, a_pad=8, b_pad=16,
         label="BM=128"),
    dict(BM=128, BN=256, BK=32, n_warps=8,  n_stages=2, a_pad=8, b_pad=8,
         label="BM=128 BN=256 BK=32"),
    dict(BM=128, BN=256, BK=16, n_warps=8,  n_stages=2, a_pad=0, b_pad=0,
         label="BM=128 BN=256 BK=16"),
    dict(BM=128, BN=128, BK=32, n_warps=8,  n_stages=2, a_pad=8, b_pad=8,
         label="BM=128 BK=32"),
    dict(BM=64,  BN=128, BK=64, n_warps=16, n_stages=2, a_pad=8, b_pad=16,
         label="nw=16"),
    dict(BM=64,  BN=256, BK=64, n_warps=16, n_stages=2, a_pad=8, b_pad=16,
         label="BN=256 nw=16"),
]


def _run_bench(cfg: dict | None) -> float:
    """Run the bench in a subprocess with ``cfg`` patched into
    ``_NAX_PER_SHAPE[(128, 2048, 4096, None)]`` (or no override if cfg
    is None). Returns reported ``Forward: NN.N ms``.
    """
    if cfg is None:
        patch = ""
    else:
        # ``quark.functional.gemm`` re-exports a ``gemm`` callable that
        # shadows the submodule when accessed via ``import quark.functional.gemm``.
        # Use importlib to grab the real module object.
        patch = f"""
import importlib
g = importlib.import_module('quark.functional.gemm')
from quark.kernels.gemm.config import GemmConfig
def _force():
    g._NAX_PER_SHAPE[(128, 2048, 4096, None)] = GemmConfig(
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

sys.argv = ['bench', '--n-frames', '8', '--warmup', '5', '--no-decode']
import bench_quark_world_engine as bq
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try: bq.main()
    except SystemExit: pass
out = buf.getvalue()
for line in out.splitlines():
    if 'Forward:' in line and 'ms avg' in line:
        ms = float(line.split('Forward:')[1].split('ms')[0].strip())
        print(json.dumps({{'forward_ms': ms}}))
        break
"""
    res = subprocess.run(
        ["uv", "run", "--offline", "python", "-c", src],
        capture_output=True, text=True, timeout=180,
    )
    for line in res.stdout.splitlines():
        try:
            obj = json.loads(line)
            if "forward_ms" in obj:
                return obj["forward_ms"]
        except (json.JSONDecodeError, ValueError):
            continue
    raise RuntimeError(
        f"bench didn't report Forward: line. stderr tail:\n{res.stderr[-400:]}\n"
        f"stdout tail:\n{res.stdout[-400:]}"
    )


def main():
    print(f"Sweeping {len(_CANDIDATES) - 1} qkv_proj overrides "
          f"(plus 1 baseline) — bench runs 8 frames + 5 warmup, "
          f"~30 s wall each.\n")
    print(f"{'config':<28s}  {'forward':>10s}  vs baseline")
    print("-" * 56)
    baseline = None
    results = []
    for cfg in _CANDIDATES:
        label = "baseline (no override)" if cfg is None else cfg["label"]
        t0 = time.perf_counter()
        try:
            ms = _run_bench(cfg)
        except Exception as e:
            print(f"  {label:<28s}  FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        elapsed = time.perf_counter() - t0
        if cfg is None:
            baseline = ms
        delta = "" if baseline is None else f"  {ms - baseline:+5.2f} ms"
        print(f"  {label:<28s}  {ms:7.2f} ms{delta}  ({elapsed:.1f}s wall)")
        results.append((label, ms, cfg))

    if baseline is not None:
        wins = [(label, ms, c) for (label, ms, c) in results
                if c is not None and ms < baseline - 0.1]
        wins.sort(key=lambda x: x[1])
        if wins:
            print("\nWinners (faster than baseline by >0.1 ms):")
            for label, ms, cfg in wins:
                print(f"  {label}: {ms:.2f} ms ({ms - baseline:+.2f} ms)")
                print(f"    cfg: {cfg}")
        else:
            print("\nNo override beats the universal config end-to-end.")


if __name__ == "__main__":
    main()
