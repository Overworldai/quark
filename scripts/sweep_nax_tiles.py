#!/usr/bin/env python3
"""End-to-end NAX tile sweep: run the real bench with each candidate
``GemmConfig`` swapped in for ``functional.gemm._universal_nax_fallback``
and report the wall-time-per-frame.

Why end-to-end and not a microbench:

  An earlier version of this script ran each config in isolation
  (lazy-batched 50× of a single GEMM) and reported the per-shape ms.
  The "best" microbench configs (n_stages=1, BN=256, etc.) regressed
  the real bench by 25-40 ms / frame:

    * ``n_stages=1`` wins in microbench because B is always L2-resident
      with the same op repeating; the double-buffered prefetch in
      ``n_stages=2`` has nothing to overlap. In the real workload B
      gets evicted between layers and ``n_stages=2`` reclaims ~40 ms.
    * Wider tiles (BN=256, BK=64) win in microbench but churn L2
      against surrounding kv_cache / norm / attn ops in real flow.
    * Per-shape configs (different tile per (M,K,N)) compile a
      distinct MSL pipeline each — the resulting shader-cache thrash
      eats the per-op gains the microbench predicted.

  Trust the end-to-end number, not the isolated per-shape number.

Usage:
    uv run --offline python scripts/sweep_nax_tiles.py
"""
from __future__ import annotations

import time

import quark.functional as qf  # noqa: F401  (warms imports)

# ── Candidate sweep ──
# Stay near the known-good config (BM=64 BN=128 BK=64 nw=8 st=2
# a_pad=8 b_pad=16) — exotic configs cost a fresh MSL pipeline compile
# and almost always regress.  Each candidate is one tweak from the
# baseline so the cause of any change is obvious.

_CANDIDATES = [
    # Baseline (sanity check that the harness reproduces the prior
    # number).
    dict(BM=64, BN=128, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=16,
         label="baseline"),
    # n_stages variants.
    dict(BM=64, BN=128, BK=64, n_warps=8, n_stages=1, a_pad=8, b_pad=16,
         label="st=1"),
    # BK variants.
    dict(BM=64, BN=128, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8,
         label="BK=32"),
    dict(BM=64, BN=128, BK=16, n_warps=8, n_stages=2, a_pad=0, b_pad=0,
         label="BK=16"),
    # BN variants (wider).
    dict(BM=64, BN=256, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=16,
         label="BN=256"),
    dict(BM=64, BN=256, BK=32, n_warps=8, n_stages=2, a_pad=8, b_pad=8,
         label="BN=256 BK=32"),
    # BM variants (taller).
    dict(BM=128, BN=128, BK=64, n_warps=8, n_stages=2, a_pad=0, b_pad=16,
         label="BM=128"),
    dict(BM=128, BN=128, BK=32, n_warps=8, n_stages=2, a_pad=0, b_pad=0,
         label="BM=128 BK=32"),
    # n_warps variants.
    dict(BM=64, BN=128, BK=64, n_warps=16, n_stages=2, a_pad=8, b_pad=16,
         label="nw=16"),
    # Pad variants on baseline tile.
    dict(BM=64, BN=128, BK=64, n_warps=8, n_stages=2, a_pad=0, b_pad=0,
         label="no pad"),
    dict(BM=64, BN=128, BK=64, n_warps=8, n_stages=2, a_pad=8, b_pad=0,
         label="a_pad=8"),
]


def _run_bench_subprocess(cfg_kwargs: dict) -> float:
    """Spawn ``bench_quark_world_engine`` in a subprocess with the
    universal NAX config patched. Returns the reported forward-ms.

    A subprocess (rather than in-process patching) gives a clean Metal
    device + autotune + pipeline cache per config — otherwise the
    first config's compile sticks and every subsequent run is biased
    by its decisions.
    """
    import json
    import subprocess

    patch_src = f"""
import sys, json, time
sys.path.insert(0, 'scripts')
import quark.functional.gemm as g
from quark.kernels.gemm.config import GemmConfig

def _override():
    return GemmConfig(
        BM={cfg_kwargs['BM']}, BN={cfg_kwargs['BN']}, BK={cfg_kwargs['BK']},
        n_warps={cfg_kwargs['n_warps']}, n_stages={cfg_kwargs['n_stages']},
        a_pad={cfg_kwargs['a_pad']}, b_pad={cfg_kwargs['b_pad']},
        main_shape='m16n32k16_nax_bf16',
    )
g._universal_nax_fallback = _override

sys.argv = ['bench', '--n-frames', '5', '--warmup', '3', '--no-decode']
import bench_quark_world_engine as bq
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try: bq.main()
    except SystemExit: pass
out = buf.getvalue()
# Parse the "Forward: NN.N ms avg" line.
for line in out.splitlines():
    if 'Forward:' in line and 'ms avg' in line:
        ms = float(line.split('Forward:')[1].split('ms')[0].strip())
        print(json.dumps({{'forward_ms': ms}}))
        break
"""
    res = subprocess.run(
        ["uv", "run", "--offline", "python", "-c", patch_src],
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
        f"bench did not report a Forward: line. stderr:\n{res.stderr[-500:]}\n"
        f"stdout tail:\n{res.stdout[-500:]}"
    )


def main():
    print(f"Sweeping {len(_CANDIDATES)} candidate NAX configs against the real "
          f"bench (5 frames, 3 warmup, no decode).\n")
    print(f"{'cfg':<22s}  {'forward':>10s}  vs baseline")
    print("-" * 50)
    baseline = None
    results = []
    for c in _CANDIDATES:
        t0 = time.perf_counter()
        try:
            ms = _run_bench_subprocess(c)
        except Exception as e:
            print(f"  {c['label']:<22s}  FAILED: {type(e).__name__}: {str(e)[:80]}")
            continue
        elapsed = time.perf_counter() - t0
        if baseline is None and c['label'] == 'baseline':
            baseline = ms
        delta = "" if baseline is None else f"  {ms - baseline:+5.1f} ms"
        print(f"  {c['label']:<22s}  {ms:7.2f} ms{delta}  ({elapsed:.1f}s wall)")
        results.append((c['label'], ms))

    if baseline is not None:
        print("\nWinners (faster than baseline):")
        for label, ms in sorted(results, key=lambda x: x[1]):
            if ms < baseline:
                print(f"  {label}: {ms:.2f} ms ({ms - baseline:+.2f} ms vs baseline)")
            else:
                break


if __name__ == "__main__":
    main()
