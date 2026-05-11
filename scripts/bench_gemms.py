#!/usr/bin/env python3
"""Compare PTX vs cuBLAS GEMM runtimes under CUDA-graph capture.

Autotune's default eager tight-loop bench over-weights per-call Python +
driver overhead, pessimizing cuBLAS+cast against PTX's fused in-smem
cvt. Production runs inside a captured graph where that overhead
largely vanishes into recorded nodes — see ``world_engine.backends
.quark_backend.QuarkBackend.gen_frame``. This script times each path
via stream capture + replay so we can see the **true steady-state
cost** that each path actually pays in production.

For every problem:
  * ``ptx  (µs)``: ``qf.gemm`` with ``QUARK_DISABLE_CUBLAS=1`` so
    autotune picks the best PTX config (mixed-dtype uses the PTX
    kernel's ``compute_dtype`` smem down-cast — cvt fused into MMA).
  * ``cublas (µs)``: direct ``dispatch_cublas`` call — includes the
    ``qf.quantize_e4m3`` cast when the spec is mixed-dtype.

Both are then captured into a ``quark.graph.capture_graph`` block,
replayed in a tight loop bracketed by CUDA events. Same warmup and
replay count per problem so the comparison is symmetric.

Usage:
    python scripts/bench_gemms.py
    python scripts/bench_gemms.py --warmup 20 --bench 500
    python scripts/bench_gemms.py --spec 512,2048,8192,bf16,e4m3,bf16
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    label: str
    M: int
    N: int
    K: int
    a_dt: str
    b_dt: str
    out_dt: str

    @property
    def shape_str(self) -> str:
        dt = f"{self.a_dt}@{self.b_dt}→{self.out_dt}"
        return f"{self.M:>4}×{self.N:>5}×{self.K:>5}  {dt}"

    @property
    def compute_dt(self) -> str | None:
        # Mixed-dtype: PTX needs compute_dtype hint so it casts A in
        # smem; cuBLAS dispatch_cublas does its own inline cast.
        return self.b_dt if self.a_dt != self.b_dt else None


# Waypoint-1.5 production Linears (720p: tpf = 16*32 = 512).
# Dtypes match the fp8-everywhere inference path: bf16 act × e4m3
# weight → bf16 out. cond_projs run bf16×bf16 at prepare() time.
_WAYPOINT = [
    Problem("qkv_proj",   512, 4096, 2048, "bf16", "e4m3", "bf16"),
    Problem("out_proj",   512, 2048, 2048, "bf16", "e4m3", "bf16"),
    Problem("mlp.fc1",    512, 8192, 2048, "bf16", "e4m3", "bf16"),
    Problem("mlp.fc2",    512, 2048, 8192, "bf16", "e4m3", "bf16"),
    Problem("out_norm",   512, 4096, 2048, "bf16", "e4m3", "bf16"),
    Problem("noise_fc1",    1, 8192,  512, "bf16", "bf16", "bf16"),
    Problem("noise_fc2",    1, 2048, 8192, "bf16", "bf16", "bf16"),
    Problem("cond_proj",    1, 2048, 2048, "bf16", "bf16", "bf16"),
]


def _make_tensors(p: Problem):
    """Allocate device buffers for one GEMM problem."""
    import numpy as np

    from quark.runtime.npconv import astype_numpy, zeros_for_dtype
    from quark.runtime.tensor import QuarkTensor

    rng = np.random.default_rng(0xC0FFEE)
    # 0.1 scale keeps values in-range for e4m3 (max ≈ 448) after matmul
    # without overflow — the bench doesn't check correctness but NaN
    # values can trip some kernels' validity checks.
    a_np = astype_numpy(
        rng.standard_normal((p.M, p.K)).astype(np.float32) * 0.1, p.a_dt
    )
    b_np = astype_numpy(
        rng.standard_normal((p.N, p.K)).astype(np.float32) * 0.1, p.b_dt
    )
    out_np = zeros_for_dtype((p.M, p.N), p.out_dt)

    # QuarkTensor.from_numpy expects bytes-correct numpy arrays. For fp8
    # combos astype_numpy emits uint8-carried arrays; from_numpy then
    # reinterprets with the explicit dtype kwarg.
    A = QuarkTensor.from_numpy(a_np, dtype=p.a_dt)
    B = QuarkTensor.from_numpy(b_np, dtype=p.b_dt)
    Out = QuarkTensor.from_numpy(out_np, dtype=p.out_dt)
    return A, B, Out


def _bench_graphed(fn, *, n_warmup: int, n_bench: int) -> float:
    """Capture ``fn`` into a CUDA graph, replay ``n_bench`` times
    bracketed by CUDA events, return µs per replay."""
    from quark.graph import capture_graph
    from quark.runtime.cuda import CudaRuntime

    rt = CudaRuntime.instance()

    # Eager warmup — prime cublasLt's internal algo cache, the autotune
    # cache, any first-call JIT / pool-warmup effects. Without this the
    # captured graph can record slow cold-start nodes.
    for _ in range(n_warmup):
        fn()
    rt.stream_synchronize(0)

    # Capture: runs fn() once, records all GPU ops as a replayable graph.
    with capture_graph() as g:
        fn()

    # Time replays.
    start = rt.event_create()
    end = rt.event_create()
    rt.event_record(start, 0)
    for _ in range(n_bench):
        g.replay()
    rt.event_record(end, 0)
    rt.event_synchronize(end)
    total_ms = rt.event_elapsed_time(start, end)
    rt.event_destroy(start)
    rt.event_destroy(end)
    rt.stream_synchronize(0)
    return (total_ms * 1000.0) / max(n_bench, 1)


def _bench_ptx(p: Problem, A, B, Out, *, n_warmup: int, n_bench: int) -> float:
    """Time the best autotuned PTX config for this problem."""
    import quark
    import quark.functional as qf

    # QUARK_DISABLE_CUBLAS=1 disables both the _try_cublas shortcut AND
    # the cuBLAS autotune candidate (via is_cublas_eligible). So
    # ``qf.gemm`` is forced onto the PTX kernel — which is what we
    # want for the PTX-only timing.
    os.environ["QUARK_DISABLE_CUBLAS"] = "1"

    kw = {"out_dtype": p.out_dt}
    if p.compute_dt is not None:
        kw["compute_dtype"] = p.compute_dt

    # Prime the autotune cache with full search so the PTX-only
    # winner is locked in before we capture.
    with quark.max_autotune():
        qf.gemm(A, B, out=Out, **kw)
    from quark.runtime.cuda import CudaRuntime

    CudaRuntime.instance().stream_synchronize(0)

    def _run():
        qf.gemm(A, B, out=Out, **kw)

    return _bench_graphed(_run, n_warmup=n_warmup, n_bench=n_bench)


def _bench_cublas(p: Problem, A, B, Out, *, n_warmup: int, n_bench: int) -> float | None:
    """Time ``dispatch_cublas`` directly — cast (if mixed-dtype) + matmul."""
    from quark.kernels.gemm.cublas_dispatch import dispatch_cublas, is_cublas_eligible
    from quark.kernels.gemm.spec import GemmSpec

    # Make sure cuBLAS is reachable for this eligibility check.
    os.environ.pop("QUARK_DISABLE_CUBLAS", None)

    spec = GemmSpec(
        M=p.M,
        N=p.N,
        K=p.K,
        a_dtype=p.a_dt,
        b_dtype=p.b_dt,
        out_dtype=p.out_dt,
        compute_dtype=p.compute_dt,
    )
    if not is_cublas_eligible(spec):
        return None

    def _run():
        dispatch_cublas(A=A, B=B, Out=Out)

    return _bench_graphed(_run, n_warmup=n_warmup, n_bench=n_bench)


def _parse_spec(s: str) -> Problem:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"--spec expects M,N,K,a_dt,b_dt,out_dt (6 fields), got {len(parts)}: {s!r}"
        )
    M, N, K, a_dt, b_dt, out_dt = parts
    return Problem("user", int(M), int(N), int(K), a_dt, b_dt, out_dt)


def main() -> int:
    import sys as _sys

    if _sys.platform == "darwin":
        print("bench_gemms.py: Metal has no cuBLAS; exiting.", file=_sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=str, default=None,
                        help="Single problem 'M,N,K,a_dt,b_dt,out_dt' instead of the Waypoint set")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Eager warmup iters before graph capture")
    parser.add_argument("--bench", type=int, default=500,
                        help="Graph replays per timed section")
    args = parser.parse_args()

    problems = [_parse_spec(args.spec)] if args.spec else _WAYPOINT

    print()
    print(f"{'label':<12} {'shape':<34}  {'PTX (µs)':>9}  {'cuBLAS (µs)':>12}  {'winner':>8}  {'speedup':>8}")
    print("-" * 96)

    for p in problems:
        A, B, Out = _make_tensors(p)
        ptx_us = _bench_ptx(p, A, B, Out, n_warmup=args.warmup, n_bench=args.bench)
        cublas_us = _bench_cublas(p, A, B, Out, n_warmup=args.warmup, n_bench=args.bench)

        if cublas_us is None:
            print(
                f"{p.label:<12} {p.shape_str:<34}  "
                f"{ptx_us:>9.2f}  {'n/a':>12}  {'PTX':>8}  {'—':>8}"
            )
            continue

        winner = "cuBLAS" if cublas_us < ptx_us else "PTX"
        ratio = max(ptx_us, cublas_us) / min(ptx_us, cublas_us)
        print(
            f"{p.label:<12} {p.shape_str:<34}  "
            f"{ptx_us:>9.2f}  {cublas_us:>12.2f}  {winner:>8}  {ratio:>7.2f}×"
        )

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
