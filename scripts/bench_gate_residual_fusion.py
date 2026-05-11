#!/usr/bin/env python3
"""Microbench: standalone GEMM + AdaGateResidual vs the fused epilogue.

Both versions compute ``Out = Residual + Gate * (A @ B.T)`` at the
Waypoint attn_gate / mlp_gate shapes (360p preset). Reports per-call
wall time after warmup, and the speedup of the fused path.

Key context: the fused kernel falls back to the simdgroup_matrix
build() path (the non-NAX path) — has_gate_residual rejects NAX in
``GemmKernel.is_valid``. So this bench is "fused-on-non-NAX" vs
"NAX GEMM + standalone AdaGateResidual". A negative result tells us
the NAX speedup outweighs the dispatch+round-trip savings; a
positive result tells us the fusion is worth porting to NAX.

    python scripts/bench_gate_residual_fusion.py
"""

from __future__ import annotations

import time

import numpy as np


def _bf16(arr_f32):
    arr_f32 = np.ascontiguousarray(arr_f32, dtype=np.float32)
    return (arr_f32.view(np.uint32) >> 16).astype(np.uint16)


def _make_inputs(M, N, K, G, seed=0xA17E):
    from quark.runtime.tensor import QuarkTensor

    rng = np.random.default_rng(seed)
    A = QuarkTensor.from_numpy(
        _bf16(rng.standard_normal((M, K)) * 0.1), dtype="bf16"
    ).reshape(M, K)
    B = QuarkTensor.from_numpy(
        _bf16(rng.standard_normal((N, K)) * 0.1), dtype="bf16"
    ).reshape(N, K)
    Gate = QuarkTensor.from_numpy(
        _bf16(rng.standard_normal((G, N)) * 0.5), dtype="bf16"
    ).reshape(G, N)
    Residual = QuarkTensor.from_numpy(
        _bf16(rng.standard_normal((M, N)) * 0.1), dtype="bf16"
    ).reshape(M, N)
    return A, B, Gate, Residual


def _sync():
    from quark.runtime.sync import synchronize

    synchronize()


def _bench(fn, *, warmup_ms=200.0, bench_ms=600.0):
    # Each fn() call ends with synchronize(): the standalone path
    # crosses a queue_launch (NAX gemm) → eager launch (ada_gate)
    # boundary, and only the explicit eval flushes the lazy queue
    # before the eager kernel reads the gemm output. Mirrors
    # ``quark.lazy()`` framing in the world-engine bench (which evals
    # at end-of-frame). Per-iter sync is intentional — both paths pay
    # it equally, so the comparison stays apples-to-apples.
    def _step():
        fn()
        _sync()

    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1000 < warmup_ms:
        _step()
    t1 = time.perf_counter()
    n = 0
    while (time.perf_counter() - t1) * 1000 < bench_ms:
        _step()
        n += 1
    elapsed_ms = (time.perf_counter() - t1) * 1000
    return elapsed_ms / max(n, 1), n


def _bench_one(M, N, K, G):
    import quark.functional as qf

    A, B, Gate, Residual = _make_inputs(M, N, K, G)

    # Standalone path: NAX GEMM, then AdaGateResidual op.
    def standalone():
        y = qf.gemm(A, B, out_dtype="bf16")
        return qf.ada_gate_residual(Residual, y, Gate)

    # Fused path: single dispatch through the extended GEMM epilogue
    # (falls back to the non-NAX simdgroup_matrix build()).
    def fused():
        return qf.gemm(A, B, gate=Gate, residual=Residual, gate_groups=G, out_dtype="bf16")

    s_us, s_n = _bench(standalone)
    f_us, f_n = _bench(fused)

    print(f"  shape M={M} N={N} K={K} G={G}")
    print(f"    standalone (NAX gemm + ada_gate): {s_us * 1000:.1f} µs / call ({s_n} iters)")
    print(f"    fused      (non-NAX gemm+epilogue): {f_us * 1000:.1f} µs / call ({f_n} iters)")
    if f_us < s_us:
        print(f"    fused is {s_us / f_us:.2f}× faster")
    else:
        print(f"    fused is {f_us / s_us:.2f}× slower")
    return s_us, f_us


def main():
    print("Waypoint attn_gate / mlp_gate shapes (360p preset, tpf=128, d=2048):")
    cases = [
        ("attn_gate (out_proj)", 128, 2048, 2048, 1),
        ("mlp_gate (mlp.fc2)", 128, 2048, 8192, 1),
    ]
    summary = []
    for name, M, N, K, G in cases:
        print(f"\n[{name}]")
        s, f = _bench_one(M, N, K, G)
        summary.append((name, s, f))

    print("\nSummary:")
    for name, s, f in summary:
        ratio = (s / f) if f > 0 else float("inf")
        sign = "faster" if f < s else "slower"
        print(f"  {name:25s}  standalone={s * 1000:.1f}µs  fused={f * 1000:.1f}µs  "
              f"({ratio if f < s else s / f:.2f}× {sign})")


if __name__ == "__main__":
    main()
