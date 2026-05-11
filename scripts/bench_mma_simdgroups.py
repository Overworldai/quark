#!/usr/bin/env python3
"""Microbench: ``matmul2d<execution_simdgroups<N>>`` vs N independent
``matmul2d<execution_simdgroup>`` calls on the same threadgroup tile.

Hypothesis: Apple's cooperative ``execution_simdgroups<N>`` API
schedules N simdgroups against one matmul tile better than N
independent single-simdgroup calls — each simdgroup re-fetches its
own slice of A and B from gmem, vs. the cooperative API which can
share A/B through smem internally.

If that hypothesis holds, it justifies extending the IR to track
``n_simdgroups`` per MmaShape and emit ``execution_simdgroups<N>``
for the new shapes — which in turn lets the owl_attn kernel use
larger BQ tiles with K/V smem sharing across simdgroups (the next
step listed in the docstring at ``kernels/owl_attn/nax.py``).

Result protocol (this script):
  1. Build two MSL kernels via raw ``_md.compile``: ``coop_<N>`` uses
     ``execution_simdgroups<N>`` on a (BM, BN, K) descriptor;
     ``manual_<N>`` runs N single-simdgroup matmul2d calls in a
     loop, each on its own (BM/W*M, BN/W*N) sub-tile.
  2. Dispatch each at fixed shape (M=512, K=2048, N=2048 — the 720p
     out_proj shape) and at fixed grid (one threadgroup processes
     one BM × BN output block; n_blocks = (M / BM) × (N / BN)).
  3. Time 100 launches inside one ``quark.lazy()`` block; report
     median per-call ms.

Usage:
    uv run --offline python scripts/bench_mma_simdgroups.py
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np


def _read_f32(addr: int, count: int) -> np.ndarray:
    arr_ty = ctypes.c_float * count
    arr = arr_ty.from_address(addr)
    return np.array(arr, dtype=np.float32, copy=True)


def _coop_msl_source(*, label: str, BM: int, BN: int, BK: int, n_simd: int,
                     M: int, K: int, N: int) -> str:
    """Cooperative matmul2d kernel: one
    ``matmul2d<execution_simdgroups<N>>`` call per BM×BN output tile
    per K-iteration. Apple manages simdgroup partitioning of the
    cooperative tensors.

    Layout:
      A: [M, K]   row-major bf16, no transpose
      B: [N, K]   row-major bf16, transpose_right (so the matmul reads B as [K, N])
      C: [M, N]   row-major f32, no transpose
    """
    # Layout convention (validated via /tmp/test_layout.py):
    #   row-major A[M,K]  → declare extents<K, M>, slice(k, m0)
    #   row-major B[N,K]  → declare extents<K, N>, slice(k, n0), transpose_right=true
    #   row-major C[M,N]  → declare extents<N, M>, slice(n0, m0)
    threads_per_tg = 32 * n_simd
    return f"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

[[kernel, max_total_threads_per_threadgroup({threads_per_tg})]]
void {label}(device bfloat *A_buf [[buffer(0)]],
              device bfloat *B_buf [[buffer(1)]],
              device float  *C_buf [[buffer(2)]],
              uint2 tgid     [[threadgroup_position_in_grid]])
{{
    auto A = tensor<device bfloat, extents<int32_t, {K}, {M}>, tensor_inline>(
        A_buf, extents<int32_t, {K}, {M}>());
    auto B = tensor<device bfloat, extents<int32_t, {K}, {N}>, tensor_inline>(
        B_buf, extents<int32_t, {K}, {N}>());
    auto C = tensor<device float,  extents<int32_t, {N}, {M}>, tensor_inline>(
        C_buf, extents<int32_t, {N}, {M}>());

    int m0 = int(tgid.y) * {BM};
    int n0 = int(tgid.x) * {BN};

    auto sC = C.slice(n0, m0);

    // First iter: multiply (no accumulate); rest: multiply_accumulate.
    constexpr auto desc0 = matmul2d_descriptor(
        {BM}, {BN}, {BK}, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc0, execution_simdgroups<{n_simd}>> op0;
    {{
        auto sA = A.slice(0, m0);
        auto sB = B.slice(0, n0);
        op0.run(sA, sB, sC);
    }}

    constexpr auto desc = matmul2d_descriptor(
        {BM}, {BN}, {BK}, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<{n_simd}>> op;
    for (int k = {BK}; k < {K}; k += {BK}) {{
        auto sA = A.slice(k, m0);
        auto sB = B.slice(k, n0);
        op.run(sA, sB, sC);
    }}
}}
"""


def _manual_msl_source(*, label: str, BM: int, BN: int, BK: int, n_simd: int,
                       M: int, K: int, N: int, WM: int, WN: int) -> str:
    """Manual multi-simdgroup composition: each simdgroup runs its
    own ``matmul2d<execution_simdgroup>`` on its (BM/WM × BN/WN)
    slice of the BM × BN output tile per K-iteration. Mirrors what
    quark's existing GEMM kernel emits via the IR ``mma`` op.
    """
    assert WM * WN == n_simd, f"WM*WN={WM * WN} != n_simd={n_simd}"
    SM = BM // WM
    SN = BN // WN
    threads_per_tg = 32 * n_simd
    return f"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

[[kernel, max_total_threads_per_threadgroup({threads_per_tg})]]
void {label}(device bfloat *A_buf [[buffer(0)]],
                device bfloat *B_buf [[buffer(1)]],
                device float  *C_buf [[buffer(2)]],
                uint  sgid      [[simdgroup_index_in_threadgroup]],
                uint2 tgid      [[threadgroup_position_in_grid]])
{{
    auto A = tensor<device bfloat, extents<int32_t, {K}, {M}>, tensor_inline>(
        A_buf, extents<int32_t, {K}, {M}>());
    auto B = tensor<device bfloat, extents<int32_t, {K}, {N}>, tensor_inline>(
        B_buf, extents<int32_t, {K}, {N}>());
    auto C = tensor<device float,  extents<int32_t, {N}, {M}>, tensor_inline>(
        C_buf, extents<int32_t, {N}, {M}>());

    int sg_row = int(sgid) / {WN};
    int sg_col = int(sgid) % {WN};
    int m0 = int(tgid.y) * {BM} + sg_row * {SM};
    int n0 = int(tgid.x) * {BN} + sg_col * {SN};

    auto sC = C.slice(n0, m0);

    constexpr auto desc0 = matmul2d_descriptor(
        {SM}, {SN}, {BK}, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc0, execution_simdgroup> op0;
    {{
        auto sA = A.slice(0, m0);
        auto sB = B.slice(0, n0);
        op0.run(sA, sB, sC);
    }}

    constexpr auto desc = matmul2d_descriptor(
        {SM}, {SN}, {BK}, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    for (int k = {BK}; k < {K}; k += {BK}) {{
        auto sA = A.slice(k, m0);
        auto sB = B.slice(k, n0);
        op.run(sA, sB, sC);
    }}
}}
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--M", type=int, default=512)
    p.add_argument("--K", type=int, default=2048)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument("--n-warmup", type=int, default=20)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import quark
    from quark.drivers import _metal_dispatch as _md
    from quark.runtime.tensor import QuarkTensor

    M, K, N = args.M, args.K, args.N
    print(f"shape: M={M} K={K} N={N}  (~720p out_proj)")

    # Use real data so we can verify correctness against numpy.
    rng = np.random.default_rng(0)
    A_np = rng.standard_normal((M, K), dtype=np.float32) * 0.1
    B_np = rng.standard_normal((N, K), dtype=np.float32) * 0.1

    def f32_to_bf16(x: np.ndarray) -> np.ndarray:
        return (x.view(np.uint32) >> 16).astype(np.uint16)

    def bf16_to_f32(x: np.ndarray) -> np.ndarray:
        return (x.astype(np.uint32) << 16).view(np.float32)

    # Round trip → both arrays now reflect the bf16 storage exactly.
    A_bf16 = f32_to_bf16(A_np.astype(np.float32))
    B_bf16 = f32_to_bf16(B_np.astype(np.float32))
    A_eff = bf16_to_f32(A_bf16)
    B_eff = bf16_to_f32(B_bf16)
    C_ref = A_eff @ B_eff.T  # bf16 inputs, f32 reference

    A_t = QuarkTensor.from_numpy(A_bf16, dtype="bf16")
    B_t = QuarkTensor.from_numpy(B_bf16, dtype="bf16")
    C_t = QuarkTensor.zeros(M, N, dtype="f32")

    candidates = []
    # Each entry: (label, src_fn, (BM, BN, n_simd))
    def add_manual(label, BM, BN, BK, n_simd, WM, WN):
        candidates.append((label,
                           lambda: _manual_msl_source(label=label, BM=BM, BN=BN, BK=BK,
                                                      n_simd=n_simd, M=M, K=K, N=N,
                                                      WM=WM, WN=WN),
                           (BM, BN, n_simd)))

    def add_coop(label, BM, BN, BK, n_simd):
        candidates.append((label,
                           lambda: _coop_msl_source(label=label, BM=BM, BN=BN, BK=BK,
                                                    n_simd=n_simd, M=M, K=K, N=N),
                           (BM, BN, n_simd)))

    add_manual("manual_BM64xBN128_w8_WM2WN4", 64, 128, 64, 8, 2, 4)
    add_coop("coop_BM64xBN128_simd8", 64, 128, 64, 8)
    add_coop("coop_BM64xBN64_simd4", 64, 64, 64, 4)
    add_coop("coop_BM128xBN64_simd4", 128, 64, 64, 4)
    add_coop("coop_BM128xBN128_simd8", 128, 128, 64, 8)
    add_coop("coop_BM128xBN256_simd16", 128, 256, 64, 16)

    _MSL_4_0 = 262144
    plan = ((0, 0, 0), (0, 1, 1), (1, 0, 2))
    for name, src_fn, (BM, BN, n_simd) in candidates:
        if M % BM != 0 or N % BN != 0:
            print(f"  {name:<40s}  SKIP (shape not divisible)")
            continue
        try:
            src = src_fn()
            pipeline = _md.compile(src, name, _MSL_4_0)
        except Exception as e:
            err = str(e)
            print(f"  {name:<40s}  COMPILE FAILED: {type(e).__name__}: {err[:80]}")
            continue

        nb_n = N // BN
        nb_m = M // BM
        grid = (nb_n * 32 * n_simd, nb_m, 1)
        tg = (32 * n_simd, 1, 1)

        A_h = A_t.metal_handle
        B_h = B_t.metal_handle
        C_h = C_t.metal_handle
        out_nbytes = C_t._storage.nbytes

        # Correctness — single launch, sync, compare to numpy reference.
        ctypes.memset(C_t._storage.ptr, 0, C_t._storage.nbytes)
        _md.queue_launch(
            pipeline,
            [A_h, B_h], [0, 0], [0, 0],
            out_nbytes, list(plan), grid, tg, 0,
            C_h,
        )
        if _md.has_lazy_pending():
            _md.eval_queue()
        from quark.functional._dispatch import launcher
        launcher().driver.sync(None)

        C_got = _read_f32(C_t._storage.ptr, M * N).reshape(M, N)
        max_err = float(np.max(np.abs(C_got - C_ref)))
        rel_err = max_err / (np.max(np.abs(C_ref)) + 1e-9)
        if rel_err > 0.02:
            print(f"  {name:<40s}  CORRECTNESS FAIL: rel_err={rel_err:.4f} max_err={max_err:.3f}")
            continue

        # Warmup
        for _ in range(args.n_warmup):
            _md.queue_launch(
                pipeline,
                [A_h, B_h], [0, 0], [0, 0],
                out_nbytes, list(plan), grid, tg, 0,
                C_h,
            )
        if _md.has_lazy_pending():
            _md.eval_queue()
        launcher().driver.sync(None)

        # Timed
        t0 = time.perf_counter()
        with quark.lazy():
            for _ in range(args.n_iters):
                _md.queue_launch(
                    pipeline,
                    [A_h, B_h], [0, 0], [0, 0],
                    out_nbytes, list(plan), grid, tg, 0,
                    C_h,
                )
        per_call_ms = (time.perf_counter() - t0) * 1000 / args.n_iters
        print(f"  {name:<40s}  {per_call_ms:7.3f} ms / call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
