"""End-to-end perf smoke for ``popcorn.functional``.

Mimics how a downstream caller drives the public ops: build inputs,
warm up once, then time in a loop. Prints TFLOPS and median runtime
per problem.

Self-contained — no imports from ``popcorn`` internals (no Problem
registry, no timing helpers). Only the public ``popcorn.functional``
surface is touched.

Usage:

    .venv/bin/python tests/e2e.py
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import popcorn
import popcorn.functional as pcf

IS_METAL = platform.system() == "Darwin"

if IS_METAL:
    import mlx.core as mx

    BF16 = mx.bfloat16
    F32 = mx.float32
    I32 = mx.int32
    FP8 = None  # CUDA-only fast path; bf16 weights on Metal

    def randn(shape, dtype=BF16):
        base = mx.random.normal(shape).astype(BF16)
        return base if dtype is BF16 else base.astype(dtype)

    def zeros(shape, dtype):
        return mx.zeros(shape, dtype=dtype)

    def i32_tensor(seq):
        return mx.array(list(seq), dtype=I32)

    def arange_mod(n, mod):
        return (mx.arange(n, dtype=I32) % mod).astype(I32)

    def full_f32(shape, value):
        return mx.full(shape, value, dtype=F32)

    def device_sync():
        pass

    def eval_out(out):
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
else:
    import torch

    BF16 = torch.bfloat16
    F32 = torch.float32
    I32 = torch.int32
    FP8 = torch.float8_e4m3fn
    DEVICE = "cuda"

    def randn(shape, dtype=BF16):
        base = torch.randn(*shape, dtype=BF16, device=DEVICE)  # ty: ignore[no-matching-overload]
        return base if dtype is BF16 else base.to(dtype)

    def zeros(shape, dtype):
        return torch.zeros(*shape, dtype=dtype, device=DEVICE)

    def i32_tensor(seq):
        return torch.tensor(list(seq), dtype=I32, device=DEVICE)

    def arange_mod(n, mod):
        return torch.arange(n, dtype=I32, device=DEVICE) % mod  # ty: ignore[no-matching-overload]

    def full_f32(shape, value):
        return torch.full(shape, value, dtype=F32, device=DEVICE)  # ty: ignore[no-matching-overload]

    def device_sync():
        torch.cuda.synchronize()

    def eval_out(out):
        torch.cuda.synchronize()


@dataclass
class Bench:
    name: str
    flops: int  # 0 → bandwidth-bound, print runtime only
    fn: Callable[[], object]


# ---- per-kernel input + call factories --------------------------------------


def gemm_bench(name, M, N, K, *, fp8_shuffle):
    A = randn((M, K), dtype=BF16)
    if fp8_shuffle:
        B_raw = randn((N, K), dtype=FP8)
        B = pcf.shuffle_b_for_gemm(A, B_raw, out_dtype="bf16", compute_dtype="e4m3")

        def call():
            return pcf.gemm(A, B, out_dtype="bf16", compute_dtype="e4m3", b_shuffled=True)
    else:
        B = randn((N, K), dtype=BF16)

        def call():
            return pcf.gemm(A, B, out_dtype="bf16")

    return Bench(name=name, flops=2 * M * N * K, fn=call)


def attention_bench(name, *, B, n_kv_heads, gqa_ratio, seq_len, kv_len, Dh):
    n_q = n_kv_heads * gqa_ratio
    Q = randn((B * n_q * seq_len, Dh))
    K = randn((B * n_kv_heads * kv_len, Dh))
    V_t = randn((B * n_kv_heads * Dh, kv_len))

    def call():
        return pcf.attention(
            Q,
            K,
            V_t,
            B=B,
            n_kv_heads=n_kv_heads,
            gqa_ratio=gqa_ratio,
            seq_len=seq_len,
            kv_len=kv_len,
        )

    flops = 4 * B * n_q * seq_len * kv_len * Dh
    return Bench(name=name, flops=flops, fn=call)


def owl_attn_bench(
    name,
    *,
    H_spatial,
    W_spatial,
    num_buckets,
    pinned_dilation,
    B=1,
    n_kv_heads=16,
    gqa_ratio=2,
    Dh=64,
    max_segments=3,
):
    tpf = H_spatial * W_spatial
    capacity = num_buckets * tpf + tpf
    n_q = n_kv_heads * gqa_ratio
    Q = randn((B * n_q * tpf, Dh))
    K_cache = randn((B * n_kv_heads * capacity, Dh))
    Vt_cache = randn((B * n_kv_heads * Dh, capacity))
    cos = randn((tpf, Dh // 2), dtype=F32)
    sin = randn((tpf, Dh // 2), dtype=F32)
    L = num_buckets * tpf
    seg_pad = B * max_segments * 2 - 4
    segments = i32_tensor([0, L, L, tpf] + [0] * seg_pad)
    n_segments = i32_tensor([2] * B)

    def call():
        return pcf.owl_attn(
            Q,
            K_cache,
            Vt_cache,
            cos,
            sin,
            segments,
            n_segments,
            B=B,
            n_kv_heads=n_kv_heads,
            gqa_ratio=gqa_ratio,
            H_spatial=H_spatial,
            W_spatial=W_spatial,
            num_buckets=num_buckets,
            pinned_dilation=pinned_dilation,
            max_segments=max_segments,
        )

    flops = 4 * B * n_q * tpf * capacity * Dh
    return Bench(name=name, flops=flops, fn=call)


def kv_cache_update_bench(
    name,
    *,
    H_spatial,
    W_spatial,
    num_buckets,
    pinned_dilation,
    B=1,
    n_kv_heads=16,
    Dh=64,
    kv_dtype=None,
    max_segments=3,
):
    tpf = H_spatial * W_spatial
    capacity = num_buckets * tpf + tpf
    kv_dt = FP8 if kv_dtype == "e4m3" else BF16
    K = randn((B * n_kv_heads * tpf, Dh), dtype=BF16)
    V = randn((B * n_kv_heads * tpf, Dh), dtype=BF16)
    cos = randn((tpf, Dh // 2), dtype=F32)
    sin = randn((tpf, Dh // 2), dtype=F32)
    frame_t = i32_tensor([0])
    K_cache = zeros((B * n_kv_heads * capacity, Dh), dtype=kv_dt)
    Vt_cache = zeros((B * n_kv_heads * Dh, capacity), dtype=kv_dt)
    segments = zeros((B * max_segments * 2,), dtype=I32)
    n_segments = zeros((B,), dtype=I32)

    def call():
        return pcf.kv_cache_update(
            K,
            V,
            cos,
            sin,
            frame_t,
            Vt_cache,
            segments,
            n_segments,
            K_cache,
            B=B,
            n_kv_heads=n_kv_heads,
            H_spatial=H_spatial,
            W_spatial=W_spatial,
            num_buckets=num_buckets,
            pinned_dilation=pinned_dilation,
            kv_dtype=kv_dtype,
            max_segments=max_segments,
        )

    return Bench(name=name, flops=0, fn=call)


def _moe_routing(M, total_slots, n_experts):
    BM = 32
    slots_per_expert = total_slots // n_experts
    token_ids = arange_mod(total_slots, M)
    slot_weights = full_f32((total_slots,), 0.5)
    wl = []
    for grp_start in range(0, total_slots, BM):
        wl.extend([grp_start, grp_start // slots_per_expert])
    return token_ids, slot_weights, i32_tensor(wl)


def moe_inproj_bench(name, *, M, D, H, n_experts=16, top_k=4, fp8_shuffle):
    total_slots = M * top_k
    X = randn((M, D), dtype=BF16)
    w_dt = FP8 if fp8_shuffle else BF16
    W_raw = randn((n_experts * H, D), dtype=w_dt)
    token_ids, _, work_list = _moe_routing(M, total_slots, n_experts)
    if fp8_shuffle:
        W_in = pcf.shuffle_b_for_moe_inproj(
            X,
            W_raw,
            n_experts=n_experts,
            top_k=top_k,
            out_dtype="bf16",
            compute_dtype="e4m3",
        )
        kw = dict(out_dtype="bf16", compute_dtype="e4m3")
    else:
        W_in = W_raw
        kw = dict(out_dtype="bf16")

    def call():
        return pcf.moe_inproj(
            X,
            W_in,
            token_ids,
            work_list,
            n_experts=n_experts,
            top_k=top_k,
            **kw,
        )

    flops = 2 * total_slots * H * D
    return Bench(name=name, flops=flops, fn=call)


def moe_outproj_bench(name, *, M, D, H, n_experts=16, top_k=4, fp8_shuffle):
    total_slots = M * top_k
    h_in = randn((total_slots, H), dtype=BF16)
    w_dt = FP8 if fp8_shuffle else BF16
    W_raw = randn((n_experts * D, H), dtype=w_dt)
    token_ids, slot_weights, work_list = _moe_routing(M, total_slots, n_experts)
    if fp8_shuffle:
        W_out = pcf.shuffle_b_for_moe_outproj(
            h_in,
            W_raw,
            M=M,
            n_experts=n_experts,
            top_k=top_k,
            out_dtype="bf16",
            compute_dtype="e4m3",
        )
        kw = dict(out_dtype="bf16", compute_dtype="e4m3")
    else:
        W_out = W_raw
        kw = dict(out_dtype="bf16")

    def call():
        return pcf.moe_outproj(
            h_in,
            W_out,
            token_ids,
            slot_weights,
            work_list,
            M=M,
            n_experts=n_experts,
            top_k=top_k,
            **kw,
        )

    flops = 2 * total_slots * D * H
    return Bench(name=name, flops=flops, fn=call)


# ---- problem set (mirrors `make bench TAG=metal` / `TAG=cuda`) --------------


def build_problems():
    fp8 = not IS_METAL
    out: list[Bench] = []

    for label, M in [("360p", 128), ("720p", 512)]:
        for layer, N, K in [
            ("qkv", 4096, 2048),
            ("attn_out", 2048, 2048),
            ("ffn_up", 8192, 2048),
            ("ffn_down", 2048, 8192),
        ]:
            out.append(gemm_bench(f"gemm/{layer}_{label}", M, N, K, fp8_shuffle=fp8))

    for label, Hs, Ws in [("360p", 16, 8), ("720p", 32, 16)]:
        for variant, dilation in [("dense", 1), ("dilated", 8)]:
            out.append(
                owl_attn_bench(
                    f"owl_attn/{variant}_{label}",
                    H_spatial=Hs,
                    W_spatial=Ws,
                    num_buckets=16,
                    pinned_dilation=dilation,
                )
            )

    kv_dt = "e4m3" if fp8 else None
    for label, Hs, Ws in [("360p", 16, 8), ("720p", 32, 16)]:
        for variant, dilation in [("dense", 1), ("dilated", 8)]:
            out.append(
                kv_cache_update_bench(
                    f"kv_cache/{variant}_{label}",
                    H_spatial=Hs,
                    W_spatial=Ws,
                    num_buckets=16,
                    pinned_dilation=dilation,
                    kv_dtype=kv_dt,
                )
            )

    for label, M in [("360p", 128), ("720p", 512)]:
        out.append(
            moe_inproj_bench(
                f"moe_inproj/{label}",
                M=M,
                D=2048,
                H=2048,
                fp8_shuffle=fp8,
            )
        )
        out.append(
            moe_outproj_bench(
                f"moe_outproj/{label}",
                M=M,
                D=2048,
                H=2048,
                fp8_shuffle=fp8,
            )
        )

    return out


# ---- driver -----------------------------------------------------------------


def time_one(bench: Bench, n_iters: int = 50) -> tuple[float, float]:
    samples: list[float] = []
    for _ in range(n_iters):
        device_sync()
        t0 = time.perf_counter()
        out = bench.fn()
        eval_out(out)
        samples.append(time.perf_counter() - t0)
    med = statistics.median(samples)
    tflops = bench.flops / med / 1e12 if bench.flops else 0.0
    return med, tflops


def main():
    backend = "metal" if IS_METAL else "cuda"
    # Force full genetic search on every cache miss during prep + warmup.
    # Covers the shuffle_b_* helpers (invoked inside build_problems) and
    # the first call of each pcf.<op>, so timed iterations hit tuned
    # configs rather than the bounded fast-search winners.
    with popcorn.max_autotune():
        problems = build_problems()
        print(f"e2e: backend={backend}, {len(problems)} problems")

        print("warmup (full autotune)...", flush=True)
        for b in problems:
            device_sync()
            out = b.fn()
            eval_out(out)

    print(f"{'name':<32} {'median (us)':>12} {'tflops':>10}")
    print("-" * 56)
    for b in problems:
        med, tflops = time_one(b)
        tf = f"{tflops:>10.2f}" if b.flops else f"{'--':>10}"
        print(f"{b.name:<32} {med * 1e6:>12.2f} {tf}")


if __name__ == "__main__":
    main()
