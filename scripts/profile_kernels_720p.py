#!/usr/bin/env python3
"""Per-kernel profiler at 720p shapes.

The 720p preset (height=16, width=32 → tpf=512) is GPU-bound on the
DiT — the bench shows ~240 ms/NFE on M5 Max. We need to know which
kernels dominate before targeting any optimization.

Strategy: build the input tensors once with 720p shapes, then run N
iterations of each kernel inside a single ``quark.lazy()`` block,
sync, divide by N. Steady-state per-call wall time, no model-side
overhead. Inside the ``lazy()`` block the dispatcher accumulates
launches into one big command buffer (auto-committing every 50
dispatches), so this measures the same encode-and-flush pattern the
real forward uses.

Per-frame call counts (Waypoint-1.5-1B, 24 layers, 5 NFE):
  - GEMM qkv_proj:  24 × 5 = 120
  - GEMM out_proj:  24 × 5 = 120
  - GEMM mlp.fc1:   24 × 5 = 120  (+ fused silu epilogue)
  - GEMM mlp.fc2:   24 × 5 = 120  (+ fused gate/residual)
  - OwlAttn:        24 × 5 = 120
  - KVCacheUpdate:  24 × 5 = 120  (only frozen=False on last sigma)
  - AdaRMSNorm × 2: 24 × 5 = 240
  - HeadRMSNorm:    24 × 5 = 120
  - AdaGate:        depends on fuse_gate_residual
  - ValueResidual:  24 × 4 = 96 (skipped on first NFE)
  - Patchify:        5
  - Unpatchify:      1

Multiply per-call ms × call count to get predicted contribution.
Sort + sum and compare to the bench's saturated forward (~244 ms).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_quark_world_engine import (  # type: ignore[import-not-found]
    _make_config,
    _move_params_to_device,
    _randn,
    _sync,
    _tensor,
    _zeros,
)


def _time_op(name: str, fn: Callable[[], None], n_iters: int, n_warmup: int = 20) -> float:
    """Run ``fn`` ``n_iters`` times inside a single ``quark.lazy()`` block.

    Returns mean per-call wall time in milliseconds. Steady-state —
    the lazy block auto-commits every 50 dispatches so the GPU is
    pipelined; the mean over many iters approaches per-call cost
    (the once-per-block sync overhead amortizes to ~0).
    """
    import quark

    # Warmup outside the timing block (kernel autotune fires here).
    for _ in range(n_warmup):
        fn()
    _sync()

    t0 = time.perf_counter()
    with quark.lazy():
        for _ in range(n_iters):
            fn()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return dt_ms / n_iters


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["360p", "720p"], default="720p")
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument("--n-warmup", type=int, default=20)
    p.add_argument(
        "--kernels",
        default="all",
        help="comma list of kernel names to run (default: all). "
        "Names: qkv,out,fc1,fc2,owl,kv,adarms,headrms,adagate,vres,silu,patch,unpatch",
    )
    args = p.parse_args()

    import quark
    import quark.nn as nn

    cfg = _make_config(args.preset)
    d = cfg.d_model
    M = cfg.tpf  # tokens per frame
    half_dt = "bf16"
    out_dt = "bf16"

    print(f"preset: {args.preset}")
    print(f"  d_model={d}  n_heads={cfg.n_heads}  n_kv_heads={cfg.n_kv_heads}  Dh={cfg.Dh}")
    print(f"  height={cfg.height}  width={cfg.width}  tpf={M}")
    print(f"  qkv_dim={cfg.qkv_dim}  mlp_dim={cfg.mlp_dim}")
    print(f"  iters={args.n_iters} (warmup={args.n_warmup})")
    print()

    # Per-call counts in one frame (24 layers × 5 NFE = 120 unless noted).
    L = cfg.n_layers
    NFE = len(cfg.scheduler_sigmas) - 1
    block_calls = L * NFE  # 120

    rope_n_frames = max(cfg.num_buckets(i) * cfg.pinned_dilation(i) + 4 for i in range(L))

    # Pre-build tensors and modules. Each module is constructed once
    # and prepare()'d, then driven from the lazy loop.
    requested = set(args.kernels.split(",")) if args.kernels != "all" else None

    def want(name: str) -> bool:
        return requested is None or name in requested

    # Inputs that flow through every kernel (residual stream).
    x = _randn(M, d, dtype=half_dt)

    # Conditioning vectors (s, b, g) per AdaRMSNorm/Gate. Single-group.
    s = _randn(1, d, dtype=half_dt)
    b = _randn(1, d, dtype=half_dt)
    g = _randn(1, d, dtype=half_dt)

    frame_t = _tensor([1], dtype="s32")

    results: list[tuple[str, float, int]] = []  # (name, per-call ms, calls/frame)

    # ── GEMMs (via nn.Linear so the cached buffers + autotune entries match the model) ──
    if want("qkv"):
        lin_qkv = nn.Linear(d, cfg.qkv_dim, out_dtype=out_dt)
        _move_params_to_device(lin_qkv)
        ms = _time_op("qkv_proj", lambda: lin_qkv(x), args.n_iters, args.n_warmup)
        results.append(("qkv_proj", ms, block_calls))

    if want("out"):
        lin_out = nn.Linear(cfg.n_heads * cfg.Dh, d, out_dtype=out_dt)
        _move_params_to_device(lin_out)
        attn_out = _randn(M, cfg.n_heads * cfg.Dh, dtype=half_dt)
        if cfg.fuse_gate_residual:
            ms = _time_op(
                "out_proj+gate",
                lambda: lin_out(attn_out, gate=g, residual=x, gate_groups=1),
                args.n_iters,
                args.n_warmup,
            )
            results.append(("out_proj+gate", ms, block_calls))
        else:
            ms = _time_op("out_proj", lambda: lin_out(attn_out), args.n_iters, args.n_warmup)
            results.append(("out_proj", ms, block_calls))

    if want("fc1"):
        lin_fc1 = nn.Linear(d, cfg.mlp_dim, out_dtype=out_dt)
        _move_params_to_device(lin_fc1)
        ms = _time_op(
            "mlp.fc1+silu",
            lambda: lin_fc1(x, activation="silu"),
            args.n_iters,
            args.n_warmup,
        )
        results.append(("mlp.fc1+silu", ms, block_calls))

    if want("fc2"):
        lin_fc2 = nn.Linear(cfg.mlp_dim, d, out_dtype=out_dt)
        _move_params_to_device(lin_fc2)
        h = _randn(M, cfg.mlp_dim, dtype=half_dt)
        if cfg.fuse_gate_residual:
            ms = _time_op(
                "mlp.fc2+gate",
                lambda: lin_fc2(h, gate=g, residual=x, gate_groups=1),
                args.n_iters,
                args.n_warmup,
            )
            results.append(("mlp.fc2+gate", ms, block_calls))
        else:
            ms = _time_op("mlp.fc2", lambda: lin_fc2(h), args.n_iters, args.n_warmup)
            results.append(("mlp.fc2", ms, block_calls))

    # ── Attention path: KVCacheUpdate + OwlAttn ──
    # Use the same params as the real model; pick layer 0 (local window).
    if want("kv") or want("owl"):
        layer_idx = 0
        nb = cfg.num_buckets(layer_idx)
        pd = cfg.pinned_dilation(layer_idx)

        kv = nn.KVCacheUpdate(
            B=1,
            n_kv_heads=cfg.n_kv_heads,
            n_q_heads=cfg.n_heads,
            H_spatial=cfg.height,
            W_spatial=cfg.width,
            Dh=cfg.Dh,
            num_buckets=nb,
            pinned_dilation=pd,
            packed_qkv=True,
            rope_n_frames=rope_n_frames,
            dtype=half_dt,
            quilt_factor=cfg.quilt_factor,
            quilt_offset=0,
        )
        _move_params_to_device(kv)

        attn = nn.OwlAttn(
            B=1,
            n_kv_heads=cfg.n_kv_heads,
            gqa_ratio=cfg.gqa_ratio,
            H_spatial=cfg.height,
            W_spatial=cfg.width,
            num_buckets=nb,
            pinned_dilation=pd,
            packed_qkv=True,
            rope_n_frames=rope_n_frames,
            compute_dtype=None,
            quilt_factor=cfg.quilt_factor,
            quilt_offset=0,
        )
        _move_params_to_device(attn)

        qkv_packed = _randn(M, cfg.qkv_dim, dtype=half_dt)

        # Pre-fill KV with frames so OwlAttn sees a saturated cache.
        # We need frame_t > global_window for steady-state timing.
        for fi in range(cfg.global_window + 4):
            kv(qkv_packed, _tensor([fi], dtype="s32"), frozen=False)
        _sync()
        steady_ft = _tensor([cfg.global_window + 4], dtype="s32")

        if want("kv"):
            ms = _time_op(
                "kv_cache_update (local)",
                lambda: kv(qkv_packed, steady_ft, frozen=True),
                args.n_iters,
                args.n_warmup,
            )
            results.append(("kv_cache_update (local)", ms, block_calls))

        if want("owl"):
            ms = _time_op(
                "owl_attn (local)",
                lambda: attn(qkv_packed, kv, steady_ft),
                args.n_iters,
                args.n_warmup,
            )
            results.append(("owl_attn (local)", ms, block_calls))

        # Now also profile the global-attention layer (layer_idx=3 hits is_global).
        # 6 of 24 layers are global on this config (period=4, offset=-1 → layers 3,7,11,15,19,23).
        global_layer_idx = 3
        nb_g = cfg.num_buckets(global_layer_idx)
        pd_g = cfg.pinned_dilation(global_layer_idx)
        kv_g = nn.KVCacheUpdate(
            B=1,
            n_kv_heads=cfg.n_kv_heads,
            n_q_heads=cfg.n_heads,
            H_spatial=cfg.height,
            W_spatial=cfg.width,
            Dh=cfg.Dh,
            num_buckets=nb_g,
            pinned_dilation=pd_g,
            packed_qkv=True,
            rope_n_frames=rope_n_frames,
            dtype=half_dt,
            quilt_factor=cfg.quilt_factor,
            quilt_offset=global_layer_idx % cfg.quilt_factor,
        )
        _move_params_to_device(kv_g)
        attn_g = nn.OwlAttn(
            B=1,
            n_kv_heads=cfg.n_kv_heads,
            gqa_ratio=cfg.gqa_ratio,
            H_spatial=cfg.height,
            W_spatial=cfg.width,
            num_buckets=nb_g,
            pinned_dilation=pd_g,
            packed_qkv=True,
            rope_n_frames=rope_n_frames,
            compute_dtype=None,
            quilt_factor=cfg.quilt_factor,
            quilt_offset=global_layer_idx % cfg.quilt_factor,
        )
        _move_params_to_device(attn_g)
        for fi in range(cfg.global_window + 4):
            kv_g(qkv_packed, _tensor([fi], dtype="s32"), frozen=False)
        _sync()

        # 6/24 of the layers are global; 18/24 are local.
        n_local_layers = sum(1 for li in range(L) if not cfg.is_global(li))
        n_global_layers = L - n_local_layers
        local_calls = n_local_layers * NFE
        global_calls = n_global_layers * NFE

        # Re-record kv/owl with split call counts (replace local entries).
        if want("kv"):
            results = [r for r in results if not r[0].startswith("kv_cache_update")]
            ms_local = _time_op(
                "kv_cache_update (local)",
                lambda: kv(qkv_packed, steady_ft, frozen=True),
                args.n_iters,
                args.n_warmup,
            )
            ms_global = _time_op(
                "kv_cache_update (global)",
                lambda: kv_g(qkv_packed, steady_ft, frozen=True),
                args.n_iters,
                args.n_warmup,
            )
            results.append(("kv_cache_update (local)", ms_local, local_calls))
            results.append(("kv_cache_update (global)", ms_global, global_calls))

        if want("owl"):
            results = [r for r in results if not r[0].startswith("owl_attn")]
            ms_local = _time_op(
                "owl_attn (local)",
                lambda: attn(qkv_packed, kv, steady_ft),
                args.n_iters,
                args.n_warmup,
            )
            ms_global = _time_op(
                "owl_attn (global)",
                lambda: attn_g(qkv_packed, kv_g, steady_ft),
                args.n_iters,
                args.n_warmup,
            )
            results.append(("owl_attn (local)", ms_local, local_calls))
            results.append(("owl_attn (global)", ms_global, global_calls))

    # ── Norms / Gates ──
    if want("adarms"):
        norm = nn.AdaRMSNorm()
        _move_params_to_device(norm)
        ms = _time_op("ada_rmsnorm", lambda: norm(x, s, b), args.n_iters, args.n_warmup)
        results.append(("ada_rmsnorm", ms, 2 * block_calls))  # pre-attn + pre-mlp

    if want("headrms"):
        head = nn.HeadRMSNorm(cfg.n_heads, cfg.n_kv_heads, cfg.Dh)
        _move_params_to_device(head)
        qkv_packed = _randn(M, cfg.qkv_dim, dtype=half_dt)
        ms = _time_op("head_rmsnorm", lambda: head(qkv_packed), args.n_iters, args.n_warmup)
        results.append(("head_rmsnorm", ms, block_calls))

    if want("adagate") and not cfg.fuse_gate_residual:
        gate = nn.AdaGateResidual()
        _move_params_to_device(gate)
        y = _randn(M, d, dtype=half_dt)
        ms = _time_op("ada_gate", lambda: gate(x, y, g), args.n_iters, args.n_warmup)
        results.append(("ada_gate", ms, 2 * block_calls))

    if want("vres") and cfg.value_residual:
        vres = nn.ValueResidualPacked(cfg.v_col_offset, cfg.v_width)
        _move_params_to_device(vres)
        qkv_packed = _randn(M, cfg.qkv_dim, dtype=half_dt)
        qkv_first = _randn(M, cfg.v_width, dtype=half_dt)
        # called only layers 1..L-1 (skipped on first), every NFE
        vres_calls = (L - 1) * NFE
        ms = _time_op(
            "value_residual_packed",
            lambda: vres(qkv_packed, qkv_first),
            args.n_iters,
            args.n_warmup,
        )
        results.append(("value_residual_packed", ms, vres_calls))

    # ── Reporting ──
    print(f"\n{'=' * 72}")
    print(f"  Per-kernel timing at {args.preset}, {args.n_iters} iters/lazy block")
    print(f"  {'─' * 68}")
    print(f"  {'kernel':<28s} {'per-call':>10s} {'calls/frame':>12s} {'total':>10s} {'%':>6s}")

    # Sort by total contribution.
    contribs = [(name, ms, calls, ms * calls) for (name, ms, calls) in results]
    contribs.sort(key=lambda r: -r[3])
    grand_total = sum(c for *_, c in contribs)

    for name, ms, calls, total_ms in contribs:
        pct = 100 * total_ms / grand_total if grand_total else 0
        print(f"  {name:<28s} {ms:8.3f} ms {calls:12d} {total_ms:8.2f} ms {pct:5.1f}%")
    print(f"  {'─' * 68}")
    print(f"  {'predicted forward':<28s} {' ' * 10} {' ' * 12} {grand_total:8.2f} ms")
    print(f"{'=' * 72}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
