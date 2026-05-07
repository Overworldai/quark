#!/usr/bin/env python3
"""Diagnostic: do kv_cache_update writes actually persist on Metal?

Calls kv_cache_update twice with non-zero inputs and checks whether the
second call sees the residue of the first (which it must if the in-place
ring writes survive eval).

If quark passes: K_cache after the second commit will differ from
K_cache after the first (different ring slots written).

If quark fails: K_cache may stay zero-filled or only show one of the
two writes — meaning the launcher is allocating a fresh output buffer
each time and Python's self.K_cache reference has gone stale.
"""
from __future__ import annotations

import numpy as np

import quark
import quark.functional as pcf
from quark.nn.layers import _set_s32
from quark.nn.module import _tensor, _zeros


def main():
    # Tiny problem: B=1 head=1 H=W=4 Dh=16 num_buckets=2
    B, Hk, Hs, Ws, Dh = 1, 1, 4, 4, 16
    num_buckets = 2
    pinned_dilation = 1
    tpf = Hs * Ws
    cap = num_buckets * tpf + tpf

    # Inputs (qkv treated as packed K and V).
    rng = np.random.default_rng(0xC0FFEE)
    qkv_np = rng.standard_normal((B * Hk * tpf, Dh)).astype(np.float32)
    qkv = _tensor(qkv_np.astype(np.float16).view(np.uint16), dtype="bf16")

    K_cache = _zeros(B * Hk * cap, Dh, dtype="bf16")
    Vt_cache = _zeros(B * Hk * Dh, cap, dtype="bf16")
    segments = _tensor([0, 0, 0, tpf, 0, 0], dtype="s32")
    n_segments = _tensor([2], dtype="s32")
    frame_t = _zeros(1, dtype="s32")
    frozen = _zeros(1, dtype="s32")

    print("Initial K_cache snapshot (should be all zeros):")
    quark.eval()
    K_before = K_cache.to_numpy().view(np.uint16).astype(np.uint32) << 16
    K_before = K_before.view(np.float32)
    print(f"  std={K_before.std():.4f} max|x|={np.abs(K_before).max():.4f}")

    # First commit at frame_t=0.
    _set_s32(frame_t, 0)
    _set_s32(frozen, 0)
    pcf.kv_cache_update(
        qkv, qkv, frame_t, frozen,
        Vt_cache, segments, n_segments, K_cache,
        B=B, n_kv_heads=Hk,
        H_spatial=Hs, W_spatial=Ws,
        num_buckets=num_buckets,
        pinned_dilation=pinned_dilation,
        packed_qkv=True, n_q_heads=Hk,
    )
    quark.eval()
    K_after_1 = K_cache.to_numpy().view(np.uint16).astype(np.uint32) << 16
    K_after_1 = K_after_1.view(np.float32)
    print("\nAfter 1st commit (frame_t=0):")
    print(f"  std={K_after_1.std():.4f} max|x|={np.abs(K_after_1).max():.4f}")
    print(f"  delta from initial: max={np.abs(K_after_1 - K_before).max():.4f}")
    if np.abs(K_after_1 - K_before).max() < 1e-6:
        print("  **WRITES DID NOT PERSIST — K_cache unchanged**")
        return 1

    # Second commit at frame_t=1 with different qkv to write a different ring slot.
    qkv2_np = rng.standard_normal((B * Hk * tpf, Dh)).astype(np.float32)
    qkv2 = _tensor(qkv2_np.astype(np.float16).view(np.uint16), dtype="bf16")
    _set_s32(frame_t, 1)
    pcf.kv_cache_update(
        qkv2, qkv2, frame_t, frozen,
        Vt_cache, segments, n_segments, K_cache,
        B=B, n_kv_heads=Hk,
        H_spatial=Hs, W_spatial=Ws,
        num_buckets=num_buckets,
        pinned_dilation=pinned_dilation,
        packed_qkv=True, n_q_heads=Hk,
    )
    quark.eval()
    K_after_2 = K_cache.to_numpy().view(np.uint16).astype(np.uint32) << 16
    K_after_2 = K_after_2.view(np.float32)
    print("\nAfter 2nd commit (frame_t=1):")
    print(f"  std={K_after_2.std():.4f} max|x|={np.abs(K_after_2).max():.4f}")
    print(f"  delta from 1st commit: max={np.abs(K_after_2 - K_after_1).max():.4f}")
    if np.abs(K_after_2 - K_after_1).max() < 1e-6:
        print("  **2nd WRITE DID NOT PERSIST — K_cache same as after 1st**")
        return 1
    print("\nWrites persist correctly. ✓")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
