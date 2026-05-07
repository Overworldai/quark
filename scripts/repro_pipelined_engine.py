#!/usr/bin/env python3
"""Two-thread submit/drain stress + perf test for ``quark.Engine``.

Thread A submits frames as fast as the engine accepts them
(``Engine.submit_frame``). Thread B drains decoded pixels
(``Engine.next_pixels``). Backpressure via a depth-1 semaphore.

At 360p with Waypoint-1.5-1B on M5 Max this pattern is *slightly
slower* than synchronous ``gen_frame`` (drainer's host-side memcpy
+ CoreML wrap contends with submitter's dispatch) — sync wins by
~2 ms / frame because its ``pipe.next()`` block gives the worker a
clean idle window. The two-thread API is kept available for future
workloads where the per-frame decode work is large enough to
exceed sync's natural idle window (e.g. 720p, multi-stage VAE);
this script is the harness for measuring that crossover.

    repro_pipelined_engine.py --n-frames 1500
    repro_pipelined_engine.py --n-frames 3000 --stats-every 200
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time

import numpy as np


def _seed_uint8(h: int, w: int, t: int = 4) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)


def _ctrl_seq(n: int, CtrlInput):
    seq = [
        CtrlInput(mouse=(0.2, 0.2)),
        CtrlInput(button={32}),
        CtrlInput(),
        CtrlInput(button={1}),
        CtrlInput(button={1, 32}),
    ] * 8
    while len(seq) < n:
        seq += [
            CtrlInput(
                button=set(random.sample(range(1, 65), random.randint(0, 4))),
                mouse=(random.random() * 0.4 - 0.2, random.random() * 0.4 - 0.2),
            )
            for _ in range(16)
        ]
    return seq[:n]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Overworld/Waypoint-1.5-1B-360P")
    p.add_argument("--n-frames", type=int, default=1500)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--stats-every", type=int, default=100)
    args = p.parse_args()

    print(f"[pipe] python={sys.version.split()[0]}")

    from quark import CtrlInput, Engine
    from quark.drivers import _metal_dispatch as md

    t0 = time.perf_counter()
    engine = Engine(args.model, quant="bf16")
    print(f"[pipe] Engine() in {time.perf_counter() - t0:.1f}s")

    cfg = engine.model_cfg
    seed_h, seed_w = getattr(cfg, "seed_target_size", None) or (
        engine._pixel_shape[2] * 8,  # fallback if model_cfg lacks the field
        engine._pixel_shape[3] * 8,
    )
    # The taehv expects (4, H, W, 3) seed in pixel space.
    if isinstance(seed_h, list | tuple):  # YAML can encode as [h, w]
        seed_h, seed_w = seed_h
    seed = _seed_uint8(seed_h, seed_w)
    engine.append_frame(seed)
    print(f"[pipe] append seed ok (pixel {seed_h}x{seed_w})")

    ctrls = _ctrl_seq(args.warmup + args.n_frames, CtrlInput)

    # Warmup synchronously via gen_frame so taehv + autotune are warm.
    print(f"[pipe] warmup ({args.warmup} frames)…")
    for i in range(args.warmup):
        engine.gen_frame(ctrl=ctrls[i])

    # Two-thread pipeline. Backpressure via a depth-1 semaphore: the
    # submitter must acquire before each submit_frame, the drainer
    # releases after each next_pixels. This is the same coordination
    # Biome will use — Engine's pipeline depth is 1, so the producer
    # is naturally throttled by the consumer's drain rate.
    submit_ms: list[float] = []
    drain_ms: list[float] = []
    submit_done = threading.Event()
    stats_lock = threading.Lock()
    slot = threading.Semaphore(1)  # 1 in-flight allowed

    def submitter():
        for i in range(args.n_frames):
            slot.acquire()
            t = time.perf_counter()
            engine.submit_frame(ctrl=ctrls[args.warmup + i])
            with stats_lock:
                submit_ms.append((time.perf_counter() - t) * 1000)
        submit_done.set()

    def drainer():
        i = 0
        while True:
            t = time.perf_counter()
            pixels = engine.next_pixels()
            dt = (time.perf_counter() - t) * 1000
            if pixels is None:
                if submit_done.is_set():
                    return
                time.sleep(0.001)
                continue
            slot.release()
            with stats_lock:
                drain_ms.append(dt)
            # Mimic Biome's per-frame work: tensor → numpy on each subframe.
            cpu_np = pixels.numpy()
            assert cpu_np.shape[0] == 4 and cpu_np.shape[-1] == 3, cpu_np.shape
            i += 1
            if i % args.stats_every == 0:
                with stats_lock:
                    s = np.median(submit_ms[-args.stats_every:])
                    d = np.median(drain_ms[-args.stats_every:])
                stats = md.stats()
                print(
                    f"[pipe] f={i:5d} | "
                    f"submit_med={s:5.1f}ms drain_med={d:5.1f}ms | "
                    f"lazy_bufs={stats['lazy_buffers']:4d} "
                    f"fence={stats['buf_to_fence']:4d} "
                    f"pend_clean={stats['pending_fence_cleanups']:3d}"
                )

    t_loop = time.perf_counter()
    th_submit = threading.Thread(target=submitter, name="submitter")
    th_drain = threading.Thread(target=drainer, name="drainer")
    th_drain.start()
    th_submit.start()
    th_submit.join()
    th_drain.join()
    elapsed = time.perf_counter() - t_loop

    print(
        f"[pipe] DONE: {args.n_frames} frames in {elapsed:.1f}s "
        f"({elapsed * 1000 / args.n_frames:.2f} ms/frame, "
        f"{args.n_frames / elapsed:.2f} LFPS)"
    )
    print(
        f"[pipe] submit median={np.median(submit_ms):5.1f}ms "
        f"drain median={np.median(drain_ms):5.1f}ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
