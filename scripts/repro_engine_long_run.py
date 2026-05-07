#!/usr/bin/env python3
"""Reproduce the long-running crash in ``quark.Engine`` on Apple Silicon.

Biome's hot loop dies silently after 600-1500 frames with a leaked-semaphore
warning at process exit. This script is the smallest standalone harness that
mirrors what Biome does — load Engine, append a seed frame, then call
``gen_frame`` in a tight loop with realistic varied inputs — while logging
the dispatcher's hidden bookkeeping every N frames so we can see what is
actually growing.

Toggle inputs/outputs to triangulate:
    --no-decode       skip TAEHV decode (DiT-only path)
    --const-ctrl      pin ctrl to a single CtrlInput across all frames
    --const-noise     pin the noise tensor (don't refresh per frame)
    --reset-every N   call engine.reset() periodically
    --no-append       skip the seed append_frame at startup
    --stats-every N   dump dispatcher counters every N frames (default 50)

If the user's hypothesis is right ("maybe it's about actual inputs"), running
with --const-ctrl + --const-noise should NOT crash, while a varied run will.

Examples:
    # Mirror Biome as closely as possible.
    repro_engine_long_run.py --n-frames 3000

    # DiT-only — rules out CoreML/ANE.
    repro_engine_long_run.py --no-decode --n-frames 3000

    # Constant inputs — tests the "varied input triggers crash" theory.
    repro_engine_long_run.py --const-ctrl --const-noise --n-frames 3000

    # Periodic reset workaround.
    repro_engine_long_run.py --reset-every 500 --n-frames 3000
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import resource
import sys
import time
import traceback


def _rss_mib() -> float:
    """Process RSS in MiB. macOS reports ru_maxrss in bytes; Linux in KB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1 << 20) if sys.platform == "darwin" else rss / 1024


def _stats_line(prefix: str) -> str:
    try:
        from quark.drivers import _metal_dispatch as md
    except Exception:
        return f"{prefix} stats=N/A"
    s = md.stats()
    return (
        f"{prefix} rss={_rss_mib():6.1f} MiB | "
        f"lazy_bufs={s['lazy_buffers']:5d} live={s['lazy_buffers_live']:4d} "
        f"freelist={s['lazy_buffers_free_list']:4d} | "
        f"queue={s['lazy_queue']:3d} fence={s['buf_to_fence']:4d} | "
        f"in_cache={s['input_cache']:3d} alias={s['output_alias_map']:3d} | "
        f"in_pool={s['input_pool_total']:4d}/{s['input_pool_buckets']:3d}b "
        f"out_pool={s['output_pool_total']:4d}/{s['output_pool_buckets']:3d}b | "
        f"rc_sum={s['refcount_sum']:5d} rc_max={s['refcount_max']:3d}"
    )


def _build_ctrl_sequence(n: int, CtrlInput):
    """Same demo sequence as ``bench_quark_world_engine.py``."""
    seq = [
        CtrlInput(mouse=(0.2, 0.2)),
        CtrlInput(button={32}),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(button={1}),
        CtrlInput(button={1, 32}),
        CtrlInput(),
        CtrlInput(button={32}),
        CtrlInput(button={65}),
        CtrlInput(button={68}),
        CtrlInput(button={83}),
    ] * 8
    while len(seq) < n:
        seq += [
            CtrlInput(
                button=set(random.sample(range(1, 65), random.randint(0, 4))),
                mouse=(random.random() * 0.4 - 0.2, random.random() * 0.4 - 0.2),
                scroll_wheel=random.choice((-1, 0, 1)),
            )
            for _ in range(16)
        ]
    return seq[:n]


def _make_seed_uint8(h: int, w: int, temporal: int = 4):
    import numpy as np

    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(temporal, h, w, 3), dtype=np.uint8)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Overworld/Waypoint-1.5-1B-360P")
    p.add_argument("--n-frames", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--stats-every", type=int, default=50)
    p.add_argument("--no-decode", action="store_true",
                   help="call gen_frame(return_img=False) — skip taehv decode")
    p.add_argument("--const-ctrl", action="store_true",
                   help="reuse a single CtrlInput across all frames")
    p.add_argument("--const-noise", action="store_true",
                   help="patch Engine to skip per-frame noise refresh")
    p.add_argument("--no-append", action="store_true",
                   help="skip append_frame(seed) — start from cold KV")
    p.add_argument("--reset-every", type=int, default=0,
                   help="call engine.reset() every N frames (0 = never)")
    p.add_argument("--gc-every", type=int, default=0,
                   help="call gc.collect() every N frames (0 = never)")
    p.add_argument("--no-prep-decode", action="store_true",
                   help="don't warm taehv decode at start (still skipped if --no-decode)")
    args = p.parse_args()

    # Make sure the user can see warnings + tracebacks if the process is
    # killed mid-run.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    print(f"[repro] platform={sys.platform} python={sys.version.split()[0]}")
    print(f"[repro] args={vars(args)}")
    print(_stats_line("[repro] pre-import"))

    t0 = time.perf_counter()
    from quark import CtrlInput, Engine
    print(f"[repro] imported quark in {time.perf_counter() - t0:.1f}s")
    print(_stats_line("[repro] post-import"))

    t0 = time.perf_counter()
    engine = Engine(args.model, quant="bf16")
    print(f"[repro] Engine() ready in {time.perf_counter() - t0:.1f}s")
    print(_stats_line("[repro] post-Engine()"))

    cfg = engine.model_cfg
    pixel_h, pixel_w = engine._pixel_shape[2], engine._pixel_shape[3]
    print(f"[repro] pixel HxW = {pixel_h}x{pixel_w}, "
          f"temporal_compression = {getattr(cfg, 'temporal_compression', '?')}")

    # ── seed (mirror Biome's first-frame init) ──
    if not args.no_append:
        seed = _make_seed_uint8(pixel_h, pixel_w, temporal=4)
        t0 = time.perf_counter()
        engine.append_frame(seed)
        print(f"[repro] append_frame() in {time.perf_counter() - t0:.2f}s")
        print(_stats_line("[repro] post-append"))

    # ── ctrl sequence ──
    n_total = args.warmup + args.n_frames
    if args.const_ctrl:
        ctrls = [CtrlInput() for _ in range(n_total)]
    else:
        ctrls = _build_ctrl_sequence(n_total, CtrlInput)

    # ── optionally pin noise ──
    # The Darwin path's per-frame noise generation is here:
    #   rng = np.random.default_rng()
    #   noise_f32 = rng.standard_normal(...).astype(...)
    #   self._noise_staged[:] = ...
    #   ctypes.memmove(self._noise_qt.data_ptr(), staged, ...)
    # If --const-noise, we monkey-patch that to be a no-op after the first
    # call so the noise buffer never gets touched again.
    if args.const_noise:
        import quark.engine as engine_mod  # noqa: F401
        orig = engine._darwin_gen_frame.__func__

        first = {"done": False}

        def patched_gen_frame(self, ctrl, return_img):
            if not first["done"]:
                # Run the original once so the staged buffer + noise_qt get
                # populated; subsequent calls below skip the refresh.
                first["done"] = True
                return orig(self, ctrl, return_img)

            # Same body as orig but without noise refresh and without ft
            # mutation either (ft is held constant too — pure const inputs).
            import ctypes as _ct  # noqa: F401
            import numpy as _np  # noqa: F401
            import quark
            ctrl_qt = self._encode_ctrl(ctrl)
            ctrl_emb = self.model.encode_ctrl(ctrl_qt) if ctrl_qt is not None else None
            x = self._noise_qt
            ft = self._frame_t_qt
            with quark.lazy():
                cur = x
                for si in range(self._n_denoise):
                    v = self.model(cur, sigma_idx=si, frame_t=ft,
                                   ctrl_emb=ctrl_emb, frozen=True)
                    cur = self._euler_steps[si](cur, v, self._dsig_tensors[si])
                self.model(cur, sigma_idx=self._n_denoise, frame_t=ft, ctrl_emb=ctrl_emb)
            self._frame_counter += 1
            if not return_img:
                return None
            latent_f32 = self._darwin_qt_to_latent_np_f32(cur)
            pixels = self._taehv.decode(latent_f32.astype(_np.float16))
            import torch
            return torch.from_numpy(pixels)

        import types as _types
        engine._darwin_gen_frame = _types.MethodType(patched_gen_frame, engine)
        print("[repro] patched _darwin_gen_frame: noise + frame_t are CONSTANT")

    # ── warmup ──
    print(f"[repro] warmup ({args.warmup} frames) …")
    for i in range(args.warmup):
        engine.gen_frame(ctrl=ctrls[i], return_img=not args.no_decode)
    print(_stats_line(f"[repro] post-warmup f={args.warmup}"))

    # ── timed loop ──
    print(f"[repro] running {args.n_frames} frames "
          f"(decode={'off' if args.no_decode else 'on'}, "
          f"reset_every={args.reset_every or 'never'}, "
          f"const_ctrl={args.const_ctrl}, const_noise={args.const_noise})")

    last_t = time.perf_counter()
    bucket_ms = []
    bucket_bytes_alloc = 0  # placeholder if we want to add mem tracking
    t_loop_start = last_t

    try:
        for i in range(args.n_frames):
            f0 = time.perf_counter()
            engine.gen_frame(ctrl=ctrls[args.warmup + i], return_img=not args.no_decode)
            dt = (time.perf_counter() - f0) * 1000
            bucket_ms.append(dt)

            if (i + 1) % args.stats_every == 0:
                import numpy as np
                arr = np.array(bucket_ms[-args.stats_every:])
                med = float(np.median(arr))
                p99 = float(np.percentile(arr, 99))
                wall = time.perf_counter() - last_t
                last_t = time.perf_counter()
                print(
                    f"[repro] f={i + 1:5d} median={med:5.1f}ms p99={p99:6.1f}ms "
                    f"wall={wall:5.1f}s | "
                    + _stats_line("").lstrip()
                )

            if args.reset_every and (i + 1) % args.reset_every == 0:
                t_r0 = time.perf_counter()
                engine.reset()
                if not args.no_append:
                    seed = _make_seed_uint8(pixel_h, pixel_w, temporal=4)
                    engine.append_frame(seed)
                print(f"[repro] f={i + 1:5d} engine.reset() + append "
                      f"{(time.perf_counter() - t_r0) * 1000:.1f}ms | "
                      + _stats_line("").lstrip())

            if args.gc_every and (i + 1) % args.gc_every == 0:
                gc.collect()
    except Exception as e:
        print(f"[repro] EXCEPTION at frame {i}: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(_stats_line(f"[repro] post-exception f={i}"))
        return 2

    elapsed = time.perf_counter() - t_loop_start
    print(f"[repro] DONE: {args.n_frames} frames in {elapsed:.1f}s "
          f"({elapsed * 1000 / args.n_frames:.2f} ms/frame)")
    print(_stats_line(f"[repro] final f={args.n_frames}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
