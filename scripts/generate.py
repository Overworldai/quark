#!/usr/bin/env python3
"""Generate video with the real waypoint-1.5 weights from HF Hub.

    python scripts/generate.py --seed-image seed.png --n-frames 60 --output demo.mp4
    python scripts/generate.py --seed-image clip.mp4 --output clip_extended.mp4
    python scripts/generate.py --seed-latent seed.npy --output out.npy
    python scripts/generate.py --b-shuffle     # pre-shuffle weights via nn.Linear.prepare()

End-to-end pipeline in one script:
  1. encode seed image/video through TAEHV → latent seed
  2. download model → load into quark nn.Module → capture CUDA graph
  3. replay per frame → collect latents
  4. decode latents through TAEHV → pixel frames
  5. write HEVC/mp4 with ``hvc1`` tag (Discord-playable)

``--output`` extension decides step 5:
  *.mp4 / *.mov / *.mkv  →  decode + ffmpeg libx265 + hvc1 tag
  *.npy                    →  save raw [T, C, H, W] f32 latents (skip decode)

On CUDA: uses QuarkTensor throughout the denoise loop, no torch
dependency in the hot path. torch is pulled in only for the TAEHV
round-trip (encode seed / decode output).

On Metal: uses the existing MLX backend via PT.
"""

from __future__ import annotations

import argparse
import sys
import time

_IS_METAL = sys.platform == "darwin"
_TEMPORAL_COMPRESS = 4  # VAE: 4 pixel frames per latent frame
_PIXEL_FPS = 60  # output video frame rate (matching training data)
_LATENT_FPS = _PIXEL_FPS // _TEMPORAL_COMPRESS  # 15 latent frames/s


# ---------------------------------------------------------------
# Backend-agnostic tensor helpers
# ---------------------------------------------------------------


def _sync():
    if _IS_METAL:
        from quark.backend import PT

        PT.synchronize()
    else:
        from quark.runtime.cuda import CudaRuntime

        CudaRuntime.instance().stream_synchronize(0)


def _randn(*shape, dtype="bf16"):
    if _IS_METAL:
        from quark.backend import PT

        _dt = {"bf16": PT.bfloat16, "f16": PT.float16, "f32": PT.float32}
        return PT.astype(PT.randn(*shape), _dt.get(dtype, PT.bfloat16))
    from quark.runtime.tensor import QuarkTensor

    return QuarkTensor.randn(*shape, dtype=dtype)


def _zeros(*shape, dtype="bf16"):
    if _IS_METAL:
        from quark.backend import PT

        _dt = {"bf16": PT.bfloat16, "f16": PT.float16, "f32": PT.float32, "s32": PT.int32}
        return PT.zeros(*shape, dtype=_dt.get(dtype, PT.bfloat16))
    from quark.runtime.tensor import QuarkTensor

    return QuarkTensor.zeros(*shape, dtype=dtype)


def _tensor(data, dtype="f32"):
    if _IS_METAL:
        from quark.backend import PT

        _dt = {"bf16": PT.bfloat16, "f32": PT.float32, "s32": PT.int32}
        return PT.tensor(data, dtype=_dt.get(dtype, PT.float32))
    from quark.runtime.tensor import QuarkTensor

    return QuarkTensor.from_list(data, dtype=dtype)


def _precompute_noise(n_frames: int, elems_per_frame: int, dtype: str = "bf16") -> list:
    """Pre-generate ``n_frames`` latent-noise tensors on host via a single
    bulk ``np.random.randn`` call and upload as a **list of ``[1, elems]``
    device tensors** — one per frame.

    Returns a list rather than a single ``[n_frames, elems]`` tensor
    because QuarkTensor slicing is not free today (contiguous-offset
    views materialize via copy_strided). Separate tensors let the hot
    path use ``noise_pool[fi]`` directly without any slice materialization.

    ``QuarkTensor.randn`` / ``PT.randn`` use scalar Python RNG loops
    (``random.gauss`` per element) that cost ~1µs/value — for a 360p
    latent that's ~60 ms of pure-Python loop *per frame*. Doing the
    whole bulk on host with numpy keeps the host cost constant.
    """
    import numpy as np

    arr = np.random.randn(n_frames, elems_per_frame).astype(np.float32)

    if _IS_METAL:
        import mlx.core as mx

        if dtype == "bf16":
            u16 = (arr.view(np.uint32) >> 16).astype(np.uint16)
            return [
                mx.array(u16[fi]).view(mx.bfloat16).reshape(1, elems_per_frame)
                for fi in range(n_frames)
            ]
        if dtype == "f16":
            f16 = arr.astype(np.float16)
            return [mx.array(f16[fi]).reshape(1, elems_per_frame) for fi in range(n_frames)]
        return [mx.array(arr[fi]).reshape(1, elems_per_frame) for fi in range(n_frames)]

    from quark.runtime.tensor import QuarkTensor

    return [
        QuarkTensor.from_numpy(arr[fi : fi + 1], dtype=dtype) for fi in range(n_frames)
    ]


def _to_numpy_f32(x):
    """Convert a device tensor to a numpy float32 array."""
    import numpy as np

    if _IS_METAL:
        from quark.backend import PT

        if PT._is_mx(x):
            import mlx.core as mx

            return np.array(x.astype(mx.float32))
    if hasattr(x, "to_numpy"):
        # QuarkTensor path: cast to f32, then to_numpy.
        t = x.astype("f32") if x.dtype != "f32" else x
        return t.to_numpy()
    import torch

    return x.float().cpu().numpy()


def _from_numpy(arr, dtype="bf16"):
    """numpy → device tensor."""
    if _IS_METAL:
        from quark.backend import PT

        import mlx.core as mx
        import numpy as np

        if dtype == "bf16" and arr.dtype == np.float32:
            u16 = (arr.view(np.uint32) >> 16).astype(np.uint16)
            return mx.array(u16).view(mx.bfloat16)
        return mx.array(arr)
    from quark.runtime.tensor import QuarkTensor

    return QuarkTensor.from_numpy(arr, dtype=dtype)


# ---------------------------------------------------------------
# TAEHV AE decoder
# ---------------------------------------------------------------


_TEMPORAL_COMPRESS = 4  # TAEHV compresses 4× along time axis

# 16:9 input resolution for the AE; internally resizes to 2:1 for
# encoding. Matches the presets in ``_make_config``.
_AE_PIXEL_PRESETS: dict[str, tuple[int, int]] = {
    "360p": (360, 640),   # (H, W)
    "720p": (720, 1280),
}


def _load_taehv(ae_uri: str):
    """Import + construct a ``ChunkedStreamingTAEHV``. Returns
    ``(ae, torch, device)`` or ``(None, None, None)`` if world_engine
    isn't importable from either of the known locations."""
    try:
        sys.path.insert(0, "/workspace/world_engine/src")
        from ae import ChunkedStreamingTAEHV
    except ImportError:
        try:
            sys.path.insert(0, "/Users/work/world_engine/src")
            from ae import ChunkedStreamingTAEHV
        except ImportError:
            print("warning: world_engine not found; AE path unavailable.")
            return None, None, None

    import torch

    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = ChunkedStreamingTAEHV.from_pretrained(ae_uri, device=device)
    return ae, torch, device


def load_taehv_decoder(ae_uri: str = "Overworld-Models/taehv1_5"):
    """Load the streaming TAEHV decoder.

    Returns a callable that takes ``[T, C, H, W]`` f32 latent array
    and returns ``[T*4, pixel_H, pixel_W, 3]`` uint8 pixel array.
    The 4× temporal upsampling is handled by the streaming decoder.
    """
    import numpy as np

    ae, torch, device = _load_taehv(ae_uri)
    if ae is None:
        return None

    @torch.inference_mode()
    def decode(latent_np: np.ndarray) -> np.ndarray:
        """Decode [T, C, H, W] latents → [T*4, H_px, W_px, 3] uint8."""
        t = torch.from_numpy(latent_np).to(device=device, dtype=torch.bfloat16)
        decoded = ae.decode(t)  # [T*4, 3, H_px, W_px] float
        return decoded.cpu().numpy()

    return decode


def encode_seed_from_path(
    path: str,
    preset: str,
    ae_uri: str = "Overworld-Models/taehv1_5",
    n_frames: int | None = None,
) -> "object":
    """Read an image or video, resize to the preset's AE input
    resolution, tile 4× for the VAE's temporal compression, and
    encode through streaming TAEHV.

    Returns ``[T, C, H, W]`` f32 numpy — the same layout the legacy
    ``--seed-latent`` path expects. Tiles a single image 4× so that
    one pixel input yields one latent frame.
    """
    import cv2
    import numpy as np

    pixel_h, pixel_w = _AE_PIXEL_PRESETS[preset]

    # Single image → 1-element list. Video → up to ``n_frames``.
    img = cv2.imread(path)
    if img is not None:
        raw = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"cannot open {path}")
        raw = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if n_frames is not None and len(raw) >= n_frames:
                break
        cap.release()
        if not raw:
            raise ValueError(f"no frames read from {path}")

    # Resize to the AE's 16:9 input, then pad to a multiple of 4 pixel
    # frames (TAEHV's temporal compression stride) by repeating the
    # last frame.
    resized = [cv2.resize(f, (pixel_w, pixel_h), interpolation=cv2.INTER_AREA) for f in raw]
    n_raw = len(resized)
    n_padded = _TEMPORAL_COMPRESS * ((n_raw + _TEMPORAL_COMPRESS - 1) // _TEMPORAL_COMPRESS)
    while len(resized) < n_padded:
        resized.append(resized[-1])
    n_latent = n_padded // _TEMPORAL_COMPRESS
    print(f"  {n_raw} input → {n_padded} tiled (4×) → {n_latent} latent frame(s)")

    pixels = np.stack(resized, axis=0)  # [T, H, W, 3] uint8

    ae, torch, _ = _load_taehv(ae_uri)
    if ae is None:
        raise RuntimeError("TAEHV not importable — can't encode seed image/video")

    with torch.inference_mode():
        latent = ae.encode(torch.from_numpy(pixels))

    return latent.float().cpu().numpy()  # [T, C, H, W] f32


def write_video_hevc(
    pixels,
    output_path: str,
    fps: int = _PIXEL_FPS,
    crf: int = 23,
) -> None:
    """Encode ``[T, H, W, 3]`` uint8 RGB frames to an HEVC/mp4 file with
    the ``hvc1`` codec tag.

    Why ``hvc1`` and not ``hev1``: Discord, QuickTime, Safari, and
    most consumer video players will play HEVC only when the sample
    description uses ``hvc1`` (the parameter-sets-in-sample-description
    variant). ``hev1`` (parameter-sets-in-band) is spec-legal but
    flatly refused by many consumer decoders, so embedded video won't
    autoplay on Discord. ``-tag:v hvc1`` tells ffmpeg to rewrite the
    tag; libx265 does the actual encode.

    Uses ``yuv420p`` pixfmt (not ``yuv420p10le`` or ``yuv444p``) so
    the Media Foundation / VideoToolbox / web-browser decoders don't
    reject the track, and ``+faststart`` so the moov atom is at the
    head of the file (non-streaming web players won't start playback
    until they see the moov).
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not on PATH — HEVC encode needs ffmpeg with libx265. "
            "apt: `apt-get install ffmpeg` / brew: `brew install ffmpeg`."
        )

    import numpy as np

    if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
        raise ValueError(
            f"write_video_hevc: expected [T,H,W,3] uint8, got shape {pixels.shape} "
            f"dtype {pixels.dtype}"
        )
    t, h, w, _ = pixels.shape

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "warning",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx265",
        "-preset", "medium",
        "-crf", str(crf),
        "-tag:v", "hvc1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        proc.stdin.write(pixels.tobytes())
        proc.stdin.close()
    finally:
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")
    print(f"saved {output_path} ({t} frames @ {fps} fps, HEVC/hvc1)")


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------


def _make_config(preset: str):
    """Return a Waypoint15Config for the given preset.

    The AE takes 16:9 pixel input and internally resizes to 2:1 for
    encoding. VAE patch size is 16×16, patchify is 2×2.

    360p: input 640×360 → AE resizes to 512×256 → latent 32×16 →
          grid 16×8 = 128 tokens/frame
    720p: input 1280×720 → AE resizes to 1024×512 → latent 64×32 →
          grid 32×16 = 512 tokens/frame
    """
    from quark.models.waypoint_15 import Waypoint15Config

    presets = {
        "360p": (8, 16),   # 128 tokens, pixel input 640×360
        "720p": (16, 32),  # 512 tokens, pixel input 1280×720
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {', '.join(presets)}")

    h, w = presets[preset]

    return Waypoint15Config(
        d_model=2048,
        n_layers=24,
        n_heads=32,
        n_kv_heads=16,
        mlp_ratio=4,
        channels=32,
        patch=(2, 2),
        height=h,
        width=w,
        local_window=16,
        global_window=128,
        global_pinned_dilation=8,
        global_attn_period=4,
        global_attn_offset=-1,
        value_residual=True,
        fourier_dim=512,
        scheduler_sigmas=(1.0, 0.9, 0.75, 0.3, 0.0),
    )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def main():
    import numpy as np

    from quark.models.waypoint_15 import CtrlInput, Waypoint15

    parser = argparse.ArgumentParser(description="Generate video with waypoint-1.5")
    parser.add_argument("--repo", default="Overworld/Waypoint-1.5-1B")
    parser.add_argument("--preset", choices=["360p", "720p"], default="720p")
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument(
        "--output",
        default="out.mp4",
        help="output path — .mp4/.mov/.mkv writes HEVC (hvc1) via ffmpeg, "
             ".npy saves the raw [T,C,H,W] f32 latent array",
    )
    parser.add_argument(
        "--seed-image",
        type=str,
        default=None,
        help="image or video to encode as the seed frame(s). Mutually "
             "exclusive with --seed-latent. Single image is tiled 4× "
             "for the VAE's temporal compression.",
    )
    parser.add_argument(
        "--seed-latent",
        type=str,
        default=None,
        help="pre-encoded seed latent .npy [T,C,H,W] (from an earlier "
             "--seed-image run or the legacy scripts/encode_frame.py)",
    )
    parser.add_argument("--fps", type=int, default=_PIXEL_FPS,
                        help="pixel fps of the output video (default 60, matching training data)")
    parser.add_argument("--crf", type=int, default=23,
                        help="libx265 quality — lower = better (default 23)")
    parser.add_argument("--ae-repo", default="Overworld-Models/taehv1_5",
                        help="HF repo for the TAEHV autoencoder")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--ctrl", type=str, default=None,
                        help="'demo' or path to JSON controller sequence")
    parser.add_argument("--b-shuffle", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--profile", action="store_true",
                        help="Profile one frame's denoise+commit and print module timings.")
    args = parser.parse_args()

    if args.seed_image and args.seed_latent:
        parser.error("--seed-image and --seed-latent are mutually exclusive")

    cfg = _make_config(args.preset)
    import dataclasses as _dc

    from quark.models.waypoint_15 import QuantConfig

    # ``--fp8`` (default) drives end-to-end fp8: Linear weight
    # quantization AND the KV cache dtype / OwlAttn MMA path. ``--bf16``
    # picks the safe fallback (no fp8 anywhere).
    if args.bf16:
        cfg = _dc.replace(cfg, quant=QuantConfig.all_bf16())
    elif not args.fp8:
        cfg = _dc.replace(cfg, quant=QuantConfig.all_bf16())
    # ctrl_conditioning is always True — even with default CtrlInput(),
    # the MLPFusion on every 3rd block transforms x through its own weights.

    ph, pw = cfg.patch
    H, W = cfg.height * ph, cfg.width * pw
    C = cfg.channels
    tpf = cfg.tpf
    half_dt = "bf16"

    print(f"preset: {args.preset} — {tpf} tokens/frame")
    print(f"  latent grid: {cfg.height}x{cfg.width}, {args.n_frames} frames")
    print(f"  model: {cfg.n_layers} layers, d_model={cfg.d_model}")

    # ── Load model ──
    print(f"loading model from {args.repo} …")
    t0 = time.perf_counter()
    repo_suffix = "-360P" if args.preset == "360p" else ""
    model = Waypoint15.from_pretrained(args.repo + repo_suffix, cfg=cfg, dtype="bf16")
    model.prepare(shuffle=args.b_shuffle)  # fp8 defaults to cfg.quant.linear
    _sync()
    print(f"  model ready ({time.perf_counter() - t0:.1f}s)")

    # ── Autotune ──
    print("autotuning …")
    import quark

    latent = _randn(1, C * H * W, dtype=half_dt)
    sigmas = cfg.scheduler_sigmas
    n_denoise = len(sigmas) - 1
    frame_t = _tensor([0], dtype="s32")

    ctrl_dev, ctrl_fill = model.make_ctrl_buffer()
    if ctrl_fill is not None:
        ctrl_fill(CtrlInput())  # zero-filled default for autotune + seed
    autotune_ctrl_emb = model.encode_ctrl(ctrl_dev)
    with quark.max_autotune():
        model(latent, sigma_idx=0, frame_t=frame_t, ctrl_emb=autotune_ctrl_emb)
        model(latent, sigma_idx=0, frame_t=frame_t, ctrl_emb=autotune_ctrl_emb)
        _sync()

    # ── Seed frames ──
    # Two ways to provide a seed:
    #   --seed-image PATH: encode an image/video inline through TAEHV,
    #                      feed the latents into the KV cache.
    #   --seed-latent NPY: load a pre-encoded latent array (produced by
    #                      an earlier run, or by the legacy
    #                      scripts/encode_frame.py helper).
    seed_frame_count = 0
    seed_np = None
    if args.seed_image:
        print(f"encoding seed from {args.seed_image} …")
        t_enc = time.perf_counter()
        seed_np = encode_seed_from_path(args.seed_image, args.preset, args.ae_repo)
        print(f"  encoded {seed_np.shape[0]} latent frame(s) in {time.perf_counter() - t_enc:.1f}s")
    elif args.seed_latent:
        print(f"seeding from {args.seed_latent} …")
        seed_np = np.load(args.seed_latent)
        if seed_np.ndim == 3:
            seed_np = seed_np[np.newaxis]

    if seed_np is not None:
        seed_frame_count = seed_np.shape[0]
        # Seed frames reuse the ctrl input buffer (already zero-filled).
        seed_ctrl_emb = model.encode_ctrl(ctrl_dev)
        for fi in range(seed_frame_count):
            latent = _from_numpy(
                seed_np[fi : fi + 1].reshape(1, C * H * W).astype(np.float32), dtype=half_dt
            )
            frame_t.set_value(fi)
            model(latent, sigma_idx=n_denoise, frame_t=frame_t, ctrl_emb=seed_ctrl_emb)
            _sync()
        print(f"  seeded {seed_frame_count} frames")

    # No warmup — it corrupts the KV cache.

    # ── Controller input sequence ──
    ctrl_sequence = None
    if args.ctrl == "demo":
        ctrl_sequence = [
            CtrlInput(mouse=(0.2, 0.2)), CtrlInput(button={32}), CtrlInput(), CtrlInput(), CtrlInput(),
            CtrlInput(button={1}), CtrlInput(), CtrlInput(), CtrlInput(button={1, 32}),
            CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(),
        ] * 4
        ctrl_sequence += [CtrlInput()] * 8
        ctrl_sequence += (
            [CtrlInput(button={32})] * 10
            + [CtrlInput(button={65})] * 10
            + [CtrlInput(button={68})] * 10
            + [CtrlInput(button={83})] * 10
        )
        ctrl_sequence += [CtrlInput()] * 10
    elif args.ctrl:
        import json

        with open(args.ctrl) as f:
            raw = json.load(f)
        ctrl_sequence = [
            CtrlInput(
                button=set(c.get("button", [])),
                mouse=tuple(c.get("mouse", [0.0, 0.0])),
                scroll_wheel=c.get("scroll_wheel", 0),
            )
            for c in raw
        ]

    # ── Profile pass ──
    if args.profile:
        import quark.nn as _nn
        from quark.nn import Module as _M

        print("\nprofiling one frame …")

        dsig_tensors = [
            _tensor([float(sigmas[si + 1] - sigmas[si])], dtype="f32")
            for si in range(n_denoise)
        ]
        # Cached EulerStep modules — one per denoise step. Each owns
        # its output buffer so the profile path doesn't auto-alloc a
        # fresh 2 MB latent buffer per step (~150 µs each), and the
        # steps show up as real leaves in the profile.
        euler_steps = _nn.ModuleList([_nn.EulerStep() for _ in range(n_denoise)])
        if ctrl_fill is not None:
            ctrl_fill(ctrl_sequence[0] if ctrl_sequence else CtrlInput())
        ctrl_emb_prof = model.encode_ctrl(ctrl_dev)
        frame_t.set_value(seed_frame_count)
        x_prof = _randn(1, C * H * W, dtype=half_dt)
        # Warmup (don't profile the first call — it's dominated by JIT).
        for si in range(n_denoise):
            v = model(x_prof, sigma_idx=si, frame_t=frame_t, ctrl_emb=ctrl_emb_prof, frozen=True)
            x_prof = euler_steps[si](x_prof, v, dsig_tensors[si])
        model(x_prof, sigma_idx=n_denoise, frame_t=frame_t, ctrl_emb=ctrl_emb_prof)
        frame_t.increment()
        _sync()

        # Profiled frame.
        frame_t.set_value(seed_frame_count + 1)
        x_prof = _randn(1, C * H * W, dtype=half_dt)
        with _M.profile(model):
            for si in range(n_denoise):
                v = model(
                    x_prof, sigma_idx=si, frame_t=frame_t, ctrl_emb=ctrl_emb_prof, frozen=True
                )
                x_prof = euler_steps[si](x_prof, v, dsig_tensors[si])
            model(x_prof, sigma_idx=n_denoise, frame_t=frame_t, ctrl_emb=ctrl_emb_prof)
        _sync()
        _M.print_profile(leaves_only=True)

    # ── Generate frames ──
    print(f"\ngenerating {args.n_frames} frames ({n_denoise} denoise steps each) …")

    latent_frames: list = []
    t_gen_start = time.perf_counter()
    frame_t.set_value(seed_frame_count)

    # Precomputed latent noise pool — one bulk np.random.randn call at
    # startup. Hot-path was paying ~60 ms/frame in Python ``random.gauss``
    # loops; the pool makes it a free device slice / D→D copy.
    noise_pool = _precompute_noise(args.n_frames, C * H * W, dtype=half_dt)

    if _IS_METAL or args.no_graph:
        # Unrolled host loop — one python dispatch per step.
        import quark.nn as _nn

        dsig_tensors = [
            _tensor([float(sigmas[si + 1] - sigmas[si])], dtype="f32")
            for si in range(n_denoise)
        ]
        # One cached EulerStep per denoise step. Using ``pcf.euler_step``
        # directly would auto-alloc a fresh 2 MB output buffer per call
        # (~150 µs ``cuMemAllocAsync`` on Blackwell); the module owns a
        # shape-keyed cache so the hot loop only allocates once.
        euler_steps = _nn.ModuleList([_nn.EulerStep() for _ in range(n_denoise)])
        for fi in range(args.n_frames):
            if ctrl_fill is not None:
                ctrl_fill(ctrl_sequence[fi % len(ctrl_sequence)] if ctrl_sequence else CtrlInput())
            ctrl_emb = model.encode_ctrl(ctrl_dev)

            x = noise_pool[fi]
            for si in range(n_denoise):
                v = model(x, sigma_idx=si, frame_t=frame_t, ctrl_emb=ctrl_emb, frozen=True)
                x = euler_steps[si](x, v, dsig_tensors[si])
            # ``x`` points at ``euler_steps[-1]``'s cached buffer, which
            # the next frame will overwrite. Clone once so the saved
            # frame is independent. (Second clone on append was
            # redundant — the commit pass only reads this buffer.)
            latent = x.clone()
            model(latent, sigma_idx=n_denoise, frame_t=frame_t, ctrl_emb=ctrl_emb)
            frame_t.increment()
            latent_frames.append(latent)
    else:
        # CUDA-graph path: one ``GenerateFrame`` module captures the
        # full pipeline (encode_ctrl → denoise → commit → increment).
        # We capture via ``Module.graph(...)`` so each call to
        # ``gen_frame(...)`` auto-copies the input tensors into the
        # stable capture buffers and returns a cloned output. No
        # manual copy_from / replay bookkeeping on the host loop.
        from quark.models.waypoint_15 import GenerateFrame

        gen_frame = GenerateFrame(model)

        if ctrl_fill is not None:
            ctrl_fill(ctrl_sequence[0] if ctrl_sequence else CtrlInput())

        print("  capturing graph …")
        t_cap = time.perf_counter()
        gen_frame.prepare_graph(noise_pool[0], ctrl_dev, start_frame_t=seed_frame_count)
        print(f"  captured in {time.perf_counter() - t_cap:.2f}s")

        for fi in range(args.n_frames):
            if ctrl_sequence and ctrl_fill is not None:
                ctrl_fill(ctrl_sequence[fi % len(ctrl_sequence)])
            # gen_frame.__call__ → _graph_replay: D2D-copies inputs into
            # the stable capture buffers, replays, returns a clone of
            # the output — so appending the clone is safe across frames.
            latent = gen_frame(noise_pool[fi], ctrl_dev)
            latent_frames.append(latent)

    _sync()
    gen_elapsed = time.perf_counter() - t_gen_start

    # ── Summary ──
    nfe = n_denoise + 1
    n = args.n_frames
    lfps = n / gen_elapsed
    print(f"\n{'='*50}")
    print(f"  {n} latent frames, {tpf} tokens/frame")
    print(f"  {nfe} NFE/frame ({n_denoise} denoise + 1 commit)")
    print(f"  NFE:  {gen_elapsed / n / nfe * 1000:.1f} ms ({lfps * nfe:.1f}/s)")
    print(f"  LFPS: {lfps:.1f} latent frames/s")
    print(f"  FPS:  {lfps * _TEMPORAL_COMPRESS:.1f} pixel frames/s (4x temporal)")
    print(f"  total: {gen_elapsed:.2f}s")
    print(f"{'='*50}")

    # ── Output ──
    latent_nps = [_to_numpy_f32(lt).reshape(1, C, H, W) for lt in latent_frames]
    all_latents = np.concatenate(latent_nps, axis=0)  # [T, C, H, W]

    out_lower = args.output.lower()
    is_video = out_lower.endswith((".mp4", ".mov", ".mkv"))
    is_npy = out_lower.endswith(".npy")
    if not (is_video or is_npy):
        print(
            f"warning: unrecognized --output extension {args.output!r}; "
            f"treating as video (HEVC/mp4)."
        )
        is_video = True

    if is_npy:
        np.save(args.output, all_latents)
        print(f"saved {args.output} ({all_latents.shape})")
    else:
        print(f"\ndecoding {all_latents.shape[0]} latent frames → pixels …")
        t_dec = time.perf_counter()
        decode = load_taehv_decoder(args.ae_repo)
        if decode is None:
            # AE not available — fall back to saving latents so the run
            # isn't wasted. Caller can decode offline once world_engine
            # is on PATH.
            fallback = args.output.rsplit(".", 1)[0] + ".npy"
            np.save(fallback, all_latents)
            print(f"  AE unavailable; saved latents to {fallback} instead")
        else:
            # Stream decode one latent frame at a time — TAEHV maintains
            # temporal state across calls, and chunked decode keeps peak
            # VRAM flat for long outputs.
            pixel_chunks = []
            for i in range(all_latents.shape[0]):
                pixel_chunks.append(decode(all_latents[i : i + 1]))
                if i == 0 or (i + 1) % 10 == 0:
                    print(f"  decoded {i + 1}/{all_latents.shape[0]}")
            pixels = np.concatenate(pixel_chunks, axis=0)  # [T*4, H, W, 3] uint8
            dec_elapsed = time.perf_counter() - t_dec
            print(
                f"  {pixels.shape[0]} pixel frames in {dec_elapsed:.1f}s "
                f"({pixels.shape[0] / dec_elapsed:.1f} fps)"
            )
            write_video_hevc(pixels, args.output, fps=args.fps, crf=args.crf)


if __name__ == "__main__":
    main()
