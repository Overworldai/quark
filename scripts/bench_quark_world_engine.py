#!/usr/bin/env python3
"""Bench quark's Waypoint-1.5 forward end-to-end on Apple Silicon.

Loads HF weights, remaps them into ``quark.nn.Waypoint15``, runs the
denoise loop, and (optionally) round-trips each latent through the
ANE TAEHV decoder (``quark.taehv``) for a real video output.

    python scripts/bench_quark_world_engine.py
    python scripts/bench_quark_world_engine.py --preset 360p --n-frames 32
    python scripts/bench_quark_world_engine.py --seed-image seed.png \\
        --output quark_demo.mp4 --n-frames 32

Output:
  *.mp4 / *.mov / *.mkv  →  HEVC (hvc1) via ffmpeg libx265
  *.npy                    →  raw [T*4, H, W, 3] uint8 frames (no encode)
  --no-decode              →  skip the AE entirely (pure DiT timing)
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import time
from pathlib import Path

import numpy as np

_TEMPORAL_COMPRESS = 4  # TAEHV: 4 pixel frames / latent frame
_PIXEL_FPS = 60
_AE_PIXEL_PRESETS: dict[str, tuple[int, int]] = {
    "360p": (360, 640),
    "720p": (720, 1280),
}

# Numpy carrier dtype for each quark dtype on Metal — matches
# ``quark.nn.module._NP_DT_MAP``. bf16 lives as ``uint16`` (raw bits).
_NP_DT_MAP = {
    "bf16": np.uint16,
    "f16": np.float16,
    "f32": np.float32,
    "s32": np.int32,
}


class _Tagged(np.ndarray):
    """ndarray subclass that carries a ``quark_dtype`` tag.

    Plain ``np.uint16`` resolves to ``DType.U16`` in ``DType.from_backend``,
    which fails the GEMM dtype check. Tagging the array as ``"bf16"``
    lets ``from_backend`` short-circuit to ``BF16`` so kernel dispatch
    works. ``__array_finalize__`` propagates the tag through views
    (reshape, slice) — concatenate / stack drop the subclass and need
    re-tagging via ``_tag``.
    """

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.quark_dtype = getattr(obj, "quark_dtype", None)


def _tag(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Tag ``arr`` (numpy carrier) with the given quark dtype."""
    out = arr.view(_Tagged)
    out.quark_dtype = dtype
    return out


# Path to the reference seed image — ``frozen_valley_sniper.jpg``,
# the same JPEG that ``world_engine.mlx_metal.benchmarks.bench_engine``
# uses. ``scripts/assets/`` is gitignored — drop the JPEG there
# manually (or symlink to the world_engine copy) for stable runs;
# without it the bench falls back to ``--seed-image '' `` (cold
# start) or whichever path you pass on the CLI.
_SEED_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "frozen_valley_sniper.jpg"


def _default_we_seed_path() -> str | None:
    return str(_SEED_IMAGE_PATH) if _SEED_IMAGE_PATH.exists() else None


# ── tensor helpers (Metal-only — model expects numpy carriers) ───


def _sync():
    from quark.runtime.sync import synchronize

    synchronize()


def _np_to_f32(x):
    """Carrier → ``np.float32``. Handles QuarkTensor + tagged numpy."""
    # QuarkTensor → host bytes first.
    if hasattr(x, "to_numpy") and not isinstance(x, np.ndarray):
        raw = x.to_numpy()
        # to_numpy returns bf16 as uint16; widen.
        if x.dtype == "bf16" or raw.dtype == np.uint16:
            return (raw.astype(np.uint32) << 16).view(np.float32).reshape(raw.shape)
        return raw.astype(np.float32)
    arr = np.asarray(x)
    qd = getattr(x, "quark_dtype", None)
    if qd == "bf16" or arr.dtype == np.uint16:
        return (arr.astype(np.uint32) << 16).view(np.float32).reshape(arr.shape)
    return arr.astype(np.float32)


def _f32_to_carrier(arr_f32, dtype: str):
    """numpy f32 → tagged numpy carrier (Metal) or QuarkTensor (SPV)."""
    if dtype == "bf16":
        raw = (arr_f32.view(np.uint32) >> 16).astype(np.uint16)
    elif dtype == "f16":
        raw = arr_f32.astype(np.float16)
    elif dtype == "f32":
        raw = arr_f32.astype(np.float32)
    elif dtype == "s32":
        raw = arr_f32.astype(np.int32)
    else:
        raise ValueError(f"unknown carrier dtype {dtype!r}")
    import sys as _sys
    if _sys.platform != "darwin":
        from quark.runtime.tensor import QuarkTensor as _QT
        return _QT.from_numpy(raw, dtype=dtype)
    return _tag(raw, dtype)


def _randn(*shape, dtype="bf16"):
    rng = np.random.default_rng()
    f32 = rng.standard_normal(shape).astype(np.float32)
    return _f32_to_carrier(f32, dtype)


def _zeros(*shape, dtype="bf16"):
    return _f32_to_carrier(np.zeros(shape, dtype=np.float32), dtype)


def _tensor(data, dtype="f32"):
    return _f32_to_carrier(np.array(data, dtype=np.float32), dtype)


# ── ctrl input ───────────────────────────────────────────────────


def _make_ctrl_input(model):
    """Allocate persistent ctrl input tensor + return ``(buffer, fill_fn)``."""
    if not hasattr(model, "ctrl_emb"):
        return None, None

    shape = model.ctrl_input_shape
    dtype = model.ctrl_input_dtype()
    n_buttons = model.cfg.n_buttons

    # The model's encode_ctrl reads this every frame — keep it as a
    # numpy carrier the kernels accept directly.
    dev = _zeros(*shape, dtype=dtype)
    host = np.zeros(shape, dtype=np.float32)

    def fill(ctrl):
        nonlocal dev
        host.fill(0.0)
        host[0, 0] = float(ctrl.mouse[0])
        host[0, 1] = float(ctrl.mouse[1])
        for b in ctrl.button:
            if 0 <= b < n_buttons:
                host[0, 2 + b] = 1.0
        host[0, 2 + n_buttons] = float(ctrl.scroll_wheel)
        dev = _f32_to_carrier(host, dtype)
        return dev

    return dev, fill


def _build_ctrl_sequence(n_frames: int, CtrlInput):
    """Mirror bench_world_engine.py's ``--ctrl demo`` sequence."""
    seq = [
        CtrlInput(mouse=(0.2, 0.2)),
        CtrlInput(button={32}),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(button={1}),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(button={1, 32}),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(),
    ] * 4
    seq += [CtrlInput()] * 8
    seq += (
        [CtrlInput(button={32})] * 10
        + [CtrlInput(button={65})] * 10
        + [CtrlInput(button={68})] * 10
        + [CtrlInput(button={83})] * 10
    )
    seq += [CtrlInput()] * 10
    while len(seq) < n_frames:
        seq += [
            CtrlInput(
                button=set(random.sample(range(1, 65), random.randint(0, 4))),
                mouse=(random.random() * 0.4 - 0.2, random.random() * 0.4 - 0.2),
                scroll_wheel=random.choice((-1, 0, 1)),
            )
            for _ in range(16)
        ]
    return seq[:n_frames]


# ── weight remap (raw HF safetensors → quark Waypoint15 sd) ──────
# Operates on numpy carrier arrays (the format the Metal model expects).


def _remap_state_dict(raw_sd: dict, cfg) -> dict:
    """Remap raw HF safetensors keys → quark Waypoint15 keys.

    Inputs are QuarkTensor (from load_from_hub); outputs are numpy
    carrier arrays (which is what the Metal Waypoint15 model expects
    as its parameter storage).
    """
    sd: dict = {}
    d = cfg.d_model
    ph, pw = cfg.patch
    C = cfg.channels

    # Materialize every value once as a (tagged_arr, quark_dtype) tuple.
    # Tagging propagates through reshape/transpose so the model sees
    # the correct quark_dtype at every dispatch site.
    raw_np: dict[str, tuple[np.ndarray, str]] = {}
    for k, v in raw_sd.items():
        if hasattr(v, "to_numpy"):
            raw_np[k] = (_tag(v.to_numpy(), v.dtype), v.dtype)
        else:
            raw_np[k] = (_tag(v, "bf16"), "bf16")

    def _carry(key: str) -> tuple[np.ndarray, str]:
        arr, dt = raw_np[key]
        return arr, dt

    # ── patchify / unpatchify ──
    pw_arr, _ = _carry("patchify.weight")
    if pw_arr.ndim == 4:
        pw_arr = pw_arr.reshape(d, C * ph * pw)
    sd["patchify.weight"] = pw_arr

    upw_arr, _ = _carry("unpatchify.weight")
    if upw_arr.ndim == 4:
        upw_arr = upw_arr.transpose(1, 2, 3, 0).reshape(C * ph * pw, d)
    sd["unpatchify.weight"] = upw_arr

    upb_arr, upb_dt = _carry("unpatchify.bias")
    if int(upb_arr.shape[0]) == C:
        b_f32 = _np_to_f32(upb_arr)
        tiled_f32 = np.repeat(b_f32, ph * pw).astype(np.float32)
        upb_arr = _f32_to_carrier(tiled_f32, upb_dt)
    sd["unpatchify.bias"] = upb_arr

    sd["noise_fc1.weight"] = _carry("denoise_step_emb.mlp.fc1.weight")[0]
    sd["noise_fc2.weight"] = _carry("denoise_step_emb.mlp.fc2.weight")[0]
    sd["out_norm_proj.weight"] = _carry("out_norm.fc.weight")[0]

    for i in range(cfg.n_layers):
        p = f"transformer.blocks.{i}."

        if p + "cond_head.bias_in" in raw_np:
            sd[f"blocks.{i}.attn_cond_bias_in"] = _carry(p + "cond_head.bias_in")[0]
            sd[f"blocks.{i}.mlp_cond_bias_in"] = _carry(p + "cond_head.bias_in")[0]
            for j in range(6):
                sd[f"blocks.{i}.cond_projs.{j}.weight"] = _carry(
                    p + f"cond_head.cond_proj.{j}.weight"
                )[0]
        else:
            sd[f"blocks.{i}.attn_cond_bias_in"] = _carry(p + "attn_cond_head.bias_in")[0]
            sd[f"blocks.{i}.mlp_cond_bias_in"] = _carry(p + "mlp_cond_head.bias_in")[0]
            for j in range(3):
                sd[f"blocks.{i}.cond_projs.{j}.weight"] = _carry(
                    p + f"attn_cond_head.cond_proj.{j}.weight"
                )[0]
                sd[f"blocks.{i}.cond_projs.{j + 3}.weight"] = _carry(
                    p + f"mlp_cond_head.cond_proj.{j}.weight"
                )[0]

        # qkv_proj: cat(q,k,v) along dim 0 — concatenate strips the
        # ndarray subclass, so re-tag the result with bf16.
        q, _ = _carry(p + "attn.q_proj.weight")
        k, _ = _carry(p + "attn.k_proj.weight")
        v, _ = _carry(p + "attn.v_proj.weight")
        sd[f"blocks.{i}.qkv_proj.weight"] = _tag(np.concatenate([q, k, v], axis=0), "bf16")
        sd[f"blocks.{i}.out_proj.weight"] = _carry(p + "attn.out_proj.weight")[0]

        fc1_key = (
            p + "mlp.fc1.weight" if (p + "mlp.fc1.weight") in raw_np else p + "dit_mlp.fc1.weight"
        )
        fc2_key = (
            p + "mlp.fc2.weight" if (p + "mlp.fc2.weight") in raw_np else p + "dit_mlp.fc2.weight"
        )
        sd[f"blocks.{i}.mlp.fc1.weight"] = _carry(fc1_key)[0]
        sd[f"blocks.{i}.mlp.fc2.weight"] = _carry(fc2_key)[0]

        if cfg.value_residual and (p + "attn.v_lamb") in raw_np:
            lamb_arr, _ = _carry(p + "attn.v_lamb")
            if lamb_arr.ndim == 0:
                lamb_arr = lamb_arr.reshape(1)
            # v_residual.lamb is an f32 parameter (per Waypoint15Config conventions).
            sd[f"blocks.{i}.v_residual.lamb"] = _np_to_f32(lamb_arr)

        if cfg.ctrl_conditioning:
            fc1_x_key = p + "ctrl_mlpfusion.fc1_x.weight"
            fc1_c_key = p + "ctrl_mlpfusion.fc1_c.weight"
            fc2_key = p + "ctrl_mlpfusion.fc2.weight"
            if fc1_x_key in raw_np:
                sd[f"blocks.{i}.ctrl_fusion.fc1_x.weight"] = _carry(fc1_x_key)[0]
                sd[f"blocks.{i}.ctrl_fusion.fc1_c.weight"] = _carry(fc1_c_key)[0]
                sd[f"blocks.{i}.ctrl_fusion.fc2.weight"] = _carry(fc2_key)[0]
            elif (p + "ctrl_mlpfusion.mlp.fc1.weight") in raw_np:
                fc1_cat, _ = _carry(p + "ctrl_mlpfusion.mlp.fc1.weight")
                d_model = int(fc1_cat.shape[1]) // 2
                sd[f"blocks.{i}.ctrl_fusion.fc1_x.weight"] = fc1_cat[:, :d_model]
                sd[f"blocks.{i}.ctrl_fusion.fc1_c.weight"] = fc1_cat[:, d_model:]
                sd[f"blocks.{i}.ctrl_fusion.fc2.weight"] = _carry(
                    p + "ctrl_mlpfusion.mlp.fc2.weight"
                )[0]

    ctrl_fc1_key = "ctrl_emb.mlp.fc1.weight"
    if cfg.ctrl_conditioning and ctrl_fc1_key in raw_np:
        fc1_arr, _ = _carry(ctrl_fc1_key)
        raw_k = int(fc1_arr.shape[1])
        padded_k = ((raw_k + 15) // 16) * 16
        if padded_k > raw_k:
            # Pad columns to a 16 multiple but keep the tile in bf16 —
            # the ctrl_emb layer is pinned to bf16 end-to-end so the
            # downstream ctrl_fusion (also bf16) reads its cond input
            # without a cast.
            w_f32 = _np_to_f32(fc1_arr)
            padded = np.zeros((w_f32.shape[0], padded_k), dtype=np.float32)
            padded[:, :raw_k] = w_f32
            fc1_arr = _f32_to_carrier(padded, "bf16")
        sd["ctrl_emb.fc1.weight"] = fc1_arr
        sd["ctrl_emb.fc2.weight"] = _carry("ctrl_emb.mlp.fc2.weight")[0]

    return sd


# ── TAEHV decoder (delegated to quark.taehv) ─────────────────────


def _load_taehv(ae_uri: str, *, ane: bool, latent_height: int, latent_width: int):
    """Thin wrapper around :func:`quark.taehv.load_taehv`.

    ``ae_uri`` is the source TAEHV PyTorch repo (e.g.
    ``Overworld-Models/taehv1_5``); the CoreML repo URI is derived
    by appending ``-coreml`` (matching ``quark.engine``'s convention).
    Override the derivation with the ``QUARK_TAEHV_COREML_URI`` env
    var when testing against a personal / staging HF repo.

    ``ane=True`` selects ``CPU_AND_NE`` compute units; ``ane=False``
    falls back to ``CPU_AND_GPU`` (Metal). CPU-only is intentionally
    not exposed by ``quark.taehv`` — for low-end Macs, ``ane=False``
    is the right answer.

    Returns ``(ae, is_ane)`` for compatibility with the bench's
    historical signature; ``is_ane`` is just ``ane`` echoed back.
    """
    import os

    from quark.taehv import load_taehv

    # Backend selection:
    #   * On darwin: CoreML (CPU_AND_NE or CPU_AND_GPU).
    #   * Elsewhere: OpenVINO (GPU / CPU / NPU).
    # Override via ``QUARK_TAEHV_BACKEND`` (``coreml`` / ``openvino``),
    # or via the URI suffix conventions ``-coreml`` / ``-openvino``.
    ov_uri_env = os.environ.get("QUARK_TAEHV_OPENVINO_URI")
    ov_dir_env = os.environ.get("QUARK_TAEHV_OPENVINO_DIR")
    coreml_uri_env = os.environ.get("QUARK_TAEHV_COREML_URI")
    backend = (os.environ.get("QUARK_TAEHV_BACKEND") or "").lower() or None
    # If the user explicitly points to an OpenVINO IR (env URI or
    # local dir), treat that as a strong opt-in even on darwin —
    # avoids requiring two env vars (DIR + BACKEND) to get the
    # OpenVINO path.
    if backend is None and (ov_uri_env or ov_dir_env):
        backend = "openvino"

    if sys.platform == "darwin" and backend != "openvino":
        uri = coreml_uri_env or f"{ae_uri}-coreml"
        compute_units = "CPU_AND_NE" if ane else "CPU_AND_GPU"
        ae = load_taehv(
            uri,
            latent_height=latent_height,
            latent_width=latent_width,
            compute_units=compute_units,
        )
    else:
        # OpenVINO path. ``uri`` may be an HF repo or a local export dir.
        uri = ov_uri_env or os.environ.get("QUARK_TAEHV_OPENVINO_DIR") or f"{ae_uri}-openvino"
        # ``ane`` is ignored on OpenVINO; ``QUARK_TAEHV_OPENVINO_DEVICE``
        # (GPU/CPU/NPU/AUTO) picks the OpenVINO device.
        device = os.environ.get("QUARK_TAEHV_OPENVINO_DEVICE", "GPU")
        ae = load_taehv(
            uri,
            latent_height=latent_height,
            latent_width=latent_width,
            compute_units=device,
            backend="openvino",
        )
    return ae, ane


def _decode_one(ae, latent_np):
    """``[1, 32, latH, latW] f32/f16`` → ``[4, H_px, W_px, 3] uint8``.

    Pure numpy in / out. Delegates to ``ae.decode`` (defined in
    :mod:`quark.taehv.coreml`).
    """
    return ae.decode(latent_np)


def _write_video_hevc(pixels, output_path: str, fps: int, crf: int) -> None:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH — needed for HEVC encode")
    if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
        raise ValueError(f"expected [T,H,W,3] uint8, got {pixels.shape} {pixels.dtype}")

    t, h, w, _ = pixels.shape
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx265",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-tag:v",
        "hvc1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
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


# ── seed encode ──────────────────────────────────────────────────


def _encode_seed(path: str, preset: str, ae_uri: str, latent_height: int, latent_width: int):
    # Diagnostic: skip CoreML encode and load a pre-saved latent (for
    # cross-host SPV demos that can't run CoreML).
    pre = os.environ.get("QUARK_PRE_ENCODED_SEED")
    if pre:
        import numpy as _np
        return _np.load(pre).astype(_np.float32)
    """Encode an image (or short video) to a ``[T, C, H, W]`` f32 latent.

    Uses **PIL + LANCZOS** for the resize, identical to
    ``world_engine.mlx_metal.benchmarks.bench_engine.load_seed_image``.
    cv2's INTER_AREA produces pixel-different output for the same
    JPEG, which would put quark and WE on slightly different input
    distributions for the "same seed" comparison.

    Encode goes through the CoreML ANE encoder
    (``quark.taehv.CoreMLTAEHV.encode``) — no torch dependency. The
    encoder is stateless so this can run alongside the per-frame
    decode without touching the decoder's MemBlock state.
    """
    from PIL import Image

    pixel_h, pixel_w = _AE_PIXEL_PRESETS[preset]
    raw: list[np.ndarray] = []
    suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    if suffix in ("mp4", "mov", "mkv", "avi", "webm"):
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"cannot open {path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not raw:
            raise ValueError(f"no frames read from {path}")
    else:
        img = Image.open(path).convert("RGB")
        raw.append(np.asarray(img))

    # PIL LANCZOS resize, identical to WE bench's ``load_seed_image``.
    def _resize(arr: np.ndarray) -> np.ndarray:
        im = Image.fromarray(arr).resize((pixel_w, pixel_h), Image.LANCZOS)
        return np.asarray(im)

    resized = [_resize(f) for f in raw]
    n_padded = _TEMPORAL_COMPRESS * ((len(resized) + _TEMPORAL_COMPRESS - 1) // _TEMPORAL_COMPRESS)
    while len(resized) < n_padded:
        resized.append(resized[-1])
    pixels = np.stack(resized, axis=0).astype(np.uint8)

    ae, _ = _load_taehv(ae_uri, ane=True, latent_height=latent_height, latent_width=latent_width)
    # ``encode`` takes 4 frames at a time; we already padded above.
    chunks = []
    for c in range(0, pixels.shape[0], _TEMPORAL_COMPRESS):
        chunks.append(ae.encode(pixels[c : c + _TEMPORAL_COMPRESS]))
    return np.concatenate(chunks, axis=0)


# ── config ───────────────────────────────────────────────────────


def _move_params_to_device(model):
    """Convert every numpy-carrier Parameter into a device-resident
    ``QuarkTensor`` (pinned in the Metal buffer pool). Stops the
    dispatcher from memcpy'ing weight bytes into a fresh pool buffer
    on every ``quark.lazy()`` eval boundary — pinned buffers survive
    eval, so the input cache hit-rate goes from 0% (per-frame re-upload)
    to 100% (uploaded once at startup).

    Walks both registered Parameters and the ad-hoc cond LUT lists
    that ``Waypoint15.prepare()`` stores on each TransformerBlock /
    on the model itself.
    """
    from quark.runtime.tensor import QuarkTensor

    def _is_numpy_carrier(t):
        return isinstance(t, np.ndarray)

    def _to_qt(arr):
        qd = getattr(arr, "quark_dtype", None) or {
            "uint16": "bf16",
            "float16": "f16",
            "float32": "f32",
            "int32": "s32",
            "uint8": "u8",
            "int8": "s8",
        }.get(arr.dtype.name, "bf16")
        return QuarkTensor.from_numpy(arr, dtype=qd)

    # ``model.named_parameters()`` yields ``param.data`` (the unwrapped
    # array), not the ``Parameter`` wrapper, so it can't help us swap
    # ``.data`` in place. Walk the module tree manually instead and
    # mutate ``Parameter.data``. We *also* pin module-level state
    # buffers (KV cache, segment table, frame_t, frozen flag) — those
    # are plain numpy carriers, not ``Parameter``s, but they need to
    # be device-resident or the dispatcher recycles the underlying
    # Metal pool buffer at every ``quark.lazy()`` eval boundary, which
    # would silently throw away every KV-cache update. (Diagnosed
    # empirically: cold-start vs seeded runs produced *pixel-identical*
    # frames until these were pinned.)
    from quark.nn.module import Module as _NN
    from quark.nn.module import Parameter as _NN_Param

    # Names of attributes on KVCacheUpdate that hold persistent device
    # state. Anything else is either a Parameter (handled above) or a
    # transient.
    _STATE_BUF_ATTRS = ("K_cache", "Vt_cache", "segments", "n_segments", "frame_t", "frozen")

    n_uploaded = 0
    total_bytes = 0

    def _walk(mod):
        nonlocal n_uploaded, total_bytes
        for _, val in mod._own_members():
            if isinstance(val, _NN_Param):
                arr = val.data
                if _is_numpy_carrier(arr):
                    arr = np.ascontiguousarray(arr)
                    val.data = _to_qt(arr)
                    n_uploaded += 1
                    total_bytes += int(arr.nbytes)
            elif isinstance(val, _NN):
                _walk(val)
        # Pin per-module persistent state buffers.
        for attr in _STATE_BUF_ATTRS:
            v = getattr(mod, attr, None)
            if _is_numpy_carrier(v):
                arr = np.ascontiguousarray(v)
                setattr(mod, attr, _to_qt(arr))
                n_uploaded += 1
                total_bytes += int(arr.nbytes)

    _walk(model)

    # Cond LUTs (per-block) + out_norm LUTs are populated by prepare()
    # as numpy carriers. Migrate them too — they're read on every
    # forward and would otherwise re-upload per eval.
    for block in getattr(model, "blocks", []):
        lut = getattr(block, "_cond_lut", None)
        if lut is None:
            continue
        new_lut = []
        for tup in lut:
            new_tup = tuple(
                _to_qt(np.ascontiguousarray(t)) if _is_numpy_carrier(t) else t for t in tup
            )
            new_lut.append(new_tup)
            for t in tup:
                if _is_numpy_carrier(t):
                    n_uploaded += 1
                    total_bytes += int(t.nbytes)
        block._cond_lut = new_lut

    on_luts = getattr(model, "_out_norm_luts", None)
    if on_luts is not None:
        new_luts = []
        for s_on, b_on in on_luts:
            new_luts.append(
                (
                    _to_qt(np.ascontiguousarray(s_on)) if _is_numpy_carrier(s_on) else s_on,
                    _to_qt(np.ascontiguousarray(b_on)) if _is_numpy_carrier(b_on) else b_on,
                )
            )
            for t in (s_on, b_on):
                if _is_numpy_carrier(t):
                    n_uploaded += 1
                    total_bytes += int(t.nbytes)
        model._out_norm_luts = new_luts

    print(
        f"  pinned {n_uploaded} weights ({total_bytes / (1 << 20):.1f} MiB) "
        f"to device — dispatcher skips host memcpy on hot path"
    )


def _make_config(preset: str):
    import sys

    from quark.models.waypoint_15 import QuantConfig, Waypoint15Config

    presets = {"360p": (8, 16), "720p": (16, 32)}
    h, w = presets[preset]
    # Metal has no native fp8 (no e4m3 type in MSL). SPV doesn't either —
    # the SpirVLowerer rejects DType.E4M3 (PORTABILITY_PLAN §3.2). Force
    # the all-bf16 quant profile on both so KV cache + OwlAttn MMAs stay
    # in the half-dt path; QuantConfig defaults to fp8-everything for
    # Hopper/Ada.
    is_metal = sys.platform == "darwin"
    # Match the auto-detect in ``quark.runtime.sync``: Linux with no
    # CUDA and a working SPV driver → SPV. Both Metal and SPV need the
    # all-bf16 profile because neither lowerer supports e4m3 today.
    is_spv = False
    if not is_metal:
        try:
            from quark.runtime.cuda import CudaRuntime
            if CudaRuntime.instance().device_count() == 0:
                from quark.drivers import spv as _spv_drivers
                is_spv = _spv_drivers.is_available()
        except Exception:
            try:
                from quark.drivers import spv as _spv_drivers
                is_spv = _spv_drivers.is_available()
            except Exception:
                pass
    quant = QuantConfig.all_bf16() if (is_metal or is_spv) else QuantConfig()
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
        quant=quant,
    )


# ── main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Bench quark Waypoint-1.5 (mirror of bench_world_engine.py)"
    )
    parser.add_argument("--repo", default="Overworld/Waypoint-1.5-1B")
    parser.add_argument("--preset", choices=["360p", "720p"], default="360p")
    parser.add_argument("--n-frames", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--f16",
        action="store_true",
        help="opt into f16 (CUDA-friendly). Default is bf16 — "
        "the only dtype the Apple NAX MMA shape supports.",
    )
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="DiT-only timing — skip TAEHV decode (matches bench_world_engine default).",
    )
    # Default seed: the exact JPEG world_engine's bench_engine.py uses,
    # resolved from the installed ``world_engine`` package. Same seed,
    # same starting KV state across both benches. Falls back to ``None``
    # (cold start) only if world_engine isn't importable.
    parser.add_argument(
        "--seed-image",
        default=_default_we_seed_path(),
        help="image/video to seed the KV cache. Default: "
        "world_engine's frozen_valley_sniper.jpg "
        "(same file bench_engine.py uses). Pass a path "
        "to override or ``--seed-image ''`` for cold start.",
    )
    # Resolve the default output under the quark repo's diagnostics
    # dir so the artifact lands somewhere predictable regardless of
    # cwd. Pass an absolute / relative ``--output`` to override.
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _DEFAULT_OUTPUT = _REPO_ROOT / "diagnostics" / "quark_demo.mp4"
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help=".mp4/.mov/.mkv writes HEVC; .npy saves pixels; "
        "ignored with --no-decode. Default: "
        "<repo>/diagnostics/quark_demo.mp4 (PNG fallback "
        "to <basename>_frames/ when ffmpeg is missing).",
    )
    parser.add_argument("--ae-repo", default="Overworld-Models/taehv1_5")
    parser.add_argument("--fps", type=int, default=_PIXEL_FPS)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument(
        "--no-lazy",
        dest="lazy_dispatch",
        action="store_false",
        help="opt out of quark.lazy() (eager per-launch commits). "
        "Default is lazy dispatch — single command buffer per "
        "frame, kernels pipeline on the GPU.",
    )
    parser.set_defaults(lazy_dispatch=True)
    parser.add_argument(
        "--no-ane",
        dest="ane",
        action="store_false",
        help="skip the ANE/CoreML TAEHV path; decode on CPU "
        "(slow). Default is ANE when available, with a "
        "silent fallback to CPU on any init error.",
    )
    parser.set_defaults(ane=True)
    parser.add_argument(
        "--no-pipeline",
        dest="pipeline",
        action="store_false",
        help="serialize forward + decode (no overlap). Default "
        "pipelines ANE decode in a background thread so it "
        "runs concurrently with the next frame's forward.",
    )
    parser.set_defaults(pipeline=True)
    parser.add_argument(
        "--full-autotune",
        action="store_true",
        help="force the deeper genetic search on every cache "
        "miss (10+ minutes on a cold cache). Default is "
        "fast search (~1 s per miss); winners persist to "
        "~/.cache/quark so subsequent runs are pure "
        "cache hits.",
    )
    parser.add_argument(
        "--no-autotune",
        action="store_true",
        help="skip autotune entirely — use bundled default "
        "kernel configs (instant startup, ~30x slower "
        "per forward). Useful when iterating on the "
        "Python side and you don't care about kernel "
        "perf. Equivalent to ``QUARK_DISABLE_AUTOTUNE=1``.",
    )
    args = parser.parse_args()

    import quark
    from quark.models.waypoint_15 import CtrlInput, Waypoint15
    from quark.nn.io import load_from_hub

    cfg = _make_config(args.preset)
    # The standalone-Engine merge dropped the per-config ``use_f16``
    # boolean — the residual stream is bf16 throughout (Apple NAX MMA
    # is ``m16n32k16_nax_bf16``-only, and the CUDA patchify / owl_attn
    # autotune caches were never seeded for f16). The ``--f16`` flag
    # is now a no-op kept for argparse-compat; warn if set so existing
    # invocations surface the dropped behaviour.
    if args.f16:
        print("[bench] WARNING: --f16 is a no-op since the QuantConfig refactor; "
              "residual stream is bf16 only.")

    ph, pw = cfg.patch
    H, W = cfg.height * ph, cfg.width * pw
    C = cfg.channels
    tpf = cfg.tpf
    half_dt = "bf16"
    sigmas = list(cfg.scheduler_sigmas)
    n_denoise = len(sigmas) - 1
    nfe = n_denoise + 1

    print(f"preset: {args.preset} — {tpf} tokens/frame")
    print(f"  latent grid: {cfg.height}x{cfg.width}, {args.n_frames} frames")
    print(f"  model: {cfg.n_layers} layers, d_model={cfg.d_model}, dtype={half_dt}")
    print(f"  scheduler: {n_denoise} denoise + 1 commit = {nfe} NFE/frame")

    # ── Load model ──
    repo_suffix = "-360P" if args.preset == "360p" else ""
    repo = args.repo + repo_suffix
    print(f"loading {repo} …")
    t0 = time.perf_counter()
    raw_sd = load_from_hub(repo, dtype="bf16")
    print(f"  safetensors loaded ({time.perf_counter() - t0:.1f}s)")

    t1 = time.perf_counter()
    sd = _remap_state_dict(raw_sd, cfg)
    # Free the QuarkTensor-side mmap buffers — we have numpy copies now.
    raw_sd.clear()
    print(f"  remap done ({time.perf_counter() - t1:.1f}s)")

    model = Waypoint15(cfg)
    model.load_state_dict(sd, strict=False)
    model.prepare()
    _move_params_to_device(model)
    _sync()
    print(f"  model ready ({time.perf_counter() - t0:.1f}s total)")

    # ── Autotune ──
    # ``quark.max_autotune()`` forces a full genetic search (64 pop ×
    # 8 gens = up to 512 compile-and-time iters) for every uncached
    # spec. With ~240 kernel sites per forward and a cold disk cache,
    # that ran 10+ minutes for almost no benefit on top of fast search.
    # Default behaviour now: fast search (~1 s per miss) on the first
    # call, persisted to disk; subsequent runs are pure cache hits.
    # ``--full-autotune`` opts back into the deeper search.
    # ``--no-autotune`` skips the search entirely (bundled defaults).
    if args.no_autotune:
        os.environ["QUARK_DISABLE_AUTOTUNE"] = "1"
        print("autotune disabled (using bundled defaults — slow per forward)")
    else:
        msg = "full search" if args.full_autotune else "fast search"
        print(f"autotuning ({msg}; pass --no-autotune to skip)…")

    t_at = time.perf_counter()
    ctrl_dev, ctrl_fill = _make_ctrl_input(model)
    if ctrl_fill is not None:
        ctrl_dev = ctrl_fill(CtrlInput())
    autotune_ctrl_emb = model.encode_ctrl(ctrl_dev)
    autotune_x = _randn(1, C * H * W, dtype=half_dt)
    autotune_ft = _zeros(1, dtype="s32")
    from contextlib import nullcontext as _nullctx

    _at_ctx = quark.max_autotune() if args.full_autotune else _nullctx()
    with _at_ctx:
        # Two warmup passes so RoPE / cond LUTs / KV state allocate;
        # the second is a fast cache hit on every kernel.
        model(autotune_x, sigma_idx=0, frame_t=autotune_ft, ctrl_emb=autotune_ctrl_emb)
        model(autotune_x, sigma_idx=0, frame_t=autotune_ft, ctrl_emb=autotune_ctrl_emb)
        _sync()
    print(f"  autotune done ({time.perf_counter() - t_at:.1f}s)")

    # ── Seed frames (optional) ──
    seed_count = 0
    if args.seed_image:
        print(f"encoding seed from {args.seed_image} …")
        t_enc = time.perf_counter()
        seed_np = _encode_seed(
            args.seed_image,
            args.preset,
            args.ae_repo,
            latent_height=H,
            latent_width=W,
        )
        print(f"  encoded {seed_np.shape[0]} latent frame(s) in {time.perf_counter() - t_enc:.1f}s")
        seed_count = seed_np.shape[0]
        if ctrl_fill is not None:
            ctrl_dev = ctrl_fill(CtrlInput())
        seed_ctrl_emb = model.encode_ctrl(ctrl_dev)
        for fi in range(seed_count):
            latent_f32 = seed_np[fi : fi + 1].reshape(1, C * H * W).astype(np.float32)
            latent = _f32_to_carrier(latent_f32, half_dt)
            ft = _tensor([fi], dtype="s32")
            model(latent, sigma_idx=n_denoise, frame_t=ft, ctrl_emb=seed_ctrl_emb)
            _sync()
        print(f"  seeded {seed_count} frames")

    # ── Build per-step euler tensors ──
    import quark.nn as nn

    dsig_tensors = [
        _tensor([float(sigmas[si + 1] - sigmas[si])], dtype="f32") for si in range(n_denoise)
    ]
    euler_steps = nn.ModuleList([nn.EulerStep() for _ in range(n_denoise)])

    ctrl_sequence = _build_ctrl_sequence(args.n_frames + args.warmup + 1, CtrlInput)

    # ── Load + warm the AE (before timed loop, so CoreML export +
    # weight loading don't pollute the per-frame timing). ──
    # quark.taehv loads only CoreML artifacts (ANE or GPU); CPU
    # fallback is intentionally not exposed.  The bench's
    # ``--no-ane`` flag now selects the GPU compute units instead.
    ae, is_ane = (None, False)
    if not args.no_decode:
        print(f"loading TAEHV ({'ANE/CoreML' if args.ane else 'GPU/CoreML'}) …")
        t_ae = time.perf_counter()
        ae, is_ane = _load_taehv(
            args.ae_repo,
            ane=args.ane,
            latent_height=H,
            latent_width=W,
        )
        # Warm one decode so the CoreML model is compiled + ANE state
        # is allocated before the hot loop sees it. CoreMLTAEHV's
        # ``_warmup`` already runs at construct time, but a second
        # warmup with the right shape catches any post-init lazy paths.
        warm_lat = np.zeros((1, C, H, W), dtype=np.float16)
        try:
            ae.decode(warm_lat)
        except Exception as e:
            print(f"  warmup decode failed ({e}); will surface real errors mid-run")
        # Reset the decoder's MemBlock state — the warmup primed it
        # with zeros, which we want to clear before the timed run.
        ae.reset()
        # Honest backend label: probe ``ae`` rather than the historical
        # ``is_ane`` flag, which only meant "CoreML w/ ANE compute".
        if hasattr(ae, "_device"):
            _backend_label = f"OpenVINO/{ae._device}"
        else:
            _backend_label = "ANE" if is_ane else "Metal-GPU"
        print(
            f"  TAEHV ready ({time.perf_counter() - t_ae:.1f}s, "
            f"backend={_backend_label})"
        )

    use_pipeline = ae is not None and args.pipeline

    # ── Frame helper that wraps gen_frame in quark.lazy() per-frame ──
    # Per-frame lazy block matches WE's gen_frame granularity (one
    # command buffer per frame). Wrapping the *entire* loop in one
    # block would give a flatter total at the cost of ill-defined
    # per-frame timing — not what we want for an apples-to-apples
    # comparison vs bench_engine.py.
    import ctypes as _ct

    # The split between Phase 1 (denoise, sync=True) and Phase 2
    # (commit, sync=False) costs an extra encoder close + command
    # buffer commit per frame (~1-2 ms on M5 Max). It's only worth
    # paying when there's *concurrent work on a separate compute
    # pool* to overlap with the commit GPU work — i.e. when the
    # TAEHV decoder is running on the ANE. With ``--no-decode`` we
    # collapse to a single lazy block: denoise+commit run in the
    # same encoder, one wait at exit, and no extra commit cycle.
    split_lazy = ae is not None and not args.no_decode

    # Bytes per element for the bench's half_dt — used by the post-eval
    # latent host-copy. bf16 / f16 are both 2 bytes.
    _BYTES_PER = 2

    def gen_frame(
        ctrl,
        fi: int,
        return_latent: bool = True,
        snapshot: bool = True,
        between_phases=None,
    ):
        """One frame: encode_ctrl → 4 euler steps → 1 commit. Mirrors
        ``MLXWorldEngine.gen_frame``.

        With a TAEHV decoder available, denoise and commit run in
        *separate* lazy blocks: denoise syncs at exit (so ``cur`` is
        host-readable), commit is committed-without-wait so it
        overlaps with the ANE decode and the next frame's Python
        dispatch. World_engine gets this for free because MLX's
        ``cache_write`` is lazy by default; we get it explicitly via
        ``quark.lazy(sync=False)``.

        Without a decoder (``--no-decode``), the split is pure
        overhead — no separate compute pool to hide the commit GPU
        work behind — so we collapse to a single lazy block, saving
        the extra encoder close + command buffer commit per frame.

        ``snapshot`` controls *who* does the host copy. The default
        (``True``) keeps the legacy behavior: gen_frame returns a
        tagged numpy carrier, the host memcpy happens inline on the
        main thread. Setting ``snapshot=False`` returns the live
        ``QuarkTensor`` instead — the storage stays alive thanks to
        the refcount-aware lazy-buffer lifetime (``b5b5d12``), and
        downstream code (typically ``PipelinedDecoder.submit``) does
        the memcpy on a worker thread where it overlaps with the
        next frame's GPU forward.

        ``between_phases``: optional callback ``fn(result)`` invoked
        between Phase 1 (denoise sync) and Phase 2 (commit lazy)
        when the split path is active. Used by the pipelined-decode
        loop to drain the previous decode and submit the current
        latent BEFORE the commit kernels run on the GPU — this
        hands the ANE worker the latent ~5 ms earlier in the iter,
        which is enough on M5 Max to push the per-iter drain wait
        below the commit's sync=False overlap budget. Mirrors the
        order in ``world_engine.MLXWorldEngine.gen_frame_pipelined``
        (submit decode → cache_write).
        """
        nonlocal ctrl_dev
        if ctrl_fill is not None:
            ctrl_dev = ctrl_fill(ctrl)
        ctrl_emb = model.encode_ctrl(ctrl_dev)
        x = _randn(1, C * H * W, dtype=half_dt)
        ft = _tensor([fi], dtype="s32")

        def _denoise(cur_in):
            cur_out = cur_in
            for si in range(n_denoise):
                v = model(cur_out, sigma_idx=si, frame_t=ft, ctrl_emb=ctrl_emb, frozen=True)
                cur_out = euler_steps[si](cur_out, v, dsig_tensors[si])
            return cur_out

        def _commit(cur_in):
            model(cur_in, sigma_idx=n_denoise, frame_t=ft, ctrl_emb=ctrl_emb)

        if not args.lazy_dispatch:
            # Eager: every kernel commits + syncs by itself; just run.
            cur = _denoise(x)
            _sync()
        elif not split_lazy:
            # No-decode (or no AE): single lazy block — denoise +
            # commit share one encoder, one wait at exit. Recovers
            # the pre-split baseline.
            with quark.lazy():
                cur = _denoise(x)
                _commit(cur)
        else:
            # With ANE decode: split for commit/decode overlap.
            with quark.lazy():
                cur = _denoise(x)

        # ── Snapshot / hand-off (between denoise and commit) ─────
        if not return_latent:
            result = None
        elif snapshot:
            # Inline host copy on the main thread (legacy + serial-
            # decode + no-decode paths). cur is read-only from here
            # on; the commit phase reads cur as input but never
            # writes it, so a concurrent worker memcpy in the
            # snapshot=False path is safe too.
            nbytes = int(np.prod(cur.shape)) * _BYTES_PER
            carrier_np = np.empty(cur.shape, dtype=np.uint16 if half_dt == "bf16" else np.float16)
            _ct.memmove(carrier_np.ctypes.data, cur.data_ptr(), nbytes)
            result = _tag(carrier_np, half_dt)
        else:
            # PipelinedDecoder path — caller's worker does the memcpy.
            result = cur

        # ── Hand-off hook (between phases, split path only) ──────
        # Callers running a pipelined decoder use this to submit the
        # current decode + drain the previous BEFORE the commit
        # encodes — the worker then has the commit's encode + GPU
        # window AND the next frame's denoise window to finish in,
        # rather than only the latter. Worth ~1–2 ms on M5 Max.
        if between_phases is not None and split_lazy:
            between_phases(result)

        # ── Phase 2: commit / cache_write (split path only) ──────
        # Lazy, no host wait — overlaps with ANE decode + next
        # frame's Python encoding. The cross-encoder fence on the
        # K_cache buffer ensures the next denoise's encoder waits
        # until commit's GPU writes are visible. The single-block /
        # eager paths above already ran the commit, so this branch
        # is skipped there.
        if not args.lazy_dispatch:
            _commit(cur)
            _sync()
            return result
        if not split_lazy:
            return result
        with quark.lazy(sync=False):
            _commit(cur)
        return result

    # ── Warmup ──
    print(f"warming up ({args.warmup} frames) …")
    t_warm = time.perf_counter()
    for i in range(args.warmup):
        gen_frame(ctrl_sequence[i], seed_count + i, return_latent=False)
    _sync()
    print(f"  warmup done ({time.perf_counter() - t_warm:.1f}s)")

    gen_frame(ctrl_sequence[args.warmup], seed_count + args.warmup, return_latent=False)
    _sync()

    # ── Generate frames (timed) ──
    if args.no_decode:
        decode_mode = "no decode"
    else:
        # Try to derive a meaningful backend label from ``ae``.
        if hasattr(ae, "_device"):
            _backend_label = f"OpenVINO/{ae._device}"
        elif is_ane:
            _backend_label = "ANE"
        else:
            _backend_label = "Metal-GPU"
        _mode_label = "pipelined" if use_pipeline else "serial"
        decode_mode = f"+ {_backend_label} decode ({_mode_label})"
    print(
        f"\ngenerating {args.n_frames} frames ({n_denoise} denoise + 1 commit each) "
        f"[{decode_mode}] …"
    )

    latents: list = []
    base_fi = seed_count + args.warmup + 1
    per_frame_ms: list = []
    pixel_chunks: list = []
    fwd_ms: list = []
    decode_ms: list = []

    # Pipelined CoreML: ``PipelinedDecoder`` runs ae.decode in a
    # background thread while the GPU is busy with the next frame's
    # forward. The ANE and the Apple GPU share unified memory but use
    # distinct compute pools, so this overlap is real (mirrors
    # world_engine's gen_frame_pipelined). The pipeline class also
    # moves the host-side latent memcpy into the worker thread, which
    # is what the inline implementation used to do on the main thread
    # — closes the ~5 ms / frame forward-time inflation we measured.
    pipe = None
    if use_pipeline and not args.no_decode:
        from quark.taehv import PipelinedDecoder

        pipe = PipelinedDecoder(ae)

    def _submit_decode(lt):
        if pipe is not None:
            # ``lt`` is a live QuarkTensor (gen_frame ran with
            # ``snapshot=False``). Pass it straight through —
            # PipelinedDecoder snapshots in its worker thread.
            pipe.submit(lt)
        else:
            latent_np = _np_to_f32(lt).reshape(1, C, H, W).astype(np.float16)
            t_d0 = time.perf_counter()
            pixel_chunks.append(_decode_one(ae, latent_np))
            decode_ms.append((time.perf_counter() - t_d0) * 1000)

    def _drain_pending():
        if pipe is None:
            return
        t_d0 = time.perf_counter()
        img = pipe.next()
        if img is not None:
            pixel_chunks.append(img)
            decode_ms.append((time.perf_counter() - t_d0) * 1000)

    # When ``pipe`` is set, gen_frame returns a live QuarkTensor (no
    # main-thread host snapshot) and PipelinedDecoder does the memcpy
    # on its worker. The serial / no-decode paths still take the
    # legacy snapshot route — they consume the latent on the main
    # thread (numpy bytes) and there's no worker to defer to.
    snapshot = pipe is None

    _sync()
    t_gen = time.perf_counter()
    for i in range(args.n_frames):
        t_f0 = time.perf_counter()
        # gen_frame owns its own quark.lazy() blocks; the trailing-edge
        # drain+submit below is intentional. We tried world_engine's
        # ordering (drain+submit BETWEEN gen_frame's denoise and commit
        # phases via the ``between_phases`` hook) and it regressed by
        # ~6 ms — there the drain wait sits on a GPU-idle window
        # because commit hasn't been encoded yet. The trailing-edge
        # order hides drain behind commit's lazy GPU work, which is
        # the right call when the iter is GPU-bound.
        _force_latent = bool(os.environ.get("QUARK_DUMP_LATENTS"))
        lt = gen_frame(
            ctrl_sequence[args.warmup + 1 + i],
            base_fi + i,
            return_latent=(not args.no_decode) or _force_latent,
            snapshot=snapshot or _force_latent,
        )
        if lt is not None and snapshot:
            # Only the snapshot path appends to ``latents`` — the
            # QuarkTensor path's storage gets handed straight to the
            # decoder's worker thread; we don't keep an N-frame fan
            # of live Metal buffers around.
            latents.append(lt)
        per_frame_ms.append((time.perf_counter() - t_f0) * 1000)
        fwd_ms.append(per_frame_ms[-1])

        if ae is not None and not args.no_decode and lt is not None:
            # Collect the previous frame's pipelined decode (no-op for
            # the first frame), then submit this one. Serial path runs
            # the decode inline.
            _drain_pending()
            _submit_decode(lt)

    # Drain the trailing pipelined decode (or the last serial one).
    if pipe is not None:
        t_d0 = time.perf_counter()
        img = pipe.flush()
        if img is not None:
            pixel_chunks.append(img)
            decode_ms.append((time.perf_counter() - t_d0) * 1000)
        pipe.shutdown()
    _sync()
    gen_elapsed = time.perf_counter() - t_gen

    # ── Summary ──
    n = args.n_frames
    lfps = n / gen_elapsed
    pf_arr = np.array(per_frame_ms)
    fwd_arr = np.array(fwd_ms)
    dec_arr = np.array(decode_ms) if decode_ms else None

    print(f"\n{'=' * 50}")
    print(
        f"  quark Waypoint-1.5 (dtype={half_dt}, "
        f"{'lazy' if args.lazy_dispatch else 'eager'} dispatch)"
    )
    print(f"  decode:      {decode_mode}")
    print(f"  {n} latent frames, {tpf} tokens/frame")
    print(f"  {nfe} NFE/frame ({n_denoise} denoise + 1 commit)")
    print(f"  Forward:     {fwd_arr.mean():6.1f} ms avg ({fwd_arr.std():5.1f} ms std)")
    # Initial vs saturated breakdown — Waypoint-1.5 sliding-window
    # attention saturates around ``cfg.global_window`` frames; report
    # both windows so PR descriptions can show cold + warm steady-state.
    init_n = min(10, len(fwd_arr))
    # Diagnostic per-frame dump for inter-frame variance analysis.
    _frame_dump = os.environ.get("QUARK_DUMP_FRAME_TIMINGS")
    if _frame_dump:
        np.save(_frame_dump, fwd_arr)
        print(f"  [diag] per-frame Forward timings saved to {_frame_dump}")
    if len(fwd_arr) >= cfg.global_window + 50:
        sat_arr = fwd_arr[cfg.global_window:]
        print(
            f"               initial {init_n}: median {np.median(fwd_arr[:init_n]):5.1f} ms "
            f"({1000/np.median(fwd_arr[:init_n]):5.2f} LFPS)"
        )
        print(
            f"               saturated (frame {cfg.global_window}+): "
            f"median {np.median(sat_arr):5.1f} ms ({1000/np.median(sat_arr):5.2f} LFPS)"
        )
    if dec_arr is not None and len(dec_arr):
        if use_pipeline:
            _ae_lbl = f"OpenVINO/{ae._device}" if hasattr(ae, "_device") else ("ANE" if is_ane else "Metal-GPU")
            print(f"  {_ae_lbl} decode:  {dec_arr.mean():6.1f} ms collect (overlapped)")
        else:
            print(f"  Decode:      {dec_arr.mean():6.1f} ms avg ({dec_arr.std():5.1f} ms std)")
    print(f"  Latent step: {pf_arr.mean():6.1f} ms forward only")
    print(f"  Wall step:   {gen_elapsed / n * 1000:6.1f} ms / frame (incl. decode)")
    print(f"  NFE:         {gen_elapsed / n / nfe * 1000:.1f} ms wall ({lfps * nfe:.1f}/s)")
    print(f"  LFPS:        {lfps:.2f}")
    print(f"  Video FPS:   {lfps * _TEMPORAL_COMPRESS:.2f}  (×{_TEMPORAL_COMPRESS} temporal)")
    print(f"  total:       {gen_elapsed:.2f}s")
    print(f"{'=' * 50}")

    if args.no_decode or (not latents and not pixel_chunks):
        # Diagnostic dump for offline cross-backend comparison.
        dump_path = os.environ.get("QUARK_DUMP_LATENTS")
        if dump_path and latents:
            latent_nps = [_np_to_f32(lt).reshape(1, C, H, W) for lt in latents]
            all_latents = np.concatenate(latent_nps, axis=0)
            np.save(dump_path, all_latents)
            print(f"  [diag] saved latents to {dump_path} {all_latents.shape}")
        return

    out_lower = args.output.lower()
    is_video = out_lower.endswith((".mp4", ".mov", ".mkv"))
    is_npy = out_lower.endswith(".npy")
    if not (is_video or is_npy):
        is_video = True

    # Make sure the output directory exists before any write paths run.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if not pixel_chunks:
        latent_nps = [_np_to_f32(lt).reshape(1, C, H, W) for lt in latents]
        all_latents = np.concatenate(latent_nps, axis=0)
        fallback = args.output.rsplit(".", 1)[0] + ".npy"
        np.save(fallback, all_latents)
        print(f"  AE unavailable; saved latents to {fallback}")
        return

    pixels = np.concatenate(pixel_chunks, axis=0)
    if is_npy:
        np.save(args.output, pixels)
        print(f"saved {args.output} ({pixels.shape})")
        return

    # Try HEVC mp4; fall back to PNG frame dump if ffmpeg isn't on PATH so
    # we always produce something to inspect visually.
    try:
        _write_video_hevc(pixels, args.output, fps=args.fps, crf=args.crf)
    except RuntimeError as e:
        print(f"  {e}")
        out_dir = args.output.rsplit(".", 1)[0] + "_frames"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        try:
            import cv2

            for i, fr in enumerate(pixels):
                cv2.imwrite(f"{out_dir}/frame_{i:04d}.png", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        except ImportError:
            from PIL import Image

            for i, fr in enumerate(pixels):
                Image.fromarray(fr).save(f"{out_dir}/frame_{i:04d}.png")
        print(f"saved {len(pixels)} PNGs to {out_dir}/")


if __name__ == "__main__":
    main()
