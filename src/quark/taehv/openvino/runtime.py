"""OpenVINO TAEHV runtime — Intel iGPU / dGPU / Arc / Battlemage.

API parity with :class:`quark.taehv.coreml.runtime.CoreMLTAEHV`:
``encode(img_np)`` / ``decode(latent_np)`` operate on plain numpy.
State (the three MemBlock buffers ``state_lo`` / ``state_mid`` /
``state_hi``) is owned by this object and threaded through each
decode call as explicit I/O — same shape as the CoreML side.

Device targets accepted by ``device=``:
  * ``"GPU"`` — Intel iGPU / dGPU / Arc / Battlemage via Level Zero
    or OpenCL. Default on systems where OpenVINO sees a GPU.
  * ``"CPU"`` — software fallback; useful for correctness baselining
    or when GPU drivers (Level Zero loader / intel-opencl-icd)
    aren't installed.
  * ``"NPU"`` — Intel Lunar/Meteor/Panther Lake NPU.
  * ``"AUTO"`` — OpenVINO picks the best device.

Tested on Battlemage (Panther Lake B390) with the ``GPU`` plugin.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class OpenVINOTAEHV:
    """Drop-in counterpart to ``CoreMLTAEHV`` on Intel hardware."""

    # Same pixel-space ↔ encoder-input maps as CoreML side. Native
    # training aspect ratios.
    _ENCODE_SIZES: ClassVar[dict[tuple[int, int], tuple[int, int]]] = {
        (720, 1280): (512, 1024),
        (360, 640): (256, 512),
    }
    _DECODE_SIZES: ClassVar[dict[tuple[int, int], tuple[int, int]]] = {
        v: k for k, v in _ENCODE_SIZES.items()
    }

    def __init__(
        self,
        encoder_xml: str,
        decoder_xml: str,
        *,
        latent_height: int,
        latent_width: int,
        device: str = "GPU",
    ):
        import openvino as ov

        self._device = device
        self._core = ov.Core()
        avail = set(self._core.available_devices)
        # ``AUTO`` is OpenVINO's own device-picker — it never fails to
        # resolve, but we still emit a one-line note so callers know
        # which device it actually picked.
        if device in ("GPU", "NPU") and device not in avail:
            # Auto-fall-back to CPU with a clear warning; matches the
            # CoreML side's behavior when ANE init fails.
            import warnings

            install_hint = (
                "To enable Intel GPU, install `intel-level-zero-gpu`, "
                "`level-zero`, `intel-opencl-icd` (apt)."
                if device == "GPU"
                else "NPU device needs `intel-driver-compiler-npu` + "
                "`intel-level-zero-npu` (apt) on Lunar/Meteor/Panther Lake."
            )
            warnings.warn(
                f"OpenVINO sees devices={sorted(avail)}; {device} not "
                f"available. Falling back to CPU. {install_hint}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._device = "CPU"

        # Compile both networks once. ``compile_model`` does the
        # device-specific optimization passes + kernel selection.
        # No per-call recompile. ``PERFORMANCE_HINT=LATENCY`` tells
        # OpenVINO to optimize for one-stream low-latency inference
        # (the right hint for our per-frame loop) rather than the
        # default which leans toward throughput.
        #
        # ``INFERENCE_NUM_THREADS``: cap how many CPU cores the AE
        # claims so the main thread's SPV dispatch (Python + Vulkan
        # cmd-buffer record) keeps enough host cycles. Tunable via
        # ``QUARK_TAEHV_OPENVINO_THREADS`` env var; default 4 is a
        # reasonable middle on an 8-core host (leaves 4 for SPV).
        # Only applied on CPU device (GPU plugin manages its own).
        import os as _os

        config = {
            "PERFORMANCE_HINT": "LATENCY",
        }
        # Precision hint: f32 wins on Intel CPU (Panther Lake / Arrow
        # Lake / Battlemage iGPU host) — measured 32 ms (f32) vs 45 ms
        # (f16) for the TAEHV decoder. CPU SIMD path is better tuned
        # for fp32 throughput than the fp16-emulation path. On GPU
        # device the trade-off may flip; keep the env var as an
        # override (``QUARK_TAEHV_OPENVINO_PRECISION=f16`` to try).
        _prec_hint = _os.environ.get("QUARK_TAEHV_OPENVINO_PRECISION")
        if _prec_hint:
            config["INFERENCE_PRECISION_HINT"] = _prec_hint
        if self._device == "CPU":
            try:
                config["INFERENCE_NUM_THREADS"] = int(
                    _os.environ.get("QUARK_TAEHV_OPENVINO_THREADS", "4")
                )
            except ValueError:
                pass
            # NUM_STREAMS: 1 (default) means single-stream latency
            # mode. Set higher (e.g. 2) only when pipelining multiple
            # AE decodes concurrently — for our single-depth
            # PipelinedDecoder, 1 is optimal. Loop iter 18 confirmed
            # ``NUM_STREAMS=2`` doesn't help in our config (see memory
            # ``project_openvino_taehv.md``).
            _streams = _os.environ.get("QUARK_TAEHV_OPENVINO_STREAMS")
            if _streams:
                try:
                    config["NUM_STREAMS"] = int(_streams)
                except ValueError:
                    pass
        self._encoder = self._core.compile_model(
            self._core.read_model(encoder_xml), self._device, config
        )
        self._decoder = self._core.compile_model(
            self._core.read_model(decoder_xml), self._device, config
        )
        # Persistent InferRequest objects — avoid the per-call
        # ``create_infer_request`` overhead and enable us to keep
        # input/output tensors pinned across calls (saves the
        # per-call ``np.asarray`` re-wrap for state buffers).
        self._enc_req = self._encoder.create_infer_request()
        self._dec_req = self._decoder.create_infer_request()

        pix_h = latent_height * 16
        pix_w = latent_width * 16
        self._target_encode_size: tuple[int, int] = (pix_h, pix_w)
        self._enc_h, self._enc_w = pix_h // 2, pix_w // 2
        self._lat_h, self._lat_w = latent_height, latent_width

        # Output port handles; we resolve them via index for stability
        # across OpenVINO versions (names can mangle on save/load).
        self._enc_out_port = self._encoder.output(0)
        # Decoder outputs: resolve indices by shape match at warmup
        # time. Trace order is (frames, new_lo, new_mid, new_hi) but
        # OpenVINO may reorder during convert/serialize.
        self._dec_frames_idx = 0
        self._dec_lo_idx = 1
        self._dec_mid_idx = 2
        self._dec_hi_idx = 3

        self._primed = False
        self._warmup()

    # ── State ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset MemBlock state. Call on every new rollout / scene
        change. Encode is stateless."""
        h, w = self._lat_h, self._lat_w
        self._state_lo = np.zeros((3, 256, h, w), dtype=np.float32)
        self._state_mid = np.zeros((3, 128, h * 2, w * 2), dtype=np.float32)
        self._state_hi = np.zeros((3, 64, h * 4, w * 4), dtype=np.float32)
        self._primed = False

    def _warmup(self) -> None:
        """Burn one encode + one decode at startup so the runtime
        warm-up + autotuning happens before the timed loop. Also
        resolves the decoder output ports by shape."""
        # Initialize state buffers so the shape-resolution loop below
        # has shapes to match against.
        h, w = self._lat_h, self._lat_w
        self.reset()
        _enc_out = self._encoder(
            {"x": np.zeros((4, 12, self._enc_h, self._enc_w), dtype=np.float32)}
        )
        # Resolve decoder output indices by shape (matches CoreML
        # backend's strategy — robust against IR output reordering
        # across export runs).
        dec_pred = self._decoder([
            np.zeros((1, 32, h, w), dtype=np.float32),
            np.zeros((3, 256, h, w), dtype=np.float32),
            np.zeros((3, 128, h * 2, w * 2), dtype=np.float32),
            np.zeros((3, 64, h * 4, w * 4), dtype=np.float32),
        ])
        # Resolve decoder output indices by shape — iterate the
        # ``outputs`` list directly (its ordering is the canonical
        # index space we'll use in ``_decode_raw``).
        for i, port in enumerate(self._decoder.outputs):
            shape = tuple(port.get_partial_shape().to_shape())
            if shape == (4, 3, h * 16, w * 16):
                self._dec_frames_idx = i
            elif shape == self._state_lo.shape:
                self._dec_lo_idx = i
            elif shape == self._state_mid.shape:
                self._dec_mid_idx = i
            elif shape == self._state_hi.shape:
                self._dec_hi_idx = i
        self.reset()

    # ── Encode ─────────────────────────────────────────────────────

    def encode(self, img_np: np.ndarray) -> np.ndarray:
        """``[4, H_pix, W_pix, 3] uint8`` → ``[1, 32, latH, latW] f32``."""
        if img_np.dtype != np.uint8:
            raise TypeError(f"encode: expected uint8, got {img_np.dtype}")
        if img_np.ndim != 4 or img_np.shape[0] != 4 or img_np.shape[-1] != 3:
            raise ValueError(f"encode: expected [4, H, W, 3] uint8, got {img_np.shape}")

        rgb = img_np.astype(np.float32) / 255.0
        rgb = rgb.transpose(0, 3, 1, 2)[None]  # [1, 4, 3, H, W]

        target = self._target_encode_size
        if rgb.shape[-2:] != target:
            rgb = _resize_f32_bilinear(rgb, target)

        rgb_4d = rgb.reshape(4, 3, rgb.shape[-2], rgb.shape[-1])
        enc_input = _pixel_unshuffle(rgb_4d, 2).astype(np.float32)

        import openvino as ov

        self._enc_req.set_input_tensor(0, ov.Tensor(enc_input))
        self._enc_req.infer()
        return np.array(self._enc_req.get_output_tensor(0).data, copy=True)

    # ── Decode ─────────────────────────────────────────────────────

    def decode(self, latent_np: np.ndarray) -> np.ndarray:
        """``[1, 32, latH, latW] f16/f32`` → ``[4, H_pix, W_pix, 3] uint8``."""
        if latent_np.ndim != 4 or latent_np.shape[1] != 32:
            raise ValueError(f"decode: expected [1, 32, latH, latW], got {latent_np.shape}")
        if latent_np.dtype != np.float32:
            latent_np = latent_np.astype(np.float32)

        if not self._primed:
            for _ in range(3):
                self._decode_raw(latent_np)
            self._primed = True

        frames_np = self._decode_raw(latent_np)  # [4, 3, dec_h, dec_w] f32

        frames_np = (np.clip(frames_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        frames_np = frames_np.transpose(0, 2, 3, 1)  # [4, H, W, 3]

        decoder_dim = (frames_np.shape[1], frames_np.shape[2])
        if decoder_dim in self._DECODE_SIZES:
            target = self._DECODE_SIZES[decoder_dim]
            if target != decoder_dim:
                frames_np = _resize_uint8_bilinear(frames_np, target)
        return frames_np

    # ── Internal ───────────────────────────────────────────────────

    def _decode_raw(self, lat_np: np.ndarray) -> np.ndarray:
        # Persistent InferRequest: set inputs by index, infer, read
        # outputs. Trace order is (x, state_lo, state_mid, state_hi);
        # see ``DecoderExplicitState.forward``.
        import openvino as ov

        req = self._dec_req
        req.set_input_tensor(0, ov.Tensor(lat_np.astype(np.float32, copy=False)))
        req.set_input_tensor(1, ov.Tensor(self._state_lo))
        req.set_input_tensor(2, ov.Tensor(self._state_mid))
        req.set_input_tensor(3, ov.Tensor(self._state_hi))
        req.infer()
        # ``get_output_tensor`` returns a view over OpenVINO-owned
        # storage — copy ONCE into our state buffers; that copy is
        # unavoidable since the next infer would overwrite the same
        # output buffer.
        frames = np.array(req.get_output_tensor(self._dec_frames_idx).data, copy=True)
        # State write-back: in-place copy into self._state_* preserves
        # buffer identity so the next infer's ``set_input_tensor`` can
        # potentially reuse it (subject to OpenVINO's own zero-copy
        # heuristics).
        np.copyto(self._state_lo, req.get_output_tensor(self._dec_lo_idx).data)
        np.copyto(self._state_mid, req.get_output_tensor(self._dec_mid_idx).data)
        np.copyto(self._state_hi, req.get_output_tensor(self._dec_hi_idx).data)
        return frames


# ── numpy helpers (lifted from coreml.runtime; identical math) ─────


def _pixel_unshuffle(x: np.ndarray, r: int) -> np.ndarray:
    N, C, H, W = x.shape
    if H % r or W % r:
        raise ValueError(f"pixel_unshuffle: H, W must be multiples of {r} (got {H}, {W})")
    x = x.reshape(N, C, H // r, r, W // r, r)
    x = x.transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(N, C * r * r, H // r, W // r)


def _resize_uint8_bilinear(frames: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    H_t, W_t = target
    out = np.empty((frames.shape[0], H_t, W_t, 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        im = Image.fromarray(frames[i])
        out[i] = np.asarray(im.resize((W_t, H_t), Image.BILINEAR))
    return out


def _resize_f32_bilinear(x: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    H_t, W_t = target
    N, T, C, H, W = x.shape
    out = np.empty((N, T, C, H_t, W_t), dtype=np.float32)
    for n in range(N):
        for t in range(T):
            for c in range(C):
                plane = x[n, t, c]
                im = Image.fromarray(plane, mode="F").resize((W_t, H_t), Image.BILINEAR)
                out[n, t, c] = np.asarray(im, dtype=np.float32)
    return out
