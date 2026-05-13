"""CoreML TAEHV runtime — numpy-only, no torch.

Wraps the two ``.mlpackage`` artifacts produced by
``quark.taehv.export`` (encoder + decoder) and exposes
``encode(img_np)`` / ``decode(latent_np)`` that operate on plain
numpy arrays. Internal state (the three MemBlock buffers
``state_lo`` / ``state_mid`` / ``state_hi``) is also numpy.

Why explicit-state instead of CoreML's built-in ``StateType``: the
state-tensor API fails to compile on ANE with error -14 regardless
of state count — passing the same buffers as regular model
inputs/outputs is the only working ANE-compatible path. (Same
trick world_engine uses; see the README for the dead-ends list.)

Aspect-ratio convention matches the legacy world_engine wrapper:

  * ``_ENCODE_SIZES``: pixel-space input → encoder-input size after
    bilinear resize. (1280, 720) and (640, 360) are the model's
    native training aspect ratios.
  * ``_DECODE_SIZES``: decoder-output size → final pixel-space size.
    Inverse map; same numbers, just the keys / values swapped.

The bilinear ops use Pillow (``Image.BILINEAR``) instead of
``torch.nn.functional.interpolate``. Pillow uses pixel-center
sample positions which match ``align_corners=False`` to ±1 LSB on
uint8 — verified on the bench seed image.

Performance under concurrent GPU load (tested on M5 Max):
``decode()`` runs ~12 ms in isolation but inflates to ~60 ms when
the world-model GPU work is concurrently saturating the SoC.
Verified by comparing the same ``predict()`` under load vs the
trailing decode call after GPU work ends. The inflation is the
same with ``compute_units="CPU_AND_NE"`` (113/113 compute ops on
ANE per ``MLComputePlan``) as with ``"CPU_AND_GPU"`` (113/113 on
GPU) — so the bottleneck is system-wide SoC budget / fabric
contention, not engine-specific.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class CoreMLTAEHV:
    """Drop-in replacement for the legacy
    ``ChunkedStreamingTAEHV`` torch path. Encode / decode operate on
    numpy arrays only; state is owned by this object and managed
    between ``predict`` calls.
    """

    # Pixel-space → encoder-input dims (after the encoder's internal
    # ``F.pixel_unshuffle(2)`` halving). Native training aspect ratios.
    # ``ClassVar`` because these are constant lookup tables, not
    # per-instance state.
    _ENCODE_SIZES: ClassVar[dict[tuple[int, int], tuple[int, int]]] = {
        (720, 1280): (512, 1024),
        (360, 640): (256, 512),
    }
    _DECODE_SIZES: ClassVar[dict[tuple[int, int], tuple[int, int]]] = {
        v: k for k, v in _ENCODE_SIZES.items()
    }

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        *,
        latent_height: int,
        latent_width: int,
        compute_units: str = "CPU_AND_NE",
    ):
        import coremltools as ct

        if compute_units not in ("CPU_AND_NE", "CPU_AND_GPU"):
            raise ValueError(
                f"compute_units must be 'CPU_AND_NE' or 'CPU_AND_GPU' "
                f"(got {compute_units!r}); CPU-only is intentionally not "
                f"a supported runtime target."
            )
        cu = getattr(ct.ComputeUnit, compute_units)

        # CoreML model load is sequential; ANE init can fail at this
        # point on hardware without a Neural Engine. We bubble that
        # up — the caller can retry with ``CPU_AND_GPU``.
        self.encoder_ml = ct.models.MLModel(encoder_path, compute_units=cu)
        self.decoder_ml = ct.models.MLModel(decoder_path, compute_units=cu)

        # Pixel-space + encoder-input + latent dims, all derived from
        # the latent grid. ``latent_height * 16`` is the AE's pixel
        # output size (8× from the encoder's spatial compression
        # times 2× from pixel_shuffle on the way out).
        pix_h = latent_height * 16
        pix_w = latent_width * 16
        self._target_encode_size: tuple[int, int] = (pix_h, pix_w)
        self._enc_h, self._enc_w = pix_h // 2, pix_w // 2
        self._lat_h, self._lat_w = latent_height, latent_width

        self._enc_key: str | None = None
        self._frames_key: str | None = None
        self._primed = False

        self._warmup()

    # ── State ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the per-decoder MemBlock state. Call on every new
        rollout / scene change. Encode is stateless."""
        h, w = self._lat_h, self._lat_w
        self._state_lo = np.zeros((3, 256, h, w), dtype=np.float16)
        self._state_mid = np.zeros((3, 128, h * 2, w * 2), dtype=np.float16)
        self._state_hi = np.zeros((3, 64, h * 4, w * 4), dtype=np.float16)
        self._primed = False

    def _warmup(self) -> None:
        """Burn one encode + one decode at startup so the ANE compile
        happens before the timed bench loop. Also discovers the
        actual output key names (CoreML mangles them per export)."""
        h, w = self._lat_h, self._lat_w
        enc_pred = self.encoder_ml.predict(
            {
                "x": np.zeros((4, 12, self._enc_h, self._enc_w), dtype=np.float16),
            }
        )
        # Encoder has a single output; CoreML may name it any one of
        # several legal identifiers, so just take the first.
        self._enc_key = next(iter(enc_pred))

        dec_pred = self.decoder_ml.predict(
            {
                "x": np.zeros((1, 32, h, w), dtype=np.float16),
                "state_lo": np.zeros((3, 256, h, w), dtype=np.float16),
                "state_mid": np.zeros((3, 128, h * 2, w * 2), dtype=np.float16),
                "state_hi": np.zeros((3, 64, h * 4, w * 4), dtype=np.float16),
            }
        )
        # The frames output is the only one with leading dim == 4.
        for k, v in dec_pred.items():
            if v.shape[0] == 4:
                self._frames_key = k
                break
        if self._frames_key is None:
            raise RuntimeError(
                "TAEHV decoder didn't return any output with leading "
                "dim 4 — the .mlpackage may be from an incompatible "
                "export. Re-run python -m quark.taehv.export."
            )
        self.reset()

    # ── Encode (one-shot, used at seed time) ───────────────────────

    def encode(self, img_np: np.ndarray) -> np.ndarray:
        """``[4, H_pix, W_pix, 3] uint8`` → ``[1, 32, latH, latW] f32``.

        Equivalent to ``ChunkedStreamingTAEHV.encode`` but on numpy
        only. The encoder is stateless so this can be called any
        number of times without affecting the decoder state.
        """
        if img_np.dtype != np.uint8:
            raise TypeError(f"encode: expected uint8, got {img_np.dtype}")
        if img_np.ndim != 4 or img_np.shape[0] != 4 or img_np.shape[-1] != 3:
            raise ValueError(f"encode: expected [4, H, W, 3] uint8, got {img_np.shape}")

        # [4, H, W, 3] uint8 → [1, 4, 3, H, W] f32 in [0, 1].
        rgb = img_np.astype(np.float32) / 255.0
        rgb = rgb.transpose(0, 3, 1, 2)[None]  # [1, 4, 3, H, W]

        # Resize to the encoder's training aspect-corrected dim.
        target = self._target_encode_size
        if rgb.shape[-2:] != target:
            rgb = self._resize_f32_bilinear(rgb, target)

        # [1, 4, 3, H, W] → [4, 3, H, W], pixel_unshuffle(2) →
        # [4, 12, H/2, W/2].
        rgb_4d = rgb.reshape(4, 3, rgb.shape[-2], rgb.shape[-1])
        enc_input = self._pixel_unshuffle(rgb_4d, 2)

        pred = self.encoder_ml.predict({"x": enc_input.astype(np.float16)})
        return pred[self._enc_key].astype(np.float32)

    # ── Decode (per-frame, hot path) ───────────────────────────────

    def decode(self, latent_np: np.ndarray) -> np.ndarray:
        """``[1, 32, latH, latW] f16/f32`` → ``[4, H_pix, W_pix, 3] uint8``.

        Returns the 4 temporally-upscaled video frames for one
        latent. State (``_state_*``) is updated in-place.
        """
        if latent_np.ndim != 4 or latent_np.shape[1] != 32:
            raise ValueError(f"decode: expected [1, 32, latH, latW], got {latent_np.shape}")
        if latent_np.dtype != np.float16:
            latent_np = latent_np.astype(np.float16)

        # Prime on first decode — the streaming TAEHV trims the first
        # 3 latents to fill the MemBlock state. We replicate that
        # warmup so the first user-visible frame is steady-state.
        if not self._primed:
            for _ in range(3):
                self._decode_raw(latent_np)
            self._primed = True

        frames_np = self._decode_raw(latent_np)  # [4, 3, dec_h, dec_w] f16

        # f16 [0, 1] → uint8 [0, 255], then transpose to channels-last
        # for downstream image consumers (PNG save, mosaic, etc.).
        frames_np = frames_np.astype(np.float32)
        frames_np = (np.clip(frames_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        frames_np = frames_np.transpose(0, 2, 3, 1)  # [4, H, W, 3]

        # Resize back to the requested pixel-space aspect ratio.
        # ``decoder_dim`` is the dim TAEHV's decoder produces; if it
        # differs from the (training-corrected) output size, resize.
        decoder_dim = (frames_np.shape[1], frames_np.shape[2])
        if decoder_dim in self._DECODE_SIZES:
            target = self._DECODE_SIZES[decoder_dim]
            if target != decoder_dim:
                frames_np = self._resize_uint8_bilinear(frames_np, target)
        return frames_np

    # ── Internal: CoreML predict + state update ────────────────────

    def _decode_raw(self, lat_np: np.ndarray) -> np.ndarray:
        """One CoreML decode call. Updates state in-place."""
        pred = self.decoder_ml.predict(
            {
                "x": lat_np,
                "state_lo": self._state_lo,
                "state_mid": self._state_mid,
                "state_hi": self._state_hi,
            }
        )
        # The state-output keys are mangled by CoreML during export
        # (they pick up a generic prefix like ``output_0``); resolve
        # by shape match.
        for _k, v in pred.items():
            if v.shape == self._state_lo.shape:
                self._state_lo = v
            elif v.shape == self._state_mid.shape:
                self._state_mid = v
            elif v.shape == self._state_hi.shape:
                self._state_hi = v
        return pred[self._frames_key]

    # ── Numpy / Pillow helpers (no torch) ──────────────────────────

    @staticmethod
    def _pixel_unshuffle(x: np.ndarray, r: int) -> np.ndarray:
        """``[N, C, H, W] → [N, C * r * r, H/r, W/r]`` — numpy
        implementation of ``F.pixel_unshuffle(x, r)`` with the same
        channel-ordering convention."""
        N, C, H, W = x.shape
        if H % r or W % r:
            raise ValueError(f"pixel_unshuffle: H, W must be multiples of {r} (got H={H}, W={W})")
        x = x.reshape(N, C, H // r, r, W // r, r)
        x = x.transpose(0, 1, 3, 5, 2, 4)
        return x.reshape(N, C * r * r, H // r, W // r)

    @staticmethod
    def _resize_uint8_bilinear(
        frames: np.ndarray,
        target: tuple[int, int],
    ) -> np.ndarray:
        """Per-frame Pillow bilinear resize of a ``[T, H, W, 3]
        uint8`` array to ``(target_h, target_w)``."""
        from PIL import Image

        H_t, W_t = target
        out = np.empty((frames.shape[0], H_t, W_t, 3), dtype=np.uint8)
        for i in range(frames.shape[0]):
            im = Image.fromarray(frames[i])
            # Pillow ``resize`` takes (W, H), opposite of numpy.
            out[i] = np.asarray(im.resize((W_t, H_t), Image.BILINEAR))
        return out

    @staticmethod
    def _resize_f32_bilinear(
        x: np.ndarray,
        target: tuple[int, int],
    ) -> np.ndarray:
        """``[1, T, 3, H, W] f32`` → ``[1, T, 3, H_t, W_t] f32`` via
        per-channel Pillow ``F``-mode (32-bit f32 grayscale) bilinear
        resize.

        Goes channel-by-channel rather than the obvious "convert
        uint8 → resize → convert back" because the uint8 round-trip
        loses ~1/255 of precision per pixel, and that quantization
        compounds through the encoder + decoder + 5-frame KV-cache
        rollout to cos=0.82 at f4 vs the torch-bilinear reference.
        Pillow ``F``-mode preserves f32 throughout the bilinear
        kernel.
        """
        from PIL import Image

        H_t, W_t = target
        N, T, C, H, W = x.shape
        out = np.empty((N, T, C, H_t, W_t), dtype=np.float32)
        for n in range(N):
            for t in range(T):
                for c in range(C):
                    plane = x[n, t, c]  # [H, W] f32
                    im = Image.fromarray(plane, mode="F").resize(
                        (W_t, H_t),
                        Image.BILINEAR,
                    )
                    out[n, t, c] = np.asarray(im, dtype=np.float32)
        return out
