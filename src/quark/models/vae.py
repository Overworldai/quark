"""Streaming TAEHV wrapper for ``quark.Engine``.

Adapts the vendored ``StreamingTAEHV`` (see ``quark.models.taehv``) to
the I/O layout the engine expects:

  * ``encode(img)``: ``[T, H, W, C] uint8`` → ``[1, C, h, w]`` latent
  * ``decode(latent)``: ``[1, C, h, w]`` → ``[T, H, W, C] uint8``

Auto aspect-ratio resize is on by default — encode input must be 720p
or 360p (16:9 → reshaped to 16:8 for the latent tile shape) and decode
output is resized back. ``auto_aspect_ratio=False`` works in raw AE
space.

Vendored from world_engine ``src/ae.py::ChunkedStreamingTAEHV`` and
trimmed to TAEHV-only (the legacy ``InferenceAE`` non-taehv path was
dropped — Waypoint-1.5 doesn't use it).
"""

from __future__ import annotations

import pathlib
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from quark.models.taehv import TAEHV, StreamingTAEHV


class ChunkedStreamingTAEHV:
    _ENCODE_SIZES: ClassVar[dict[tuple[int, int], tuple[int, int]]] = {
        (720, 1280): (512, 1024),
        (360, 640): (256, 512),
    }
    _DECODE_SIZES: ClassVar[dict[tuple[int, int], tuple[int, int]]] = {
        v: k for k, v in _ENCODE_SIZES.items()
    }

    def __init__(
        self,
        ae_model,
        auto_aspect_ratio: bool = True,
        device=None,
        dtype=torch.bfloat16,
        height: int | None = None,
        width: int | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.auto_aspect_ratio = auto_aspect_ratio
        scale = ae_model.patch_size * 2 ** sum(
            getattr(m, "stride", None) == (2, 2) for m in ae_model.encoder
        )
        self._img_size = (
            None if height is None else self._DECODE_SIZES[(height * scale, width * scale)]
        )
        self.streaming_ae_model = StreamingTAEHV(ae_model.eval().to(device=device, dtype=dtype))

    @classmethod
    def from_pretrained(cls, model_uri: str, auto_aspect_ratio: bool = True, **kwargs):
        try:
            import huggingface_hub

            base = pathlib.Path(huggingface_hub.snapshot_download(model_uri))
        except Exception:
            base = pathlib.Path(model_uri)

        ckpt = base if base.is_file() else base / "taehv1_5.pth"
        return cls(TAEHV(str(ckpt)), auto_aspect_ratio=auto_aspect_ratio, **kwargs)

    def reset(self) -> None:
        # Rebuild streaming state, reuse same weights model.
        self.streaming_ae_model = StreamingTAEHV(self.streaming_ae_model.taehv)

    def _resize(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        return F.interpolate(x[0], size=size, mode="bilinear", align_corners=False)[None]

    @torch.inference_mode()
    def encode(self, img: Tensor) -> Tensor:
        """``img``: ``[T, H, W, C]`` uint8 (T == ``t_downscale``).
        Returns latent ``[1, C, h, w]``."""
        t = self.streaming_ae_model.taehv.t_downscale
        assert img.dim() == 4 and img.shape[-1] == 3 and img.shape[0] == t, (
            f"Expected [{t}, H, W, 3] RGB uint8"
        )

        rgb = (
            img.unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .div(255)
        )

        if self.auto_aspect_ratio:
            if img.shape[1] * 16 != img.shape[2] * 9:
                raise ValueError(f"Expected 16:9 input, got {img.shape[1:3]}")
            rgb = self._resize(rgb, self._ENCODE_SIZES[self._img_size or img.shape[1:3]])

        return self.streaming_ae_model.encode(rgb).squeeze(1)

    @torch.inference_mode()
    def decode(self, latent: Tensor) -> Tensor:
        """``latent``: ``[1, C, h, w]``.
        Returns frames ``[T, H, W, C]`` uint8."""
        assert latent.dim() == 4, "Expected [B, C, h, w] latent tensor"

        z = latent.unsqueeze(1).to(device=self.device, dtype=self.dtype)

        if self.streaming_ae_model.n_frames_decoded == 0:
            for _ in range(self.streaming_ae_model.taehv.frames_to_trim):
                self.streaming_ae_model.decode(z)
                self.streaming_ae_model.flush_decoder()

        first = self.streaming_ae_model.decode(z)
        assert first is not None, "Expected decoded output after a latent"
        frames = [first, *self.streaming_ae_model.flush_decoder()]

        decoded = torch.cat(frames, dim=1)

        if self.auto_aspect_ratio:
            decoded = self._resize(
                decoded, self._img_size or self._DECODE_SIZES[decoded.shape[-2:]]
            )

        decoded = (decoded.clamp(0, 1) * 255).round().to(torch.uint8)
        return decoded.squeeze(0).permute(0, 2, 3, 1)[..., :3]
