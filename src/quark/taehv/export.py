#!/usr/bin/env python3
"""One-time CoreML export of TAEHV encoder + decoder for ANE inference.

Loads the upstream TAEHV PyTorch model (from the ``taehv`` PyPI
package) and traces it into CoreML ``.mlpackage`` artifacts that
``quark.taehv.load_taehv`` reads at runtime. Run once per
``(latent_height, latent_width)`` pair; the artifacts cache to
``./taehv_cache/<latH>x<latW>/`` (override with ``--cache-dir``)
and never need re-exporting unless the upstream TAEHV changes.
The cache directory is keyed on the latent dims (matching the CLI
args + model-config fields) — pre-2026-05-06 builds keyed it on
the encoder pixel dim ``8 × latH × 8 × latW``; rename or re-export
existing dirs to migrate.

The runtime path (``quark.taehv``) does NOT import torch or
``taehv``. This module is the only torch-touching code in the
quark tree — ``quark.taehv.__init__`` deliberately does NOT import
it, so installing the runtime ``[taehv]`` extras leaves torch off
the dependency graph. Pulled in only by ``[export]``.

Architecture:
  * Encoder is stateless — straight ``torch.jit.trace`` → ``ct.convert``.
  * Decoder is stateful via 3 MemBlock buffers. CoreML ``StateType``
    fails to compile on ANE (error -14 regardless of state count),
    so we wrap the decoder to take state as regular inputs and
    return updated state as regular outputs. This matches the
    explicit-I/O-state design used in the legacy ``world_engine``
    wrapper that quark replaces here.

Usage:
    uv run --extra export python -m quark.taehv.export \\
        --ae-repo Overworld-Models/taehv1_5 \\
        --latent-height 16 --latent-width 32

To export a different resolution (e.g. 720p):
    --latent-height 32 --latent-width 64
"""

from __future__ import annotations

import argparse
import os
import pathlib

import coremltools as ct
import torch
import torch.nn as nn
import torch.nn.functional as F
from taehv import TAEHV

# ────────────────────────────────────────────────────────────────────
# Encoder — stateless, traced as a flat sequence of ``taehv.encoder``
# blocks. Hardcoded to ``taehv1_5`` (T=4, patch_size=2). If you're
# exporting a different TAEHV variant, you'll need to update the
# block indices below.
# ────────────────────────────────────────────────────────────────────


class _EncoderStatic(nn.Module):
    """Stateless encoder.

    Input:  ``[4, 12, H, W]`` (4 frames after pixel_unshuffle(2))
    Output: ``[1, 32, H/8, W/8]``  (1 latent for the 4-frame chunk)
    """

    def __init__(self, taehv: TAEHV, h: int, w: int):
        super().__init__()
        self.blocks = nn.ModuleList(list(taehv.encoder))
        self._h, self._w = h, w

    def forward(self, x):
        h, w = self._h, self._w
        x = self.blocks[0](x)  # Conv [4, 64, h, w]
        x = self.blocks[1](x)  # ReLU

        x = self.blocks[2].conv(x.reshape(2, 128, h, w))  # ty: ignore[call-non-callable]  # TPool(2): 4→2
        x = self.blocks[3](x)  # Conv stride=2: [2, 64, h/2, w/2]

        for i in (4, 5, 6):
            past = torch.cat([torch.zeros_like(x[:1]), x[:-1]], dim=0)
            x = self.blocks[i](x, past)

        x = self.blocks[7].conv(x.reshape(1, 128, h // 2, w // 2))  # ty: ignore[call-non-callable]  # TPool(2): 2→1
        x = self.blocks[8](x)  # Conv stride=2: [1, 64, h/4, w/4]

        for i in (9, 10, 11):
            x = self.blocks[i](x, torch.zeros_like(x))

        x = self.blocks[12].conv(x)  # ty: ignore[call-non-callable]  # TPool(1): no change
        x = self.blocks[13](x)  # Conv stride=2: [1, 64, h/8, w/8]

        for i in (14, 15, 16):
            x = self.blocks[i](x, torch.zeros_like(x))

        return self.blocks[17](x)  # Conv 64→32: [1, 32, h/8, w/8]


# ────────────────────────────────────────────────────────────────────
# Decoder — stateful via explicit I/O state. CoreML ``StateType``
# fails on ANE (error -14 regardless of state count), so we pass
# the 3 MemBlock buffers as regular inputs and return their updated
# values as regular outputs. ``torch.cat`` (rather than zeros + scatter)
# is ~40 ms faster on ANE in this graph.
# ────────────────────────────────────────────────────────────────────


class _DecoderExplicitState(nn.Module):
    """Stateful decoder, ANE-compatible.

    Inputs:  ``x [1, 32, latH, latW]``,
             ``state_lo  [3, 256, latH,    latW]``,
             ``state_mid [3, 128, latH*2,  latW*2]``,
             ``state_hi  [3,  64, latH*4,  latW*4]``
    Outputs: ``frames [4, 3, latH*16, latW*16]``,
             updated state_lo / state_mid / state_hi
    """

    def __init__(self, taehv: TAEHV, lat_h: int, lat_w: int):
        super().__init__()
        self.blocks = nn.ModuleList(list(taehv.decoder))
        # Spatial dims at each decoder resolution level.
        self._h2, self._w2 = lat_h * 4, lat_w * 4  # after 2× upsample twice
        self._h4, self._w4 = lat_h * 8, lat_w * 8  # after 3× upsample

    def forward(self, x, state_lo, state_mid, state_hi):
        x = self.blocks[0](x)  # Clamp
        x = self.blocks[1](x)  # Conv 32→256
        x = self.blocks[2](x)  # ReLU

        # Group 1 (blocks 3-5, T=1): save INPUT to each block as new state.
        save_3 = x
        x = self.blocks[3](x, state_lo[0:1])
        save_4 = x
        x = self.blocks[4](x, state_lo[1:2])
        save_5 = x
        x = self.blocks[5](x, state_lo[2:3])
        new_lo = torch.cat([save_3, save_4, save_5], dim=0)

        x = self.blocks[6](x)  # Upsample(2)
        x = self.blocks[7].conv(x)  # ty: ignore[call-non-callable]  # TGrow(1)
        x = self.blocks[8](x)  # Conv 256→128

        # Group 2 (blocks 9-11, T=1)
        save_9 = x
        x = self.blocks[9](x, state_mid[0:1])
        save_10 = x
        x = self.blocks[10](x, state_mid[1:2])
        save_11 = x
        x = self.blocks[11](x, state_mid[2:3])
        new_mid = torch.cat([save_9, save_10, save_11], dim=0)

        x = self.blocks[12](x)  # Upsample(2)
        x = self.blocks[13].conv(x)  # ty: ignore[call-non-callable]  # TGrow(2): 1→2
        x = x.reshape(2, 128, self._h2, self._w2)
        x = self.blocks[14](x)  # Conv 128→64

        # Group 3 (blocks 15-17, T=2): save LAST frame's input.
        past = torch.cat([state_hi[0:1], x[:1]], dim=0)
        save_15 = x[1:2]
        x = self.blocks[15](x, past)

        past = torch.cat([state_hi[1:2], x[:1]], dim=0)
        save_16 = x[1:2]
        x = self.blocks[16](x, past)

        past = torch.cat([state_hi[2:3], x[:1]], dim=0)
        save_17 = x[1:2]
        x = self.blocks[17](x, past)
        new_hi = torch.cat([save_15, save_16, save_17], dim=0)

        x = self.blocks[18](x)  # Upsample(2)
        x = self.blocks[19].conv(x)  # ty: ignore[call-non-callable]  # TGrow(2): 2→4
        x = x.reshape(4, 64, self._h4, self._w4)
        x = self.blocks[20](x)  # Conv 64→64
        x = self.blocks[21](x)  # ReLU
        x = self.blocks[22](x)  # Conv 64→12

        x = F.pixel_shuffle(x, 2)  # [4, 3, H_out, W_out]
        x = x.clamp(0, 1)

        return x, new_lo, new_mid, new_hi


# ────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────


def _resolve_taehv_checkpoint(ae_repo: str) -> pathlib.Path:
    """Snapshot-download the TAEHV repo from HF and return the
    .pth checkpoint path. Falls back to treating ``ae_repo`` as a
    local path."""
    try:
        import huggingface_hub
    except ImportError:
        # Soft-fail: pure local path is still allowed.
        base = pathlib.Path(ae_repo)
    else:
        try:
            base = pathlib.Path(huggingface_hub.snapshot_download(ae_repo))
        except Exception:
            base = pathlib.Path(ae_repo)
    if base.is_file():
        return base
    return base / "taehv1_5.pth"


def export(
    ae_repo: str,
    *,
    latent_height: int,
    latent_width: int,
    cache_dir: str,
) -> tuple[str, str]:
    """Run the export. Returns ``(encoder_path, decoder_path)``.

    The output directory is keyed on the latent dims (``<latH>x<latW>``)
    so the cache layout matches the runtime ``load_taehv`` lookup, the
    export CLI args, and the model-config fields. The 8× to encoder
    pixel space happens internally below.
    """
    enc_h = latent_height * 8
    enc_w = latent_width * 8
    out_dir = pathlib.Path(cache_dir) / f"{latent_height}x{latent_width}"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_path = out_dir / "taehv_encoder.mlpackage"
    dec_path = out_dir / "taehv_decoder_ane.mlpackage"

    if enc_path.exists() and dec_path.exists():
        print(
            f"[export_taehv] {out_dir} already populated — skipping. "
            f"(Delete the .mlpackage dirs to force re-export.)"
        )
        return str(enc_path), str(dec_path)

    print(f"[export_taehv] Loading TAEHV checkpoint for {ae_repo!r}…")
    ckpt = _resolve_taehv_checkpoint(ae_repo)
    taehv = TAEHV(str(ckpt)).eval().to(torch.float32)

    # Encoder ───────────────────────────────────────────────────────
    if not enc_path.exists():
        print(f"[export_taehv] Tracing encoder ({enc_h}×{enc_w} input)…")
        enc = _EncoderStatic(taehv, h=enc_h, w=enc_w).eval()
        with torch.no_grad():
            traced = torch.jit.trace(
                enc,
                torch.randn(4, 12, enc_h, enc_w),
            )
        ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=(4, 12, enc_h, enc_w))],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        ).save(str(enc_path))
        print(f"[export_taehv]   wrote {enc_path}")

    # Decoder ───────────────────────────────────────────────────────
    if not dec_path.exists():
        print(f"[export_taehv] Tracing decoder (latent {latent_height}×{latent_width})…")
        dec = _DecoderExplicitState(
            taehv,
            lat_h=latent_height,
            lat_w=latent_width,
        ).eval()
        dummy = (
            torch.randn(1, 32, latent_height, latent_width),
            torch.zeros(3, 256, latent_height, latent_width),
            torch.zeros(3, 128, latent_height * 2, latent_width * 2),
            torch.zeros(3, 64, latent_height * 4, latent_width * 4),
        )
        with torch.no_grad():
            traced = torch.jit.trace(dec, dummy, strict=False)
        ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="x",
                    shape=(1, 32, latent_height, latent_width),
                ),
                ct.TensorType(
                    name="state_lo",
                    shape=(3, 256, latent_height, latent_width),
                ),
                ct.TensorType(
                    name="state_mid",
                    shape=(3, 128, latent_height * 2, latent_width * 2),
                ),
                ct.TensorType(
                    name="state_hi",
                    shape=(3, 64, latent_height * 4, latent_width * 4),
                ),
            ],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        ).save(str(dec_path))
        print(f"[export_taehv]   wrote {dec_path}")

    return str(enc_path), str(dec_path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export TAEHV encoder + decoder to CoreML "
        ".mlpackage artifacts for ANE inference. Run once per "
        "(latent_height, latent_width) pair; quark.taehv reuses the "
        "cached artifacts forever after.",
    )
    p.add_argument(
        "--ae-repo",
        default="Overworld-Models/taehv1_5",
        help="HuggingFace TAEHV repo (or local path to a checkpoint dir / .pth).",
    )
    p.add_argument(
        "--latent-height",
        type=int,
        required=True,
        help="Latent grid height (16 for 360p, 32 for 720p).",
    )
    p.add_argument(
        "--latent-width",
        type=int,
        required=True,
        help="Latent grid width (32 for 360p, 64 for 720p).",
    )
    p.add_argument(
        "--cache-dir",
        default=os.path.join(os.getcwd(), "taehv_cache"),
        help="Where to write the .mlpackage artifacts. Default is "
        "./taehv_cache under the current working directory.",
    )
    args = p.parse_args()
    enc, dec = export(
        args.ae_repo,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        cache_dir=args.cache_dir,
    )
    print(f"[export_taehv] Done.\n  encoder: {enc}\n  decoder: {dec}")


if __name__ == "__main__":
    main()
