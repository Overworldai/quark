#!/usr/bin/env python3
"""One-time OpenVINO export of TAEHV encoder + decoder for Intel iGPU /
dGPU / Arc / Battlemage inference.

Mirrors :mod:`quark.taehv.coreml.export` but targets OpenVINO IR
(``.xml`` + ``.bin``) via ``openvino.convert_model``. The PyTorch
trace classes (``EncoderStatic``, ``DecoderExplicitState``) come
from :mod:`quark.taehv._pt_traces`, shared with the CoreML path.

Run once per ``(latent_height, latent_width)``; artifacts cache to
``<cache_dir>/<latH>x<latW>/{encoder.{xml,bin}, decoder.{xml,bin}}``.

Usage:
    python -m quark.taehv.openvino.export \\
        --ae-repo Overworld-Models/taehv1_5 \\
        --latent-height 16 --latent-width 32 \\
        --cache-dir ./openvino_cache
"""

from __future__ import annotations

import argparse
import pathlib


def export(
    ae_repo: str,
    *,
    latent_height: int,
    latent_width: int,
    cache_dir: str,
    fp16: bool = True,
) -> tuple[str, str]:
    """Export the TAEHV encoder + decoder to OpenVINO IR.

    Returns ``(encoder_xml_path, decoder_xml_path)``. The ``.bin``
    weights live next to each ``.xml`` and are loaded by OpenVINO
    automatically when the ``.xml`` is read.
    """
    import openvino as ov
    import torch

    from quark.taehv._pt_traces import (
        DecoderExplicitState,
        EncoderStatic,
        load_taehv_pytorch,
    )

    enc_h = latent_height * 8
    enc_w = latent_width * 8
    out_dir = pathlib.Path(cache_dir) / f"{latent_height}x{latent_width}"
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_xml = out_dir / "encoder.xml"
    dec_xml = out_dir / "decoder.xml"

    if enc_xml.exists() and dec_xml.exists():
        print(f"[export_taehv_openvino] {out_dir} populated — skipping.")
        return str(enc_xml), str(dec_xml)

    print(f"[export_taehv_openvino] loading TAEHV checkpoint for {ae_repo!r}...")
    taehv = load_taehv_pytorch(ae_repo)
    target_dtype = ov.Type.f16 if fp16 else ov.Type.f32

    if not enc_xml.exists():
        print(f"[export_taehv_openvino] tracing encoder ({enc_h}x{enc_w})...")
        enc = EncoderStatic(taehv, h=enc_h, w=enc_w).eval()
        example_x = torch.randn(4, 12, enc_h, enc_w)
        with torch.no_grad():
            ov_model = ov.convert_model(
                enc,
                example_input=example_x,
                input=[ov.PartialShape([4, 12, enc_h, enc_w])],
            )
        # ``compress_to_fp16=True`` writes fp16 weights to disk while
        # keeping the runtime free to upcast for compute precision.
        ov.save_model(ov_model, str(enc_xml), compress_to_fp16=fp16)
        print(f"[export_taehv_openvino]   wrote {enc_xml}")

    if not dec_xml.exists():
        print(f"[export_taehv_openvino] tracing decoder (latent {latent_height}x{latent_width})...")
        dec = DecoderExplicitState(taehv, lat_h=latent_height, lat_w=latent_width).eval()
        h, w = latent_height, latent_width
        example_x = torch.randn(1, 32, h, w)
        example_lo = torch.zeros(3, 256, h, w)
        example_mid = torch.zeros(3, 128, h * 2, w * 2)
        example_hi = torch.zeros(3, 64, h * 4, w * 4)
        with torch.no_grad():
            ov_model = ov.convert_model(
                dec,
                example_input=(example_x, example_lo, example_mid, example_hi),
                input=[
                    ov.PartialShape([1, 32, h, w]),
                    ov.PartialShape([3, 256, h, w]),
                    ov.PartialShape([3, 128, h * 2, w * 2]),
                    ov.PartialShape([3, 64, h * 4, w * 4]),
                ],
            )
        ov.save_model(ov_model, str(dec_xml), compress_to_fp16=fp16)
        print(f"[export_taehv_openvino]   wrote {dec_xml}")

    return str(enc_xml), str(dec_xml)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ae-repo", default="Overworld-Models/taehv1_5")
    ap.add_argument("--latent-height", type=int, required=True)
    ap.add_argument("--latent-width", type=int, required=True)
    ap.add_argument("--cache-dir", default="./openvino_cache")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    args = ap.parse_args()
    export(
        args.ae_repo,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        cache_dir=args.cache_dir,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
