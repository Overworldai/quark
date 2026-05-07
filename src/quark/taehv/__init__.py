"""TAEHV decoder for Apple Neural Engine — torch-free runtime.

Loads CoreML-compiled TAEHV encoder/decoder ``.mlpackage`` artifacts
and provides numpy-only encode/decode plus a ``ThreadPoolExecutor``-
based pipeline so the ANE decode overlaps the next frame's GPU
forward without contending for the main thread's GIL.

Runtime deps: ``coremltools``, ``numpy``, ``Pillow``,
``huggingface_hub``. NO torch, NO ``world_engine``.

Artifacts live in a pre-built HF model repo (one repo per source
TAEHV checkpoint, e.g. ``Overworld-Models/taehv1_5-coreml``). The
runtime calls ``huggingface_hub.snapshot_download`` with
``allow_patterns=["<latH>x<latW>/**"]`` to pull only the resolution
the caller asked for, into HF's standard cache
(``~/.cache/huggingface/hub/...``). No local ``./taehv_cache/``
fallback — the HF cache IS the cache, and offline-mode
(``HF_HUB_OFFLINE=1``) reuses it without needing a network round trip.

Adding a new resolution
-----------------------
Run the maintainer-only publish script on any macOS Apple-Silicon
machine with the ``[export]`` extras installed:

    python scripts/publish_taehv_coreml.py

Edit ``RESOLUTIONS`` at the top of that script to add new latent
``(H, W)`` pairs. The script traces TAEHV → CoreML, uploads to the
target HF repo, and writes ``manifest.json`` with the source
revision pin. Subsequent runtime callers pick up the new resolution
on next ``snapshot_download``.

Usage
-----

    from quark.taehv import load_taehv, PipelinedDecoder

    ae = load_taehv("Overworld-Models/taehv1_5-coreml",
                    latent_height=16, latent_width=32)
    pipe = PipelinedDecoder(ae)

    for fi, latent_qt in enumerate(latents):
        pipe.submit(latent_qt)
        img = pipe.next()                 # None on first call
        if img is not None:
            save(img)
    img = pipe.flush()
    if img is not None:
        save(img)
"""

from __future__ import annotations

from quark.taehv.coreml import CoreMLTAEHV
from quark.taehv.fetch import fetch_coreml_artifacts
from quark.taehv.pipeline import PipelinedDecoder

__all__ = [
    "CoreMLTAEHV",
    "PipelinedDecoder",
    "decode_one",
    "fetch_coreml_artifacts",
    "load_taehv",
]


def load_taehv(
    coreml_uri: str,
    *,
    latent_height: int,
    latent_width: int,
    revision: str | None = None,
    cache_dir: str | None = None,
    compute_units: str = "CPU_AND_NE",
) -> CoreMLTAEHV:
    """Load a pre-exported CoreML TAEHV decoder from HF.

    ``coreml_uri``: HF model URI of the pre-built CoreML artifacts —
    typically ``<source-taehv-repo>-coreml`` (e.g.
    ``Overworld-Models/taehv1_5-coreml``). The repo layout is
    ``<latH>x<latW>/{taehv_encoder,taehv_decoder_ane}.mlpackage``.
    On first call ``huggingface_hub.snapshot_download`` pulls only
    the requested resolution's subdirectory into the HF cache;
    subsequent calls are a fast cache hit.

    ``latent_height`` / ``latent_width``: latent-grid spatial dims
    (e.g. ``(16, 32)`` for 360p, ``(32, 64)`` for 720p). The encoder
    input pixel dim is ``8x`` these.

    ``revision``: optional HF revision pin (commit SHA, tag, or
    branch). ``None`` = ``main`` (latest published export). Set this
    when a model config requires a specific source-TAEHV revision —
    the publish script writes the source rev into the repo's
    ``manifest.json`` so consumers can pin to it.

    ``cache_dir``: explicit root for the materialised local mirror.
    ``None`` falls through to ``$QUARK_TAEHV_CACHE`` then to
    ``~/.cache/quark/taehv``. Host applications that want the
    cache to live inside their own data dir (so cleanup operations
    don't leave artifacts in unexpected user paths) should pass
    something like ``f"{biome_app_dir}/taehv"``.

    ``compute_units``: ``"CPU_AND_NE"`` (default — Apple Neural
    Engine, ~22 ms / decode, 0% GPU) or ``"CPU_AND_GPU"`` (Metal
    GPU fallback for non-ANE Macs). CPU-only is intentionally not
    exposed.

    Raises ``FileNotFoundError`` if the requested resolution isn't
    in the HF repo, with a pointer to the publish script. Raises
    a network error from ``huggingface_hub`` if the repo can't be
    reached at all and isn't already cached locally.
    """
    enc_path, dec_path = fetch_coreml_artifacts(
        coreml_uri,
        latent_height=latent_height,
        latent_width=latent_width,
        revision=revision,
        cache_dir=cache_dir,
    )

    return CoreMLTAEHV(
        enc_path,
        dec_path,
        latent_height=latent_height,
        latent_width=latent_width,
        compute_units=compute_units,
    )


def decode_one(ae: CoreMLTAEHV, latent_np):
    """``[1, 32, latH, latW] f32/f16 numpy → [4, H_pix, W_pix, 3] uint8``.

    Single-call convenience for callers that don't need pipelining.
    For per-frame loops with overlap against another GPU op, use
    :class:`PipelinedDecoder` instead.
    """
    return ae.decode(latent_np)
