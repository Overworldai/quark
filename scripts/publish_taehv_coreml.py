#!/usr/bin/env python3
"""Publish pre-built CoreML TAEHV artifacts to a HF model repo.

Maintainer-only — run on any macOS Apple Silicon machine with the
``[export]`` extras installed and an HF write token. Traces the
upstream TAEHV PyTorch model into ``.mlpackage`` artifacts at every
``(latent_height, latent_width)`` pair listed in ``RESOLUTIONS``,
writes a ``manifest.json`` recording the source TAEHV revision, then
uploads the whole tree to ``COREML_REPO`` on HF.

The runtime path (``quark.taehv.load_taehv``) reads from this repo
via ``huggingface_hub.snapshot_download`` — pulling only the
resolution it needs into HF's standard cache.

Adding a new resolution
-----------------------
Edit ``RESOLUTIONS`` below, run::

    python scripts/publish_taehv_coreml.py

The script is idempotent: existing ``.mlpackage`` directories in the
staging dir are reused (delete them or pass ``--rebuild`` to force
re-trace), and ``upload_folder`` only re-uploads files whose contents
changed (HF's content addressing handles dedup).

Setup
-----
Once per machine:

    uv venv .venv --python 3.12
    VIRTUAL_ENV=.venv uv pip install -e '.[export]'
    huggingface-cli login   # paste a token with write scope on Overworld-Models

Run
---

    python scripts/publish_taehv_coreml.py
    python scripts/publish_taehv_coreml.py --rebuild     # force re-export
    python scripts/publish_taehv_coreml.py --dry-run     # export only, skip upload
    python scripts/publish_taehv_coreml.py --staging-dir /tmp/staging
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys

# ────────────────────────────────────────────────────────────────────
# Edit this list to add or remove resolutions. Each entry is a
# ``(latent_height, latent_width)`` pair — the same numbers a model
# config carries, the same numbers ``quark.taehv.load_taehv`` takes.
# Encoder pixel dim = 8× these, but that derivation is internal to
# the export module; the key throughout is latent dims.
# ────────────────────────────────────────────────────────────────────

RESOLUTIONS: list[tuple[int, int]] = [
    (16, 32),    # Waypoint-1.5-1B 360P
    (32, 64),    # Waypoint-1.5-1B 720P
]

# Source upstream TAEHV PyTorch checkpoint.
SOURCE_REPO = "Overworld-Models/taehv1_5"

# Target HF repo for the published CoreML artifacts. Convention:
# ``<source>-coreml`` so ``quark.engine`` derives this URI by appending
# ``-coreml`` to the model config's ``ae_uri``.
COREML_REPO = "Overworld-Models/taehv1_5-coreml"

# Default staging dir: ``./taehv_publish_staging/`` under the repo
# checkout. Disposable — delete to force a clean rebuild. Override
# with ``--staging-dir``.
DEFAULT_STAGING = pathlib.Path(__file__).resolve().parent.parent / "taehv_publish_staging"


def _resolve_source_revision(source_repo: str) -> str:
    """Look up the current HEAD revision of the source TAEHV HF repo.

    Stored in ``manifest.json`` so consumers can pin against it. Uses
    ``HfApi().repo_info`` — a single REST call, no clone needed.
    """
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id=source_repo, repo_type="model")
    sha = info.sha
    if not sha:
        raise RuntimeError(f"HfApi().repo_info({source_repo!r}) returned no SHA")
    return sha


def _export_one(latent_height: int, latent_width: int, *, staging: pathlib.Path,
                rebuild: bool) -> tuple[pathlib.Path, pathlib.Path]:
    """Trace TAEHV → CoreML for one resolution. Idempotent: skips if
    both ``.mlpackage`` directories already exist (override with
    ``--rebuild``).
    """
    res_dir = staging / f"{latent_height}x{latent_width}"
    enc = res_dir / "taehv_encoder.mlpackage"
    dec = res_dir / "taehv_decoder_ane.mlpackage"

    if rebuild and res_dir.exists():
        import shutil
        shutil.rmtree(res_dir)

    if enc.exists() and dec.exists():
        print(f"  skip {res_dir.name}/ — already populated")
        return enc, dec

    # Delegate to the existing export module — same code that's been
    # producing artifacts locally for months. We just point its
    # ``--cache-dir`` at our staging area.
    from quark.taehv import export as _exp

    print(f"  exporting {res_dir.name}/ …")
    enc_path, dec_path = _exp.export(
        ae_repo=SOURCE_REPO,
        latent_height=latent_height,
        latent_width=latent_width,
        cache_dir=str(staging),
    )
    return pathlib.Path(enc_path), pathlib.Path(dec_path)


def _build_manifest(staging: pathlib.Path, source_repo: str, source_revision: str) -> dict:
    """Write ``staging/manifest.json`` recording every published
    resolution and the pinned source revision.
    """
    resolutions = []
    for latH, latW in sorted(RESOLUTIONS):
        res_dir = staging / f"{latH}x{latW}"
        if not res_dir.exists():
            continue
        resolutions.append({
            "latent_height": latH,
            "latent_width": latW,
            "encoder_input_pixels": [latH * 8, latW * 8],
        })

    manifest = {
        "schema_version": 1,
        "source_repo": source_repo,
        "source_revision": source_revision,
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "resolutions": resolutions,
    }

    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote {manifest_path}")
    return manifest


def _upload(staging: pathlib.Path, target_repo: str, source_revision: str) -> None:
    """Upload the staging tree to the HF target repo. Uses
    ``upload_folder`` — content-addressed, only changed files
    re-uploaded.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi()
    try:
        api.repo_info(repo_id=target_repo, repo_type="model")
    except RepositoryNotFoundError:
        print(f"  creating {target_repo} …")
        api.create_repo(repo_id=target_repo, repo_type="model", private=False, exist_ok=True)

    print(f"  uploading {staging} → {target_repo} …")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=target_repo,
        repo_type="model",
        commit_message=f"publish CoreML artifacts (source rev {source_revision[:12]})",
    )
    print(f"  done — https://huggingface.co/{target_repo}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-repo", default=SOURCE_REPO,
                   help=f"Upstream TAEHV PyTorch repo (default: {SOURCE_REPO})")
    p.add_argument("--coreml-repo", default=COREML_REPO,
                   help=f"Target HF model repo for the artifacts (default: {COREML_REPO})")
    p.add_argument("--staging-dir", type=pathlib.Path, default=DEFAULT_STAGING,
                   help=f"Local staging dir (default: {DEFAULT_STAGING})")
    p.add_argument("--rebuild", action="store_true",
                   help="Delete + re-trace each resolution even if already present")
    p.add_argument("--dry-run", action="store_true",
                   help="Run the export pass only — skip the HF upload")
    args = p.parse_args()

    if sys.platform != "darwin":
        print("ERROR: coremltools tracing only runs on macOS", file=sys.stderr)
        return 2

    if not RESOLUTIONS:
        print("ERROR: RESOLUTIONS list is empty — nothing to publish", file=sys.stderr)
        return 2

    args.staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"source repo:  {args.source_repo}")
    print(f"target repo:  {args.coreml_repo}")
    print(f"staging dir:  {args.staging_dir}")
    print(f"resolutions:  {RESOLUTIONS}")
    print()

    source_revision = _resolve_source_revision(args.source_repo)
    print(f"source revision: {source_revision}")
    print()

    print("Exporting:")
    for latH, latW in RESOLUTIONS:
        _export_one(latH, latW, staging=args.staging_dir, rebuild=args.rebuild)

    print()
    print("Manifest:")
    _build_manifest(args.staging_dir, args.source_repo, source_revision)

    if args.dry_run:
        print()
        print(f"--dry-run set; staging tree is at {args.staging_dir}, skipping upload.")
        return 0

    print()
    print("Uploading:")
    _upload(args.staging_dir, args.coreml_repo, source_revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
