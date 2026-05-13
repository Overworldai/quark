"""HF snapshot fetch for pre-built OpenVINO TAEHV IR artifacts.

Mirror of :mod:`quark.taehv.coreml.fetch` but for OpenVINO IR
(``encoder.xml`` + ``encoder.bin``, ``decoder.xml`` + ``decoder.bin``).
Resolution-scoped pull via ``snapshot_download(allow_patterns=...)``.

Mirror lives under ``~/.cache/quark/taehv-openvino/<safe-repo>/<rev>/``
unless ``QUARK_TAEHV_OPENVINO_CACHE`` overrides.

When no remote repo exists yet, callers can pass a ``local_dir``
that already contains the export output directly — the export script
writes the same ``<latH>x<latW>/`` layout, so the resolver just
checks that path before falling back to HF.
"""

from __future__ import annotations

import os
import pathlib

_ENCODER_FILE = "encoder.xml"
_DECODER_FILE = "decoder.xml"


def _mirror_dir(
    ov_uri: str,
    revision: str | None,
    *,
    cache_dir: str | os.PathLike | None = None,
) -> pathlib.Path:
    safe_repo = ov_uri.replace("/", "--")
    rev = revision or "main"
    if cache_dir is not None:
        root = pathlib.Path(os.fspath(cache_dir))
    else:
        root = pathlib.Path(
            os.environ.get("QUARK_TAEHV_OPENVINO_CACHE")
            or os.path.expanduser("~/.cache/quark/taehv-openvino")
        )
    return root / safe_repo / rev


def fetch_openvino_artifacts(
    ov_uri: str,
    *,
    latent_height: int,
    latent_width: int,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> tuple[str, str]:
    """Return ``(encoder_xml_path, decoder_xml_path)``.

    Resolution order:
      1. If ``ov_uri`` resolves as a local directory containing
         ``<latH>x<latW>/{encoder.xml, decoder.xml}``, use it directly.
         (Used by the export script's output dir.)
      2. Otherwise, ``snapshot_download(ov_uri)`` from HF with
         ``allow_patterns=["<latH>x<latW>/**"]``.
    """
    res_key = f"{latent_height}x{latent_width}"

    # Local-path shortcut.
    if os.path.isdir(ov_uri):
        res_dir = pathlib.Path(ov_uri) / res_key
        enc, dec = res_dir / _ENCODER_FILE, res_dir / _DECODER_FILE
        if enc.exists() and dec.exists():
            return os.fspath(enc), os.fspath(dec)

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
    except ImportError as e:
        raise ImportError(
            "quark.taehv.openvino requires 'huggingface_hub' for remote "
            "artifact fetch — install with ``pip install huggingface_hub`` "
            "or pass a local directory path as ``ov_uri``."
        ) from e

    mirror = _mirror_dir(ov_uri, revision, cache_dir=cache_dir)
    try:
        snapshot_path = snapshot_download(
            repo_id=ov_uri,
            revision=revision,
            allow_patterns=[f"{res_key}/**"],
            local_dir=str(mirror),
        )
    except (RepositoryNotFoundError, RevisionNotFoundError) as e:
        raise FileNotFoundError(
            f"OpenVINO TAEHV repo {ov_uri!r} (rev {revision or 'main'!r}) "
            f"not found on HF and not present as a local directory. "
            f"Either point ``ov_uri`` at a local export dir produced by "
            f"``python -m quark.taehv.openvino.export``, or publish the IR "
            f"to an HF repo."
        ) from e

    res_dir = pathlib.Path(snapshot_path) / res_key
    enc, dec = res_dir / _ENCODER_FILE, res_dir / _DECODER_FILE
    if not (enc.exists() and dec.exists()):
        raise FileNotFoundError(
            f"OpenVINO TAEHV resolution {res_key!r} missing under {snapshot_path!r}. "
            f"Re-export with ``python -m quark.taehv.openvino.export "
            f"--latent-height {latent_height} --latent-width {latent_width}``."
        )
    return os.fspath(enc), os.fspath(dec)
