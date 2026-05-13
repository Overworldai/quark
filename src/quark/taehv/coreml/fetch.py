"""HF snapshot fetch for pre-built CoreML TAEHV artifacts.

Resolution-scoped pull: ``snapshot_download`` with
``allow_patterns=[f"{latH}x{latW}/**"]`` so we only download the
~50-150 MB ``.mlpackage`` directory the caller actually needs,
not every resolution in the repo.

We pass ``local_dir`` to force HF to materialize real files (or
hard links) at a stable mirror path, NOT symlinks back into HF's
``blobs/`` store. CoreML's ``.mlpackage`` → ``.mlmodelc`` compiler
silently follows the symlink chain when copying files into its
temp build dir and produces "file not found" errors at compile
time — the symptom is a path like ``foo/weight.bin/`` (note
trailing slash) appearing in a ``RuntimeError`` from
``MLModel.predict()``. Materialised files sidestep this entirely.

Mirror lives under ``~/.cache/quark/taehv/<safe-repo-id>/<rev>/``
(rev defaults to ``main`` when not pinned). Repo IDs with ``/``
get sanitised to ``--`` to keep the path POSIX-clean. The HF cache
proper (``~/.cache/huggingface/hub/...``) is still populated as a
side effect — that's where the blobs live, and the mirror's hard
links share storage with them. Disk cost per resolution is the
``.mlpackage`` size (≈50 MB), counted once across all callers
hitting the same ``(repo, rev)``.
"""

from __future__ import annotations

import os
import pathlib

_ENCODER_FILE = "taehv_encoder.mlpackage"
_DECODER_FILE = "taehv_decoder_ane.mlpackage"


def _mirror_dir(
    coreml_uri: str,
    revision: str | None,
    *,
    cache_dir: str | os.PathLike | None = None,
) -> pathlib.Path:
    """Stable local-mirror path for one ``(repo, revision)`` pair.

    Resolution order (first match wins):
      1. ``cache_dir`` arg passed through from ``fetch_coreml_artifacts``
         (which is itself plumbed from ``load_taehv`` and ``Engine``) —
         this is how a host application like Biome anchors the cache
         under its own data dir.
      2. ``QUARK_TAEHV_CACHE`` env var.
      3. ``~/.cache/quark/taehv`` — XDG-style user cache fallback.
    """
    safe_repo = coreml_uri.replace("/", "--")
    rev = revision or "main"
    if cache_dir is not None:
        root = pathlib.Path(os.fspath(cache_dir))
    else:
        root = pathlib.Path(
            os.environ.get("QUARK_TAEHV_CACHE") or os.path.expanduser("~/.cache/quark/taehv")
        )
    return root / safe_repo / rev


def fetch_coreml_artifacts(
    coreml_uri: str,
    *,
    latent_height: int,
    latent_width: int,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> tuple[str, str]:
    """Download (or cache-hit) the requested resolution's encoder
    and decoder ``.mlpackage`` directories from the HF repo.

    Returns ``(encoder_path, decoder_path)`` — absolute paths into
    the local mirror. Both are directories (CoreML's ``.mlpackage``
    is a folder of files), suitable for direct ``ct.models.MLModel``
    construction by ``CoreMLTAEHV``.

    ``coreml_uri``: HF model URI (e.g.
    ``Overworld-Models/taehv1_5-coreml``).
    ``latent_height`` / ``latent_width``: latent-grid spatial dims;
    matches the directory key inside the repo (``<latH>x<latW>/``).
    ``revision``: optional HF revision pin. ``None`` resolves to
    ``main`` (= the latest published export).
    ``cache_dir``: explicit root for the local mirror. ``None``
    falls through to ``$QUARK_TAEHV_CACHE`` then to
    ``~/.cache/quark/taehv``. Host applications (e.g. Biome) that
    want artifacts to live inside their own data dir should pass
    this — anything stored under the dir is fully managed by quark
    so the host can clear the cache by removing the dir.

    Raises ``FileNotFoundError`` if the resolution subdir isn't
    present in the repo. Raises ``huggingface_hub`` errors if the
    repo can't be reached AND isn't cached.
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
    except ImportError as e:
        raise ImportError(
            "quark.taehv requires the 'huggingface_hub' package — "
            "install with ``pip install 'quark[taehv]'`` or directly "
            "via ``pip install huggingface_hub``."
        ) from e

    res_key = f"{latent_height}x{latent_width}"
    mirror = _mirror_dir(coreml_uri, revision, cache_dir=cache_dir)

    try:
        snapshot_path = snapshot_download(
            repo_id=coreml_uri,
            revision=revision,
            allow_patterns=[f"{res_key}/**"],
            local_dir=str(mirror),
        )
    except (RepositoryNotFoundError, RevisionNotFoundError) as e:
        raise FileNotFoundError(
            f"CoreML TAEHV repo {coreml_uri!r} (revision {revision or 'main'!r}) "
            f"not found on HF. Has it been published yet? Run the maintainer "
            f"publish script:\n"
            f"  python scripts/publish_taehv_coreml.py\n"
            f"(needs the ``[export]`` extras and an HF write token)."
        ) from e

    res_dir = pathlib.Path(snapshot_path) / res_key
    enc_path = res_dir / _ENCODER_FILE
    dec_path = res_dir / _DECODER_FILE

    if not (enc_path.exists() and dec_path.exists()):
        raise FileNotFoundError(
            f"CoreML TAEHV resolution {res_key!r} is not present in "
            f"{coreml_uri!r} (revision {revision or 'main'!r}).\n"
            f"Looked for:\n"
            f"  {enc_path}\n"
            f"  {dec_path}\n"
            f"Add this resolution to ``RESOLUTIONS`` in "
            f"``scripts/publish_taehv_coreml.py`` and re-run that script."
        )

    return os.fspath(enc_path), os.fspath(dec_path)
