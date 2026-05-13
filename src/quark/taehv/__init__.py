"""TAEHV decoder runtime — backend-dispatched.

Two backends:

  * **coreml** (Apple Silicon, ANE / Metal GPU) — see
    :mod:`quark.taehv.coreml`. Compiled to ``.mlpackage`` via
    ``coremltools``; runs on the Neural Engine (preferred) or the
    Apple GPU.

  * **openvino** (Intel iGPU / dGPU / Arc / Battlemage / NPU) — see
    :mod:`quark.taehv.openvino`. Compiled to OpenVINO IR (``.xml`` +
    ``.bin``) via ``openvino.convert_model``; runs on Intel GPU.

Both backends expose the same encode/decode/reset API so callers
swap between them transparently. :func:`load_taehv` picks the
right one based on the URI (``...-coreml`` → coreml, ``...-openvino``
→ openvino) or via an explicit ``backend=`` arg.

Usage
-----

    from quark.taehv import load_taehv, PipelinedDecoder

    ae = load_taehv(
        "Clyde013/taehv1_5-coreml",   # or -openvino
        latent_height=16, latent_width=32,
    )
    pipe = PipelinedDecoder(ae)

    for fi, latent_qt in enumerate(latents):
        pipe.submit(latent_qt)
        img = pipe.next()
        if img is not None:
            save(img)
    img = pipe.flush()
    if img is not None:
        save(img)
"""

from __future__ import annotations

import os
import sys

__all__ = [
    "PipelinedDecoder",
    "load_taehv",
]


def _resolve_backend(coreml_uri: str, backend: str | None) -> str:
    if backend is not None:
        b = backend.lower()
        if b not in ("coreml", "openvino"):
            raise ValueError(f"backend must be 'coreml' or 'openvino' (got {backend!r})")
        return b
    # URI-suffix heuristic
    u = coreml_uri.lower()
    if u.endswith("-openvino"):
        return "openvino"
    if u.endswith("-coreml"):
        return "coreml"
    # Platform default: CoreML on darwin, OpenVINO elsewhere.
    return "coreml" if sys.platform == "darwin" else "openvino"


def load_taehv(
    uri: str,
    *,
    latent_height: int,
    latent_width: int,
    revision: str | None = None,
    cache_dir: str | None = None,
    compute_units: str | None = None,
    backend: str | None = None,
):
    """Load a pre-exported TAEHV from an HF repo.

    ``uri``: HF model URI. ``...-coreml`` or ``...-openvino`` suffix
    selects the backend by default; pass ``backend=`` to override.

    ``compute_units``: backend-specific compute unit selector. CoreML
    accepts ``"CPU_AND_NE"`` (default) or ``"CPU_AND_GPU"``. OpenVINO
    accepts ``"GPU"`` (default on Intel hardware), ``"CPU"``, or
    ``"NPU"``. ``None`` lets each backend pick its default.
    """
    b = _resolve_backend(uri, backend)
    if b == "coreml":
        from quark.taehv.coreml import load as _load
        kwargs = {} if compute_units is None else {"compute_units": compute_units}
        return _load(
            uri,
            latent_height=latent_height,
            latent_width=latent_width,
            revision=revision,
            cache_dir=cache_dir,
            **kwargs,
        )
    # openvino
    from quark.taehv.openvino import load as _load
    kwargs = {} if compute_units is None else {"device": compute_units}
    return _load(
        uri,
        latent_height=latent_height,
        latent_width=latent_width,
        revision=revision,
        cache_dir=cache_dir,
        **kwargs,
    )


def __getattr__(name: str):
    """Lazy-load PipelinedDecoder from whichever backend is installed.

    Both backends ship the same class shape. The CoreML one is
    imported by default on darwin; the OpenVINO one elsewhere. Keeps
    cross-platform `from quark.taehv import PipelinedDecoder` working
    without hard-importing both runtimes.
    """
    if name == "PipelinedDecoder":
        if sys.platform == "darwin" and os.environ.get("QUARK_TAEHV_BACKEND", "").lower() != "openvino":
            from quark.taehv.coreml.pipeline import PipelinedDecoder as _PD
        else:
            from quark.taehv.openvino.pipeline import PipelinedDecoder as _PD
        return _PD
    raise AttributeError(f"module 'quark.taehv' has no attribute {name!r}")
