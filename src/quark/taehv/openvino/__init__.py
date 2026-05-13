"""OpenVINO TAEHV backend — Intel iGPU / dGPU / Arc / Battlemage.

Compiled artifacts: OpenVINO IR (``.xml`` + ``.bin``) produced by
:mod:`quark.taehv.openvino.export`. The runtime
(:class:`OpenVINOTAEHV`) loads both files via ``ov.Core().read_model``
and compiles them once for the target device (``"GPU"`` /
``"CPU"`` / ``"NPU"`` / ``"AUTO"``).

State is owned by the runtime and passed as explicit I/O on each
decode call, matching the CoreML backend's shape so the two are
swappable.
"""

from __future__ import annotations

from quark.taehv.openvino.fetch import fetch_openvino_artifacts
from quark.taehv.openvino.pipeline import PipelinedDecoder
from quark.taehv.openvino.runtime import OpenVINOTAEHV

__all__ = [
    "OpenVINOTAEHV",
    "PipelinedDecoder",
    "fetch_openvino_artifacts",
    "load",
]


def load(
    uri: str,
    *,
    latent_height: int,
    latent_width: int,
    revision: str | None = None,
    cache_dir: str | None = None,
    device: str = "GPU",
) -> OpenVINOTAEHV:
    """Load a pre-exported OpenVINO TAEHV.

    ``uri``: either an HF repo id (e.g. ``"yourorg/taehv1_5-openvino"``)
    or a local directory path produced by
    :func:`quark.taehv.openvino.export.export`.
    """
    enc, dec = fetch_openvino_artifacts(
        uri,
        latent_height=latent_height,
        latent_width=latent_width,
        revision=revision,
        cache_dir=cache_dir,
    )
    return OpenVINOTAEHV(
        enc, dec,
        latent_height=latent_height,
        latent_width=latent_width,
        device=device,
    )
