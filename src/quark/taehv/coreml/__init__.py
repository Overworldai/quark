"""CoreML TAEHV backend — Apple Silicon (ANE / Metal GPU).

Compiled artifacts: ``.mlpackage`` directories (encoder + decoder)
loaded via ``coremltools.models.MLModel``. State is owned by the
runtime object and passed as explicit input/output tensors per call
(the in-graph ``StateType`` path fails on ANE with error -14;
see runtime docstring).
"""

from __future__ import annotations

from quark.taehv.coreml.fetch import fetch_coreml_artifacts
from quark.taehv.coreml.pipeline import PipelinedDecoder
from quark.taehv.coreml.runtime import CoreMLTAEHV

__all__ = [
    "CoreMLTAEHV",
    "PipelinedDecoder",
    "fetch_coreml_artifacts",
    "load",
]


def load(
    coreml_uri: str,
    *,
    latent_height: int,
    latent_width: int,
    revision: str | None = None,
    cache_dir: str | None = None,
    compute_units: str = "CPU_AND_NE",
) -> CoreMLTAEHV:
    """Load a pre-exported CoreML TAEHV from an HF repo."""
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
