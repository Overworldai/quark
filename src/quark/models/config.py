"""YAML config loader for quark inference (no omegaconf).

Loads a Waypoint-1.5-style ``config.yaml`` from a local path or HF
repo id, applies a small set of repo-wide defaults, and resolves
``${var}`` references against the merged dict. The result is a
plain ``dict[str, Any]`` — ``Waypoint15Config.from_dict`` consumes
it and ``quark.Engine`` reads the AE / timing fields off it
directly.

PyYAML is the only dependency, and it's already pulled in
transitively by ``huggingface_hub`` so we don't widen the dep
surface.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Defaults mirror the world_engine training repo's MODEL_CONFIG_DEFAULTS
# so a checkpoint trained there round-trips through quark.Engine
# without per-field overrides. The two ``${var}`` strings are resolved
# below by ``_resolve``.
MODEL_CONFIG_DEFAULTS: dict[str, Any] = {
    "auto_aspect_ratio": True,
    "gated_attn": False,
    "inference_fps": "${base_fps}",
    "model_type": "waypoint-1",
    "n_kv_heads": "${n_heads}",
    "patch": [1, 1],
    "prompt_conditioning": None,
    "prompt_encoder_uri": "google/umt5-xl",
    "rope_nyquist_frac": 0.8,
    "rope_theta": 10000.0,
    "taehv_ae": False,
    "temporal_compression": 1,
    "value_residual": False,
}


_VAR_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _resolve(d: dict[str, Any], *, max_passes: int = 4) -> dict[str, Any]:
    """Replace ``${var}`` strings in ``d`` with the value of ``d[var]``.

    Single-pass per call but iterates up to ``max_passes`` so chained
    references (rare) resolve. Non-string leaf values are passed
    through; nested dicts/lists aren't recursed into because the
    Waypoint config schema only uses interpolation at the top level.
    """
    out = dict(d)
    for _ in range(max_passes):
        changed = False
        for k, v in list(out.items()):
            if isinstance(v, str):
                m = _VAR_RE.fullmatch(v.strip())
                if m and m.group(1) in out:
                    out[k] = out[m.group(1)]
                    changed = True
        if not changed:
            break
    return out


def _read_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"load_yaml_config: expected a mapping at top level of {path!r}")
    return cfg


def _resolve_path(model_uri: str, filename: str = "config.yaml") -> str:
    """Return a local path to ``filename`` for ``model_uri``.

    Accepts: a path to ``filename`` directly, a directory containing
    ``filename``, or an HF repo id (downloaded via ``hf_hub_download``).
    """
    if os.path.isfile(model_uri):
        return model_uri
    if os.path.isdir(model_uri):
        return os.path.join(model_uri, filename)
    import huggingface_hub

    return huggingface_hub.hf_hub_download(repo_id=model_uri, filename=filename)


def load_yaml_config(model_uri: str, *, filename: str = "config.yaml") -> dict[str, Any]:
    """Load a model config YAML and merge with ``MODEL_CONFIG_DEFAULTS``.

    ``model_uri`` may be a config file path, a directory holding
    ``config.yaml``, or an HF repo id. Returns a plain ``dict`` with
    interpolations (``${base_fps}`` etc.) resolved.
    """
    cfg_path = _resolve_path(model_uri, filename=filename)
    raw = _read_yaml(cfg_path)
    merged = {**MODEL_CONFIG_DEFAULTS, **raw}
    return _resolve(merged)
