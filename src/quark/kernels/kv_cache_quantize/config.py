"""KVQuantizeConfig — minimal config for the first-cut quant kernel."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class KVQuantizeConfig(KernelConfig):
    """Single-warp WG, one token per WG. Default tracks the validated
    GLSL probe layout (32 threads cover Dh=64 with 2 elements/lane)."""

    n_warps: int = 1
