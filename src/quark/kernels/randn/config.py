"""RandnConfig — tile shape for the Philox randn kernel."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class RandnConfig(KernelConfig):
    # Each thread runs Philox once per group of 4 outputs, so
    # ``elems_per_block`` must be a multiple of ``n_warps * 32 * 4``.
    n_warps: int = 4
    elems_per_block: int = 2048

    @classmethod
    def default_for(cls, spec) -> RandnConfig:
        # Pick a config that divides typical latent sizes (65536 for 360p).
        return cls(n_warps=4, elems_per_block=2048)
