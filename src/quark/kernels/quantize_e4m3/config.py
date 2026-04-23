"""QuantizeE4M3Config."""

from __future__ import annotations

from dataclasses import dataclass

from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class QuantizeE4M3Config(KernelConfig):
    n_warps: int = 4
    elems_per_block: int = 2048

    @classmethod
    def default_for(cls, spec) -> QuantizeE4M3Config:
        # `epb` is bf16/f16/f32 elements per block. is_valid requires
        # epb * src_b / 16 and epb / 16 to each divide n_threads — i.e.
        # the cp.async line count and the vec_store count are both
        # per-thread integers. Walk from the largest feasible epb down
        # so we prefer fewer, larger blocks.
        src_b = spec.src_dtype.bytes
        for n_warps, candidates in (
            (4, (4096, 2048, 1024)),
            (2, (2048, 1024, 512)),
            (8, (8192, 4096, 2048)),
        ):
            n_threads = n_warps * 32
            for epb in candidates:
                lines = (epb * src_b) // 16
                vecs = epb // 16
                if (
                    spec.N % epb == 0
                    and epb % n_threads == 0
                    and (epb * src_b) % 16 == 0
                    and epb % 16 == 0
                    and lines % n_threads == 0
                    and vecs % n_threads == 0
                ):
                    return cls(n_warps=n_warps, elems_per_block=epb)
        # Fallback for tiny N — single-warp, tightest block.
        return cls(n_warps=1, elems_per_block=128)
