"""QuantizeE4M3 — bf16/f16/f32 → e4m3 quantization kernel.

Each thread processes a pair of elements using packed_convert, which
maps to PTX's ``cvt.rn.satfinite.e4m3x2.f32 d, a, b``. The output
is a u8 tensor with one byte per e4m3 value.

Grid: (1, N/2 / n_threads, 1) — one pair per thread, blocks tile the
flat array.
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.quantize_e4m3.config import QuantizeE4M3Config
from quark.kernels.quantize_e4m3.spec import QuantizeE4M3Spec


@kernel(
    "quantize_e4m3",
    spec=QuantizeE4M3Spec,
    config=QuantizeE4M3Config,
    output_idx=-1,
    problems=lambda: [],
    baselines=lambda: [],
    reference=lambda spec, *, X, Out=None: X,
)
class QuantizeE4M3Kernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("X", dtype=lambda s, c: s.src_dtype, shape=lambda s, c: (s.N,)),
        TensorDecl("Out", dtype=lambda s, c: DType.E4M3, shape=lambda s, c: (s.N,), role="out"),
    ]

    spec: QuantizeE4M3Spec
    config: QuantizeE4M3Config

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        n_threads = c.n_warps * 32
        # Each thread handles one pair (2 elements).
        pairs = s.N // 2
        return (
            pairs % c.elems_per_block == 0
            and c.elems_per_block % n_threads == 0
            and 1 <= c.n_warps <= 32
        )

    def grid(self) -> tuple[int, int, int]:
        pairs = self.spec.N // 2
        return (1, pairs // self.config.elems_per_block, 1)

    def flops(self) -> int:
        return self.spec.N

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [2, 4, 8], "elems_per_block": [128, 256, 512, 1024]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy

        spec = QuantizeE4M3Spec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "X": astype_numpy(rng.standard_normal(spec.N).astype(np.float32), spec.src_dtype),
            # Out is a packed-b16 buffer (two e4m3 per 16-bit slot).
            "Out": np.zeros(spec.N // 2, dtype=np.uint16),
        }

    def build(self) -> None:
        _, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_threads = c.n_warps * 32
        epb = c.elems_per_block
        epl = epb // n_threads

        block = qk.block_idx("y")
        base = block * bctx.c(epb, dtype=DType.U32)
        n_threads_c = bctx.c(n_threads, dtype=DType.U32)
        c2 = bctx.c(2, dtype=DType.U32)

        for i in range(epl):
            # pair_idx: index into the pairs array
            pair_idx = base + bctx.c(i) * n_threads_c + bctx.tid

            # Load two source elements and convert to f32.
            src_idx_lo = pair_idx * c2
            src_idx_hi = src_idx_lo + bctx.c(1, dtype=DType.U32)

            val_lo = qk.convert(g.X[src_idx_lo], DType.F32)
            val_hi = qk.convert(g.X[src_idx_hi], DType.F32)

            # packed_convert: two f32 → one b16 (two packed e4m3 bytes).
            # PTX: cvt.rn.satfinite.e4m3x2.f32 d, a, b
            packed = qk.packed_convert(val_lo, val_hi, DType.E4M3)

            # Store b16 (2 bytes) at pair_idx * 2. Out is [N] e4m3 (1 byte
            # each), so indexing by pair_idx * 2 hits the right byte offset.
            # The b16 store writes both e4m3 values in one 2-byte write.
            dst_idx = pair_idx * c2
            qk.store(g.Out, packed, dst_idx)
