"""QuantizeE4M3 — bf16/f16/f32 → e4m3 via cp.async smem staging.

Three-phase per-block pipeline (mirrors the GEMM epilogue staged store
pattern):

  1. Cooperative cp.async of 16 B lines from gmem X → smem ``sm_x``
     (src dtype: bf16 / f16 / f32). No pipelining — one commit,
     one wait(0), one barrier.
  2. Scalar convert from ``sm_x`` to ``sm_out`` (e4m3). Uses the
     scalar fp8 cvt lowering (`src/quark/lower/ptx/lower.py` →
     `_visit_convert`), which synthesizes a single-element cvt via
     the packed ``cvt.<fp8>x2.f32`` form + zero partner. Works on
     sm_89+ without needing the PTX 9.1+ ``bf16x2→e4m3x2`` direct form.
  3. Cooperative ``st.global.v4.b32`` from ``sm_out`` → gmem Out.
     Every thread emits one 16 B line (same shape as the GEMM
     epilogue), so the store path is pure 16 B traffic with no
     per-thread scalar b8/b16 stores.

Grid: ``(1, N / elems_per_block, 1)`` — one block per ``epb`` elements.
Output: ``[N]`` e4m3 tensor, one byte per element.
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
        epb = c.elems_per_block
        src_b = s.src_dtype.bytes
        # Per-phase divisibility:
        #   phase 1 (cp.async 16 B lines of src):  epb*src_b % 16 == 0
        #   phase 2 (scalar cvt, 1 elem per tid):  epb % n_threads == 0
        #   phase 3 (16 B vec_store of e4m3):      epb % 16 == 0
        # Plus each thread does an integer number of lines / vecs / elems.
        if s.N % epb != 0:
            return False
        if epb % n_threads != 0:
            return False
        if (epb * src_b) % 16 != 0:
            return False
        if epb % 16 != 0:
            return False
        lines = (epb * src_b) // 16
        vecs = epb // 16
        if lines % n_threads != 0 or vecs % n_threads != 0:
            return False
        return 1 <= c.n_warps <= 32

    def grid(self) -> tuple[int, int, int]:
        return (1, self.spec.N // self.config.elems_per_block, 1)

    def flops(self) -> int:
        return self.spec.N

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {"n_warps": [2, 4, 8], "elems_per_block": [1024, 2048, 4096, 8192]}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy

        spec = QuantizeE4M3Spec(**problem)
        rng = np.random.default_rng(seed)
        return {
            "X": astype_numpy(rng.standard_normal(spec.N).astype(np.float32), spec.src_dtype),
            # Out is packed as uint16 (two e4m3 per slot) so host-side
            # diffing matches the b16 stores the kernel emits.
            "Out": np.zeros(spec.N // 2, dtype=np.uint16),
        }

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        bctx = self.bctx

        n_threads = c.n_warps * 32
        epb = c.elems_per_block
        src_b = s.src_dtype.bytes
        elems_per_line = 16 // src_b  # 8 for bf16/f16, 4 for f32

        block = qk.block_idx("y")
        gmem_base = block * bctx.c(epb, dtype=DType.U32)
        line_c = bctx.c(elems_per_line, dtype=DType.U32)
        byte16_c = bctx.c(16, dtype=DType.U32)

        # Smem staging — src-dtype input tile + e4m3 output tile.
        # The smem layout pass coalesces these into the block's shared
        # pool and handles alignment; separate allocations keep the
        # address math simple (no manual byte offsets across dtypes).
        sm_x = qk.smem_alloc("sm_x", s.src_dtype, (epb,))
        sm_out = qk.smem_alloc("sm_out", DType.E4M3, (epb,))

        # ── Phase 1: cp.async gmem X → sm_x ────────────────────────────
        # One 16 B line per cp.async issue. Each thread issues
        # ``lines_per_thread`` lines; ``is_valid`` ensures the total
        # line count divides n_threads so no predication is needed.
        total_lines = (epb * src_b) // 16
        lines_per_thread = total_lines // n_threads
        for p in range(lines_per_thread):
            line_id = bctx.tid + bctx.c(p * n_threads, dtype=DType.U32)
            gmem_elem = gmem_base + line_id * line_c
            smem_elem = line_id * line_c
            qk.async_copy(
                sm_x,
                g.X,
                dst_idx=(smem_elem,),
                src_idx=(gmem_elem,),
                count=16,
            )
        qk.async_commit()
        qk.async_wait(0)
        qk.barrier("block")

        # ── Phase 2: scalar convert sm_x → sm_out ──────────────────────
        # Each thread owns ``epb / n_threads`` consecutive-by-stride
        # elements (stride = n_threads). The scalar convert lowers to
        # the synthesized packed form on sm_89 (see Phase 1 note in
        # lower.py `_visit_convert`) — one packed cvt + one narrow per
        # element, no vec paths or pair-packing needed here.
        elems_per_thread = epb // n_threads
        for i in range(elems_per_thread):
            idx = bctx.tid + bctx.c(i * n_threads, dtype=DType.U32)
            v = qk.load(sm_x, idx)
            v_e4m3 = qk.convert(v, DType.E4M3)
            qk.store(sm_out, v_e4m3, idx)
        qk.barrier("block")

        # ── Phase 3: 16 B vec_store sm_out → gmem Out ──────────────────
        # Same shape as the GEMM epilogue staged store: every thread
        # emits one ``st.global.v4.b32`` per pass. The B32 dtype
        # override on ``vec_load`` bypasses the fp8-reg-class canon
        # (lower.py `_canonicalize_vec`) — we're treating the 16 e4m3
        # bytes as 4 b32 words for the cooperative transfer.
        total_vecs = epb // 16
        vecs_per_thread = total_vecs // n_threads
        for p in range(vecs_per_thread):
            vec_id = bctx.tid + bctx.c(p * n_threads, dtype=DType.U32)
            byte_off = vec_id * byte16_c
            v_b32 = qk.vec_load(sm_out, byte_off, width=4, dtype=DType.B32)
            qk.vec_store(g.Out, v_b32, gmem_base + byte_off)
