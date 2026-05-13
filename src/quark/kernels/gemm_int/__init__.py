"""Int8 GEMM kernel — C[M, N] = (A_s8[M, K] @ B_s8[K, N]) × A_scales[M] × B_scales[N].

Per-row dequant via separate scale tensors; s8 × s8 → s32 MMA via
the `m8n16k32_intel_s8_s32` cooperative_matrix shape on Battlemage
Xe2. The epilogue bypasses the F32-only frag_apply / frag_for_each
IR ops via a direct store to a smem scratch region followed by a
per-lane scatter that does the s32→f32 conversion, scale-multiply,
output-dtype cast, and gmem store.

Validated end-to-end at the GLSL level by
``/tmp/probe_int8_gemm_scales.comp`` + ``/tmp/run_int8_gemm_scales.py``
prior to this module landing (cos > 0.9999 vs bf16 reference).
"""

from quark.kernels.gemm_int.config import GemmIntConfig
from quark.kernels.gemm_int.kernel import GemmIntKernel
from quark.kernels.gemm_int.spec import GemmIntSpec

__all__ = ["GemmIntConfig", "GemmIntKernel", "GemmIntSpec"]
