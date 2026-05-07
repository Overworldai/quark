"""GemmSpec — immutable problem definition for the universal GEMM.

C[M, N] = A[M, K] @ B^T[N, K]^T

A can be >2D (batch/seq dims get flattened into M). B is always 2D.
Output C is [M, N]. The kernel is parameterized over four dtype axes:

  a_dtype × b_dtype → acc_dtype → out_dtype

Supported dtypes for A/B: bf16, fp16, e4m3, e5m2.
Accumulator is always f32 (no tf32 — we avoid that PTX surface).
Output: bf16, fp16, f32, e4m3, e5m2.
"""

from __future__ import annotations

from dataclasses import dataclass

from quark.ir import DType
from quark.kernels.base import KernelSpec

_VALID_AB = frozenset({DType.BF16, DType.F16, DType.F32, DType.E4M3, DType.E5M2})
_VALID_OUT = frozenset({DType.BF16, DType.F16, DType.F32, DType.E4M3, DType.E5M2})
_VALID_ACTIVATIONS = frozenset({None, "silu"})
# Dtypes that can actually drive an MMA fragment (F32 doesn't — it must
# be downcast on load, the same way bf16×e4m3 works via ``compute_dtype``).
_VALID_COMPUTE = frozenset({DType.BF16, DType.F16, DType.E4M3, DType.E5M2})


@dataclass(frozen=True)
class GemmSpec(KernelSpec):
    """C[M, N] = A[M, K] @ B^T[N, K]^T.

    M is the flattened product of all leading dims of A — the caller
    reshapes [B, S, K] → [B*S, K] before building the spec. B is
    always [N, K] (weight, stored as B^T so K is the fast axis).

    ``compute_dtype`` controls the dtype the mma.sync operates in.
    Source A/B are cast to compute_dtype during the gmem→smem load,
    so the on-device smem tiles and the mma fragments are always in
    compute_dtype. Defaults to ``a_dtype`` (no cast) for back-compat.
    Example: a_dtype=BF16, b_dtype=E4M3, compute_dtype=E4M3 → A is
    downcast bf16→e4m3 on load; the mma is e4m3×e4m3→f32.
    """

    M: int
    N: int
    K: int
    a_dtype: DType = DType.BF16
    b_dtype: DType = DType.BF16
    acc_dtype: DType = DType.F32  # always F32 for now
    out_dtype: DType = DType.BF16
    compute_dtype: DType | None = None  # None → same as a_dtype (no cast)
    # Fused activation applied to the accumulator in the epilogue, before
    # cast to out_dtype. None (default) is a pure GEMM; "silu" is the
    # fused SiLU path used by MLP fc1. Routed through
    # ``pop.store_acc(activation=...)`` which already lowers SiLU through
    # the fast rcp+ex2 path on PTX and MSL.
    activation: str | None = None
    has_bias: bool = False
    # Fused AdaGate-residual epilogue: when True, the kernel reads two
    # extra inputs (Gate[G, N] and Residual[M, N]) at store time and
    # writes ``Out = Residual + Gate_bcast * (A @ Bᵀ)`` instead of the
    # plain ``Out = A @ Bᵀ`` — folds the standalone AdaGateResidual op
    # (post-attn / post-MLP residual) into the GEMM, saving one
    # dispatch + one bf16 round-trip on the accumulator. ``G`` is the
    # gate-broadcast group count (each group of M//G consecutive rows
    # reads the same gate row), matching ``AdaGateResidualSpec.G``.
    # Like ``has_bias``/``activation``, the per-block fused path is only
    # correct at split_k=1 — see ``is_valid`` for the rejection.
    has_gate_residual: bool = False
    G: int = 1
    # Whether the caller will hand this kernel a preshuffled B tensor
    # (bytes already permuted to match the vectorized fragment-load
    # layout). This is a property of the B tensor layout, hence a spec
    # field, not a tunable. When True the kernel emits the vectorized
    # ``ld.shared.v{N}.b32`` frag loads and ``prepare_launch_tensors``
    # caches the shuffled result per B tensor.
    b_shuffle: bool = False

    def __post_init__(self):
        # Coerce any string values (from Problem.params) into DType.
        for field in ("a_dtype", "b_dtype", "acc_dtype", "out_dtype"):
            val = getattr(self, field)
            if isinstance(val, str) and not isinstance(val, DType):
                object.__setattr__(self, field, DType(val))
        if (
            self.compute_dtype is not None
            and isinstance(self.compute_dtype, str)
            and not isinstance(self.compute_dtype, DType)
        ):
            object.__setattr__(self, "compute_dtype", DType(self.compute_dtype))
        if self.a_dtype not in _VALID_AB:
            raise ValueError(f"GemmSpec: a_dtype {self.a_dtype!r} not in {_VALID_AB}")
        if self.b_dtype not in _VALID_AB:
            raise ValueError(f"GemmSpec: b_dtype {self.b_dtype!r} not in {_VALID_AB}")
        if self.out_dtype not in _VALID_OUT:
            raise ValueError(f"GemmSpec: out_dtype {self.out_dtype!r} not in {_VALID_OUT}")
        if self.acc_dtype is not DType.F32:
            raise ValueError("GemmSpec: acc_dtype must be F32 (no tf32 path)")
        if self.compute_dtype is not None and self.compute_dtype not in _VALID_COMPUTE:
            raise ValueError(
                f"GemmSpec: compute_dtype {self.compute_dtype!r} not in {_VALID_COMPUTE} "
                "(F32 has no tensor-core MMA; must be BF16/F16/E4M3/E5M2)"
            )
        if self.activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"GemmSpec: activation {self.activation!r} not in {_VALID_ACTIVATIONS}"
            )
        if self.has_gate_residual:
            if self.G < 1 or self.M % self.G != 0:
                raise ValueError(
                    f"GemmSpec: has_gate_residual requires M ({self.M}) divisible by G ({self.G})"
                )
        # When A (or B) is F32 and the caller didn't pick a compute_dtype,
        # we have to downcast on load — F32 has no MMA path. Pick B's
        # dtype if it's MMA-capable, otherwise A's.
        if self.compute_dtype is None and (self.a_dtype is DType.F32 or self.b_dtype is DType.F32):
            fallback = (
                self.b_dtype
                if self.b_dtype in _VALID_COMPUTE
                else self.a_dtype
                if self.a_dtype in _VALID_COMPUTE
                else DType.BF16
            )
            object.__setattr__(self, "compute_dtype", fallback)

    @property
    def compute_dtype_resolved(self) -> DType:
        """Effective compute dtype — source a_dtype when not set, unless
        a_dtype is F32 (no tensor-core F32 MMA) in which case __post_init__
        has already filled ``compute_dtype`` with a valid fallback."""
        return self.compute_dtype if self.compute_dtype is not None else self.a_dtype
