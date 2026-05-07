"""Increment — ``t[0] += 1`` on a single-element s32 tensor.

Exists solely for ``QuarkTensor.increment()``: a 1-block, 1-thread
kernel that keeps the op async (graph-capturable) so CUDA graphs work
end-to-end. The previous host-round-trip path stalls the stream and
can't be captured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel, KernelConfig, KernelSpec
from quark.kernels.decorator import kernel


@dataclass(frozen=True)
class IncrementSpec(KernelSpec):
    dtype: DType = DType.S32

    def __post_init__(self):
        if isinstance(self.dtype, str) and not isinstance(self.dtype, DType):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if self.dtype not in (DType.S32, DType.U32):
            raise ValueError(f"IncrementSpec: dtype must be S32 or U32, got {self.dtype!r}")


@dataclass(frozen=True)
class IncrementConfig(KernelConfig):
    @classmethod
    def default_for(cls, spec) -> IncrementConfig:
        return cls()


def _reference(spec, *, T):

    from quark.runtime.npconv import astype_numpy, to_f32_numpy

    val = to_f32_numpy(T, dtype_hint=spec.dtype.value)
    return astype_numpy(val + 1.0, spec.dtype).reshape(T.shape if hasattr(T, "shape") else (1,))


@kernel(
    "increment",
    spec=IncrementSpec,
    config=IncrementConfig,
    output_idx=-1,
    problems=lambda: [],
    baselines=lambda: [],
    reference=_reference,
)
class IncrementKernel(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("T", dtype=lambda s, c: s.dtype, shape=lambda s, c: (1,), role="out"),
    ]

    spec: IncrementSpec
    config: IncrementConfig

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    def is_valid(self) -> bool:
        return True

    def grid(self) -> tuple[int, int, int]:
        return (1, 1, 1)

    def block(self) -> tuple[int, int, int]:
        return (1, 1, 1)

    def flops(self) -> int:
        return 1

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        return {}

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        from quark.runtime.npconv import zeros_for_dtype

        del seed
        spec = IncrementSpec(**problem)
        return {"T": zeros_for_dtype((1,), spec.dtype)}

    @classmethod
    def spec_from_tensors(cls, T) -> IncrementSpec:
        return IncrementSpec(dtype=DType.from_backend(T))

    def build(self) -> None:
        g = self.g
        bctx = self.bctx

        # 1 block × 1 thread. Only tid==0 runs (trivially). Single
        # load-add-store at offset 0.
        dtype = self.spec.dtype
        zero = bctx.c(0, dtype=DType.U32)
        one = bctx.c(1, dtype=dtype)
        val = qk.load(g.T, zero)
        g.T[zero] = val + one
