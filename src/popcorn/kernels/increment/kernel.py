"""Increment — ``t[0] += 1`` on a single-element s32 tensor.

Exists solely for ``PopcornTensor.increment()``: a 1-block, 1-thread
kernel that keeps the op async (graph-capturable) so CUDA graphs work
end-to-end. The previous host-round-trip path stalls the stream and
can't be captured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import popcorn.lang as pop
from popcorn.blocks import TensorDecl
from popcorn.ir import DType
from popcorn.kernels.base import Kernel, KernelConfig, KernelSpec
from popcorn.kernels.decorator import kernel


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


def _reference(kernel, T):
    from popcorn.backend import PT

    one = PT.tensor([1], dtype=T.dtype)
    return T + one  # returns new tensor; caller compares flat values


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
    def make_tensors(cls, problem: dict) -> dict:
        from popcorn.backend import PT

        spec = IncrementSpec(**problem)
        return {"T": PT.zeros(1, dtype=spec.dtype.backend)}

    @classmethod
    def spec_from_tensors(cls, T) -> IncrementSpec:
        return IncrementSpec(dtype=DType.from_backend(T.dtype))

    def build(self) -> None:
        g = self.g
        bctx = self.bctx

        # 1 block × 1 thread. Only tid==0 runs (trivially). Single
        # load-add-store at offset 0.
        dtype = self.spec.dtype
        zero = bctx.c(0, dtype=DType.U32)
        one = bctx.c(1, dtype=dtype)
        val = pop.load(g.T, zero)
        g.T[zero] = val + one
