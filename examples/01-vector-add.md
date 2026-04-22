# 01 — Vector add

> `Out[i] = A[i] + B[i]` — the simplest possible GPU kernel. Our job
> is to introduce the `@kernel` surface with as little else going on
> as possible.

## New ideas

* `@kernel` decorator — wires the kernel into quark's registry and
  binds `self.bctx` / `self.g` / `self.m_base` before `build()` runs.
* `TensorDecl` manifest — the kernel's gmem parameter list.
* `build(self)` body — authors IR via `quark.lang as qk`.
* 1D grid + per-thread element loop.

## Shape

One block per `BM`-element tile. Inside the block, `n_warps * 32`
threads split the tile. For a problem of size `N = 4096` and `BM =
256`, the grid is `(N // BM, 1, 1) = (16, 1, 1)` — 16 blocks, each
handling 256 elements.

```
gmem:  A [ 0 1 2 ... 255 | 256 ... 511 | ... ]
block0:       ^^^^^^^^^^        (BM=256 per block)
             threads 0..(n_warps*32)-1 cooperate on this tile
```

## Spec / Config

```python
# spec.py
from dataclasses import dataclass
from quark.ir import DType
from quark.kernels.base import KernelSpec


@dataclass(frozen=True)
class VecAddSpec(KernelSpec):
    N: int
    dtype: DType = DType.F32


# config.py
from dataclasses import dataclass
from quark.kernels.base import KernelConfig


@dataclass(frozen=True)
class VecAddConfig(KernelConfig):
    BM: int = 256       # elements per block
    n_warps: int = 4    # → 128 threads per block

    @classmethod
    def default_for(cls, spec):
        return cls()
```

## build()

```python
# kernel.py
from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel


@kernel("vec_add", spec=VecAddSpec, config=VecAddConfig)
class VecAdd(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("A",   dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,)),
        TensorDecl("B",   dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,)),
        TensorDecl("Out", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.N,), role="out"),
    ]

    spec: VecAddSpec
    config: VecAddConfig

    def grid(self) -> tuple[int, int, int]:
        return (self.spec.N // self.config.BM, 1, 1)

    def flops(self) -> int:
        return self.spec.N  # one add per element

    def is_valid(self) -> bool:
        return self.spec.N % self.config.BM == 0

    @classmethod
    def tune_space(cls):
        return {"BM": [128, 256, 512], "n_warps": [2, 4, 8]}

    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        bctx = self.bctx

        # Per-block starting offset in the flat N-length vector.
        block_base = qk.block_idx("x") * c.BM
        n_threads = c.n_warps * 32
        per_thread = c.BM // n_threads     # each thread handles this many elements

        with qk.for_range(0, per_thread, 1, iv_name="i") as (i, _):
            # Flat index this thread touches on this loop iteration.
            local_idx = bctx.tid * per_thread + i
            gmem_idx = block_base + local_idx

            a = g.A[gmem_idx]              # scalar gmem load
            b = g.B[gmem_idx]
            g.Out[gmem_idx] = a + b        # store with overloaded arithmetic
```

That's it. `@kernel` handles everything else — the decorator:

* Publishes `self.bctx` (a `BlockContext`) and `self.g`
  (`SimpleNamespace` of `GlobalTensor`s keyed by `TensorDecl.name`).
* Emits the function signature + parameter declarations from
  `TENSORS`.
* Calls `build()`, then closes the IR module.
* Registers the kernel in `quark.kernels.registry` so
  `pop functional.vec_add(a, b)` can find it (once the name
  conflict-free helpers are added in `quark.functional`).

## Things to notice

* **`qk.for_range(0, per_thread, 1)`** — Python ints are
  auto-promoted to U32 constants. The loop's body runs per IR
  iteration; the induction variable `i` is a `Value` you use in
  expressions.
* **`g.A[gmem_idx]`** — subscripting a `GlobalTensor` emits a load
  of the element; `g.Out[idx] = value` emits a store.
* **`a + b`** — `Value` has the arithmetic operators overloaded;
  equivalent to `qk.add(a, b)`.
* **No smem, no barriers, no MMA.** Vector add is embarrassingly
  parallel — every thread works independently.

## Authoring surface recap

```python
import quark.lang as qk                  # free-function IR ops
from quark.blocks import TensorDecl       # DSL primitives
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
```

Next up: we introduce shared memory so threads within a block can
cooperate on per-row reductions.

→ [02-block-reduction.md](02-block-reduction.md)
