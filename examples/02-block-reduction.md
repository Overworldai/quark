# 02 — Row-wise block reduction

> `Out[i] = sum(A[i, :])` over a row of length `K`. One block per
> row; threads in the block cooperate on the reduction.

## New ideas

* `qk.smem_alloc(name, dtype, shape)` — allocate shared memory.
* `qk.barrier("block")` — synchronize threads within a block.
* `qk.subgroup_reduce(value, op)` — cross-lane reduction inside one
  warp (32 threads, in one `simd`-shuffle on Metal / one `shfl.sync`
  tree on CUDA).
* Two-level reduction: per-thread partial → per-warp reduce →
  barrier → one warp finalizes.

## Shape

```
A: [M, K]
Out: [M]

grid: (M, 1, 1)                  — one block per row
block: (n_warps * 32, 1, 1)
```

Each thread sums `K / n_threads` elements, subgroup-reduces within
its warp into warp-wide partial sums, writes those to smem, then
warp 0 reduces across warps.

## Spec / Config

```python
@dataclass(frozen=True)
class RowSumSpec(KernelSpec):
    M: int
    K: int
    dtype: DType = DType.F32


@dataclass(frozen=True)
class RowSumConfig(KernelConfig):
    n_warps: int = 4

    @classmethod
    def default_for(cls, spec):
        return cls()
```

## build()

```python
from quark.blocks import TensorDecl
from quark.ir import DType
import quark.lang as qk


@kernel("row_sum", spec=RowSumSpec, config=RowSumConfig)
class RowSum(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("A",   dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.M, s.K)),
        TensorDecl("Out", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.M,), role="out"),
    ]

    spec: RowSumSpec
    config: RowSumConfig

    def grid(self): return (self.spec.M, 1, 1)
    def is_valid(self):
        n_threads = self.config.n_warps * 32
        return self.spec.K % n_threads == 0

    @classmethod
    def tune_space(cls):
        return {"n_warps": [1, 2, 4, 8]}

    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        bctx = self.bctx
        n_threads = c.n_warps * 32
        per_thread = s.K // n_threads

        # One block per row: block_idx("x") is the row index.
        row = qk.block_idx("x")

        # Per-warp partial-sum scratch.
        partials = qk.smem_alloc("partials", s.dtype, (c.n_warps,))

        # ── Step 1: per-thread partial sum ──
        acc = qk.const(s.dtype, 0.0)
        with qk.for_range(0, per_thread, 1, iv_name="j") as (j, _):
            col = bctx.tid * per_thread + j
            acc = acc + g.A[row, col]

        # ── Step 2: reduce within the warp (32 lanes → 1 lane) ──
        warp_sum = qk.subgroup_reduce(acc, op="add")

        # Lane 0 of each warp writes its warp's sum into smem.
        with qk.if_(bctx.lane_id == 0) as _:
            partials[bctx.warp_id] = warp_sum
        qk.barrier("block")

        # ── Step 3: warp 0 reduces the n_warps partials ──
        with qk.if_(bctx.warp_id == 0) as _:
            val = qk.const(s.dtype, 0.0)
            with qk.if_(bctx.lane_id < c.n_warps) as _:
                val = partials[bctx.lane_id]
            total = qk.subgroup_reduce(val, op="add")
            with qk.if_(bctx.lane_id == 0) as _:
                g.Out[row] = total
```

## Things to notice

* **`qk.smem_alloc`** — returns a `SharedRegion`. Shape is static.
  Lifetime defaults to `AUTO` (the smem-layout pass infers from use).
* **`qk.barrier("block")`** — synchronises every thread in the
  block. Required between the warp partials writing to smem and warp
  0 reading them back.
* **`qk.subgroup_reduce(val, op="add")`** — 32-lane tree reduction.
  One shuffle per halving step; on Apple it lowers to
  `simd_sum(...)`, on NVIDIA to `__shfl_xor_sync`.
* **`bctx.warp_id` / `bctx.lane_id`** — cached thread-identity
  Values; the `BlockContext` hoists them to one compute per kernel.
* **`qk.if_(pred) as _`** — context manager emitting an
  `IfRegionOp`. Structured control flow only — no `goto`, no early
  return.
* **`partials[bctx.warp_id] = warp_sum`** — subscript on a
  `SharedRegion` emits an smem store.

## Authoring surface recap

```python
import quark.lang as qk

# Additions from step 01:
qk.smem_alloc(name, dtype, shape)
qk.barrier("block")
qk.subgroup_reduce(value, op)
qk.if_(pred)

# bctx helpers (cached on the BlockContext):
bctx.tid / bctx.warp_id / bctx.lane_id
```

Next: row-wise softmax — same shape, but we need **two passes** over
the row (max, then normalize), which means we stage the row in smem
instead of re-reading it from gmem.

→ [03-softmax.md](03-softmax.md)
