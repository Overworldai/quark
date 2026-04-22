# 03 — Row-wise softmax

> `Out[i, :] = softmax(A[i, :])` — stable softmax over a row. We need
> the row max (pass 1), then exp-and-sum (pass 2), then normalize
> (pass 3). That means re-reading every element at least twice.

## New ideas

* `SmemTile(name, dtype, shape, pad=)` — declarative smem tile with
  auto-allocation.
* `smem_tile.load_from(gmem, row=, col=, cast=)` — cooperative
  gmem→smem load. Dispatches cp.async (CUDA) or synchronous
  vec_load (Metal) automatically.
* `qk.store_acc(...)` — unified epilogue helper (we'll see the
  MMA-driven form in step 4; here we use the scalar path).

## Shape

```
A: [M, K]
Out: [M, K]

grid:  (M, 1, 1)                      — one block per row
block: (n_warps * 32, 1, 1)
```

Instead of reading `A[i, :]` from gmem three times, we load it into
smem once and do all three passes over the smem copy. `pad=0` is
fine since `K` is typically a multiple of the smem row alignment.

## Spec / Config

```python
@dataclass(frozen=True)
class SoftmaxSpec(KernelSpec):
    M: int
    K: int
    dtype: DType = DType.F32


@dataclass(frozen=True)
class SoftmaxConfig(KernelConfig):
    n_warps: int = 4

    @classmethod
    def default_for(cls, spec):
        return cls()
```

## build()

```python
import math
import quark.lang as qk
from quark.blocks import SmemTile, TensorDecl
from quark.ir import DType


_LOG2E = math.log2(math.e)


@kernel("softmax", spec=SoftmaxSpec, config=SoftmaxConfig)
class Softmax(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("A",   dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.M, s.K)),
        TensorDecl("Out", dtype=lambda s, c: s.dtype, shape=lambda s, c: (s.M, s.K), role="out"),
    ]

    spec: SoftmaxSpec
    config: SoftmaxConfig

    def grid(self): return (self.spec.M, 1, 1)
    def is_valid(self):
        n_threads = self.config.n_warps * 32
        return self.spec.K % n_threads == 0

    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        bctx = self.bctx
        n_threads = c.n_warps * 32
        per_thread = s.K // n_threads
        row = qk.block_idx("x")

        # Stage A[row, :] in smem so we can pass over it multiple times.
        row_tile = SmemTile("row", s.dtype, (1, s.K))
        row_tile.load_from(g.A, row=row, col=0)
        qk.barrier("block")

        # Scratch for the warp-level partials (max then sum).
        scratch = qk.smem_alloc("scratch", s.dtype, (c.n_warps,))

        # ── Pass 1: row max ──
        local_max = qk.const(s.dtype, -1e30)
        with qk.for_range(0, per_thread, 1, iv_name="j") as (j, _):
            col = bctx.tid * per_thread + j
            local_max = qk.max(local_max, row_tile.smem[0, col])

        warp_max = qk.subgroup_reduce(local_max, op="max")
        with qk.if_(bctx.lane_id == 0) as _:
            scratch[bctx.warp_id] = warp_max
        qk.barrier("block")

        # Every thread reads the n_warps values from smem and reduces.
        row_max = qk.const(s.dtype, -1e30)
        with qk.for_range(0, c.n_warps, 1, iv_name="w") as (w, _):
            row_max = qk.max(row_max, scratch[w])

        # ── Pass 2: exp(x - max) and sum ──
        log2e = qk.const(s.dtype, _LOG2E)
        local_sum = qk.const(s.dtype, 0.0)
        with qk.for_range(0, per_thread, 1, iv_name="j") as (j, _):
            col = bctx.tid * per_thread + j
            # exp(x) = exp2(x * log2(e)); ex2_approx is fast-math.
            e = qk.ex2_approx((row_tile.smem[0, col] - row_max) * log2e)
            row_tile.smem[0, col] = e     # overwrite smem with exp values
            local_sum = local_sum + e

        warp_sum = qk.subgroup_reduce(local_sum, op="add")
        with qk.if_(bctx.lane_id == 0) as _:
            scratch[bctx.warp_id] = warp_sum
        qk.barrier("block")

        row_sum = qk.const(s.dtype, 0.0)
        with qk.for_range(0, c.n_warps, 1, iv_name="w") as (w, _):
            row_sum = row_sum + scratch[w]

        inv_sum = qk.rcp_approx(row_sum)

        # ── Pass 3: normalize and store ──
        with qk.for_range(0, per_thread, 1, iv_name="j") as (j, _):
            col = bctx.tid * per_thread + j
            g.Out[row, col] = row_tile.smem[0, col] * inv_sum
```

## Things to notice

* **`SmemTile("row", dtype, (1, K))`** — declarative. Auto-emits the
  allocation via the active `BlockContext`; you never call `.emit()`.
  The `.smem` attribute gives you the `SharedRegion` for direct
  subscript R/W.
* **`row_tile.load_from(g.A, row=row, col=0)`** — cooperative tile
  load. Under the hood: chooses cp.async 16B lines (CUDA) or
  `simdgroup_load`/`vec_load` (Metal), spreads the work across the
  block's threads, handles the ragged tail. Barrier after is your
  responsibility.
* **`qk.ex2_approx(x)` / `qk.rcp_approx(x)`** — fast-math
  approximations. Lower to `ex2.approx.f32` on PTX and `fast::exp2`
  / `1.0f / x` on MSL.
* **`qk.max(a, b)`** — two-arg scalar max. (There's also
  `qk.max(frag, dim=)` sugar we'll use in the attention example
  that routes to `frag_reduce` over an MMA fragment.)
* **`row_tile.smem[0, col] = ...`** — we overwrite the smem tile
  with the exp values in-place to save one more barrier + register
  pass. The only constraint is that writes happen within the same
  pass as their later reads, which the per-thread `col` stride
  guarantees.

## Authoring surface recap

```python
# Additions from steps 01-02:
SmemTile(name, dtype, shape, pad=)           # DSL primitive
smem_tile.load_from(gmem, row=, col=, cast=) # cooperative gmem→smem
smem_tile.smem[row, col]                     # subscript R/W
qk.ex2_approx(x)  /  qk.rcp_approx(x)
qk.max(a, b) / qk.min(a, b)                # scalar; dim= sugar later
```

Next: the first kernel that uses the tensor cores. We drop the
scalar arithmetic and start composing with `SmemPlan`,
`Accumulators`, and `MmaBody`.

→ [04-gemm-basic.md](04-gemm-basic.md)
