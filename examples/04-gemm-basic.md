# 04 — GEMM, synchronous pipeline

> `C[M, N] = A[M, K] @ B^T[N, K]^T`, one block per `(BM, BN)` output
> tile, MMA tensor cores for the inner product. This is the first
> kernel where popcorn's L1 / L2 abstractions pay off.

## New ideas

* `SmemPlan.staged_pairs(dtype, *, a_shape, b_shape, …, n_stages)` —
  allocates paired A/B smem tiles with MMA-aware per-lane views. One
  plan per pipeline stage.
* `Accumulators.from_mma(mma_cfg, BM=, BN=, n_warps=)` — declarative
  MMA accumulator grid. `MT × NT` vec-4 f32 Values per warp.
* `MmaBody(acc=acc)` — the triple-nested `(K_inner, MT, NT)` MMA
  tile loop. Callable directly as the pipeline's `consume`.
* `PipelineBody(stages, produce, consume, carry).run(n_iters=,
  n_stages=)` — the closure-body K-loop. Single code path covers
  `n_stages ∈ {1, 2}`.
* `pop.store_acc(dst, acc, row=, col=, cast=)` — unified epilogue.
  Stages through smem + cooperative vec_store by default.

## Shape

```
A: [M, K]    B: [N, K]    (B stored as [N, K] — K is the fast axis)
Out: [M, N]

grid:  (N // BN, M // BM, 1)           — 2D grid; x=N, y=M
block: (n_warps * 32, 1, 1)
```

K-loop iterates `K_outer = K // BK` times, each iteration loading
one `(BM, BK)` A-tile + one `(BN, BK)` B-tile into smem and
executing `MT × NT × K_inner` MMAs.

`MT = BM / shape.m`, `NT = (BN / shape.n) / n_warps` — the `NT` is
the per-warp slice of the BN columns.

## Spec / Config

```python
@dataclass(frozen=True)
class GemmSpec(KernelSpec):
    M: int
    N: int
    K: int
    a_dtype: DType = DType.BF16
    b_dtype: DType = DType.BF16
    out_dtype: DType = DType.BF16

    def __post_init__(self):
        for f in ("a_dtype", "b_dtype", "out_dtype"):
            v = getattr(self, f)
            if isinstance(v, str) and not isinstance(v, DType):
                object.__setattr__(self, f, DType(v))


@dataclass(frozen=True)
class GemmConfig(KernelConfig):
    BM: int = 64
    BN: int = 64
    BK: int = 16
    n_warps: int = 4
    n_stages: int = 1            # step 05 flips this to 2
    a_pad: int = 8
    b_pad: int = 8
    main_shape: str = ""         # autotune fills from caps.matmul_shapes

    @classmethod
    def default_for(cls, spec):
        return cls()
```

## build()

```python
import popcorn.lang as pop
from popcorn.blocks import (
    Accumulators, IterCtx, MmaBody, PipelineBody, SmemPlan, TensorDecl,
)
from popcorn.kernels.base import Kernel, MmaSite


@kernel("gemm_basic", spec=GemmSpec, config=GemmConfig)
class GemmBasic(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("A",   dtype=lambda s, c: s.a_dtype, shape=lambda s, c: (s.M, s.K)),
        TensorDecl("B",   dtype=lambda s, c: s.b_dtype, shape=lambda s, c: (s.N, s.K)),
        TensorDecl("Out", dtype=lambda s, c: s.out_dtype, shape=lambda s, c: (s.M, s.N),
                   role="out"),
    ]

    spec: GemmSpec
    config: GemmConfig

    def grid(self):
        return (self.spec.N // self.config.BN, self.spec.M // self.config.BM, 1)

    def flops(self):
        return 2 * self.spec.M * self.spec.N * self.spec.K

    def is_valid(self):
        s, c = self.spec, self.config
        return (s.M % c.BM == 0 and s.N % c.BN == 0 and s.K % c.BK == 0)

    @classmethod
    def tune_space(cls):
        return {
            "BM": [32, 64, 128], "BN": [32, 64, 128], "BK": [16, 32],
            "n_warps": [2, 4, 8], "n_stages": [1, 2],
            "a_pad": [0, 8], "b_pad": [0, 8],
        }

    @classmethod
    def mma_sites(cls, spec):
        if spec is None:
            return []
        return [MmaSite(name="main", a_dtype=spec.a_dtype, b_dtype=spec.b_dtype)]

    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        m_base, n_base = self.m_base, self.n_base         # auto-bound by @kernel
        mma_cfg = self._mma_cfg()

        # One paired A/B SmemPlan per pipeline stage. For n_stages=1
        # the list has one entry.
        stages = SmemPlan.staged_pairs(
            s.a_dtype,                                    # compute dtype
            a_shape=(c.BM, c.BK), b_shape=(c.BN, c.BK),
            a_pad=c.a_pad, b_pad=c.b_pad,
            mma_cfg=mma_cfg, n_warps=c.n_warps,
            b_shuffled=False,
            n_stages=c.n_stages,
        )

        # Accumulator grid — one vec-4 f32 per (mt, nt) per warp.
        acc = Accumulators.from_mma(mma_cfg, BM=c.BM, BN=c.BN, n_warps=c.n_warps)

        # MMA body. shape defaults to active_bctx().mma_cfg; K_inner
        # is inferred from the B tile's shape[1] at call time.
        mma = MmaBody(acc=acc)

        K_outer = s.K // c.BK

        def produce(ictx: IterCtx) -> None:
            # Load the (BM, BK) A slice and (BN, BK) B slice for this K iter.
            k_col = ictx.iter_idx * c.BK
            ictx.stage.a.load_from(g.A, row=m_base, col=k_col)
            ictx.stage.b.load_from(g.B, row=n_base, col=k_col)

        # The pipeline. With n_stages=1 this is a straight sync loop:
        #   produce → barrier → consume → barrier → yield new carry.
        PipelineBody(
            stages=stages, produce=produce, consume=mma, carry=acc,
        ).run(n_iters=K_outer, n_stages=c.n_stages)

        # After .run: acc.results is populated with the loop-final Values.
        # Per-warp N offset — each warp owns a BN/n_warps stripe of N.
        BN_per_warp = (c.BN // mma_cfg.shape.n // c.n_warps) * mma_cfg.shape.n
        pop.store_acc(
            g.Out, acc,
            row=m_base,
            col=n_base + self.bctx.warp_id * BN_per_warp,
            cast=s.out_dtype,
        )
```

## Things to notice

* **`self.m_base` / `self.n_base`** — the `@kernel` decorator binds
  these from `config.BM` / `config.BN` since grid uses the `(N/BN,
  M/BM, 1)` convention. Equivalent to
  `pop.block_base("y", BM)` / `pop.block_base("x", BN)`.
* **`SmemPlan.staged_pairs(...)`** — collapses `n_stages` stage
  constructions + per-lane B view fixups into one call. Each
  returned `SmemPlan` has `.a` / `.b` `SmemTile`s and `.a_lane[0]` /
  `.b_lane[0]` per-lane `SharedRegion` views for MMA frag loads.
  `b_shuffled=True` activates the vectorized ld.shared.v4 path.
* **`Accumulators.from_mma`** — works out MT / NT / width from the
  MMA config so you never hardcode them against a specific
  `m16n8k16` / `m8n8k8` shape. The `width` field is a per-lane vec
  length; `acc.init()` emits the zero-initialized Values.
* **`MmaBody(acc=acc)`** — no `shape=` argument needed. It defaults
  to `active_bctx().mma_cfg` (the decorator set this from
  `config.main_shape`). `K_inner` is inferred from the B tile's
  `shape[1]` at call time. Only when the tile's K dim isn't axis 1
  (a rare layout) do you pass `K_inner=` explicitly.
* **`PipelineBody(...).run(n_iters=, n_stages=)`** — the fluent
  form. Internally just calls `run_pipeline(body=self, n_iters=,
  n_stages=)`; using `.run` keeps the construction + launch in one
  expression.
* **`consume=mma`** — `MmaBody` is callable; `mma(ictx)` reads the
  SmemPlan stage on `ictx.stage` + the carry on `ictx.carry` and
  returns the updated carry. When you pass `consume=mma` the
  pipeline calls `mma(ictx)` every iteration.
* **`produce` runs every iteration** at `n_stages=1`: prefetch →
  barrier → compute → barrier → yield. The barriers are implicit;
  `run_pipeline` adds them.
* **`pop.store_acc`** — takes the `Accumulators` by identity; pulls
  the tile values from `acc.results` and writes them to gmem.
  Defaults to the staged-smem fast path — scalar scatter only fires
  for problems too small to vectorize.

## Authoring surface recap

```python
# Additions from steps 01-03:
from popcorn.blocks import (
    Accumulators, IterCtx, MmaBody, PipelineBody, SmemPlan,
)

SmemPlan.staged_pairs(dtype, a_shape=, b_shape=, ..., n_stages=)
Accumulators.from_mma(mma_cfg, BM=, BN=, n_warps=)
MmaBody(acc=)                                   # shape + K_inner inferred
PipelineBody(stages=, produce=, consume=, carry=).run(n_iters=, n_stages=)
pop.store_acc(dst, acc, row=, col=, cast=)
```

Next: flip `n_stages` to 2 and see the produce/consume contract pay
off with an asynchronous prefetch.

→ [05-gemm-pipelined.md](05-gemm-pipelined.md)
