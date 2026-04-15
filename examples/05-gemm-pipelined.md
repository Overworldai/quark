# 05 — GEMM, double-buffered pipeline

> Same kernel as step 04 — but `n_stages=2`. The compute for chunk
> `k` runs while the loader prefetches chunk `k+1`. On CUDA the
> prefetch fires via `cp.async`; on Metal it's a synchronous load
> that still benefits from two buffer slots (the compiler schedules
> independent loads + MMAs across the K dimension).

## New ideas

* How `produce` / `consume` separate so the pipeline can overlap
  them.
* Automatic `cp.async` orchestration: `run_pipeline` watches
  `Builder.async_emissions` before and after each `produce` call and
  emits the matching `async_commit` / `async_wait` only when produce
  actually fired cp.async.
* Software-pipelined prologue + epilogue — the only contract on the
  kernel body is that `produce(ictx)` is safe to call up to one
  iteration past end-of-K.

## Shape

```
K-dim iteration, n_stages=2:

  prologue: produce(0) → stage[0],  produce(1) → stage[1]
  steady state (K_outer/2 - 1 rounds):
      wait stage[0] → consume(iter=k, stage[0]) → produce(k+2)
      wait stage[1] → consume(iter=k+1, stage[1]) → produce(k+3)
  epilogue: wait stage[0] → consume(K-2, stage[0])
             wait stage[1] → consume(K-1, stage[1])
```

The only difference from step 04 is that `produce` + `consume` run
interleaved rather than sequentially. You don't write the prologue /
steady-state / epilogue yourself — `run_pipeline` lays them out.

## What changes from step 04

**Nothing** in the kernel body. The only move is bumping
`n_stages=2` in the config. The same `build()`, the same
`PipelineBody(...).run(...)` call, the same `MmaBody` — every
abstraction composes forward.

```python
@dataclass(frozen=True)
class GemmConfig(KernelConfig):
    BM: int = 64
    BN: int = 64
    BK: int = 16
    n_warps: int = 4
    n_stages: int = 2           # ← the only change vs step 04
    a_pad: int = 8
    b_pad: int = 8
    main_shape: str = ""
```

When the pipeline runs with `n_stages=2`:

* `SmemPlan.staged_pairs(..., n_stages=2)` allocates **two** A / B
  pairs — stage 0 and stage 1. Each iteration's `ictx.stage` is one
  of the two; the scheduler rotates them.
* `produce(ictx)` fires `ictx.stage.a.load_from(...)` — and
  `load_from` picks cp.async automatically (it's 16B-aligned and
  `cast=None`). So `produce` now emits one or more `cp.async` ops.
* `run_pipeline` sees `Builder.async_emissions` advance during
  `produce`; it wraps the produce in `async_commit` and the
  consume's predecessor in `async_wait(0)`. Kernels with scalar-only
  loaders skip the async stuff.
* The prologue fires `produce(0)` and `produce(1)` before the first
  `consume`. The epilogue calls `consume_tail` (defaults to
  `consume`) on the final two iterations so the last MMA body runs
  against the post-epilogue drained state.

## build() — unchanged

```python
def build(self) -> None:
    s, c, g = self.spec, self.config, self.g
    m_base, n_base = self.m_base, self.n_base
    mma_cfg = self._mma_cfg()

    stages = SmemPlan.staged_pairs(
        s.a_dtype,
        a_shape=(c.BM, c.BK), b_shape=(c.BN, c.BK),
        a_pad=c.a_pad, b_pad=c.b_pad,
        mma_cfg=mma_cfg, n_warps=c.n_warps, b_shuffled=False,
        n_stages=c.n_stages,                              # 2 now
    )
    acc = Accumulators.from_mma(mma_cfg, BM=c.BM, BN=c.BN, n_warps=c.n_warps)
    mma = MmaBody(acc=acc)

    K_outer = s.K // c.BK

    def produce(ictx: IterCtx) -> None:
        k_col = ictx.iter_idx * c.BK
        ictx.stage.a.load_from(g.A, row=m_base, col=k_col)
        ictx.stage.b.load_from(g.B, row=n_base, col=k_col)

    PipelineBody(
        stages=stages, produce=produce, consume=mma, carry=acc,
    ).run(n_iters=K_outer, n_stages=c.n_stages)

    BN_per_warp = (c.BN // mma_cfg.shape.n // c.n_warps) * mma_cfg.shape.n
    pop.store_acc(
        g.Out, acc,
        row=m_base, col=n_base + self.bctx.warp_id * BN_per_warp,
        cast=s.out_dtype,
    )
```

## `consume_tail`

Some kernels (attention in step 06) want the final two iterations
to skip part of the compute — e.g. attention skips the online-softmax
rescale because there's no next chunk to carry `m` / `l` into. Pass
`consume_tail=fn` to `PipelineBody`; `ictx.is_tail` is `True` on
those calls. For plain GEMM you don't need it.

## Runtime `n_iters`

`n_stages=2` needs the iteration count at compile time so the
prologue / epilogue can be laid out. For runtime-variable counts
(owl_attn's inner segment loop) use `n_stages=1` and pass a `Value`
for `n_iters` — the scheduler falls back to the synchronous form.

## Graduation

At this point you've got the full GEMM authoring surface. The real
`kernels/gemm/kernel.py` adds:

* `compute_dtype` path (on-the-fly gmem→smem cast, e.g. bf16→e4m3).
* `b_shuffle=True` with pad-baked-into-stride — the vectorized
  shuffled-B fragment load.
* `is_valid_for(caps)` filtering in the autotuner.
* Real `problems` / `baselines` / `reference` for bench / fuzz /
  autotune.

Look at `src/popcorn/kernels/gemm/kernel.py` — it's under 220 lines
with all of that on top. The primitives we just covered do the
lifting.

## Authoring surface recap

```python
# No new imports from step 04.
# The additions are all behavioural:
n_stages=2                       # async prefetch + double buffer
consume_tail=fn                  # optional per-tail override
# PipelineBody.run + SmemPlan.staged_pairs + load_from all auto-adjust.
```

Next: attention. Multi-GEMM per K iteration, a loop carry that
isn't just a single `Accumulators`, and online softmax that stays
in registers.

→ [06-flash-attention.md](06-flash-attention.md)
