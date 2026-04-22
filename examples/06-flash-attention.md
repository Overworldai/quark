# 06 — Flash attention

> `O = softmax(Q @ K^T / √d) @ V` — in registers, streaming over K.
> Q is loaded once per block and lives in MMA A-fragment registers
> for the whole KV loop. We never materialize the `S` matrix; we
> fold `softmax → P → P @ V` into one pass per K-chunk.

This is the capstone. It composes every abstraction we've seen plus
two new ones: `Carry` for the multi-part loop state (`O`, `m`, `l`)
and the `MmaBody(a=, b=, acc=)` Triton-style call form.

## New ideas

* `qk.q_register_load(g_q, smem=, q_row=, warp_id=, MT=, Dh=)` —
  cp.async Q → per-warp smem → register A-fragments. Returns
  `q_frags[mt][kk_step]` — a nested list of MMA-ready Values.
* `Stage.staged(n, **SmemTile.spec(...))` — factory for `n` stages
  with templated tiles. Replaces the per-kernel stage-loop.
* `Carry(**slots)` — typed multi-subset loop carry. Attribute access
  (`ictx.carry.o`, `.m`, `.l`); flatten/unflatten handled
  internally.
* `MmaBody(a=, b=, acc=)` Triton-style form — pass operands
  directly. `a` accepts pre-loaded register fragments
  (`list[list[Value]]` indexed `[mt][inner_k]`), `b` accepts a
  `SmemTile` for load_matrix.
* `OnlineSoftmax` — the one L1 Block specific to attention. Returns
  the rescaled O, new m, new l, and the P A-fragments ready to feed
  GEMM2.
* `qk.store_acc(..., per_warp=True, row_scale=l)` — attention-style
  epilogue: `O /= l`, cast, stage through a per-warp smem slice,
  cooperative vec_store.

## Shape

```
Q: [B * n_q_heads * seq_len, Dh]
K: [B * n_kv_heads * kv_len, Dh]
V^T: [B * n_kv_heads * Dh, kv_len]                (V transposed)
O: [B * n_q_heads * seq_len, Dh]

grid: (seq_len // BlockQRows, B * n_kv_heads, 1)

BlockQRows = NCW * MTiles * 16          (query rows per block)
NCW  = N-contiguous warps (columns of query rows per warp group)
GQA  = n_q_heads / n_kv_heads           (Q heads per KV group)
MTiles = query-row tiles per warp
Dh = head dim
KvTile = K/V chunk size
```

Per-warp loop invariant: Q fragments stored in registers, O / m / l
accumulated across K chunks.

## Flash-attention math in one paragraph

For each K-chunk:

1. `S = Q @ K^T / √d` — one MMA per K chunk (GEMM1).
2. `m_new = max(m_old, rowmax(S))`.
3. `rescale = exp(m_old - m_new)`.
4. `O *= rescale`; `l *= rescale`.
5. `P = exp(S - m_new)`.
6. `l += rowsum(P)`.
7. `O += P @ V` — one MMA per K chunk (GEMM2).

At the end: `O /= l`. `m` and `l` are per-row-class scalars (one per
m-tile per row class); `O` is a full MT×N_DH accumulator grid.
Steps 2-6 are the "online softmax" — they all fold into one
register-level pass (`OnlineSoftmax.__call__`).

## Spec / Config

```python
@dataclass(frozen=True)
class AttnSpec(KernelSpec):
    B: int
    n_kv_heads: int
    gqa_ratio: int
    seq_len: int
    kv_len: int
    Dh: int
    a_dtype: DType = DType.BF16
    b_dtype: DType = DType.BF16
    out_dtype: DType = DType.BF16

    @property
    def n_q_heads(self) -> int:
        return self.n_kv_heads * self.gqa_ratio


@dataclass(frozen=True)
class AttnConfig(KernelConfig):
    KvTile: int = 64
    MTiles: int = 1
    NCW: int = 1
    KvPad: int = 0
    n_stages: int = 1
    main_shape: str = ""

    @classmethod
    def default_for(cls, spec):
        return cls()
```

## build()

```python
import math

import quark.lang as qk
from quark.blocks import (
    Accumulators, Carry, IterCtx, MmaBody, PipelineBody, SmemTile, Stage, TensorDecl,
)
from quark.ir import DType
from quark.kernels.attn.online_softmax_block import OnlineSoftmax


@kernel("attn", spec=AttnSpec, config=AttnConfig)
class Attn(Kernel):
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("Q",      dtype=lambda s, c: s.a_dtype,
                             shape=lambda s, c: (s.B * s.n_q_heads * s.seq_len, s.Dh)),
        TensorDecl("K",      dtype=lambda s, c: s.b_dtype,
                             shape=lambda s, c: (s.B * s.n_kv_heads * s.kv_len, s.Dh)),
        TensorDecl("V_t",    dtype=lambda s, c: s.b_dtype,
                             shape=lambda s, c: (s.B * s.n_kv_heads * s.Dh, s.kv_len)),
        TensorDecl("output", dtype=lambda s, c: s.a_dtype,
                             shape=lambda s, c: (s.B * s.n_q_heads * s.seq_len, s.Dh),
                             role="out"),
    ]

    spec: AttnSpec
    config: AttnConfig
    CORRECTNESS_THRESHOLD: ClassVar[float] = 0.99

    def grid(self):
        s, c = self.spec, self.config
        block_q_rows = c.NCW * c.MTiles * 16
        return (s.seq_len // block_q_rows, s.B * s.n_kv_heads, 1)

    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        bctx, ctx = self.bctx, self.ctx
        Dh, KvTile, MTiles, NCW, GQA = s.Dh, c.KvTile, c.MTiles, c.NCW, s.gqa_ratio
        mma_cfg = self._mma_cfg()
        m_tile, n_tile = mma_cfg.shape.m, mma_cfg.shape.n
        BlockQRows = NCW * MTiles * m_tile
        NK = KvTile // n_tile                    # S n-tiles
        N_DH = Dh // n_tile                      # O n-tiles
        KV_CHUNKS = s.kv_len // KvTile

        # ── Grid decomposition ──
        q_tile_idx = ctx.block_idx("x")
        bh_idx = ctx.block_idx("y")
        warp_id = bctx.warp_id
        gqa_idx, m_idx = warp_id // NCW, warp_id % NCW
        kv_h, b_idx = bh_idx % s.n_kv_heads, bh_idx // s.n_kv_heads
        q_head = kv_h * GQA + gqa_idx

        q_row_warp = (b_idx * s.n_q_heads + q_head) * s.seq_len + (
            q_tile_idx * BlockQRows + m_idx * (MTiles * m_tile)
        )
        k_row_base = bh_idx * s.kv_len
        vt_row_base = bh_idx * Dh

        # ── Q in registers (flash-attn canonical pattern) ──
        # cp.async Q tile into per-warp smem, extract A-frags, register-resident
        # for the whole KV loop. The smem is dead after this — the layout pass
        # can alias its storage with O_stage.
        lcs = mma_cfg.lane_col_step
        q_smem = qk.smem_alloc("Q_smem", s.a_dtype, (BlockQRows, Dh), pad=c.KvPad)
        q_frags = qk.q_register_load(
            g.Q, smem=q_smem, q_row=q_row_warp, warp_id=warp_id,
            MT=MTiles, Dh=Dh, kv_pad=c.KvPad,
        )

        # ── KV pipeline stages ──
        stages = Stage.staged(
            c.n_stages,
            k=SmemTile.spec(s.b_dtype,  (KvTile, Dh), pad=c.KvPad, lane_col_step=lcs),
            vt=SmemTile.spec(s.b_dtype, (Dh, KvTile), pad=c.KvPad, lane_col_step=lcs),
        )

        # ── Accumulators ──
        o_acc = Accumulators(MT=MTiles, NT=N_DH, width=mma_cfg.shape.c_regs)
        s_acc = Accumulators(MT=MTiles, NT=NK,   width=mma_cfg.shape.c_regs)

        mma1    = MmaBody(acc=s_acc)             # Q_regs @ K^T → S
        softmax = OnlineSoftmax(s_acc=s_acc, o_acc=o_acc, scale=1.0 / math.sqrt(Dh))
        mma2    = MmaBody(acc=o_acc)             # P @ V → O

        # Per-row-class count: m16n8 has 2 (dr ∈ {0, 8}); m8n8k8 has 1.
        n_rc = len({dr for dr, _ in mma_cfg.cd_offsets})
        n_ml = MTiles * n_rc

        # ── Typed carry: O + per-(mt, rc) m, l ──
        carry = Carry(
            o=o_acc,                                        # Accumulators
            m=(n_ml, -1e30, DType.F32),                     # (count, init, dtype)
            l=(n_ml,  0.0,  DType.F32),
        )

        def produce(ictx: IterCtx) -> None:
            kv_offset = ictx.iter_idx * KvTile
            ictx.stage.k.load_from(g.K,   row=k_row_base  + kv_offset)
            ictx.stage.vt.load_from(g.V_t, row=vt_row_base, col=kv_offset)

        def consume(ictx: IterCtx) -> Carry:
            stage = ictx.stage
            carry = ictx.carry                              # a Carry, slots rebound

            # GEMM1: S = Q_regs @ K^T. A is pre-loaded register tile.
            s_vals = mma1(a=q_frags, b=stage.k, acc=s_acc.init())

            # Online softmax — pure register math; no barrier.
            o_vals, m_vals, l_vals, p_frags = softmax(
                s_acc_vals=s_vals,
                o_vals=carry.o, m_vals=carry.m, l_vals=carry.l,
            )

            # GEMM2: O += P @ V. A is the P fragment from softmax; B is V^T smem.
            carry.o = mma2(a=p_frags, b=stage.vt, acc=o_vals)
            carry.m = m_vals
            carry.l = l_vals
            return carry                                    # pipeline flattens for yield

        final = PipelineBody(
            stages=stages, produce=produce, consume=consume, carry=carry,
        ).run(n_iters=KV_CHUNKS, n_stages=c.n_stages)

        # ── Epilogue: O /= l, scale + cast + per-warp staged → vec_store ──
        # `final` is the Carry rebound to loop-end values. `o_acc.results`
        # was populated by the pipeline. Staging smem is auto-allocated
        # (AUTO lifetime — the smem-layout pass aliases it over the now-dead
        # Q_smem); `warp_id` defaults to bctx.warp_id.
        qk.store_acc(
            g.output, o_acc,
            row=q_row_warp, col=0,
            cast=s.a_dtype,
            per_warp=True,
            row_scale=final.l,                              # O /= l
        )
```

## Things to notice

* **`qk.q_register_load(...)`** — issues cp.async, waits, barriers,
  then extracts the MMA A-fragments into a nested
  `list[list[Value]]`. The smem allocation you pass in is yours —
  the lifetime pass will alias it with the output staging smem
  later, saving you a separate O_stage buffer.
* **`Stage.staged(c.n_stages, k=..., vt=...)`** — one line replaces
  the old `stages = []; for i in range(c.n_stages): stages.append(
  {"k": ..., "vt": ...})` loop. Each `SmemTile.spec(...)` is a
  frozen factory; `Stage.staged` names the per-stage tiles `K_s0`,
  `K_s1`, `Vt_s0`, `Vt_s1`, etc.
* **`Carry(**slots)`** — declares the loop-carried state with
  mixed-type slots. `o=o_acc` (an Accumulators with a full grid of
  vec-Values), `m=(count, init_val, dtype)` (a scalar array). Inside
  `consume`, `ictx.carry.o` returns the bound `list[Value]`. After
  the pipeline completes, `final.o`, `final.m`, `final.l` read the
  loop-end values directly.
* **`MmaBody(a=q_frags, b=stage.k, acc=s_acc.init())`** — the
  Triton-style form. `a` is a `list[list[Value]]` indexed
  `[mt][inner_k]`; `b` is a `SmemTile` (internal B fragment loads go
  through `load_matrix`). Same `MmaBody` class used for the
  smem-backed A in the GEMM example — the only difference is what
  you hand it.
* **`OnlineSoftmax(...)`** — one of the few L1 Blocks that has real
  structure (it composes `frag_apply`, `frag_reduce`, `frag_convert`
  to stay in registers throughout). Callable:
  `softmax(s_acc_vals=, o_vals=, m_vals=, l_vals=)` returns the
  updated tuple + P A-fragments for GEMM2.
* **`qk.store_acc(..., per_warp=True, row_scale=final.l)`** —
  attention epilogue. `per_warp=True` stages the block's output into
  a per-warp slice of smem and stores warp-local row ranges with the
  warp's 32 lanes. `row_scale=final.l` applies `O /= l` during the
  scatter. Staging smem auto-allocates and the lifetime pass aliases
  it with the dead Q_smem — zero extra smem budget.

## What we skipped

The kernel above is close to production. The real
`src/quark/kernels/attn/kernel.py` adds:

* `consume_tail=` for the `n_stages=2` last-iter skip.
* `is_valid_for(caps)` filtering.
* Real problem / baseline / reference wiring.

And `kernels/owl_attn/kernel.py` is a bigger superset: fused RoPE
on Q between the cp.async and the ldmatrix, segment-variable KV
iteration, and a custom outer `for_range` that yields nested Carry
state. It's ~700 lines end-to-end; every new abstraction pulls its
weight.

## Authoring surface recap

```python
# New from step 05:
from quark.blocks import Carry, Stage, SmemTile, SmemTileSpec
import quark.lang as qk

q_frags = qk.q_register_load(g_q, smem=, q_row=, warp_id=, MT=, Dh=, kv_pad=)
stages = Stage.staged(n, **SmemTile.spec(dtype, shape, pad=, lane_col_step=))
carry = Carry(o=..., m=(count, init, dtype), l=(count, init, dtype))
s_vals = mma(a=q_frags, b=stage.k, acc=s_acc.init())      # Triton-style
qk.store_acc(..., per_warp=True, row_scale=final.l)
```

---

That's the full progression — from a 10-line vector add to a
production flash-attention kernel, with each abstraction introduced
only when the previous one started creaking. The same primitives
cover GEMM, MoE, KV-cache append, and the various attention
variants. Every registered kernel in `src/quark/kernels/` is built
from this toolkit.

If you're about to write a new kernel, start from
[ADDING_A_KERNEL.md](../docs/ADDING_A_KERNEL.md) for the folder
contract, and reach back here when you need to remember which
abstraction fits which shape.
