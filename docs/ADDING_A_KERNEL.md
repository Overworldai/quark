# Adding a kernel

Copy an existing kernel folder that matches the pattern (`gemm` for
matmul-like, `attn` for flash-attention-like, `moe_inproj` for
gather-scatter) and edit from there. This doc walks the contract.

For a gentler introduction that builds the abstractions from scratch,
see the [examples/](../examples/) walk-through.

## Folder

```
src/popcorn/kernels/my_kernel/
    __init__.py            imports kernel.py to trigger @kernel registration
    spec.py                frozen KernelSpec — problem definition
    config.py              frozen KernelConfig — tuning knobs + default_for(spec)
    kernel.py              @kernel class + build() body
    reference.py           backend-agnostic correctness oracle
    problems.py            bench/fuzz problem list
    baselines.py           reference implementations to measure against
```

Zero edits elsewhere. `kernels/__init__.py` auto-imports every
subfolder; the `@kernel` decorator registers the class.

## 1. spec.py

Frozen dataclass. Fields are the problem dimensions. Must be
hashable (enforced by autotune cache). DTypes are first-class
`popcorn.ir.DType` values — no parallel `*_ir_dtype` properties.

```python
from dataclasses import dataclass
from popcorn.ir import DType
from popcorn.kernels.base import KernelSpec

_VALID_AB = frozenset({DType.BF16, DType.F16, DType.E4M3, DType.E5M2})


@dataclass(frozen=True)
class MySpec(KernelSpec):
    M: int
    N: int
    K: int
    a_dtype: DType = DType.BF16
    b_dtype: DType = DType.BF16
    out_dtype: DType = DType.BF16
    compute_dtype: DType | None = None    # optional; defaults to a_dtype (no cast)

    def __post_init__(self):
        # Coerce strings-from-Problem.params into DType.
        for field in ("a_dtype", "b_dtype", "out_dtype"):
            val = getattr(self, field)
            if isinstance(val, str) and not isinstance(val, DType):
                object.__setattr__(self, field, DType(val))
        if self.a_dtype not in _VALID_AB:
            raise ValueError(f"MySpec: a_dtype {self.a_dtype!r} not in {_VALID_AB}")
        # ... validate the rest ...

    @property
    def compute_dtype_resolved(self) -> DType:
        return self.compute_dtype if self.compute_dtype is not None else self.a_dtype
```

Guidelines:

- DTypes are `popcorn.ir.DType` directly. Problem dicts may pass
  strings (`"bf16"`); the `__post_init__` coerces.
- Property-derived fields (`compute_dtype_resolved`, `total_slots`)
  stay out of the dataclass footprint so frozen+hashable stays clean.
- Reject bad inputs in `__post_init__` with a clear message.

## 2. config.py

Frozen dataclass with `default_for(spec)` classmethod.

```python
from dataclasses import dataclass
from popcorn.kernels.base import KernelConfig


@dataclass(frozen=True)
class MyConfig(KernelConfig):
    BM: int = 64
    BN: int = 64
    BK: int = 16
    n_warps: int = 4
    n_stages: int = 2
    a_pad: int = 0
    b_pad: int = 0
    # Per-MMA-site shape (see MmaSite). Autotune fills from
    # device.caps.matmul_shapes filtered by `mma_sites()`. Empty
    # string falls back to the kernel's compute-dtype default.
    main_shape: str = ""

    @classmethod
    def default_for(cls, spec) -> "MyConfig":
        # Spec-dependent conservative defaults — the "compiles everywhere"
        # fallback when no tuned config is on disk. The autotuner
        # overrides via from_problem_cached. Leave `main_shape` empty;
        # `_mma_cfg` resolves to the compute-dtype default.
        fp8 = spec.compute_dtype_resolved.bytes == 1
        return cls(BM=64, BN=64, BK=32 if fp8 else 16, n_warps=4, n_stages=2)
```

The `default_for` classmethod is what `Kernel.from_problem` reaches
for. Keep it backend-agnostic (no `IS_METAL` here).

For each MMA call site the kernel emits, add one
`<site_name>_shape` field. The autotuner fills them from
`device.caps.matmul_shapes` intersected with the site's dtype axes.

## 3. reference.py

Backend-agnostic torch-or-mlx reference via `PT`.

```python
from popcorn.backend import PT


def my_reference(kernel, A, B):
    s = kernel.spec
    A_f = PT.astype(A, PT.float32)
    B_f = PT.astype(B, PT.float32)
    C = PT.matmul(A_f, PT.transpose(B_f))
    return PT.astype(C, s.out_dtype.backend)
```

The decorator's `reference=` hook passes `(kernel, *tensors)`. Pull
spec / config off `kernel` instead of taking them as args so the
signature stays aligned with the kernel's `make_tensors` dict
(which `bench` / `fuzz` unpack positionally).

Rule: **no raw torch / mlx calls**. Everything goes through PT. See
[BACKEND.md](BACKEND.md).

## 4. problems.py

Production and regression problems. Tag them so tag-filtered bench
sweeps pull the right ones.

```python
from popcorn.kernels.base import Problem


def my_problems() -> list[Problem]:
    return [
        Problem("smoke",
                {"M": 256, "N": 256, "K": 256},
                tags={"smoke", "small"}),
        Problem("prod_360p_metal",
                {"M": 128, "N": 4096, "K": 2048},
                tags={"metal", "metal-dense", "production"}),
        Problem("prod_360p_cuda",
                {"M": 128, "N": 4096, "K": 2048,
                 "b_dtype": "e4m3", "compute_dtype": "e4m3",
                 "b_shuffle": True},
                tags={"cuda", "cuda-dense", "production"}),
    ]
```

Canonical tag scheme (see [BENCHMARKING.md](BENCHMARKING.md#tags)):

- `smoke` — CI-fast regression (tag **at least one** problem).
- `metal` / `metal-dense` / `metal-moe` — Metal model paths.
- `cuda` / `cuda-dense` / `cuda-moe` — CUDA model paths (fp8 + shuffle).
- `production` — superset of real model workloads.

## 5. baselines.py

```python
from popcorn.backend import IS_METAL, PT
from popcorn.kernels.base import Baseline


def my_baselines(kernel, tensors: dict) -> list[Baseline]:
    A, B = tensors["A"], tensors["B"]

    def run():
        C = PT.matmul(A, PT.transpose(B))
        PT.synchronize()

    return [Baseline("PT.matmul", run)]
```

`baselines=` hook signature: `fn(kernel, tensors) -> list[Baseline]`.
Each `Baseline(name, fn)` is a callable the bench harness times.

Backend-specialized baselines (`torch._scaled_mm`, `flex_attention`,
`mx.fast.scaled_dot_product_attention`) are allowed here — baselines
by definition measure backend-fast references. Use `IS_METAL` to
route.

## 6. kernel.py

```python
from typing import ClassVar

import popcorn.lang as pop
from popcorn.blocks import (
    Accumulators, IterCtx, MmaBody, PipelineBody, SmemPlan, TensorDecl,
)
from popcorn.ir import DType
from popcorn.kernels.base import Kernel, MmaSite
from popcorn.kernels.decorator import kernel
from popcorn.kernels.my_kernel.baselines import my_baselines
from popcorn.kernels.my_kernel.config import MyConfig
from popcorn.kernels.my_kernel.problems import my_problems
from popcorn.kernels.my_kernel.reference import my_reference
from popcorn.kernels.my_kernel.spec import MySpec


@kernel(
    "my_kernel",
    spec=MySpec,
    config=MyConfig,
    output_idx=-1,                 # which TENSORS entry is the output (default: last)
    problems=my_problems,
    baselines=my_baselines,
    reference=my_reference,
)
class MyKernel(Kernel):
    # ── Optional: override correctness gate ──
    CORRECTNESS_THRESHOLD: ClassVar[float | None] = None   # None = dtype table

    # ── Gmem buffer manifest ──
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl("A",   dtype=lambda s, c: s.a_dtype,
                          shape=lambda s, c: (s.M, s.K)),
        TensorDecl("B",   dtype=lambda s, c: s.b_dtype,
                          shape=lambda s, c: (s.N, s.K)),
        TensorDecl("Out", dtype=lambda s, c: s.out_dtype,
                          shape=lambda s, c: (s.M, s.N),
                          role="out"),
    ]

    spec: MySpec                   # type aliases — read by build()
    config: MyConfig

    # ── Required: grid / flops / validation ──
    def grid(self) -> tuple[int, int, int]:
        return (self.spec.N // self.config.BN, self.spec.M // self.config.BM, 1)

    def flops(self) -> int:
        return 2 * self.spec.M * self.spec.N * self.spec.K

    def is_valid(self) -> bool:
        c, s = self.config, self.spec
        return (s.M % c.BM == 0 and s.N % c.BN == 0 and s.K % c.BK == 0
                and c.BK >= 16)

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        # Don't include `<site>_shape` here — `tune_space_resolved`
        # injects it at autotune time from `device.caps.matmul_shapes`.
        return {
            "BM": [32, 64, 128],
            "BN": [32, 64, 128],
            "BK": [16, 32],
            "n_warps": [2, 4, 8],
            "n_stages": [1, 2],
            "a_pad": [0, 8],
            "b_pad": [0, 8],
        }

    @classmethod
    def mma_sites(cls, spec) -> list[MmaSite]:
        if spec is None:
            return []
        compute = spec.compute_dtype_resolved
        return [MmaSite(name="main", a_dtype=compute, b_dtype=compute)]

    @classmethod
    def make_tensors(cls, problem: dict) -> dict:
        from popcorn.backend import PT
        spec = MySpec(**problem)
        return {
            "A":   PT.astype(PT.randn(spec.M, spec.K), spec.a_dtype.backend),
            "B":   PT.astype(PT.randn(spec.N, spec.K), spec.b_dtype.backend),
            "Out": PT.zeros(spec.M, spec.N, dtype=spec.out_dtype.backend),
        }

    # param_spec() / _mma_cfg() / block() / entry_name() have sensible
    # defaults on Kernel. Only override if the kernel needs custom logic
    # (e.g. attn's _mma_cfg uses spec.a_dtype / spec.b_dtype instead of
    # the compute-dtype default).

    # ── The actual kernel ──
    # @kernel auto-publishes self.bctx, and binds self.m_base /
    # self.n_base from config.BM / config.BN before build() runs.
    def build(self) -> None:
        s, c, g = self.spec, self.config, self.g
        bctx, m_base, n_base = self.bctx, self.m_base, self.n_base
        compute = s.compute_dtype_resolved
        mma_cfg = self._mma_cfg()
        a_cast = compute if s.a_dtype is not compute else None
        b_cast = compute if s.b_dtype is not compute else None

        stages = SmemPlan.staged_pairs(
            compute,
            a_shape=(c.BM, c.BK), b_shape=(c.BN, c.BK),
            a_pad=c.a_pad, b_pad=c.b_pad,
            mma_cfg=mma_cfg, n_warps=c.n_warps, b_shuffled=False,
            n_stages=c.n_stages,
        )
        acc = Accumulators.from_mma(mma_cfg, BM=c.BM, BN=c.BN, n_warps=c.n_warps)
        mma = MmaBody(acc=acc)                        # shape + K_inner inferred

        def produce(ictx: IterCtx) -> None:
            k_col = ictx.iter_idx * c.BK
            ictx.stage.a.load_from(g.A, row=m_base, col=k_col, cast=a_cast)
            ictx.stage.b.load_from(g.B, row=n_base, col=k_col, cast=b_cast)

        PipelineBody(
            stages=stages, produce=produce, consume=mma, carry=acc,
        ).run(n_iters=s.K // c.BK, n_stages=c.n_stages)

        BN_per_warp = (c.BN // mma_cfg.shape.n // c.n_warps) * mma_cfg.shape.n
        pop.store_acc(
            g.Out, acc,
            row=m_base,
            col=n_base + bctx.warp_id * BN_per_warp,
            cast=s.out_dtype,
        )
```

Notes:

- `TensorDecl` fields take **callables of (spec, config)** so shapes
  can depend on config knobs (e.g. shuffle-padded B).
- `output_idx=-1` means the last TENSORS entry is the one checked
  against `reference()`. Override when the output isn't last.
- `self.g` is a `SimpleNamespace` of `GlobalTensor` objects — one
  per TENSORS entry, keyed by name.
- The `@kernel` decorator publishes `self.bctx` (BlockContext) and
  binds `self.m_base = block_base("y", config.BM)` +
  `self.n_base = block_base("x", config.BN)` when those config
  fields exist — kernels whose grid follows the `(N/BN, M/BM, 1)`
  convention get these for free. Kernels with custom grids compute
  their own bases inside `build()` and leave `BM/BN` off the config.
- Author IR via `pop.*` (`pop.add`, `pop.mul`, `pop.for_range`,
  `pop.barrier`, `pop.store_acc`, …) rather than `bctx.bld.*`
  directly. The `Builder` stays for module-setup calls
  (`begin_function`, `param`, `register_shape`).
- Kernel-authoring helpers (`pop.work_list_load`, `pop.index_cache`,
  `pop.q_register_load`, `pop.silu`, `pop.cast`) live in
  `popcorn.lang`; see [BLOCKS.md](BLOCKS.md#retired-classes-use-the-free-function-replacement).
- `MmaBody(acc=acc)` without more args is the usual shape — `shape`
  defaults to `active_bctx().mma_cfg` and `K_inner` is inferred from
  the B tile's shape[1] at call time. Only override when those
  defaults don't fit.

## 7. __init__.py

```python
from popcorn.kernels.my_kernel.kernel import MyKernel  # noqa: F401
```

That's it — the `@kernel` decorator handles the registry wiring.

## 8. Verify

```bash
make test KERNEL=my_kernel       # smoke test auto-picks up the kernel
make fuzz KERNEL=my_kernel       # correctness sweep across all problems
make bench KERNEL=my_kernel      # perf vs baselines
make autotune KERNEL=my_kernel   # save best configs to configs/
```

## Gotchas

- Missing `@kernel` decorator → kernel invisible to tools.
  `TypeError` at import lists the missing hooks.
- `make_tensors()` dict keys must match TENSORS names / `ParamSpec`
  buffer order.
- Non-frozen Spec / Config → autotune cache dies (unhashable).
- Raw `torch.*` / `mlx.*` in `reference.py` / `make_tensors` →
  breaks on the other backend. Route through `PT`.
- Tag at least one problem `smoke` — that's what CI exercises.
- `config_overrides` on a Problem (in `problems.py`) pins config
  fields for that problem before autotune runs. Useful for pinning
  `b_shuffle=True` on a shuffled variant.
- `EXEMPT FROM 500-LINE RULE` in the module docstring if `kernel.py`
  goes over. The pre-commit hook is strict.
