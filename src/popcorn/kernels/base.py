"""Kernel base classes: KernelSpec, KernelConfig, Kernel, Autotuner.

The contract:
    - KernelSpec describes WHAT to compute (immutable problem definition).
    - KernelConfig describes HOW to compute it (tunable knobs).
    - Kernel(spec, config) is a compileable, launchable, benchmarkable unit.
    - Autotuner runs a genetic search over a tune space and returns the
      fastest valid (Spec, Config) pair.
"""

from __future__ import annotations

import json
import os
import time as _time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch

from popcorn.correctness import check_correctness
from popcorn.ir import DType
from popcorn.ir import Module as _Module
from popcorn.launcher import ParamSpec
from popcorn.launcher.launcher import Launcher


def _walk_ops(module: _Module):
    """Recursively yield every Op in the module — descends into nested
    Regions (for-loop / if-region bodies) so the full op set is
    visible to `is_valid_for`'s IR-scan. Order is depth-first, parent
    first. Used as the source of truth for "what will this kernel
    actually emit"; no `predicted` hooks on the kernel subclass.
    """
    for fn in module.functions:
        yield from _walk_region(fn.body)


def _walk_region(region):
    for op in region.ops:
        yield op
        for sub in op.regions:
            yield from _walk_region(sub)


def _count_mma_accumulator_chains(mma_ops: list) -> int:
    """Count distinct MMA accumulator chains — the per-warp
    simdgroup_matrix accumulator grid size in the emitted function.

    Each MmaOp takes an accumulator input ``c`` and produces a new
    accumulator ``d``. Within one function, chains are identified by
    tracing ``c`` back through prior MmaOps to a "fresh" root
    (either a constant zero, a ForLoopOp carried Value from outside
    the loop, or any non-MmaOp producer). The number of distinct
    chain roots == the accumulator grid size that the kernel holds
    live.

    Used by ``Kernel.is_valid_for`` against
    ``caps.max_mma_accumulator_tiles`` — Metal's shader compiler
    scales poorly with the number of live simdgroup_matrix variables,
    so kernels whose accumulator grid balloons past the cap take
    pathologically long to JIT.
    """
    if not mma_ops:
        return 0
    mma_results = {op.results[0]: op for op in mma_ops if op.results}
    # For each MmaOp, its C operand is the third positional operand.
    # Count those whose C is NOT the result of another MmaOp —
    # those are the chain roots.
    roots = 0
    for op in mma_ops:
        if len(op.operands) < 3:
            continue
        c_operand = op.operands[2]
        if c_operand not in mma_results:
            roots += 1
    return roots


def _tensor_ptr(t) -> int:
    """Extract the device pointer from a torch (or compatible) tensor.

    Falls back to ``t.data.ptr`` (the cupy attribute) only when
    ``data_ptr`` isn't present. Bundle 4 tightens the contract so
    every kernel passes torch tensors; the fallback is left in place
    so legacy call sites that haven't migrated yet still work without
    importing cupy at module load.
    """
    if hasattr(t, "data_ptr"):
        return t.data_ptr()  # torch
    return t.data.ptr  # cupy (lazy — only hit on legacy call sites)


# ═══════════════════════════════════════════════════════════════════
# Spec / Config
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class KernelSpec:
    """Immutable problem definition. Subclasses add concrete fields."""

    pass


@dataclass(frozen=True)
class KernelConfig:
    """Tunable parameters. Subclasses add concrete fields."""

    @classmethod
    def default_for(cls, spec) -> KernelConfig:
        """Conservative default config for a given spec.

        Subclasses override when the safe "compiles everywhere" config
        depends on the spec (fp8 compute dtype, K divisibility, smem
        budget). Default: bare ``cls()``.
        """
        return cls()


@dataclass(frozen=True)
class MmaSite:
    """One MMA call site a kernel emits.

    A kernel can have multiple sites with different dtype axes —
    flash attention's ``gemm1_qk`` (Q × K) and ``gemm2_pv`` (P × V)
    are the canonical example; both operate in the compute dtype but
    may autotune to different shapes (Q × K has inner-K = Dh while
    P × V has inner-K = KvTile).

    The autotuner walks ``cls.mma_sites(spec)`` and injects one
    ``<site.name>_shape`` knob per site into the resolved tune space
    (see ``Kernel.tune_space_resolved``). Legal values for each knob
    are the intersection of ``device.caps.matmul_shapes`` and the
    shapes whose dtypes match the site's declared axes.

    Per MMA_SHAPES proposal §4.2.
    """

    name: str
    a_dtype: DType
    b_dtype: DType
    acc_dtype: DType = DType.F32

    def shape_matches(self, shape_id: str) -> bool:
        """Return True iff the descriptor for ``shape_id`` has dtypes
        compatible with this site. Looked up against the live registry."""
        from popcorn.ir.mma_registry import ALL_SHAPES

        for cfg in ALL_SHAPES:
            if cfg.shape_id != shape_id:
                continue
            return (
                cfg.shape.a_dtype is self.a_dtype
                and cfg.shape.b_dtype is self.b_dtype
                and cfg.shape.acc_dtype is self.acc_dtype
            )
        return False


@dataclass
class Problem:
    """A named, tagged benchmark problem.

    name:   Human-readable label (e.g. "small_sq", "prod_8k").
            Used in bench table output and config filenames.
    params: Dict of spec field values — passed to Spec(**params).
    tags:   Set of tags for filtering (e.g. {"small", "smoke"},
            {"production", "long_kv"}). Tools accept --tag to
            restrict to problems matching a tag.
    config_overrides:
            Dict of config-field → value that MUST be pinned for this
            problem. Lets a problem carry config-level constraints
            without the tools (fuzz/autotune/bench) knowing about them.
            Example: ``{"b_shuffle": True, "vec_epilogue": True}`` on
            a "preshuffle" variant ensures the autotuner searches only
            b_shuffle=True configs for that problem name, and fuzz
            instantiates the kernel with those knobs set. Default: none.
    """

    name: str
    params: dict
    tags: frozenset[str] = frozenset()
    config_overrides: dict = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        params: dict,
        tags: set[str] | frozenset[str] | None = None,
        config_overrides: dict | None = None,
    ):
        self.name = name
        self.params = params
        self.tags = frozenset(tags) if tags else frozenset()
        self.config_overrides = dict(config_overrides) if config_overrides else {}


# ═══════════════════════════════════════════════════════════════════
# Baseline for bench comparisons
# ═══════════════════════════════════════════════════════════════════


class Baseline:
    """A named comparison for the bench harness.

    `fn` is a no-arg callable that runs ONE iteration of the baseline. The
    bench script warms up + times it the same way it times the kernel under
    test. `reported_time_scale` multiplies the measured time after the fact —
    used when one fused baseline call covers multiple logical operations.
    """

    __slots__ = ("fn", "name", "reported_time_scale")

    def __init__(self, name: str, fn: Callable, *, reported_time_scale: float = 1.0):
        self.name = name
        self.fn = fn
        self.reported_time_scale = reported_time_scale

    def __repr__(self):
        return f"Baseline({self.name!r})"


# ═══════════════════════════════════════════════════════════════════
# Kernel base class
# ═══════════════════════════════════════════════════════════════════


def _current_sm() -> int:
    """Compute capability of the active CUDA device, e.g. 89 for sm_89.

    Reads from torch — no cupy. Used by the legacy `Kernel.compile()`
    path that still wraps `popcorn.compiler.Compiler`. New IR-based
    kernels go through `popcorn.launcher.Launcher`, which picks the
    target SM from `device.caps.compute_capability` directly.
    """
    cap = torch.cuda.get_device_capability(0)
    return cap[0] * 10 + cap[1]


def _format_problem_label(problem: dict) -> str:
    """Stable filename-friendly label for a bench/autotune problem dict.

    Used by both ``tools/autotune.py`` (when saving winning configs) and
    :meth:`Kernel.load_tuned_config` (when reading them back). Underscores
    are used between key/value pairs and any key starting with ``_`` is
    skipped (those are scratch context like ``_torch_Q``).
    """
    return "_".join(f"{k}{v}" for k, v in problem.items() if not str(k).startswith("_"))


def _default_configs_dir() -> str:
    """Resolve where tuned config JSONs live.

    Looks at ``$POPCORN_CONFIGS_DIR`` first, then falls back to
    ``<repo_root>/configs`` for editable installs (``parents[3]`` from
    ``base.py`` lands at the repo root). Final fallback is the cwd.
    """
    env = os.environ.get("POPCORN_CONFIGS_DIR")
    if env:
        return env
    repo_configs = Path(__file__).resolve().parents[3] / "configs"
    if repo_configs.exists():
        return str(repo_configs)
    return "configs"


class Kernel(ABC):
    """A specific (spec, config) pair that can be compiled and launched.

    Subclasses must implement:
        is_valid()       — config sanity check (no compile)
        smem_estimate()  — bytes
        emit()           — build the Program
        global_tensors() — list[GlobalTensor] in launch arg order
        grid()           — launch grid
        reference()      — pure cupy/torch reference
        flops()          — total FLOPs of the operation

    Subclasses MAY set:
        TARGET_NAME      — string used by tools/autotune.py and the
                           tuned-config JSON loader. Set this so
                           ``Kernel.load_tuned_config()`` works.
    """

    # Override in subclasses to enable tuned-config loading.
    TARGET_NAME: str | None = None

    def __init__(self, spec: KernelSpec, config: KernelConfig):
        self.spec = spec
        self.config = config
        self._compiled = None

    # ── Tuned config JSON loader ──

    @classmethod
    def load_tuned_config(
        cls,
        problem: dict,
        *,
        configs_dir: str | None = None,
    ) -> dict | None:
        """Load a previously-autotuned config dict for ``(cls, problem)``.

        Returns the raw ``config`` dict from the JSON, or ``None`` if no
        matching file exists. Caller is responsible for constructing the
        kernel's ``KernelConfig`` from the dict and validating it.

        Looks for ``<configs_dir>/<TARGET_NAME>_<problem_label>.json``,
        which is the path written by ``tools/autotune.py --save``.

        Setting the ``POPCORN_FORCE_DEFAULT_CONFIG=1`` environment variable
        makes this always return ``None`` so kernels fall back to their
        hardcoded ``_pick_default_cfg``. Used by ``tools/test.py`` to test
        the "default" path independently of the "tuned" path.
        """
        if os.environ.get("POPCORN_FORCE_DEFAULT_CONFIG") == "1":
            return None
        if cls.TARGET_NAME is None:
            return None

        cdir = Path(configs_dir or _default_configs_dir())
        path = cdir / f"{cls.TARGET_NAME}_{_format_problem_label(problem)}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return data.get("config")

    # ── Subclass interface ──

    @abstractmethod
    def is_valid(self) -> bool: ...

    def is_valid_for(self, caps) -> bool:
        """Validate this kernel against device capabilities.

        Gates on `caps.max_smem_per_block`, then emits the IR module
        and walks it against caps:
          * MmaOp / LoadMatrixOp / StoreMatrixOp shape_ids must all be
            in `caps.matmul_shapes` (or the set is empty = no filter).
          * AtomicRmwOp value dtypes must be in `caps.atomic_add_dtypes`.
          * Total op count per function must be ≤ `caps.max_ir_ops`.
            This is a proxy for emitted-MSL LoC / SSA variable count,
            correlating with JIT compile time on backends whose shader
            compiler scales poorly on large kernels (Metal).
        Finally delegates to the config-local `is_valid()` check.

        The IR scan means kernels don't have to predict what ops they
        will emit — the Module *is* the source of truth. New kernels
        don't need to add hooks; if they emit a shape or op a backend
        can't lower, `is_valid_for` catches it automatically.

        Subclasses generally should NOT override this.
        """
        if not self.is_valid():
            return False

        # Emit the module and scan. If emit() raises (e.g. an invalid
        # mma_k for this compute dtype), the config is not valid.
        try:
            module = self.emit()
        except Exception:
            return False

        # Authoritative smem check: the ``smem_layout`` pass computes
        # the post-aliasing total from the emitted IR. Replaces the
        # old ``kernel.smem_estimate()`` predict-then-verify dance —
        # the layout pass is the single source of truth.
        try:
            from popcorn.lower.smem_layout import compute_smem_layout

            fn = module.functions[0]
            plan = compute_smem_layout(fn, enable_aliasing=True)
            if plan.total_bytes > caps.max_smem_per_block:
                return False
        except Exception:
            return False

        from popcorn.ir import (  # local to avoid import cycles
            AtomicRmwOp,
            LoadMatrixOp,
            MmaOp,
            StoreMatrixOp,
        )

        per_fn_counts: dict[str, int] = {}
        per_fn_acc_tiles: dict[str, int] = {}
        for fn in module.functions:
            n = 0
            mma_ops: list[MmaOp] = []
            for op in _walk_region(fn.body):
                n += 1
                # Matmul shape support
                if isinstance(op, (MmaOp, LoadMatrixOp, StoreMatrixOp)):
                    shape_id = op.attrs.get("shape_id")
                    if (
                        shape_id is not None
                        and caps.matmul_shapes
                        and shape_id not in caps.matmul_shapes
                    ):
                        return False
                # Atomic dtype support
                if isinstance(op, AtomicRmwOp):
                    if caps.atomic_add_dtypes and op.attrs.get("op") == "add":
                        if op.operands[0].dtype not in caps.atomic_add_dtypes:
                            return False
                if isinstance(op, MmaOp):
                    mma_ops.append(op)
            per_fn_counts[fn.name] = n
            per_fn_acc_tiles[fn.name] = _count_mma_accumulator_chains(mma_ops)

        if caps.max_ir_ops is not None:
            for _fn_name, n in per_fn_counts.items():
                if n > caps.max_ir_ops:
                    return False
        if caps.max_mma_accumulator_tiles is not None:
            for _fn_name, tiles in per_fn_acc_tiles.items():
                if tiles > caps.max_mma_accumulator_tiles:
                    return False
        return True

    def smem_estimate(self) -> int:
        """Total smem bytes this kernel will use. Computed by emitting
        the IR and running the ``smem_layout`` pass — the authoritative
        post-aliasing number. Kernels no longer need (or should)
        override this; the old predict-then-verify pattern is gone.
        """
        try:
            module = self.emit()
        except Exception:
            return 0
        from popcorn.lower.smem_layout import compute_smem_layout

        fn = module.functions[0]
        plan = compute_smem_layout(fn, enable_aliasing=True)
        return plan.total_bytes

    @abstractmethod
    def emit(self) -> _Module: ...

    def global_tensors(self) -> list:
        return []

    @abstractmethod
    def grid(self) -> tuple[int, ...]: ...

    @abstractmethod
    def reference(self, *tensors) -> Any: ...

    @abstractmethod
    def flops(self) -> int: ...

    # ── Bundle 6: registry-aware classmethods (concrete stubs) ──
    #
    # These are NOT @abstractmethod so existing kernels keep
    # instantiating without overriding them. The registry decorator
    # in `popcorn.kernels.registry` validates that any class passed
    # to `@register(...)` has overridden every required hook below;
    # legacy kernels that don't go through the registry pay nothing.
    #
    # New kernels (the universal GEMM, future migrations) override
    # all of them and become first-class registry entries with
    # tools/fuzz.py + tools/bench.py + tools/autotune.py support.

    NAME: str = ""  # Stable kernel name. Used as the registry key.
    SPEC_CLS: type | None = None
    CONFIG_CLS: type | None = None
    OUTPUT_IDX: int = -1  # Index of the output buffer in param_spec.buffers
    # Per-kernel override for the cos-sim correctness gate. ``None``
    # means "use the dtype-table default from popcorn.correctness";
    # set a float in a subclass when the kernel's precision profile
    # doesn't match the table (online softmax chains, fp32 RoPE casts,
    # etc.). Honored via ``cls.correctness_threshold(out, accum)``.
    CORRECTNESS_THRESHOLD: float | None = None
    # Name of the TENSORS entry that holds the shuffleable B-matrix,
    # or None for kernels without a shuffled-B fast path. When set
    # AND ``config.b_shuffle`` is truthy, the base
    # ``prepare_launch_tensors`` routes the named tensor through
    # ``cached_shuffle_b`` using the resolved MMA + (BK, b_pad). Every
    # GEMM-shaped kernel sets this to the name of its weight tensor
    # ("B" / "W_in" / "W_out"); attention-family kernels leave it None.
    SHUFFLE_TENSOR: str | None = None

    @classmethod
    def correctness_threshold(cls, out_dtype, accum_dtype=None) -> float:
        """Resolve the kernel's cos-sim threshold.

        Subclass override on ``CORRECTNESS_THRESHOLD`` wins; otherwise
        the dtype-table default from ``popcorn.correctness`` applies.
        """
        if cls.CORRECTNESS_THRESHOLD is not None:
            return float(cls.CORRECTNESS_THRESHOLD)
        from popcorn.correctness import _threshold_for
        from popcorn.ir import DType

        return _threshold_for(out_dtype, accum_dtype or DType.F32)

    # Attributes injected by the @kernel decorator's generated emit()
    # on every instance right before build() runs. Declared here so
    # type checkers see them as legitimate attributes inside build().
    # (Runtime values are set by decorator.py — the defaults below
    # are placeholders.)
    bld: Any = None
    ctx: Any = None
    bctx: Any = None
    g: Any = None
    # Block-base convention: set automatically by the @kernel decorator
    # when config has BM/BN fields. Kernels whose grid doesn't follow
    # the GEMM-shaped (N/BN, M/BM, 1) convention don't declare BM/BN
    # and compute their own bases inside build().
    m_base: Any = None
    n_base: Any = None

    def make_bctx(self, mma_cfg: Any = None) -> Any:
        """Register the MMA shape + create + publish the BlockContext.

        Bundles the two-line ritual (register_shape + make_ctx) used
        inside every ``build()`` body. Stores the bctx on ``self`` and
        returns it so callers can reference it locally for readability.
        Called from build() bodies — the @kernel decorator sets
        ``self.ctx`` just before ``build()`` runs.
        """
        if mma_cfg is not None:
            # Idempotent — the @kernel decorator now calls make_bctx
            # before build() runs (A9), so build() bodies that still
            # repeat the call shouldn't double-register the shape.
            if mma_cfg.shape.name not in self.ctx.bld.module.kernel_shapes:
                self.ctx.bld.register_shape(mma_cfg.shape)
        self.bctx = self.ctx.make_ctx(mma_cfg)
        return self.bctx

    @classmethod
    def problems(cls) -> list[Problem]:
        """Canonical fuzz/bench problem list.

        Returns a list of Problem instances, each with a name, params
        dict (spec field values), and optional tags for filtering.
        Tools accept --tag to restrict to problems matching a tag.

        Default: defer to the legacy ``bench_problems`` classmethod
        if the kernel still defines that, otherwise raise. Registry-
        aware kernels override this directly.
        """
        legacy = getattr(cls, "bench_problems", None)
        if callable(legacy) and legacy.__func__ is not Kernel.problems.__func__:
            try:
                return legacy()
            except (AttributeError, NotImplementedError):
                pass
        raise NotImplementedError(
            f"{cls.__name__}.problems(): registry-aware kernels must "
            f"override this. See popcorn cleanup proposal §4.3."
        )

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        """Autotune search space — knob name → candidate values.

        Cartesian product over the dict produces all candidate
        configs the bounded search will consider. Names must match
        CONFIG_CLS field names. Default: empty (skip autotune).
        """
        return {}

    @classmethod
    def mma_sites(cls, spec) -> list[MmaSite]:
        """Declared MMA call sites — one per distinct (dtype-axis)
        mma.sync / simdgroup_multiply_accumulate emitted by this kernel.

        Default behavior: one ``"main"`` site with the spec's
        ``compute_dtype_resolved`` on both axes. Covers every GEMM-shaped
        kernel (gemm, moe_inproj, moe_outproj, owl_attn). Attn-like
        kernels that use distinct A/B dtype axes override. Kernels without
        MMAs (kv_cache_update) override to return ``[]``.

        ``spec is None`` during autotune space introspection; return
        ``[]`` in that case — we can't resolve compute dtype without a
        concrete spec.

        Per MMA_SHAPES proposal §4.2.
        """
        if spec is None:
            return []
        compute = getattr(spec, "compute_dtype_resolved", None)
        if compute is None:
            return []
        return [MmaSite(name="main", a_dtype=compute, b_dtype=compute)]

    def _mma_cfg(self):
        """Resolve the MMA descriptor from ``config.main_shape``, falling
        back to the compute-dtype default (mma_k=16) when empty.

        Default handles every kernel whose primary MMA site uses the
        spec's ``compute_dtype_resolved`` on both axes (every GEMM-shaped
        kernel). Kernels with a different fallback (attn uses ``a_dtype``
        / ``b_dtype`` directly) override. Kernels without any MMA don't
        call this method.
        """
        from popcorn.ir.mma_registry import ALL_SHAPES, lookup_mma

        cfg_any: Any = cast(Any, self.config)
        spec_any: Any = cast(Any, self.spec)
        main_shape = cfg_any.main_shape
        if main_shape:
            for cfg in ALL_SHAPES:
                if cfg.shape_id == main_shape:
                    return cfg
            raise KeyError(f"{type(self).__name__}: main_shape={main_shape!r} not in registry")
        compute = spec_any.compute_dtype_resolved
        return lookup_mma(compute, compute, 16)

    def _validate_gemm_tile(self, mma) -> bool:
        """Shared GEMM-shaped tile validity.

        Checks common to every kernel whose build() body follows the
        GEMM pattern (gemm / moe_inproj / moe_outproj / owl_attn / attn):
        BK/BM/BN divisibility by the MMA shape, per-warp N partition,
        n_threads divisibility of tile element counts, n_stages domain,
        and the fp8-vs-bf16 pad granule rule.

        Kernels layer on their own spec-level checks before calling.
        Returns True iff all shared checks pass.
        """
        c: Any = cast(Any, self.config)
        spec_any: Any = cast(Any, self.spec)
        if c.BK % mma.mma_k != 0:
            return False
        if c.BM % mma.shape.m != 0 or c.BN % mma.shape.n != 0:
            return False
        if (c.BN // mma.shape.n) % c.n_warps != 0:
            return False
        n_threads = c.n_warps * 32
        if (c.BM * c.BK) % n_threads != 0:
            return False
        if (c.BN * c.BK) % n_threads != 0:
            return False
        if c.n_stages not in (1, 2):
            return False
        # 1-byte compute dtype (fp8) wants pad=16 granule; 2-byte wants 8.
        pad_nonzero = 16 if spec_any.compute_dtype_resolved.bytes == 1 else 8
        if getattr(c, "a_pad", 0) not in (0, pad_nonzero):
            return False
        if getattr(c, "b_pad", 0) not in (0, pad_nonzero):
            return False
        return True

    @classmethod
    def tune_space_resolved(cls, spec, device) -> dict[str, list]:
        """Full device-resolved tune space: ``tune_space()`` merged with
        per-MMA-site shape knobs filtered by the active device.

        For each ``MmaSite`` the kernel declares, adds one knob
        ``<site.name>_shape`` whose values are the shape ids in
        ``device.caps.matmul_shapes`` that match the site's declared
        dtype axes. Kernels that declare no sites get their
        ``tune_space()`` through unchanged.

        This method is what ``tools/autotune.py`` walks — kernels
        never hand-maintain ``mma_k`` or other per-shape knobs in
        their own ``tune_space()``; the machinery here injects them
        from the authoritative registry.
        """
        space: dict[str, list] = dict(cls.tune_space())
        sites = cls.mma_sites(spec)
        if not sites:
            return space
        matmul_shapes = getattr(device.caps, "matmul_shapes", frozenset())
        for site in sites:
            legal = sorted(s for s in matmul_shapes if site.shape_matches(s))
            if legal:
                space[f"{site.name}_shape"] = legal
        return space

    def param_spec(self):
        """Return the kernel's `popcorn.launcher.ParamSpec`.

        Default: derive from the IR Module the kernel emits, since
        the new IR-based kernels expose their param shape through
        `emit() -> Module`. Legacy Program-based kernels override
        this to walk `global_tensors() + extra_args` instead.
        """
        emitted = self.emit()
        if isinstance(emitted, _Module):
            if not emitted.functions:
                raise ValueError(
                    f"{type(self).__name__}.emit() returned a Module with no functions"
                )
            return ParamSpec.from_function(emitted.functions[0])
        raise NotImplementedError(
            f"{type(self).__name__}.param_spec(): legacy Program-based "
            f"kernels must override this method explicitly."
        )

    # ── Legacy method preserved further down in this file ──
    #
    # `from_problem`, `make_tensors`, and `baselines` already exist
    # below as legacy instance/classmethods used by tools/bench.py
    # and tools/autotune.py. The registry-aware contract overrides
    # them at the new-kernel level (Python permits classmethod-over-
    # instance-method overrides on subclasses) so we don't shadow
    # them in the base.

    def block(self) -> tuple[int, int, int]:
        """Default: ``(n_warps * 32, 1, 1)`` — single-row block from
        ``config.n_warps``. Kernels with a computed warp count
        (attn/owl_attn use ``gqa_ratio * NCW``) override."""
        n_warps = getattr(self.config, "n_warps", None)
        if n_warps is None:
            raise NotImplementedError(
                f"{type(self).__name__}.block() must be implemented or config "
                f"must have an n_warps field"
            )
        return (n_warps * 32, 1, 1)

    def entry_name(self) -> str:
        """Symbol name for the compiled kernel. Defaults to the
        registered ``NAME`` (set by ``@kernel``). Override only when the
        MSL/PTX entry point should diverge from the registry name."""
        return type(self).NAME or type(self).__name__.lower()

    # ── Provided by base class ──

    def compile(self):
        """Compile via the new Launcher path."""
        if self._compiled is None:
            launcher = Launcher()
            self._compiled = launcher.compile(type(self), self.spec, self.config)
        return self._compiled

    def launch(self, tensors: Sequence, extras: Sequence = ()):
        compiled = self.compile()
        buf_list = list(tensors) if not isinstance(tensors, list) else tensors
        compiled.launch(buffers=buf_list)

    def prepare_launch_tensors(self, tensors: dict) -> dict:
        """Hook: transform launch tensors for this (spec, config) pair.

        Default: if ``SHUFFLE_TENSOR`` is set and ``config.b_shuffle`` is
        truthy, route the named tensor through ``cached_shuffle_b`` and
        return an updated dict. Otherwise identity.

        Kernels with custom pre-launch transforms override this directly.
        The ``reference()`` call should always receive the *plain* tensors
        from ``make_tensors`` (so the reference stays simple); only the
        launch path goes through this hook.
        """
        name = type(self).SHUFFLE_TENSOR
        if name is None:
            return tensors
        # b_shuffle lives on the spec for kernels where it's a property
        # of the caller-supplied tensor layout (GemmKernel), and on the
        # config for kernels where it's a tuning knob (MoE in/out).
        shuffle = getattr(self.spec, "b_shuffle", False) or getattr(self.config, "b_shuffle", False)
        if not shuffle:
            return tensors
        from popcorn.weight_shuffle import cached_shuffle_b

        mma = self._mma_cfg()
        cfg_any: Any = cast(Any, self.config)
        shuffled = cached_shuffle_b(
            tensors[name],
            K_CHUNK=cfg_any.BK,
            mma_k=mma.mma_k,
            bpad=cfg_any.b_pad,
        )
        return {**tensors, name: shuffled}

    def time(
        self,
        tensors: Sequence,
        *,
        warmup_ms: float = 10.0,
        bench_ms: float = 50.0,
    ) -> float:
        """Average runtime in microseconds.

        Three phases:
        1. Probe: run+sync until >=1ms to estimate per-kernel runtime.
        2. Warmup: async-queue warmup_ms worth of runs, sync once.
        3. Bench: async-queue bench_ms worth of runs between two
           CUDA events, sync, report elapsed / n_iters.

        No sync between individual launches in warmup/bench — this
        minimizes host launch overhead and measures closer to the
        true kernel runtime under async stream queuing.
        """
        buf_list = list(tensors) if not isinstance(tensors, list) else tensors
        compiled = self.compile()

        # Phase 1: Probe — get rough runtime estimate
        compiled.launch(buffers=buf_list)
        torch.cuda.synchronize()

        elapsed_ms = 0.0
        probe_iters = 0
        while elapsed_ms < 1.0:
            t0 = _time.perf_counter()
            compiled.launch(buffers=buf_list)
            torch.cuda.synchronize()
            elapsed_ms += (_time.perf_counter() - t0) * 1000.0
            probe_iters += 1
        rough_us = (elapsed_ms / probe_iters) * 1000.0

        # Phase 2: Warmup — async-queued, single sync at end
        n_warmup = max(1, int(warmup_ms * 1000.0 / rough_us))
        for _ in range(n_warmup):
            compiled.launch(buffers=buf_list)
        torch.cuda.synchronize()

        # Phase 3: Bench — async-queued between two CUDA events
        n_bench = max(1, int(bench_ms * 1000.0 / rough_us))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_bench):
            compiled.launch(buffers=buf_list)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / n_bench * 1000.0

    def tflops(self, runtime_us: float) -> float:
        return self.flops() / (runtime_us * 1e-6) / 1e12

    def correctness(
        self,
        tensors: Sequence,
        ref: Any,
        out_idx: int = -1,
        cos_threshold: float | None = None,
        **kwargs,
    ) -> dict:
        """Launch the kernel and compare tensors[out_idx] to ref.

        Thin wrapper around `popcorn.correctness.check_correctness`,
        kept for backward compatibility with the few call sites that
        still expect a dict result. The legacy multi-metric
        implementation that handled both cupy and torch tensors has
        been removed — see Bundle 4 of the cleanup proposal §7.

        New code should call `popcorn.correctness.check_correctness`
        directly and consult `result.passed` / `result.cos_sim`.
        """
        self.launch(tensors)
        torch.cuda.synchronize()
        out = tensors[out_idx]
        out_t = out if isinstance(out, torch.Tensor) else torch.from_dlpack(out)
        ref_t = ref if isinstance(ref, torch.Tensor) else torch.from_dlpack(ref)

        # The kernel doesn't necessarily declare an out_dtype field
        # yet — fall back to inferring from the actual tensor dtype.
        torch_to_ir = {
            torch.float32: DType.F32,
            torch.bfloat16: DType.BF16,
            torch.float16: DType.F16,
        }
        out_dtype = torch_to_ir.get(out_t.dtype, DType.F32)

        result = check_correctness(
            out_t,
            ref_t,
            out_dtype=out_dtype,
            threshold=cos_threshold,
        )
        return {
            "match": result.passed,
            "cos_sim": result.cos_sim,
            "max_abs": result.max_abs if result.max_abs is not None else 0.0,
            "norm_err": result.max_abs if result.max_abs is not None else 0.0,
        }

    def prune_score(self) -> float:
        """Optional hook for the genetic autotuner. Returns a value in [0, 1]
        where 0 means "definitely worth trying" and 1 means "skip this config".

        Subclasses can override this with cheap heuristics (register pressure,
        wave-tail waste, etc.) so the autotuner spends compile budget on
        candidates likely to win. Default: 0 (no prior).
        """
        return 0.0

    # ── Bench harness hooks (override per kernel) ──

    @classmethod
    def bench_problems(cls) -> list[dict]:
        """Canonical list of (problem-dict) the `make bench` harness sweeps.

        Each dict is a kwargs-style payload for ``cls.from_problem``.
        Subclasses override this; the default returns an empty list which
        causes the bench harness to skip the kernel.
        """
        return []

    @classmethod
    def from_problem(cls, problem: dict) -> Kernel:
        """Construct a (spec, config, kernel) triple from a bench problem dict.

        Builds ``cls.SPEC_CLS(**problem)`` and pairs it with
        ``cls.CONFIG_CLS.default_for(spec)``. Config subclasses own
        their spec→config default logic via ``default_for``.
        """
        if cls.SPEC_CLS is None or cls.CONFIG_CLS is None:
            raise NotImplementedError(
                f"{cls.__name__} has no SPEC_CLS/CONFIG_CLS — override from_problem()"
            )
        spec = cls.SPEC_CLS(**problem)
        config_cls: Any = cast(Any, cls.CONFIG_CLS)
        return cls(spec, config_cls.default_for(spec))

    @classmethod
    def from_problem_cached(cls, problem: dict, cache=None) -> Kernel:
        """Like ``from_problem``, but consults an AutotuneCache first.

        When ``cache`` has an entry for this (kernel, spec), returns a
        kernel built with that tuned config. Otherwise falls back to
        ``cls.from_problem(problem)`` verbatim. This closes the
        default-vs-autotune gap every kernel's ``from_problem`` hardcodes
        — the autotuner finds the fast path, but only ``compile()`` on
        the launcher consulted it before. Now any caller with access to
        the launcher's cache (bench, fuzz, autotune, downstream apps)
        can get the tuned config directly.

        Non-breaking: callers that can't provide a cache just keep
        calling ``from_problem``.
        """
        if cache is None or cls.SPEC_CLS is None:
            return cls.from_problem(problem)
        spec = cls.SPEC_CLS(**problem)
        cached = cache.lookup(cls, spec)
        if cached is not None:
            return cls(spec, cached)
        return cls.from_problem(problem)

    @classmethod
    def make_tensors(cls, problem: dict) -> dict:
        """Allocate launch tensors for a bench problem.

        Returns a dict whose keys match the ``ParamSpec`` buffer names.
        Subclasses override with a ``@classmethod`` that takes the same
        ``problem`` dict consumed by ``from_problem``.
        """
        raise NotImplementedError(f"{cls.__name__} doesn't implement make_tensors()")

    def baselines(self, tensors: dict) -> list[Baseline]:
        """Named baselines (default: none) for the speed comparison column."""
        return []

    def launch_tensor_list(self, tensors: dict) -> list:
        """Helper: build the positional launch list from a name-keyed dict."""
        return [tensors[gt.name] for gt in self.global_tensors()]

    def bench_label(self) -> str:
        """Short human-readable label for one row of the bench table."""
        return type(self).__name__


# ═══════════════════════════════════════════════════════════════════
# Autotuner
# ═══════════════════════════════════════════════════════════════════


@dataclass
class AutotuneResult:
    config: KernelConfig
    runtime_us: float
    correct: bool
    norm_err: float  # informational (not gating)
    smem_bytes: int
    reg_count_estimate: int
    error: str | None = None  # if compile/launch failed


class Autotuner:
    """Genetic search over a kernel config space.

    Usage:
        autotuner = Autotuner(
            spec=MySpec(...),
            kernel_cls=MyKernel,
            config_cls=MyConfig,             # optional; inferred from kernel_cls if omitted
            tune_space={
                "BM": [16, 32, 64],
                "n_warps": [2, 4, 8],
                ...
            },
            fixed={"BK": 16},                # values shared by every candidate
        )
        best = autotuner.best(tensors, ref, verbose=True)
        print(best.config, best.runtime_us)

    For each candidate config the cross-product produces, the autotuner:
        1. Builds Kernel(spec, config) and skips if `is_valid()` is False.
        2. Compiles, runs once for correctness against `ref`.
        3. Times `n_iters` runs and records the median.
    The genetic loop keeps an `elite_frac` slice each generation, breeds
    the rest via crossover + truncated-Gaussian index mutation, and biases
    sampling against high `prune_score` configs. Set `pop_size` and
    `generations` explicitly or let `_auto_pop_size` choose them based on
    the valid space cardinality.
    """

    def __init__(
        self,
        spec: KernelSpec,
        kernel_cls: type[Kernel],
        tune_space: dict[str, list],
        *,
        config_cls: type[KernelConfig] | None = None,
        fixed: dict[str, Any] | None = None,
        caps: Any = None,
    ):
        self.spec = spec
        self.kernel_cls = kernel_cls
        self.config_cls = config_cls or _infer_config_cls(kernel_cls)
        self.tune_space = dict(tune_space)
        self.fixed = dict(fixed or {})
        # Device caps used to filter the tune_space. When supplied,
        # `_enumerate_valid` consults `is_valid_for(caps)` — which
        # applies the backend-level gates (smem, vec_epilogue,
        # max_unrolled_mma_ops, …) — instead of the config-local
        # `is_valid()`. Optional so unit tests can still autotune
        # against a mocked spec.
        self.caps = caps
        # Pre-compiled kernels keyed by cfg_key. Value is either a Kernel
        # instance (with `_compiled` populated) or a string error message
        # captured during a failed compile in the worker pool. Persists across
        # generations so a re-bred config doesn't recompile.
        self._compile_cache: dict[tuple, Kernel | str] = {}

    # ── Public API ──

    def search(
        self,
        tensors: Sequence,
        ref: Any,
        *,
        n_iters: int = 20,
        warmup: int = 5,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        pop_size: int | None = 16,
        generations: int | None = None,
        elite_frac: float = 0.125,
        mutate_prob: float = 0.3,
        early_stop_stale: int = 6,
        seed: int | None = None,
        max_compile_workers: int | None = None,
        verbose: bool = False,
    ) -> list[AutotuneResult]:
        """Run the genetic search and return all evaluated configs sorted by runtime."""
        import random as _random

        rng = _random.Random(seed)

        all_valid = self._enumerate_valid()
        if not all_valid:
            raise RuntimeError(f"Autotuner: no valid configs in tune_space {self.tune_space}")

        auto_pop, auto_gen = _auto_pop_size(len(all_valid))
        _pop = pop_size if pop_size is not None else auto_pop
        _gens = generations if generations is not None else auto_gen
        n_elite = max(2, int(_pop * elite_frac))

        dense = len(all_valid) <= _pop
        if dense:
            pop = list(all_valid)
        else:
            weights = [
                max(1e-3, 1.0 - self.kernel_cls(self.spec, c).prune_score()) for c in all_valid
            ]
            pop = rng.choices(all_valid, weights=weights, k=_pop)

        if verbose:
            mode = "dense" if dense else f"genetic (pop={_pop}, gen={_gens})"
            print(f"  Autotuner: {len(all_valid)} valid configs, {mode}")

        seen: dict[tuple, AutotuneResult] = {}
        gens_stale = 0
        prev_best = float("inf")

        for gen in range(_gens):
            # Pre-compile every config in this generation that we haven't
            # already evaluated, in parallel. Compiled kernels land in
            # self._compile_cache and persist across generations.
            to_compile = [
                c
                for c in pop
                if self._cfg_key(c) not in seen and self._cfg_key(c) not in self._compile_cache
            ]
            if to_compile:
                self._parallel_compile(
                    to_compile,
                    max_workers=max_compile_workers,
                    verbose=verbose,
                    gen=gen,
                )

            scored: list[tuple[float, KernelConfig, AutotuneResult]] = []
            for cfg in pop:
                key = self._cfg_key(cfg)
                if key in seen:
                    res = seen[key]
                else:
                    res = self._evaluate(
                        cfg,
                        tensors,
                        ref,
                        n_iters=n_iters,
                        warmup=warmup,
                        atol=atol,
                        rtol=rtol,
                    )
                    seen[key] = res
                    if verbose:
                        if res.correct:
                            print(f"  gen {gen} OK   {cfg}  {res.runtime_us:.2f} us")
                        elif res.error:
                            print(f"  gen {gen} FAIL {cfg}  {res.error[:60]}")
                        else:
                            print(
                                f"  gen {gen} WRONG{cfg}  norm={res.norm_err:.2e} err={res.error}"
                            )
                scored.append((res.runtime_us, cfg, res))

            scored.sort(key=lambda t: t[0])
            gen_best = scored[0][0]
            if gen_best < prev_best * 0.99:
                prev_best = gen_best
                gens_stale = 0
            else:
                gens_stale += 1
            if verbose:
                n_ok = sum(1 for ms, _, _ in scored if ms < float("inf"))
                print(f"  gen {gen}: best={gen_best:.2f} us, valid={n_ok}/{len(pop)}")

            if dense or gen == _gens - 1:
                break
            if gens_stale >= early_stop_stale:
                if verbose:
                    print(f"  early stop (stale {gens_stale} gens)")
                break

            elites = [cfg for _, cfg, _ in scored[:n_elite]]
            pop = self._breed(
                elites,
                all_valid,
                _pop,
                mutate_prob=mutate_prob,
                rng=rng,
            )

        results = list(seen.values())
        results.sort(key=lambda r: (not r.correct, r.runtime_us))
        return results

    def best(self, tensors, ref, **kwargs) -> AutotuneResult:
        results = self.search(tensors, ref, **kwargs)
        valid = [r for r in results if r.correct]
        if not valid:
            raise RuntimeError(
                f"Autotuner.best: no valid + correct configs (out of {len(results)} evaluated)"
            )
        return valid[0]

    # ── Internals ──

    def _cfg_key(self, cfg: KernelConfig) -> tuple:
        return tuple(getattr(cfg, k) for k in self.tune_space)

    def _make_cfg(self, values: dict) -> KernelConfig:
        kwargs = dict(self.fixed)
        kwargs.update(values)
        return self.config_cls(**kwargs)

    def _enumerate_valid(self) -> list[KernelConfig]:
        import itertools as _it

        names = list(self.tune_space.keys())
        value_lists = [self.tune_space[n] for n in names]
        out: list[KernelConfig] = []
        for combo in _it.product(*value_lists):
            try:
                cfg = self._make_cfg(dict(zip(names, combo, strict=False)))
            except TypeError:
                continue
            k = self.kernel_cls(self.spec, cfg)
            valid = k.is_valid_for(self.caps) if self.caps is not None else k.is_valid()
            if valid:
                out.append(cfg)
        return out

    def _parallel_compile(
        self,
        configs: list[KernelConfig],
        *,
        max_workers: int | None,
        verbose: bool,
        gen: int,
    ) -> None:
        """Compile all `configs` in parallel and stash the resulting Kernel
        instances (with `_compiled` populated) in `self._compile_cache`.

        cupy's RawModule compilation is a CUDA driver call which releases the
        GIL, so a thread pool gives near-linear speedup until ptxas itself is
        the bottleneck. We default to ``min(os.cpu_count() // 2, len(configs))``
        workers — match the convention from owl_kernels' autotuner.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not configs:
            return

        n_workers = max_workers if max_workers is not None else max(1, (os.cpu_count() or 2) // 2)
        n_workers = max(1, min(n_workers, len(configs)))

        if verbose:
            print(f"  gen {gen}: compiling {len(configs)} configs with {n_workers} workers...")

        def _worker(cfg: KernelConfig):
            kernel = self.kernel_cls(self.spec, cfg)
            try:
                kernel.compile()  # populates kernel._compiled
                return cfg, kernel, None
            except Exception as e:
                return cfg, None, f"compile: {e}"[:200]

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_worker, c) for c in configs]
            for fut in as_completed(futs):
                cfg, kernel, err = fut.result()
                key = self._cfg_key(cfg)
                self._compile_cache[key] = err if err is not None else kernel

    def _evaluate(
        self,
        cfg: KernelConfig,
        tensors: Sequence,
        ref: Any,
        *,
        n_iters: int,
        warmup: int,
        atol: float,
        rtol: float,
    ) -> AutotuneResult:
        # Pull the precompiled kernel from the cache. If it's not there
        # (single-shot evaluate path used by tests), compile inline.
        key = self._cfg_key(cfg)
        cached = self._compile_cache.get(key)
        if isinstance(cached, str):
            return AutotuneResult(
                config=cfg,
                runtime_us=float("inf"),
                correct=False,
                norm_err=float("inf"),
                smem_bytes=0,
                reg_count_estimate=0,
                error=cached,
            )
        if isinstance(cached, Kernel):
            kernel = cached
            compiled = kernel.compile()  # cached on the instance
        else:
            kernel = self.kernel_cls(self.spec, cfg)
            try:
                compiled = kernel.compile()
            except Exception as e:
                self._compile_cache[key] = f"compile: {e}"[:200]
                return AutotuneResult(
                    config=cfg,
                    runtime_us=float("inf"),
                    correct=False,
                    norm_err=float("inf"),
                    smem_bytes=0,
                    reg_count_estimate=0,
                    error=f"compile: {e}"[:200],
                )
            self._compile_cache[key] = kernel
        try:
            check = kernel.correctness(tensors, ref, atol=atol, rtol=rtol)
        except Exception as e:
            return AutotuneResult(
                config=cfg,
                runtime_us=float("inf"),
                correct=False,
                norm_err=float("inf"),
                smem_bytes=getattr(getattr(compiled, "footprint", None), "smem_bytes", 0),
                reg_count_estimate=0,
                error=f"launch: {e}"[:200],
            )
        if not check["match"]:
            return AutotuneResult(
                config=cfg,
                runtime_us=float("inf"),
                correct=False,
                norm_err=check["norm_err"],
                smem_bytes=getattr(getattr(compiled, "footprint", None), "smem_bytes", 0),
                reg_count_estimate=0,
                error=f"wrong: cos={check.get('cos_sim', 0):.4f}",
            )
        try:
            rt = kernel.time(tensors)
        except Exception as e:
            return AutotuneResult(
                config=cfg,
                runtime_us=float("inf"),
                correct=False,
                norm_err=check["norm_err"],
                smem_bytes=getattr(getattr(compiled, "footprint", None), "smem_bytes", 0),
                reg_count_estimate=0,
                error=f"time: {e}"[:200],
            )
        return AutotuneResult(
            config=cfg,
            runtime_us=rt,
            correct=True,
            norm_err=check["norm_err"],
            smem_bytes=getattr(getattr(compiled, "footprint", None), "smem_bytes", 0),
            reg_count_estimate=0,
        )

    def _breed(
        self,
        elites: list[KernelConfig],
        all_valid: list[KernelConfig],
        pop_size: int,
        *,
        mutate_prob: float,
        rng,
    ) -> list[KernelConfig]:
        names = list(self.tune_space.keys())
        value_lists = [self.tune_space[n] for n in names]

        # Always carry elites forward
        next_pop: list[KernelConfig] = list(elites)
        valid_keys = {self._cfg_key(c) for c in all_valid}
        valid_by_key = {self._cfg_key(c): c for c in all_valid}

        attempts = 0
        while len(next_pop) < pop_size and attempts < pop_size * 20:
            attempts += 1
            a, b = rng.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
            child = {}
            for k in names:
                child[k] = getattr(a if rng.random() < 0.5 else b, k)
            # Truncated-Gaussian index mutation
            for k, vals in zip(names, value_lists, strict=False):
                if rng.random() < mutate_prob and len(vals) > 1:
                    cur = child[k]
                    if cur in vals:
                        idx = vals.index(cur)
                        n = len(vals)
                        sigma = max(0.8, n * 0.2)
                        for _ in range(10):
                            offset = rng.gauss(0, sigma)
                            new_idx = max(0, min(n - 1, round(idx + offset)))
                            if new_idx != idx:
                                child[k] = vals[new_idx]
                                break
            try:
                cfg = self._make_cfg(child)
            except TypeError:
                continue
            key = self._cfg_key(cfg)
            if key not in valid_keys:
                continue
            kernel = self.kernel_cls(self.spec, valid_by_key[key])
            if rng.random() < kernel.prune_score():
                continue
            next_pop.append(valid_by_key[key])
        # If we couldn't fill the population (very pruned space), pad with
        # random valid configs.
        while len(next_pop) < pop_size:
            next_pop.append(rng.choice(all_valid))
        return next_pop


def _infer_config_cls(kernel_cls: type[Kernel]) -> type[KernelConfig]:
    """Look up the `config: SomeConfig` annotation on a Kernel subclass.

    Walks the MRO and uses ``typing.get_type_hints`` so PEP 563 string
    annotations (``from __future__ import annotations``) get resolved to the
    actual classes via the defining module's globals.
    """
    import sys
    import typing

    for cls in kernel_cls.__mro__:
        ann = getattr(cls, "__annotations__", {}) or {}
        if "config" not in ann:
            continue
        module_globals = getattr(sys.modules.get(cls.__module__), "__dict__", {})
        try:
            hints = typing.get_type_hints(cls, globalns=module_globals)
        except Exception:
            hints = {}
        resolved = hints.get("config", ann.get("config"))
        if isinstance(resolved, str):
            # Fall back: look it up directly in the module globals.
            resolved = module_globals.get(resolved, resolved)
        if isinstance(resolved, type):
            return resolved
    raise TypeError(
        f"Autotuner: cannot popcorn config_cls for {kernel_cls.__name__}; "
        f"pass config_cls= explicitly"
    )


def _auto_pop_size(n_valid: int) -> tuple[int, int]:
    """Heuristic from owl_kernels' autotuner.

    For small spaces (≤ DENSE_THRESHOLD) we test exhaustively. Otherwise the
    population grows ~ ``80 * log2(n_valid)`` capped at MAX_POP, and the
    generation count rises with the size of the space.
    """
    import math as _math

    DENSE_THRESHOLD = 256
    MIN_POP = 32
    MAX_POP = 512
    POP_SCALE = 80

    if n_valid <= DENSE_THRESHOLD:
        return n_valid, 1
    pop = int(POP_SCALE * _math.log2(max(n_valid, 2)))
    pop = max(MIN_POP, min(pop, MAX_POP))
    gens = 3 if n_valid < 2000 else (4 if n_valid < 10000 else 5)
    return pop, gens
