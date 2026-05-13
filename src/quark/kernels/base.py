"""Kernel base classes: KernelSpec, KernelConfig, Kernel.

The contract:
    - KernelSpec describes WHAT to compute (immutable problem definition).
    - KernelConfig describes HOW to compute it (tunable knobs).
    - Kernel(spec, config) is a compileable, launchable, benchmarkable unit.

Autotuning lives in ``quark.autotune`` (``AutotuneCache`` +
``genetic_search``).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from quark.ir import DType
from quark.ir import Module as _Module
from quark.launcher import ParamSpec
from quark.launcher.launcher import Launcher


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
    """Tunable parameters. Subclasses add concrete fields.

    ``subgroup_size`` (Intel iGPU only) opts the kernel into a
    specific SIMD width at pipeline-create time. ``None`` keeps the
    driver default (32 on Battlemage). Set to 16 to opt a kernel into
    SIMD16 — better VE utilisation + 2× per-work-item GRF budget.
    Ignored on Metal / CUDA paths.
    """

    subgroup_size: int | None = None

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
        from quark.ir.mma_registry import ALL_SHAPES

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

    Looks at ``$QUARK_CONFIGS_DIR`` first, then falls back to
    ``<repo_root>/configs`` for editable installs (``parents[3]`` from
    ``base.py`` lands at the repo root). Final fallback is the cwd.
    """
    env = os.environ.get("QUARK_CONFIGS_DIR")
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
        # Optional explicit caps binding. None means emit() falls back
        # to ``current_device().caps`` (probed lazily). Tests and the
        # autotune harness set this directly via ``bind_caps`` to pin
        # a kernel to a specific device profile without touching the
        # process-wide ``current_device`` cache.
        self._caps = None

    def bind_caps(self, caps) -> Kernel:
        """Pin this kernel to a specific ``DeviceCaps`` for emit-time
        backend dispatch (``build_<family>`` selection). Returns self
        for chaining."""
        self._caps = caps
        return self

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

        Setting the ``QUARK_FORCE_DEFAULT_CONFIG=1`` environment variable
        makes this always return ``None`` so kernels fall back to their
        hardcoded ``_pick_default_cfg``. Used by ``tools/test.py`` to test
        the "default" path independently of the "tuned" path.
        """
        if os.environ.get("QUARK_FORCE_DEFAULT_CONFIG") == "1":
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
        #
        # Backend-aware aliasing: Metal / CUDA emitters apply the
        # aliasing plan (overlap disjoint-lifetime regions in the
        # same byte slot), so the aliased total matches what they
        # allocate. The OCL emitter today gives every SmemAllocOp
        # its own ``OpVariable Workgroup`` — no aliasing — so we must
        # check against the un-aliased total or we accept configs
        # that fit on Metal / CUDA but bust Intel's 48KB smem cap
        # at runtime.
        try:
            from quark.device import DeviceFamily  # noqa: PLC0415
            from quark.lower.smem_layout import compute_smem_layout  # noqa: PLC0415

            fn = module.functions[0]
            alias = getattr(caps, "family", None) is not DeviceFamily.INTEL_GPU
            plan = compute_smem_layout(fn, enable_aliasing=alias)
            if plan.total_bytes > caps.max_smem_per_block:
                return False
        except Exception:
            return False

        from quark.ir import (  # local to avoid import cycles
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
                    if op.attrs.get("op") == "add":
                        atomic_type = op.attrs.get("atomic_type")
                        # Vector atomics: type string encodes the packed form
                        # ("bf16x2", "f16x2"). Gate on caps.atomic_add_vector
                        # instead of the scalar dtype set.
                        if atomic_type in ("bf16x2", "f16x2"):
                            elem = DType.BF16 if atomic_type == "bf16x2" else DType.F16
                            if (elem, 2) not in caps.atomic_add_vector:
                                return False
                        elif caps.atomic_add_dtypes:
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
        from quark.lower.smem_layout import compute_smem_layout

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
    def flops(self) -> int: ...

    # ── Bundle 6: registry-aware classmethods (concrete stubs) ──
    #
    # These are NOT @abstractmethod so existing kernels keep
    # instantiating without overriding them. The registry decorator
    # in `quark.kernels.registry` validates that any class passed
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
    # means "use the dtype-table default from quark.correctness";
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
        the dtype-table default from ``quark.correctness`` applies.
        """
        if cls.CORRECTNESS_THRESHOLD is not None:
            return float(cls.CORRECTNESS_THRESHOLD)
        from quark.correctness import _threshold_for
        from quark.ir import DType

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
            f"override this. See quark cleanup proposal §4.3."
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
        from quark.ir.mma_registry import ALL_SHAPES, lookup_mma

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
        n_threads = c.n_warps * self.resolve_subgroup_size()
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
        """Return the kernel's `quark.launcher.ParamSpec`.

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

    def resolve_subgroup_size(self) -> int:
        """SIMD width the kernel runs at, derived from the active MMA shape.

        Intel's ``cl_intel_subgroup_matrix_multiply_accumulate`` pins SG=16
        for every shape currently in the registry; the lowerer enforces
        the same via ``OpExecutionMode SubgroupSize 16``. CUDA/Metal
        default to 32. Autotuners DO NOT set ``config.subgroup_size`` —
        it's only an explicit-override knob (intended for opt-in SIMD16
        on Apple, etc.), and is honored here when set.

        Single source of truth — used by ``block()`` to size the CTA, by
        the decorator to seed ``KernelContext.subgroup_size``, and
        readable from kernel build bodies via ``bctx.subgroup_size``.
        """
        cfg_sgs = getattr(self.config, "subgroup_size", None)
        if cfg_sgs is not None:
            return int(cfg_sgs)
        if type(self).mma_sites(self.spec):
            import contextlib
            with contextlib.suppress(KeyError, NotImplementedError):
                shape_id = self._mma_cfg().shape_id
                if "_intel_" in shape_id:
                    return 16
        return 32

    def block(self) -> tuple[int, int, int]:
        """Default: ``(n_warps * subgroup_size, 1, 1)`` — single-row block
        from ``config.n_warps``. Kernels with a computed warp count
        (attn/owl_attn use ``gqa_ratio * NCW``) override.

        ``subgroup_size`` resolves via :meth:`resolve_subgroup_size` —
        derived from the active MMA shape (Intel → 16, default 32).
        Sizing block in subgroup-size units guarantees
        ``num_subgroups_per_WG == n_warps`` regardless of width, which
        the per-warp slot-partition math depends on (``warp_id =
        qk.subgroup_id()`` must range 0..n_warps-1).
        """
        n_warps = getattr(self.config, "n_warps", None)
        if n_warps is None:
            raise NotImplementedError(
                f"{type(self).__name__}.block() must be implemented or config "
                f"must have an n_warps field"
            )
        return (n_warps * self.resolve_subgroup_size(), 1, 1)

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
        from quark.weight_shuffle import cached_shuffle_b

        mma = self._mma_cfg()
        cfg_any: Any = cast(Any, self.config)
        shuffled = cached_shuffle_b(
            tensors[name],
            K_CHUNK=cfg_any.BK,
            mma_k=mma.mma_k,
            bpad=cfg_any.b_pad,
        )
        return {**tensors, name: shuffled}

    def autotune_input_key(self) -> tuple:
        """Identify which config knobs change the autotune input tensors.

        Default: empty tuple — every config shares the cached inputs +
        reference produced by ``make_tensors_numpy`` /
        ``reference_numpy``.

        Kernels whose inputs depend on a config knob (MoE in/out:
        ``work_list`` step is ``config.BM``) override this so the
        autotune harness recomputes inputs + reference once per
        distinct key, and reuses across configs that share it.
        """
        return ()

    def rebuild_autotune_inputs(self, base_inputs_np: dict) -> dict:
        """Return a fresh ``inputs_np`` tailored to this kernel's config.

        Only called by the autotune harness when ``autotune_input_key``
        differs from the base. Default: identity. Override alongside
        ``autotune_input_key`` so the test fixture's tensor layout
        tracks the active config.
        """
        return base_inputs_np

    def tflops(self, runtime_us: float) -> float:
        return self.flops() / (runtime_us * 1e-6) / 1e12

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
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        """Allocate numpy inputs + output stubs for a bench problem.

        Returns a ``{name: np.ndarray}`` dict matching the kernel's
        ``TENSORS`` manifest (every entry, including ``role="out"``
        stubs zero-filled in the target carrier dtype). Seeded via
        ``numpy.random.default_rng(seed)`` so the same
        ``(kernel, problem, seed)`` triple yields identical arrays
        across sessions — essential for on-disk ``RefCache``.

        Subclasses override with a ``@classmethod`` taking the same
        ``problem`` dict consumed by ``from_problem``. Device-side
        conversion happens at the tool boundary (``fuzz`` / ``bench``
        / ``autotune``), not here.
        """
        raise NotImplementedError(f"{cls.__name__} doesn't implement make_tensors_numpy()")

    def baselines(self, tensors: dict) -> list[Baseline]:
        """Named baselines (default: none) for the speed comparison column."""
        return []

    def launch_tensor_list(self, tensors: dict) -> list:
        """Helper: build the positional launch list from a name-keyed dict."""
        return [tensors[gt.name] for gt in self.global_tensors()]

    def bench_label(self) -> str:
        """Short human-readable label for one row of the bench table."""
        return type(self).__name__
