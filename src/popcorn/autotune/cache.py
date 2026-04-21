"""``AutotuneCache`` — three-level config cache + fast search.

EXEMPT FROM 500-LINE RULE: the cache class + fast-search generation loop
(with its per-config correctness and timing helpers) naturally cluster
around the same state (hot dict, launcher hook, timer hook). Splitting
would force a circular-import-shaped carve-out between the cache and
its own search path.

Sibling modules in the ``autotune`` package:

  * ``popcorn.autotune`` (this package's ``__init__``) owns the depth
    contextvar and generic GA primitives (``_crossover`` / ``_mutate`` /
    ``_parallel_compile``).
  * ``popcorn.autotune.io`` owns disk I/O + JSON (de)serialization.
  * ``popcorn.autotune.full_search`` owns the genetic loop used by
    ``pcf.<op>.autotune()`` / ``popcorn.max_autotune()``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from popcorn.autotune import (
    _FAST_EARLY_STOP_STALE_GENS,
    _FAST_N_GENS,
    _FAST_POP_PER_GEN,
    _log,
    _parallel_compile,
    current_search_depth,
)
from popcorn.autotune.io import (
    autotune_disabled,
    default_bundled_dir,
    default_cache_dir,
    load_bundled_default,
    load_from_disk,
    load_warm_seeds,
    make_key,
    save_to_disk,
)

if TYPE_CHECKING:
    from popcorn.device import Device


# ---------------------------------------------------------------------------
# Helpers shared by both search paths
# ---------------------------------------------------------------------------


def _cfg_key_simple(cfg) -> tuple:
    """Stable identity key for a config — used for seed dedup."""
    if dataclasses.is_dataclass(cfg):
        return tuple(sorted(dataclasses.asdict(cfg).items()))
    return (id(type(cfg)), repr(cfg))


def _cartesian(lists: list[list]) -> Iterable[tuple]:
    if not lists:
        yield ()
        return
    head, *tail = lists
    for a in head:
        for rest in _cartesian(tail):
            yield (a, *rest)


def _is_valid_for_device(kernel, caps) -> bool:
    is_valid_for = getattr(kernel, "is_valid_for", None)
    if callable(is_valid_for):
        try:
            return bool(is_valid_for(caps))
        except (TypeError, NotImplementedError):
            pass
    legacy = getattr(kernel, "is_valid", None)
    if callable(legacy):
        try:
            return bool(legacy())
        except (TypeError, NotImplementedError):
            pass
    return True


def _prune_score(kernel) -> float:
    fn = getattr(kernel, "prune_score", None)
    if callable(fn):
        try:
            return float(fn())
        except (TypeError, NotImplementedError):
            pass
    return 0.0


def _spec_summary(spec) -> str:
    """Short, human-readable form of a KernelSpec for autotune logs.

    Lists the dataclass field values on one line. Lets you correlate a
    ``[launch err]`` config with the actual problem it came from —
    which Linear / layer in the model was calling into the kernel.
    """
    if dataclasses.is_dataclass(spec):
        parts = []
        for f in dataclasses.fields(spec):
            val = getattr(spec, f.name)
            # Compact dtype repr — the full ``DType.BF16`` is noisy.
            val_str = str(val).removeprefix("DType.")
            parts.append(f"{f.name}={val_str}")
        return ", ".join(parts)
    return repr(spec)


# ---------------------------------------------------------------------------
# AutotuneCache
# ---------------------------------------------------------------------------


_DEFAULT_WARMUP = 3
_DEFAULT_ITERS = 10


@dataclass
class AutotuneCache:
    """Three-level cache for kernel configs with two search depths.

    **Fast search** (default): bounded N-candidate pass seeded from existing
    disk configs. Runs inline on first miss.

    **Full search**: full genetic algorithm, same warm seeding. Triggered by
    ``pcf.<op>.autotune()``, the ``POPCORN_MAX_AUTOTUNE=1`` env var, or
    ``with popcorn.max_autotune()``.

    Owned by :class:`popcorn.launcher.Launcher` (one per device per
    process). The Launcher injects two hooks at construction time:
    ``_compile_and_time`` (for fast search timing) and ``_launcher`` (for
    full search compilation).
    """

    device: Device
    cache_dir: Path = field(default_factory=default_cache_dir)
    bundled_dir: Path = field(default_factory=default_bundled_dir)
    # Fast-search params (per-generation budget; see _FAST_* constants in
    # ``autotune.py`` for the per-gen pop size + max-gens schedule).
    warmup: int = _DEFAULT_WARMUP
    iters: int = _DEFAULT_ITERS
    parallel_compile: int = 4
    # Full-search params: 128 per-gen population × up to 16 generations
    # with 4-stale-gen early stop. Bigger pop than fast (deeper
    # exploration) but still bounded so a slow kernel doesn't burn the
    # whole afternoon. ``pop_size=None`` would auto-size based on
    # log2(n_valid); we pin a concrete number for predictability.
    pop_size: int | None = 64
    n_gens: int = 8
    early_stop_gens: int = 3
    mutate_prob: float = 0.3
    rng_seed: int = 42
    bench_ms: float = 50.0
    max_workers: int | None = None
    _hot: dict[tuple, Any] = field(default_factory=dict, init=False)
    # Injected by Launcher at construction time. See Launcher.__init__.
    _compile_and_time: Any = field(default=None, init=False)
    _launcher: Any = field(default=None, init=False)
    # Optional ``(kernel_cls, spec, cfg, exc, launcher) -> str | None`` hook
    # fired on compile failures during full search. ``tools/autotune.py``
    # attaches a PTX-dump closure here.
    _compile_error_hook: Any = field(default=None, init=False)

    # ---- public lookup API --------------------------------------------------

    def lookup(self, kernel_cls, spec) -> Optional[Any]:
        """Walk the three-level chain. Returns the cached config or None."""
        key = self._make_key(kernel_cls, spec)
        if key in self._hot:
            return self._hot[key]

        from_disk = load_from_disk(self.cache_dir, key, self.device.fingerprint())
        if from_disk is not None:
            self._hot[key] = from_disk
            return from_disk

        bundled = load_bundled_default(self.bundled_dir, kernel_cls, spec)
        if bundled is not None:
            self._hot[key] = bundled
            return bundled

        return None

    def lookup_or_search(self, kernel_cls, spec) -> Any:
        """Cache lookup; runs search on miss and persists the winner.

        Search depth comes from ``current_search_depth()`` —
        ``POPCORN_MAX_AUTOTUNE=1`` or the ``popcorn.max_autotune()``
        context manager flip it from ``"fast"`` to ``"full"``.
        """
        cached = self.lookup(kernel_cls, spec)
        if cached is not None:
            return cached

        if autotune_disabled():
            return self._fallback_default(kernel_cls, spec)

        config = self._search(kernel_cls, spec, depth=current_search_depth())
        if config is None:
            config = self._fallback_default(kernel_cls, spec)
        self.store(kernel_cls, spec, config)
        return config

    def store(self, kernel_cls, spec, config) -> None:
        """Explicit write: hot tier + disk."""
        key = self._make_key(kernel_cls, spec)
        self._hot[key] = config
        save_to_disk(
            self.cache_dir,
            key,
            kernel_cls,
            spec,
            config,
            self.device.fingerprint(),
            runtime_us=None,
        )

    def clear_hot(self) -> None:
        self._hot.clear()

    # ---- key construction ---------------------------------------------------

    def _make_key(self, kernel_cls, spec) -> tuple:
        return make_key(kernel_cls, spec, self.device.fingerprint())

    # ---- warm seeding -------------------------------------------------------

    def _load_warm_seeds(self, kernel_cls, spec) -> list:
        return load_warm_seeds(
            self.cache_dir,
            self.bundled_dir,
            kernel_cls,
            spec,
            self.device.fingerprint(),
            self.device.caps,
        )

    # ---- search dispatch ----------------------------------------------------

    def _search(self, kernel_cls, spec, *, depth: str = "fast") -> Optional[Any]:
        seeds = self._load_warm_seeds(kernel_cls, spec)
        if depth == "full":
            return self._search_full(kernel_cls, spec, seeds)
        return self._search_fast(kernel_cls, spec, seeds)

    # ---- fast (bounded) search ----------------------------------------------

    def _search_fast(self, kernel_cls, spec, seeds: list) -> Optional[Any]:
        """Bounded multi-generation correctness-gated fast search.

        Lifecycle:
          - Build ONE tensors dict + reference for the whole search.
          - Random-sample the valid-config pool into ``n_gens`` small
            generations of ``n_per_gen`` configs each (warm-seeded
            disk configs front-loaded into gen 0).
          - Per generation: parallel-compile the batch (worker
            threads), then serial-check-and-time each config in the
            main thread. Update overall best_us / best_cfg.
          - Early stop after a generation that fails to improve
            ``best_us``. Return the fastest correct config.
        """
        prep = self._prepare_fast_search_inputs(kernel_cls, spec)
        if prep is None:
            return None
        name, all_valid, knob_names, tensors, reference, pspec, out_idx = prep
        lnch = getattr(self, "_launcher", None)
        timer = getattr(self, "_compile_and_time", None)

        # Legacy / unit-test path: launcher absent or reference setup
        # failed. Without a real launcher we can't run the per-config
        # correctness check, so fall through to the simple
        # enumerate-and-time loop (or prune-score sort if no timer).
        if lnch is None or tensors is None:
            return _legacy_fast_search(kernel_cls, spec, all_valid, timer)

        def _say(msg: str) -> None:
            _log(name, msg)

        _say(f"spec={_spec_summary(spec)}")

        # ── Build the candidate pool: warm seeds first, then a random
        # shuffle of the remaining valid configs. Each config tested at
        # most once across all generations.
        seed_keys = {_cfg_key_simple(c) for c in seeds}
        rest = [c for c in all_valid if _cfg_key_simple(c) not in seed_keys]
        rng = random.Random(self.rng_seed)
        rng.shuffle(rest)
        pool: list = list(seeds) + rest

        best_cfg: Any = None
        best_us = float("inf")
        stale_gens = 0
        pool_idx = 0

        for gen in range(_FAST_N_GENS):
            batch: list = []
            while pool_idx < len(pool) and len(batch) < _FAST_POP_PER_GEN:
                batch.append(pool[pool_idx])
                pool_idx += 1
            if not batch:
                break

            prev_best_us = best_us
            gen_best_us = self._run_fast_generation(
                kernel_cls,
                spec,
                batch,
                knob_names,
                tensors,
                reference,
                pspec,
                out_idx,
                gen,
                best_us,
                _say,
            )

            # The generation hook records wins via ``best_us`` mutations,
            # returning the gen's own best. Fold in and early-stop.
            if gen_best_us[0] < best_us:
                best_us = gen_best_us[0]
                best_cfg = gen_best_us[1]

            improved = best_cfg is not None and best_us < prev_best_us * 0.999
            if improved:
                stale_gens = 0
                _say(
                    f"gen{gen}: best improved {prev_best_us:.2f} → {best_us:.2f} μs "
                    f"(gen this round: {gen_best_us[0]:.2f} μs)"
                )
            elif best_cfg is not None:
                stale_gens += 1
                _say(
                    f"gen{gen}: best stayed at {best_us:.2f} μs "
                    f"(gen this round: {gen_best_us[0]:.2f} μs); stale={stale_gens}"
                )
                if stale_gens >= _FAST_EARLY_STOP_STALE_GENS:
                    _say(f"early-stop after gen {gen}")
                    break

        if best_cfg is None:
            _say("no config passed correctness across all generations")
        return best_cfg

    def _prepare_fast_search_inputs(self, kernel_cls, spec) -> Optional[tuple]:
        """Common setup for ``_search_fast``: enumerate valid configs,
        resolve knob names, build the one-shot tensors+reference. Returns
        ``(name, all_valid, knob_names, tensors, reference, pspec, out_idx)``
        or None if no valid configs.
        """
        import dataclasses as _dc

        name = kernel_cls.NAME
        all_valid = list(self._enumerate_valid(kernel_cls, spec))
        if not all_valid:
            return None

        knob_names: list[str] = []
        try:
            tune_space_fn = getattr(kernel_cls, "tune_space_resolved", None)
            if tune_space_fn is not None:
                tune_space = tune_space_fn(spec, self.device)
            else:
                tune_space = kernel_cls.tune_space()
            knob_names = list(tune_space.keys())
        except Exception:
            if dataclasses.is_dataclass(all_valid[0]):
                knob_names = [f.name for f in dataclasses.fields(all_valid[0])]

        lnch = getattr(self, "_launcher", None)
        spec_dict = _dc.asdict(spec) if _dc.is_dataclass(spec) else {}
        tensors = None
        reference = None
        pspec = None
        out_idx = None
        if lnch is not None:
            try:
                tensors = kernel_cls.make_tensors(spec_dict)
                default_kernel = kernel_cls.from_problem(spec_dict)
                pspec = default_kernel.param_spec()
                bufs = [tensors[b.name] for b in pspec.buffers]
                out_idx = kernel_cls.OUTPUT_IDX
                if out_idx < 0:
                    out_idx = len(bufs) + out_idx
                inputs = [bufs[i] for i in range(len(bufs)) if i != out_idx]
                reference = default_kernel.reference(*inputs)
            except Exception as e:
                _log(name, f"reference setup failed ({type(e).__name__}: {e})")
                tensors = None

        return name, all_valid, knob_names, tensors, reference, pspec, out_idx

    def _run_fast_generation(
        self,
        kernel_cls,
        spec,
        batch: list,
        knob_names: list[str],
        tensors,
        reference,
        pspec,
        out_idx,
        gen: int,
        best_us_snapshot: float,
        say,
    ) -> tuple[float, Any]:
        """Compile, correctness-check, and time each config in ``batch``.

        Returns the ``(gen_best_us, gen_best_cfg)`` pair found this gen
        (inf / None when nothing passed). Side effect: none — caller
        folds the return into the overall best tracking.
        """
        import math as _math

        from popcorn.backend import IS_METAL, PT
        from popcorn.correctness import check_correctness

        lnch = self._launcher
        timer = self._compile_and_time
        spec_cls = kernel_cls.SPEC_CLS

        import dataclasses as _dc

        spec_dict = _dc.asdict(spec) if _dc.is_dataclass(spec) else {}

        compile_cache: dict = {}
        if knob_names and not IS_METAL:
            try:
                compile_cache = _parallel_compile(
                    kernel_cls, spec, batch, lnch, knob_names=knob_names
                )
            except Exception as e:
                say(
                    f"gen{gen}: parallel compile failed ({type(e).__name__}: {e}); "
                    f"falling back to serial"
                )

        def _check_config(cfg) -> tuple[bool, float, str | None]:
            # Drain any async error queued by the prior config before
            # this one starts. Otherwise CUDA reports the prior
            # launch's misalignment / OOB on the first call of the
            # next config (typically ``PT.zero_``) and blames the
            # wrong config. Surface it here, move on.
            with contextlib.suppress(BaseException):
                PT.synchronize()
            try:
                cfg_kernel = kernel_cls(spec_cls(**spec_dict), cfg)
                launch_tensors = cfg_kernel.prepare_launch_tensors(tensors)
                cfg_buffers = [launch_tensors[b.name] for b in pspec.buffers]
                cfg_output_buf = PT.zero_(cfg_buffers[out_idx])
                cfg_buffers[out_idx] = cfg_output_buf
                compiled = lnch.compile(kernel_cls, spec, cfg)
                _launch_result = compiled.launch(buffers=cfg_buffers)
                PT.synchronize()
                # On CUDA the kernel writes ``cfg_output_buf`` in place;
                # on Metal MLX returns fresh mx.arrays for every non-
                # readonly buffer in pspec order, and the original
                # ``cfg_output_buf`` stays zero. Pick the launch's
                # actual output on Metal.
                if IS_METAL and _launch_result:
                    out_pos = sum(1 for b in pspec.buffers[:out_idx] if not b.readonly)
                    actual_output = _launch_result[out_pos]
                else:
                    actual_output = cfg_output_buf
                out_dtype = PT.backend_dtype_to_ir(actual_output.dtype)
                # NOTE (regression test): we tried switching this to
                # GPU-resident comparison to avoid the heavy D2H per
                # config, but doing so broke the cast+cpu workaround in
                # call_with_bindings — the previously-stable ground-
                # truth fix stopped working. Restore D2H here to keep
                # that ground truth stable while we hunt the root cause.
                cr = check_correctness(
                    actual_output,
                    reference,
                    out_dtype=out_dtype,
                    threshold=kernel_cls.correctness_threshold(out_dtype),
                )
                return cr.passed, cr.cos_sim, None
            except Exception as e:
                return False, float("nan"), f"{type(e).__name__}: {e}"

        gen_best_us = float("inf")
        gen_best_cfg: Any = None
        for cfg in batch:
            if knob_names and compile_cache:
                key = tuple(getattr(cfg, k) for k in knob_names)
                cached = compile_cache.get(key)
                if cached is not None and cached[0] is None:
                    say(f"gen{gen} [cerr] {cfg}: {cached[1]}")
                    continue

            ok, cos, err = _check_config(cfg)
            if err is not None:
                say(f"gen{gen} [err]  {cfg}: {err}")
                continue
            if not ok:
                cos_s = "nan" if _math.isnan(cos) else f"{cos:.4f}"
                say(f"gen{gen} [fail] cos={cos_s}  {cfg}")
                continue
            try:
                us = timer(kernel_cls, spec, cfg, tensors=tensors)
            except Exception as e:
                say(f"gen{gen} [terr] {cfg}: {type(e).__name__}: {e}")
                continue
            tag = " ***" if us < best_us_snapshot else ""
            say(f"gen{gen} [ok]   {us:8.2f} μs  cos={cos:.4f}  {cfg}{tag}")
            if us < gen_best_us:
                gen_best_us = us
                gen_best_cfg = cfg

        return gen_best_us, gen_best_cfg

    # ---- full (genetic) search ----------------------------------------------

    def _search_full(
        self,
        kernel_cls,
        spec,
        seeds: list,
        *,
        tune_space: dict | None = None,
    ) -> Optional[Any]:
        """Full genetic search. Falls back to fast search when the launcher
        or make_tensors/reference are unavailable.

        ``tune_space`` overrides the kernel's default space — used by
        ``tools/autotune.py`` to pin problem-specific config overrides.
        """
        result = self._search_full_result(kernel_cls, spec, seeds, tune_space=tune_space)
        if result is None:
            return None
        return result[0]

    def _search_full_result(
        self,
        kernel_cls,
        spec,
        seeds: list,
        *,
        tune_space: dict | None = None,
    ) -> Optional[tuple]:
        """Like ``_search_full`` but returns ``(best_cfg, best_us)``.

        ``tools/autotune.py`` calls this directly so it can report the
        winning runtime + save it to disk.
        """
        from popcorn.autotune.full_search import run_full_search

        return run_full_search(self, kernel_cls, spec, seeds, tune_space=tune_space)

    # ---- candidate enumeration ----------------------------------------------

    def _enumerate_valid(self, kernel_cls, spec) -> Iterable[Any]:
        """Cartesian product of tune_space_resolved, filtered to valid configs."""
        space: dict | None = None
        fn_resolved = getattr(kernel_cls, "tune_space_resolved", None)
        if fn_resolved is not None:
            try:
                space = fn_resolved(spec, self.device)
            except Exception:
                space = None
        if not space:
            fn = getattr(kernel_cls, "tune_space", None)
            if fn is None:
                return
            try:
                space = fn()
            except NotImplementedError:
                return

        if not isinstance(space, dict) or not space:
            return

        config_cls = getattr(kernel_cls, "CONFIG_CLS", None)
        if config_cls is None:
            return

        knob_names: list[str] = [str(k) for k in space]
        knob_values = [space[k] for k in knob_names]
        for combo in _cartesian(knob_values):
            kwargs: dict[str, Any] = dict(zip(knob_names, combo, strict=False))
            try:
                cfg = config_cls(**kwargs)
            except TypeError:
                continue
            try:
                kernel = kernel_cls(spec, cfg)
            except (TypeError, ValueError):
                continue
            if _is_valid_for_device(kernel, self.device.caps):
                yield cfg

    # Alias kept for existing test imports.
    def _enumerate_candidates(self, kernel_cls, spec) -> Iterable[Any]:
        return self._enumerate_valid(kernel_cls, spec)

    # ---- fallback -----------------------------------------------------------

    def _fallback_default(self, kernel_cls, spec) -> Any:
        """Return a config valid for this (kernel, spec, device), or raise.

        Called when the main search path (``lookup_or_search``) didn't
        produce a winner — either because autotune is disabled, or
        because every candidate failed correctness. The returned config
        must pass ``is_valid_for(caps)`` so the subsequent
        ``Launcher.compile`` call can't raise into user code.

        Order:
          1. Kernel's ``_pick_default_cfg(spec)`` hook if present.
          2. First config yielded by ``_enumerate_valid`` (already
             caps-filtered).
          3. ``config_cls()`` default — only if it happens to pass
             caps validation for this spec.

        If nothing passes, raise ``RuntimeError`` with the spec so the
        mismatch surfaces here rather than as a ``CUDA error: misaligned
        address`` / ``smem > max_smem_per_block`` at the call site.
        """
        picker = getattr(kernel_cls, "_pick_default_cfg", None)
        if callable(picker):
            try:
                cfg = picker(spec)
                if cfg is not None and _is_valid_for_device(
                    kernel_cls(spec, cfg), self.device.caps
                ):
                    return cfg
            except (TypeError, NotImplementedError):
                pass

        for cfg in self._enumerate_valid(kernel_cls, spec):
            return cfg

        config_cls = getattr(kernel_cls, "CONFIG_CLS", None)
        if config_cls is None:
            raise RuntimeError(
                f"AutotuneCache: no fallback config available for {kernel_cls.__qualname__}"
            )
        try:
            default_cfg = config_cls()
        except TypeError as e:
            raise RuntimeError(
                f"AutotuneCache: cannot construct default {config_cls.__name__}()"
            ) from e
        try:
            if _is_valid_for_device(kernel_cls(spec, default_cfg), self.device.caps):
                return default_cfg
        except (TypeError, ValueError):
            pass
        raise RuntimeError(
            f"AutotuneCache: no valid config for {kernel_cls.__qualname__} "
            f"with spec {spec!r} on device {self.device.caps.name!r}. "
            f"Every candidate from tune_space + the default config was rejected "
            f"by is_valid_for (typical causes: K not divisible by any legal BK, "
            f"smem budget exceeded, or the spec's dtype axes demanding an MMA "
            f"shape the device doesn't advertise)."
        )


# ---------------------------------------------------------------------------
# Free-function helpers extracted from the big search methods
# ---------------------------------------------------------------------------


def _legacy_fast_search(kernel_cls, spec, all_valid, timer) -> Any:
    """Launcher-absent fast path: either time with the timer callback or
    sort by ``prune_score`` if no timer was injected (unit tests).
    """
    if timer is None:

        def _score(cfg):
            try:
                return _prune_score(kernel_cls(spec, cfg))
            except Exception:
                return 0.0

        return min(all_valid, key=_score)

    best_cfg: Any = None
    best_t = float("inf")
    for cfg in all_valid:
        try:
            t = timer(kernel_cls, spec, cfg)
        except Exception:
            continue
        if t < best_t:
            best_t = t
            best_cfg = cfg
    return best_cfg
