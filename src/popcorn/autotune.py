"""Runtime autotune cache + shared genetic search engine.

Three-level config cache (hot dict / on-disk JSON / bundled defaults) plus
two search depths:

  ``"fast"``  — bounded N-candidate search, seeded from existing disk configs.
                Runs inline on first cache miss; typically < 2 s.
  ``"full"``  — full genetic search (pop × gens), same warm seeding.
                Used by ``pcf.<op>.autotune(...)`` and ``make autotune``.

The active depth is controlled by (highest priority first):

  1. ``POPCORN_MAX_AUTOTUNE=1`` env var (forces ``"full"`` process-wide)
  2. ``with popcorn.max_autotune():`` context manager
  3. ``AutotuneCache.search_depth`` field (default ``"fast"``)

Cache key (per §9.1):

    (kernel_qualname, spec_fingerprint, device_fingerprint, source_hash)

Public symbols consumed by ``tools/autotune.py``:

    genetic_search(kernel_cls, spec, tune_space, launcher, *, tensors,
                   reference, out_idx, seeds, compile_batch_fn, ...)
    _crossover(parent_a, parent_b, knob_names, rng) -> dict
    _mutate(child, knob_names, knob_values, mutate_prob, rng) -> dict
"""

from __future__ import annotations

import contextvars
import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from popcorn.device import Device


# ---------------------------------------------------------------------------
# Search-depth control
# ---------------------------------------------------------------------------

_SEARCH_DEPTH: contextvars.ContextVar[str] = contextvars.ContextVar(
    "popcorn_search_depth", default="fast"
)


def current_search_depth() -> str:
    """Return the active search depth: ``"full"`` if
    ``POPCORN_MAX_AUTOTUNE=1`` is set, else the ContextVar value."""
    val = os.environ.get("POPCORN_MAX_AUTOTUNE", "")
    if val.lower() in ("1", "true", "yes", "on"):
        return "full"
    return _SEARCH_DEPTH.get()


# ---------------------------------------------------------------------------
# Defaults / env vars
# ---------------------------------------------------------------------------

# Fast search: 4 generations × 8 configs each = 32 configs max,
# early-stop after a generation with no improvement (so most
# searches terminate after 1-2 gens once a fast-enough config is
# found). Tuned to keep cold-start latency low (sub-second when
# winners are obvious) while still exploring enough of the search
# space to escape the cartesian-order bias on diverse tune spaces.
_FAST_POP_PER_GEN = 8
_FAST_N_GENS = 4
_FAST_EARLY_STOP_STALE_GENS = 1

_DEFAULT_WARMUP = 3
_DEFAULT_ITERS = 10
_KILL_SWITCH_ENV = "POPCORN_DISABLE_AUTOTUNE"
_CACHE_DIR_ENV = "POPCORN_CACHE_DIR"


def _default_cache_dir() -> Path:
    env = os.environ.get(_CACHE_DIR_ENV)
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "popcorn"
    return Path.home() / ".cache" / "popcorn"


def _default_bundled_dir() -> Path:
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        candidate = ancestor / "configs"
        if candidate.is_dir():
            return candidate
    return here.parent.parent / "configs"


def _autotune_disabled() -> bool:
    val = os.environ.get(_KILL_SWITCH_ENV, "")
    return val.lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Spec fingerprinting
# ---------------------------------------------------------------------------


def _spec_fingerprint(spec: Any) -> tuple:
    if dataclasses.is_dataclass(spec):
        return tuple((f.name, getattr(spec, f.name)) for f in dataclasses.fields(spec))
    return ((spec.__class__.__name__, repr(spec)),)


def _spec_to_json_dict(spec: Any) -> dict:
    if dataclasses.is_dataclass(spec):
        return dataclasses.asdict(spec)
    return {"_repr": repr(spec)}


def _config_to_json_dict(config: Any) -> dict:
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return {"_repr": repr(config)}


def _config_from_json_dict(config_cls: type, payload: dict) -> Any:
    if not dataclasses.is_dataclass(config_cls):
        raise TypeError(f"_config_from_json_dict: {config_cls.__name__} is not a dataclass")
    field_names = {f.name for f in dataclasses.fields(config_cls)}
    kwargs = {k: v for k, v in payload.items() if k in field_names}
    return config_cls(**kwargs)


_SOURCE_HASH_CACHE: dict[type, str] = {}


def _source_hash(kernel_cls: type) -> str:
    # Process-lifetime memo: source files don't change mid-run, and
    # ``inspect.getsource`` + SHA256 on a multi-KB kernel class shows
    # up as ~1ms per call on the hot path (every ``lookup_or_search``
    # rebuilds the cache key). Kernel classes are identity-stable, so
    # keying by the class object is safe.
    cached = _SOURCE_HASH_CACHE.get(kernel_cls)
    if cached is not None:
        return cached
    try:
        src = inspect.getsource(kernel_cls)
    except (OSError, TypeError):
        src = kernel_cls.__qualname__
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    _SOURCE_HASH_CACHE[kernel_cls] = digest
    return digest


# ---------------------------------------------------------------------------
# Genetic algorithm primitives (shared with tools/autotune.py)
# ---------------------------------------------------------------------------


def _crossover(
    parent_a: Any,
    parent_b: Any,
    knob_names: list[str],
    rng: random.Random,
) -> dict:
    """Uniform crossover: pick each knob from one parent at random."""
    return {k: getattr(parent_a if rng.random() < 0.5 else parent_b, k) for k in knob_names}


def _mutate(
    child: dict,
    knob_names: list[str],
    knob_values: list[list],
    mutate_prob: float,
    rng: random.Random,
) -> dict:
    """Gaussian index mutation: nudge each knob with probability mutate_prob."""
    for k, vals in zip(knob_names, knob_values, strict=False):
        if rng.random() < mutate_prob and len(vals) > 1:
            cur = child[k]
            if cur in vals:
                idx = vals.index(cur)
                sigma = max(0.8, len(vals) * 0.2)
                for _ in range(10):
                    new_idx = max(0, min(len(vals) - 1, round(idx + rng.gauss(0, sigma))))
                    if new_idx != idx:
                        child[k] = vals[new_idx]
                        break
    return child


def _parallel_compile(
    kernel_cls,
    spec,
    configs: list,
    launcher,
    *,
    knob_names: list[str],
    max_workers: int | None = None,
    on_compile_error=None,
) -> dict:
    """Compile ``configs`` in parallel threads.

    Returns ``{cfg_key: (compiled_kernel_or_None, error_str_or_None)}``.

    ``on_compile_error(kernel_cls, spec, cfg, exc, launcher) -> str | None``
    is an optional callback for extra error annotation (e.g. PTX dumps in
    the offline CLI). The return value is appended to the error string.
    """
    n_workers = min(max_workers or max(1, (os.cpu_count() or 2) // 2), max(len(configs), 1))
    _runtime = getattr(launcher.driver, "runtime", None)

    def _cfg_key(cfg):
        return tuple(getattr(cfg, k) for k in knob_names)

    raw: dict = {}

    def _worker(cfg):
        if _runtime is not None:
            _runtime.retain_primary_context(0)
        try:
            return _cfg_key(cfg), launcher.compile(kernel_cls, spec, cfg), None, cfg
        except Exception as e:
            return _cfg_key(cfg), None, e, cfg

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_worker, c) for c in configs]
        for fut in as_completed(futs):
            key, compiled, exc, cfg = fut.result()
            raw[key] = (compiled, exc, cfg)

    result = {}
    for key, (compiled, exc, cfg) in raw.items():
        if exc is None:
            result[key] = (compiled, None)
        else:
            err_str = str(exc)[:200]
            if on_compile_error is not None:
                try:
                    extra = on_compile_error(kernel_cls, spec, cfg, exc, launcher)
                    if extra:
                        err_str = f"{err_str}  {extra}"
                except Exception:
                    pass
            result[key] = (None, err_str)

    return result


# ---------------------------------------------------------------------------
# Genetic search (shared by AutotuneCache._search_full and tools/autotune.py)
# ---------------------------------------------------------------------------


def genetic_search(
    kernel_cls,
    spec,
    tune_space: dict,
    launcher,
    *,
    tensors: dict,
    reference,
    out_idx: int,
    seeds: list | None = None,
    compile_batch_fn=None,
    pop_size: int | None = None,
    n_gens: int = 32,
    early_stop_gens: int = 2,
    mutate_prob: float = 0.3,
    rng_seed: int = 42,
    bench_ms: float = 50.0,
    max_workers: int | None = None,
    verbose: bool = True,
) -> tuple | None:
    """Run the genetic autotune search. Returns ``(best_config, best_us)`` or
    ``None`` if no valid+correct config is found.

    Parameters
    ----------
    kernel_cls, spec, tune_space:
        Kernel class, frozen spec, and device-resolved knob space
        (from ``kernel_cls.tune_space_resolved(spec, device)``).
    launcher:
        Active ``Launcher`` instance used for compilation and timing.
    tensors:
        Input+output tensors from ``kernel_cls.make_tensors(...)``.
    reference:
        Pre-computed reference output for correctness gating.
    out_idx:
        Index of the output buffer in ``kernel.param_spec().buffers``.
    seeds:
        Configs to front-load in the initial population (warm seeding).
    compile_batch_fn:
        ``fn(kernel_cls, spec, configs, launcher, *, knob_names,
        max_workers) -> {key: (compiled, err)}`` — defaults to the
        built-in ``_parallel_compile``. Pass ``tools.autotune._parallel_compile``
        for PTX-dump-on-error behaviour in the offline CLI.
    """
    from popcorn.backend import IS_METAL, PT
    from popcorn.correctness import check_correctness

    if compile_batch_fn is None:
        compile_batch_fn = _parallel_compile

    if not tune_space:
        return None

    spec_cls = kernel_cls.SPEC_CLS
    config_cls = kernel_cls.CONFIG_CLS
    if config_cls is None:
        return None

    # Buffer layout comes from the default config's param_spec —
    # ordering is config-independent for every kernel in this repo.
    import dataclasses as _dc

    spec_dict = _dc.asdict(spec) if _dc.is_dataclass(spec) else {}
    default_kernel = kernel_cls.from_problem(spec_dict)
    pspec = default_kernel.param_spec()
    buffers_list = [tensors[b.name] for b in pspec.buffers]
    if out_idx < 0:
        out_idx = len(buffers_list) + out_idx

    knob_names = list(tune_space.keys())
    knob_values = [tune_space[k] for k in knob_names]

    # Fixed (non-tunable) config fields come from the default config.
    all_cfg_fields = (
        {f.name for f in _dc.fields(default_kernel.config)}
        if _dc.is_dataclass(default_kernel.config)
        else set()
    )
    fixed = {k: getattr(default_kernel.config, k) for k in all_cfg_fields if k not in tune_space}

    # Enumerate valid configs — full cartesian product up to a sample cap.
    from itertools import product as _product

    _sample_budget = max(1024, (pop_size or 64) * 8)
    total_combos = 1
    for vals in knob_values:
        total_combos *= max(len(vals), 1)

    _rng_enum = random.Random(rng_seed)

    def _build(combo):
        kw = dict(fixed)
        kw.update(dict(zip(knob_names, combo, strict=False)))
        try:
            cfg = config_cls(**kw)
            k = kernel_cls(spec_cls(**spec_dict), cfg)
            if k.is_valid_for(launcher.device.caps):
                return cfg
        except (TypeError, ValueError):
            pass
        return None

    valid_configs: list = []
    if total_combos <= _sample_budget:
        for combo in _product(*knob_values):
            cfg = _build(combo)
            if cfg is not None:
                valid_configs.append(cfg)
    else:
        seen_combos: set[tuple] = set()
        attempts = 0
        attempt_cap = _sample_budget * 4
        while len(valid_configs) < _sample_budget and attempts < attempt_cap:
            combo = tuple(_rng_enum.choice(v) for v in knob_values)
            attempts += 1
            if combo in seen_combos:
                continue
            seen_combos.add(combo)
            cfg = _build(combo)
            if cfg is not None:
                valid_configs.append(cfg)
        if verbose:
            print(
                f"  sampled {len(valid_configs)} valid / {attempts} attempts "
                f"from {total_combos:,} combos"
            )

    n_valid = len(valid_configs)
    if n_valid == 0:
        if verbose:
            print("  (no valid configs)")
        return None

    # Decide exhaustive vs genetic.
    auto_pop = max(16, int(16 * math.log2(max(n_valid, 2))))
    _pop_size = pop_size if pop_size is not None else auto_pop
    dense = n_valid <= _pop_size

    if dense:
        _pop_size = n_valid
        _n_gens = 1
        mode = "exhaustive"
    else:
        _n_gens = n_gens
        mode = f"genetic (pop={_pop_size}, gens={_n_gens})"

    if verbose:
        print(f"  {n_valid} valid configs, {mode}")

    rng = random.Random(rng_seed)

    # Build initial population — warm seeds at the front.
    if dense:
        population = list(valid_configs)
    else:
        _seeds = seeds or []
        seed_keys = {tuple(getattr(c, k) for k in knob_names) for c in _seeds}
        non_seeds = [
            c for c in valid_configs if tuple(getattr(c, k) for k in knob_names) not in seed_keys
        ]
        seed_slots = min(len(_seeds), _pop_size)
        rand_slots = _pop_size - seed_slots
        population = list(_seeds[:seed_slots]) + rng.sample(
            non_seeds, min(rand_slots, len(non_seeds))
        )
        # Pad if breeding produced fewer than pop_size.
        while len(population) < min(_pop_size, n_valid):
            population.append(rng.choice(valid_configs))

    seen: dict[tuple, tuple[float, object]] = {}
    valid_set = {tuple(getattr(c, k) for k in knob_names) for c in valid_configs}
    valid_by_key = {tuple(getattr(c, k) for k in knob_names): c for c in valid_configs}

    best_us = float("inf")
    best_cfg = None
    stale_gens = 0

    for gen in range(_n_gens):
        prev_best_us = best_us
        to_compile = [c for c in population if tuple(getattr(c, k) for k in knob_names) not in seen]

        if to_compile:
            n_threads = max(1, min(max_workers or (os.cpu_count() or 2) // 2, len(to_compile)))
            if verbose:
                print(
                    f"  gen {gen}: compiling {len(to_compile)} new configs ({n_threads} threads)..."
                )

            compile_cache = compile_batch_fn(
                kernel_cls,
                spec,
                to_compile,
                launcher,
                knob_names=knob_names,
                max_workers=max_workers,
            )

            _first_err_logged = False
            for cfg in to_compile:
                key = tuple(getattr(cfg, k) for k in knob_names)
                compiled, err = compile_cache.get(key, (None, "not compiled"))
                if compiled is None and err is None:
                    err = "compile returned None"
                if err is not None:
                    if verbose and not _first_err_logged:
                        print(f"    [compile err] {cfg}: {err}")
                        _first_err_logged = True
                    seen[key] = (float("inf"), cfg)
                    continue

                cfg_kernel = kernel_cls(spec_cls(**spec_dict), cfg)
                try:
                    launch_tensors = cfg_kernel.prepare_launch_tensors(tensors)
                except Exception as _e:
                    if verbose and not _first_err_logged:
                        print(f"    [prep err] {cfg}: {_e}")
                        _first_err_logged = True
                    seen[key] = (float("inf"), cfg)
                    continue

                cfg_buffers = [launch_tensors[b.name] for b in pspec.buffers]
                cfg_output_buf = PT.zero_(cfg_buffers[out_idx])
                cfg_buffers[out_idx] = cfg_output_buf

                _launch_err: BaseException | None = None
                _launch_result = None
                try:
                    _launch_result = compiled.launch(buffers=cfg_buffers)
                    PT.synchronize()
                except BaseException as _e:
                    _launch_err = _e

                if _launch_err is not None:
                    if verbose and not _first_err_logged:
                        print(
                            f"    [launch err] {cfg}: {type(_launch_err).__name__}: {_launch_err}"
                        )
                        _first_err_logged = True
                    seen[key] = (float("inf"), cfg)
                    continue

                if IS_METAL and _launch_result:
                    out_pos = sum(1 for b in pspec.buffers[:out_idx] if not b.readonly)
                    actual_output = _launch_result[out_pos]
                else:
                    actual_output = cfg_output_buf

                out_dtype = PT.backend_dtype_to_ir(actual_output.dtype)
                # Pass GPU tensors directly — check_correctness uses
                # PT.cosine_sim / PT.has_nan / PT.all_finite which run
                # on-device and only sync a tiny scalar at the end.
                # Avoiding the full-tensor D2H here matters for the
                # post-Phase-2 launch state (see FIRST_LAUNCH_BUG.md).
                cr = check_correctness(
                    actual_output,
                    reference,
                    out_dtype=out_dtype,
                    threshold=kernel_cls.correctness_threshold(out_dtype),
                )
                if not cr.passed:
                    if verbose and not _first_err_logged:
                        print(f"    [wrong] {cfg}: cos={cr.cos_sim:.4f}")
                        _first_err_logged = True
                    seen[key] = (float("inf"), cfg)
                    continue

                # Zero output again for timing run.
                cfg_output_buf = PT.zero_(cfg_buffers[out_idx])
                cfg_buffers[out_idx] = cfg_output_buf
                try:
                    from popcorn.backend import PT as _PT

                    _compiled = compiled
                    _bufs = cfg_buffers
                    us = _PT.time_callable(
                        lambda _c=_compiled, _b=_bufs: _c.launch(buffers=_b),
                        warmup_ms=10.0,
                        bench_ms=bench_ms,
                    )
                except Exception:
                    seen[key] = (float("inf"), cfg)
                    continue

                seen[key] = (us, cfg)
                if verbose:
                    tag = " ***" if us < best_us else ""
                    print(f"    {us:8.2f} μs  cos={cr.cos_sim:.6f}  {cfg}{tag}")

                if us < best_us:
                    best_us = us
                    best_cfg = cfg

        # Generation summary.
        gen_results = [
            (seen[tuple(getattr(c, k) for k in knob_names)], c)
            for c in population
            if tuple(getattr(c, k) for k in knob_names) in seen
        ]
        gen_results.sort(key=lambda x: x[0][0])
        gen_best = gen_results[0][0][0] if gen_results else float("inf")
        n_ok = sum(1 for (us, _), _ in gen_results if us < float("inf"))
        if verbose:
            print(f"  gen {gen}: best={gen_best:.2f} μs, valid={n_ok}/{len(population)}")

        if gen > 0:
            if gen_best >= prev_best_us * 0.99:
                stale_gens += 1
            else:
                stale_gens = 0
            if stale_gens >= early_stop_gens:
                if verbose:
                    print(f"  early stop (no >1% improvement for {early_stop_gens} gens)")
                break

        if dense or gen == _n_gens - 1:
            continue

        # Breed next generation.
        n_elite = max(2, _pop_size // 4)
        elites = [c for (_, _), c in gen_results[:n_elite]]

        next_pop = list(elites)
        attempts = 0
        while len(next_pop) < _pop_size and attempts < _pop_size * 20:
            attempts += 1
            a, b = rng.sample(elites, min(2, len(elites)))
            child = _crossover(a, b, knob_names, rng)
            child = _mutate(child, knob_names, knob_values, mutate_prob, rng)
            kw = dict(fixed)
            kw.update(child)
            try:
                cfg = config_cls(**kw)
            except TypeError:
                continue
            key = tuple(getattr(cfg, k) for k in knob_names)
            if key not in valid_set:
                continue
            next_pop.append(valid_by_key[key])

        while len(next_pop) < _pop_size:
            next_pop.append(rng.choice(valid_configs))

        population = next_pop

    if best_cfg is None:
        return None
    return (best_cfg, best_us)


# ---------------------------------------------------------------------------
# Result struct
# ---------------------------------------------------------------------------


@dataclass
class _SearchResult:
    config: Any
    runtime_us: float
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# AutotuneCache
# ---------------------------------------------------------------------------


@dataclass
class AutotuneCache:
    """Three-level cache for kernel configs with two search depths.

    **Fast search** (``search_depth="fast"``, default): bounded N-candidate
    pass seeded from existing disk configs. Runs inline on first miss.

    **Full search** (``search_depth="full"``): full genetic algorithm, same
    warm seeding. Triggered by ``pcf.<op>.autotune()``, the
    ``POPCORN_MAX_AUTOTUNE=1`` env var, or ``with popcorn.max_autotune()``.

    Owned by :class:`popcorn.launcher.Launcher` (one per device per
    process). The Launcher injects two hooks at construction time:
    ``_compile_and_time`` (for fast search timing) and ``_launcher`` (for
    full search compilation).
    """

    device: Device
    cache_dir: Path = field(default_factory=_default_cache_dir)
    bundled_dir: Path = field(default_factory=_default_bundled_dir)
    # Fast-search params (per-generation budget; see _FAST_* constants
    # above for the per-gen pop size + max-gens schedule).
    warmup: int = _DEFAULT_WARMUP
    iters: int = _DEFAULT_ITERS
    parallel_compile: int = 4
    # Full-search params: 64 per-gen population × up to 8 generations
    # with 2-stale-gen early stop. Bigger pop than fast (deeper
    # exploration) but still bounded so a slow kernel doesn't burn
    # the whole afternoon. ``pop_size=None`` would let
    # ``genetic_search`` auto-size based on log2(n_valid); we pin a
    # concrete number for predictability.
    search_depth: str = "fast"
    pop_size: int | None = 64
    n_gens: int = 8
    early_stop_gens: int = 2
    mutate_prob: float = 0.3
    rng_seed: int = 42
    bench_ms: float = 50.0
    max_workers: int | None = None
    # Internal hot dict — not a constructor argument.
    _hot: dict[tuple, Any] = field(default_factory=dict, init=False)
    # Injected by Launcher at construction time. See Launcher.__init__.
    _compile_and_time: Any = field(default=None, init=False)
    _launcher: Any = field(default=None, init=False)

    # ---- public lookup API --------------------------------------------------

    def lookup(self, kernel_cls, spec) -> Optional[Any]:
        """Walk the three-level chain. Returns the cached config or None."""
        key = self._make_key(kernel_cls, spec)
        if key in self._hot:
            return self._hot[key]

        from_disk = self._load_from_disk(key)
        if from_disk is not None:
            self._hot[key] = from_disk
            return from_disk

        bundled = self._load_bundled_default(kernel_cls, spec)
        if bundled is not None:
            self._hot[key] = bundled
            return bundled

        return None

    def lookup_or_search(self, kernel_cls, spec, *, depth: str | None = None) -> Any:
        """Cache lookup; runs search on miss and persists the winner.

        ``depth`` overrides the instance ``search_depth`` and the
        ``POPCORN_MAX_AUTOTUNE`` env var for this single call.
        Pass ``depth="full"`` from the ``.autotune()`` warmup API.
        """
        cached = self.lookup(kernel_cls, spec)
        if cached is not None:
            return cached

        if _autotune_disabled():
            return self._fallback_default(kernel_cls, spec)

        effective_depth = depth or current_search_depth() or self.search_depth
        config = self._search(kernel_cls, spec, depth=effective_depth)
        if config is None:
            config = self._fallback_default(kernel_cls, spec)
        self.store(kernel_cls, spec, config)
        return config

    def store(self, kernel_cls, spec, config) -> None:
        """Explicit write: hot tier + disk."""
        key = self._make_key(kernel_cls, spec)
        self._hot[key] = config
        self._save_to_disk(key, kernel_cls, spec, config, runtime_us=None)

    def clear_hot(self) -> None:
        self._hot.clear()

    # ---- key construction ---------------------------------------------------

    def _make_key(self, kernel_cls, spec) -> tuple:
        return (
            kernel_cls.__qualname__,
            _spec_fingerprint(spec),
            self.device.fingerprint(),
            _source_hash(kernel_cls),
        )

    def _key_to_filename(self, key: tuple) -> str:
        kernel_name = key[0]
        digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:24]
        return f"{kernel_name}__{digest}.json"

    # ---- disk I/O -----------------------------------------------------------

    def _load_from_disk(self, key: tuple) -> Optional[Any]:
        path = self.cache_dir / self._key_to_filename(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("source_hash") != key[3]:
            return None
        if payload.get("device") != self.device.fingerprint():
            return None
        config_payload = payload.get("config")
        if not isinstance(config_payload, dict):
            return None
        config_cls = _resolve_config_cls(payload.get("kernel"))
        if config_cls is None:
            return None
        try:
            return _config_from_json_dict(config_cls, config_payload)
        except TypeError:
            return None

    def _save_to_disk(
        self,
        key: tuple,
        kernel_cls,
        spec,
        config,
        *,
        runtime_us: Optional[float],
    ) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

        record = {
            "kernel": kernel_cls.__qualname__,
            "spec": _spec_to_json_dict(spec),
            "device": self.device.fingerprint(),
            "source_hash": key[3],
            "config": _config_to_json_dict(config),
            "runtime_us": runtime_us,
            "timestamp": int(time.time()),
        }
        path = self.cache_dir / self._key_to_filename(key)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(self.cache_dir),
                prefix=".popcorn_cache_",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(record, tmp)
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        except OSError:
            return

    # ---- bundled defaults ---------------------------------------------------

    def _load_bundled_default(self, kernel_cls, spec) -> Optional[Any]:
        if not self.bundled_dir.exists():
            return None
        kernel_name = getattr(kernel_cls, "NAME", None) or kernel_cls.__name__.lower()
        spec_label = _format_spec_label(spec)
        candidates = [
            self.bundled_dir / f"{kernel_name}_{spec_label}.json",
            self.bundled_dir / f"{kernel_cls.__qualname__}_{spec_label}.json",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            config_payload = payload.get("config")
            if not isinstance(config_payload, dict):
                continue
            config_cls = getattr(kernel_cls, "CONFIG_CLS", None)
            if config_cls is None:
                continue
            try:
                return _config_from_json_dict(config_cls, config_payload)
            except TypeError:
                continue
        return None

    # ---- warm seeding -------------------------------------------------------

    def _load_warm_seeds(self, kernel_cls, spec) -> list:
        """Return configs from disk/bundled cache that are valid for ``spec``.

        Scans all JSON files for this (kernel, device) pair regardless of
        spec. Configs that pass ``is_valid_for`` become warm seeds — they
        are prepended to the initial search population so the first
        generation evaluates proven-good configs first.
        """
        seeds: list = []
        seen: set = set()

        for d in (self.cache_dir, self.bundled_dir):
            if not d.exists():
                continue
            for path in d.glob("*.json"):
                try:
                    payload = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if payload.get("kernel") not in (
                    kernel_cls.__qualname__,
                    getattr(kernel_cls, "NAME", None),
                ):
                    continue
                if payload.get("device") != self.device.fingerprint():
                    continue
                cfg_dict = payload.get("config")
                if not isinstance(cfg_dict, dict):
                    continue
                config_cls = getattr(kernel_cls, "CONFIG_CLS", None)
                if config_cls is None:
                    continue
                try:
                    cfg = _config_from_json_dict(config_cls, cfg_dict)
                except (TypeError, KeyError):
                    continue
                k = tuple(sorted(cfg_dict.items()))
                if k in seen:
                    continue
                seen.add(k)
                try:
                    if kernel_cls(spec, cfg).is_valid_for(self.device.caps):
                        seeds.append(cfg)
                except Exception:
                    continue

        return seeds

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

        Why this shape:
          - **Random sampling** beats taking the first N from the
            cartesian product, which biases toward one corner of the
            search space (small NCW, small MTiles, etc.).
          - **Generations + early stop** spend the budget where it
            matters: if gen 0 already found something fast, we don't
            need to test the remaining 12.
          - **Parallel compile, serial launch**: compile is CPU-bound
            (PTX assembly + cuModuleLoadData) and parallelizes
            cleanly. Kernel launches are μs-scale and parallelizing
            them poisons their timing measurements (and on CUDA has
            been observed to corrupt subsequent main-thread launches
            via worker-thread context state).
          - A config that fails correctness is dropped for THIS search;
            if it's a real bug the fix belongs in the kernel or
            ``is_valid``, not a retry/sampling band-aid.
        """
        import dataclasses as _dc
        import math as _math

        verbose = os.environ.get("POPCORN_AUTOTUNE_VERBOSE", "").lower() in ("1", "true", "yes")

        def _log(msg: str) -> None:
            if verbose:
                print(f"[autotune:{kernel_cls.NAME}] {msg}", flush=True)

        all_valid = list(self._enumerate_valid(kernel_cls, spec))
        if not all_valid:
            return None

        # Determine knob_names (for _parallel_compile and dedup keys).
        knob_names: list[str] = []
        try:
            tune_space_fn = getattr(kernel_cls, "tune_space_resolved", None)
            if tune_space_fn is not None:
                tune_space = tune_space_fn(spec, self.device)
            else:
                tune_space = kernel_cls.tune_space()
            knob_names = list(tune_space.keys())
        except Exception:
            # Fall back to the dataclass field names in declaration order.
            if dataclasses.is_dataclass(all_valid[0]):
                knob_names = [f.name for f in dataclasses.fields(all_valid[0])]

        lnch = getattr(self, "_launcher", None)
        timer = getattr(self, "_compile_and_time", None)

        # ── Shared tensors (built once for the whole search) ──────────────
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
                _log(f"reference setup failed ({type(e).__name__}: {e})")
                tensors = None

        from popcorn.backend import IS_METAL, PT
        from popcorn.correctness import check_correctness

        # ``SPEC_CLS`` is only required on the per-config correctness
        # path; toy kernels in unit tests don't define it but also
        # never reach ``_check_config`` (they hit the no-launcher
        # legacy branch below). Resolve lazily.
        spec_cls = getattr(kernel_cls, "SPEC_CLS", None)

        def _check_config(cfg) -> tuple[bool, float, str | None]:
            assert spec_cls is not None, (
                f"{kernel_cls.__name__}.SPEC_CLS is required for fast-search correctness"
            )
            assert pspec is not None
            assert lnch is not None
            try:
                cfg_kernel = kernel_cls(spec_cls(**spec_dict), cfg)
                launch_tensors = cfg_kernel.prepare_launch_tensors(tensors)
                cfg_buffers = [launch_tensors[b.name] for b in pspec.buffers]
                assert out_idx is not None
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
                # config, but doing so broke the cast+cpu workaround
                # in call_with_bindings — the previously-stable
                # ground-truth fix stopped working. Restore D2H here
                # to keep that ground truth stable while we hunt the
                # root cause.
                cr = check_correctness(
                    actual_output,
                    reference,
                    out_dtype=out_dtype,
                    threshold=kernel_cls.correctness_threshold(out_dtype),
                )
                return cr.passed, cr.cos_sim, None
            except Exception as e:
                return False, float("nan"), f"{type(e).__name__}: {e}"

        def _time_config(cfg) -> float:
            assert timer is not None
            assert tensors is not None
            return timer(kernel_cls, spec, cfg, tensors=tensors)

        # ── Build the candidate pool: warm seeds first, then a random
        # shuffle of the remaining valid configs. We sample WITHOUT
        # replacement (each config tested at most once across all gens).
        seed_keys = {_cfg_key_simple(c) for c in seeds}
        rest = [c for c in all_valid if _cfg_key_simple(c) not in seed_keys]
        rng = random.Random(self.rng_seed)
        rng.shuffle(rest)
        pool: list = list(seeds) + rest

        # ── Generation schedule: 8 configs per gen × 4 gens max,
        # early-stop after one stale gen. See _FAST_* module constants
        # for the rationale and tuning.
        n_gens = _FAST_N_GENS
        n_per_gen = _FAST_POP_PER_GEN
        early_stop_after_stale_gens = _FAST_EARLY_STOP_STALE_GENS

        # Legacy / unit-test path: launcher absent or reference setup
        # failed. Without a real launcher we can't run the per-config
        # correctness check, so fall through to the simple
        # enumerate-and-time loop (or prune-score sort if no timer).
        if lnch is None or tensors is None:
            if timer is None:
                # Sort by prune_score; deterministic tiebreak.
                def _score(cfg):
                    try:
                        return _prune_score(kernel_cls(spec, cfg))
                    except Exception:
                        return 0.0

                return min(all_valid, key=_score)
            best_cfg_legacy: Any = None
            best_t_legacy = float("inf")
            for cfg in all_valid:
                try:
                    t = timer(kernel_cls, spec, cfg)
                except Exception:
                    continue
                if t < best_t_legacy:
                    best_t_legacy = t
                    best_cfg_legacy = cfg
            return best_cfg_legacy

        best_cfg: Any = None
        best_us = float("inf")
        prev_best_us = float("inf")  # best_us at the START of this gen
        stale_gens = 0
        pool_idx = 0

        for gen in range(n_gens):
            batch: list = []
            while pool_idx < len(pool) and len(batch) < n_per_gen:
                batch.append(pool[pool_idx])
                pool_idx += 1
            if not batch:
                break
            prev_best_us = best_us

            # Parallel compile the whole batch up front (CUDA only —
            # MLX's compile path isn't thread-safe; on Metal we let
            # _check_config compile each cfg serially via the launcher
            # cache).
            from popcorn.backend import IS_METAL as _IS_METAL

            compile_cache: dict = {}
            if knob_names and not _IS_METAL:
                try:
                    compile_cache = _parallel_compile(
                        kernel_cls, spec, batch, lnch, knob_names=knob_names
                    )
                except Exception as e:
                    _log(
                        f"gen{gen}: parallel compile failed ({type(e).__name__}: {e}); "
                        f"falling back to serial"
                    )

            gen_best_us = float("inf")
            for cfg in batch:
                # If parallel compile flagged a compile error, skip early.
                if knob_names and compile_cache:
                    key = tuple(getattr(cfg, k) for k in knob_names)
                    cached = compile_cache.get(key)
                    if cached is not None and cached[0] is None:
                        _log(f"gen{gen} [cerr] {cfg}: {cached[1]}")
                        continue

                ok, cos, err = _check_config(cfg)
                if err is not None:
                    _log(f"gen{gen} [err]  {cfg}: {err}")
                    continue
                if not ok:
                    cos_s = "nan" if _math.isnan(cos) else f"{cos:.4f}"
                    _log(f"gen{gen} [fail] cos={cos_s}  {cfg}")
                    continue
                try:
                    us = _time_config(cfg)
                except Exception as e:
                    _log(f"gen{gen} [terr] {cfg}: {type(e).__name__}: {e}")
                    continue
                tag = " ***" if us < best_us else ""
                _log(f"gen{gen} [ok]   {us:8.2f} μs  cos={cos:.4f}  {cfg}{tag}")
                if us < best_us:
                    best_us = us
                    best_cfg = cfg
                if us < gen_best_us:
                    gen_best_us = us

            # Early-stop: did THIS gen improve on the prior best?
            # gen_best_us == best_us in gen 0 since best_us starts at
            # inf, so we compare against ``prev_best_us`` snapshotted
            # at gen entry — that way gen 0 always counts as
            # improvement and we don't bail before doing real work.
            improved = best_cfg is not None and best_us < prev_best_us * 0.999
            if improved:
                stale_gens = 0
                _log(
                    f"gen{gen}: best improved {prev_best_us:.2f} → {best_us:.2f} μs "
                    f"(gen this round: {gen_best_us:.2f} μs)"
                )
            elif best_cfg is not None:
                stale_gens += 1
                _log(
                    f"gen{gen}: best stayed at {best_us:.2f} μs "
                    f"(gen this round: {gen_best_us:.2f} μs); stale={stale_gens}"
                )
                if stale_gens >= early_stop_after_stale_gens:
                    _log(f"early-stop after gen {gen}")
                    break

        if best_cfg is None:
            _log("no config passed correctness across all generations")
        return best_cfg

    # ---- full (genetic) search ----------------------------------------------

    def _search_full(self, kernel_cls, spec, seeds: list) -> Optional[Any]:
        """Full genetic search. Falls back to fast search if the launcher
        or make_tensors/reference are unavailable."""
        lnch = getattr(self, "_launcher", None)
        if lnch is None:
            return self._search_fast(kernel_cls, spec, seeds)

        import dataclasses as _dc

        spec_dict = _dc.asdict(spec) if _dc.is_dataclass(spec) else {}

        try:
            tensors = kernel_cls.make_tensors(spec_dict)
        except Exception:
            return self._search_fast(kernel_cls, spec, seeds)

        try:
            default_kernel = kernel_cls.from_problem(spec_dict)
            pspec = default_kernel.param_spec()
            buffers = [tensors[b.name] for b in pspec.buffers]
            out_idx = kernel_cls.OUTPUT_IDX
            if out_idx < 0:
                out_idx = len(buffers) + out_idx
            inputs = [buffers[i] for i in range(len(buffers)) if i != out_idx]
            reference = default_kernel.reference(*inputs)
        except Exception:
            return self._search_fast(kernel_cls, spec, seeds)

        try:
            tune_space = kernel_cls.tune_space_resolved(spec, lnch.device)
        except Exception:
            try:
                tune_space = kernel_cls.tune_space()
            except Exception:
                return self._search_fast(kernel_cls, spec, seeds)

        if not tune_space:
            return self._search_fast(kernel_cls, spec, seeds)

        result = genetic_search(
            kernel_cls,
            spec,
            tune_space,
            lnch,
            tensors=tensors,
            reference=reference,
            out_idx=out_idx,
            seeds=seeds,
            pop_size=self.pop_size,
            n_gens=self.n_gens,
            early_stop_gens=self.early_stop_gens,
            mutate_prob=self.mutate_prob,
            rng_seed=self.rng_seed,
            bench_ms=self.bench_ms,
            max_workers=self.max_workers,
            verbose=False,
        )
        return result[0] if result is not None else None

    # ---- candidate enumeration ----------------------------------------------

    def _enumerate_valid(self, kernel_cls, spec) -> Iterable[Any]:
        """Cartesian product of tune_space_resolved, filtered to valid configs."""
        # Prefer the device-resolved space (includes MMA shape knobs).
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

    # Keep the old name as an alias so existing callers (tests) don't break.
    def _enumerate_candidates(self, kernel_cls, spec) -> Iterable[Any]:
        return self._enumerate_valid(kernel_cls, spec)

    # ---- fallback -----------------------------------------------------------

    def _fallback_default(self, kernel_cls, spec) -> Any:
        picker = getattr(kernel_cls, "_pick_default_cfg", None)
        if callable(picker):
            try:
                cfg = picker(spec)
                if cfg is not None:
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
            return config_cls()
        except TypeError as e:
            raise RuntimeError(
                f"AutotuneCache: cannot construct default {config_cls.__name__}()"
            ) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg_key_simple(cfg) -> tuple:
    """Stable identity key for a config — used for seed dedup."""
    if dataclasses.is_dataclass(cfg):
        return tuple(sorted(dataclasses.asdict(cfg).items()))
    return (id(type(cfg)), repr(cfg))


def _format_spec_label(spec: Any) -> str:
    if not dataclasses.is_dataclass(spec):
        return hashlib.sha256(repr(spec).encode("utf-8")).hexdigest()[:16]
    fields = sorted(dataclasses.fields(spec), key=lambda f: f.name)
    parts = []
    for f in fields:
        v = getattr(spec, f.name)
        parts.append(f"{f.name}{v}")
    return "_".join(parts)


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


def _resolve_config_cls(kernel_qualname: Optional[str]) -> Optional[type]:
    if not kernel_qualname:
        return None
    try:
        from popcorn.kernels.registry import all_kernels
    except ImportError:
        return None
    short = kernel_qualname.rsplit(".", 1)[-1]
    for cls in all_kernels():
        if cls.__qualname__ == kernel_qualname or cls.__name__ == short:
            return getattr(cls, "CONFIG_CLS", None)
    return None
