"""Autotune plumbing: depth contextvar, verbose helpers, GA primitives.

The three-level ``AutotuneCache`` class itself lives in
``quark.autotune.cache``; disk I/O + JSON helpers live in
``quark.autotune.io``; the full genetic search body lives in
``quark.autotune.full_search``. This package's ``__init__`` just
hosts:

  * ``_SEARCH_DEPTH`` / ``current_search_depth()`` — fast vs full toggle.
  * ``_verbose()`` / ``_log()`` — single source of truth for diagnostic
    output gated on ``QUARK_AUTOTUNE_VERBOSE``.
  * ``_crossover`` / ``_mutate`` / ``_parallel_compile`` — genetic
    primitives reused by the cache's full search.

Search depth priority (highest first):

  1. ``QUARK_MAX_AUTOTUNE=1`` env var (forces ``"full"`` process-wide).
  2. ``with quark.max_autotune():`` context manager.
  3. Otherwise ``"fast"``.
"""

from __future__ import annotations

import contextvars
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# Re-exported so existing callers can keep doing
# ``from quark.autotune import AutotuneCache``. Import at the bottom to
# avoid a circular import with ``.cache`` (which imports this
# module's helpers).

# ---------------------------------------------------------------------------
# Search-depth control
# ---------------------------------------------------------------------------

_SEARCH_DEPTH: contextvars.ContextVar[str] = contextvars.ContextVar(
    "quark_search_depth", default="fast"
)


def current_search_depth() -> str:
    """Return the active search depth: ``"full"`` if
    ``QUARK_MAX_AUTOTUNE=1`` is set, else the ContextVar value."""
    val = os.environ.get("QUARK_MAX_AUTOTUNE", "")
    if val.lower() in ("1", "true", "yes", "on"):
        return "full"
    return _SEARCH_DEPTH.get()


# ---------------------------------------------------------------------------
# Diagnostic output
# ---------------------------------------------------------------------------

_VERBOSE_ENV = "QUARK_AUTOTUNE_VERBOSE"


def _verbose() -> bool:
    """Whether diagnostic autotune prints are enabled."""
    return os.environ.get(_VERBOSE_ENV, "").lower() in ("1", "true", "yes")


def _log(kernel_name: str, msg: str) -> None:
    """Emit an ``[autotune:<kernel>]`` line when verbose is on."""
    if _verbose():
        print(f"[autotune:{kernel_name}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Fast-search generation schedule (consumed by AutotuneCache._search_fast)
# ---------------------------------------------------------------------------

# Fast search: up to 8 generations × 16 configs = 128 max; early-stop
# after a stale generation keeps typical cold-starts sub-second when
# winners are obvious while still exploring enough to escape the
# cartesian-order bias on diverse tune spaces.
_FAST_POP_PER_GEN = 16
_FAST_N_GENS = 8
_FAST_EARLY_STOP_STALE_GENS = 2


# ---------------------------------------------------------------------------
# Genetic algorithm primitives
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


# Re-export at the bottom to avoid import cycles — ``.cache`` imports
# the private helpers above.
from quark.autotune.cache import AutotuneCache  # noqa: E402

__all__ = [
    "_SEARCH_DEPTH",
    "AutotuneCache",
    "_crossover",
    "_log",
    "_mutate",
    "_parallel_compile",
    "_verbose",
    "current_search_depth",
]
