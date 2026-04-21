#!/usr/bin/env python
"""Registry-driven kernel autotuner — CLI wrapper over ``AutotuneCache``.

For each (kernel, problem), runs a full genetic search (warm-seeded from
existing on-disk configs) and saves the winner to ``configs/``.

Usage:
  python tools/autotune.py                        # all kernels
  python tools/autotune.py --kernel gemm           # one kernel
  python tools/autotune.py --tag owl               # tagged problems
  python tools/autotune.py --kernel gemm --problem owl_720p
  python tools/autotune.py --pop 64 --gens 16      # override search params
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────


def _configs_dir() -> Path:
    env = os.environ.get("POPCORN_CONFIGS_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[1] / "configs"
    if repo.exists():
        return repo
    return Path("configs")


def _config_to_dict(cfg) -> dict:
    if is_dataclass(cfg):
        return asdict(cfg)
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


# ── PTX-dump compile-error hook ──────────────────────────────────────


def _make_ptx_error_hook():
    """Return a closure that deduplicates PTX dumps across compile errors.

    Attached to ``AutotuneCache._compile_error_hook`` in ``autotune_one``.
    """
    dumped: dict[tuple[str, str], str] = {}

    def _hook(kernel_cls, spec, cfg, exc, launcher):
        from popcorn.utils.ptx_dump import classify_compile_error, write_ptx_for

        if classify_compile_error(exc) != "ptx":
            return None
        err_name = type(exc).__name__
        err_msg = str(exc)
        dedup_key = (err_name, err_msg[:80])
        path_s = dumped.get(dedup_key)
        if path_s is None:
            path = write_ptx_for(kernel_cls, spec, cfg, launcher.device.caps)
            path_s = str(path) if path else "(ptx dump failed)"
            dumped[dedup_key] = path_s
            print(f"    compile failed ({err_name}): {err_msg[:160]}")
            print(f"      ptx dumped to: {path_s}")
        return f"(ptx: {path_s})"

    return _hook


# ── Main autotune loop ───────────────────────────────────────────────


def autotune_one(
    kernel_cls,
    problem,
    launcher,
    *,
    bench_ms: float,
    save: bool,
    pop_size: int | None,
    n_gens: int,
    early_stop_gens: int,
    mutate_prob: float,
    seed: int,
    max_workers: int | None,
) -> None:
    from popcorn.utils.pretty import print_header, style

    name = kernel_cls.NAME
    print_header(f"{name} / {problem.name}")

    spec_cls = kernel_cls.SPEC_CLS
    from popcorn.device import current_device

    _device = current_device()
    _problem_spec = None
    if spec_cls is not None:
        try:
            _problem_spec = spec_cls(**problem.params)
        except Exception:
            _problem_spec = None
    tune_space = kernel_cls.tune_space_resolved(_problem_spec, _device)

    if not tune_space:
        print("  (no tune_space, skipping)")
        return

    # Apply problem.config_overrides: each pinned knob collapses the
    # search space for that dimension to the single override value.
    overrides = getattr(problem, "config_overrides", None)
    if overrides:
        pinned = []
        for knob, value in overrides.items():
            if knob in tune_space:
                tune_space = {**tune_space, knob: [value]}
                pinned.append(f"{knob}={value}")
        if pinned:
            print(f"  [problem config_overrides: {', '.join(pinned)}]")

    default_kernel = kernel_cls.from_problem(problem.params)
    spec = spec_cls(**problem.params) if spec_cls is not None else None
    print(f"  tune_space: {tune_space}")

    # Temporarily route the shared cache through tools/autotune's tuning
    # overrides (pop size, gens, PTX-dump hook), run the full search,
    # then restore the defaults so subsequent calls to the same cache
    # aren't affected.
    cache = launcher._autotune
    prev = (
        cache.pop_size,
        cache.n_gens,
        cache.early_stop_gens,
        cache.mutate_prob,
        cache.rng_seed,
        cache.bench_ms,
        cache.max_workers,
        cache._compile_error_hook,
    )
    cache.pop_size = pop_size
    cache.n_gens = n_gens
    cache.early_stop_gens = early_stop_gens
    cache.mutate_prob = mutate_prob
    cache.rng_seed = seed
    cache.bench_ms = bench_ms
    cache.max_workers = max_workers
    cache._compile_error_hook = _make_ptx_error_hook()

    os.environ["POPCORN_AUTOTUNE_VERBOSE"] = "1"
    try:
        seeds = cache._load_warm_seeds(kernel_cls, spec)
        result = cache._search_full_result(kernel_cls, spec, seeds, tune_space=tune_space)
    finally:
        (
            cache.pop_size,
            cache.n_gens,
            cache.early_stop_gens,
            cache.mutate_prob,
            cache.rng_seed,
            cache.bench_ms,
            cache.max_workers,
            cache._compile_error_hook,
        ) = prev

    if result is None:
        print("  NO valid+correct configs found")
        return

    best_cfg, best_us = result
    print(f"\n  {style('WINNER', 'bold_green')}: {best_us:.2f} μs — {best_cfg}")
    try:
        tflops = default_kernel.flops() / (best_us * 1e-6) / 1e12
        print(f"  TFLOPS: {tflops:.2f}")
    except Exception:
        pass

    if save:
        cdir = _configs_dir()
        cdir.mkdir(parents=True, exist_ok=True)
        fname = cdir / f"{name}_{problem.name}.json"
        payload = {
            "kernel": name,
            "problem": problem.params,
            "problem_name": problem.name,
            "config": _config_to_dict(best_cfg),
            "runtime_us": best_us,
        }
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  saved → {fname}")


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kernel", help="restrict to one kernel name")
    parser.add_argument("--problem", help="restrict to one problem name")
    parser.add_argument("--tag", help="only tune problems with this tag")
    parser.add_argument("--exclude-tag", help="skip problems with this tag")
    parser.add_argument("--bench-ms", type=float, default=50.0)
    parser.add_argument(
        "--pop",
        type=int,
        default=None,
        dest="pop_size",
        help="population size (default: 16*log2(search_space))",
    )
    parser.add_argument("--gens", type=int, default=32, help="max generations (default: 32)")
    parser.add_argument(
        "--early-stop",
        type=int,
        default=2,
        dest="early_stop_gens",
        help="stop after N gens with <1%% improvement (default: 2)",
    )
    parser.add_argument("--mutate", type=float, default=0.3, help="mutation probability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None, help="compile threads")
    parser.add_argument("--no-save", action="store_true", help="skip saving configs")
    args = parser.parse_args(argv)

    from popcorn.device import current_device
    from popcorn.kernels import all_kernels, get
    from popcorn.launcher import Launcher
    from popcorn.utils.pretty import print_header, print_kv

    kernels = [get(args.kernel)] if args.kernel else all_kernels()
    launcher = Launcher(device=current_device())

    print_header(f"Autotune — {launcher.device.caps.name}")
    print_kv("Configs", str(_configs_dir()))

    for cls in kernels:
        problems = cls.problems()
        if args.problem:
            problems = [p for p in problems if p.name == args.problem]
        if args.tag:
            problems = [p for p in problems if args.tag in p.tags]
        if args.exclude_tag:
            problems = [p for p in problems if args.exclude_tag not in p.tags]

        for problem in problems:
            autotune_one(
                cls,
                problem,
                launcher,
                bench_ms=args.bench_ms,
                save=not args.no_save,
                pop_size=args.pop_size,
                n_gens=args.gens,
                early_stop_gens=args.early_stop_gens,
                mutate_prob=args.mutate,
                seed=args.seed,
                max_workers=args.workers,
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
