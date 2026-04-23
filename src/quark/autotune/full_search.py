"""Full genetic search — the body of ``AutotuneCache._search_full_result``.

EXEMPT FROM 500-LINE RULE: the config enumeration, population breeding,
per-config correctness+timing (``_evaluate_config``), and the outer
generational loop (``run_full_search``) are all tightly coupled around
the same compile-cache / reference / tensors state. Splitting would
either duplicate that state threading across modules or force a
circular import between the search driver and its per-config
evaluator.

Fast search on the launcher-absent fallback path is delegated back to
``cache._search_fast`` so we don't duplicate that logic here.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
from typing import Any, Optional

from quark.autotune import _crossover, _log, _mutate, _parallel_compile


def _enumerate_full_valid_configs(
    kernel_cls,
    spec_cls,
    config_cls,
    spec_dict: dict,
    caps,
    knob_names: list[str],
    knob_values: list[list],
    fixed: dict,
    *,
    sample_budget: int,
    rng_seed: int,
    say,
) -> list:
    """Cartesian product of ``knob_values`` with ``fixed``, sampled to
    ``sample_budget`` when the space is too big to enumerate fully.
    """
    from itertools import product as _product

    total_combos = 1
    for vals in knob_values:
        total_combos *= max(len(vals), 1)

    rng_enum = random.Random(rng_seed)

    def _build(combo):
        kw = dict(fixed)
        kw.update(dict(zip(knob_names, combo, strict=False)))
        try:
            cfg = config_cls(**kw)
            k = kernel_cls(spec_cls(**spec_dict), cfg)
            if k.is_valid_for(caps):
                return cfg
        except (TypeError, ValueError):
            pass
        return None

    valid: list = []
    if total_combos <= sample_budget:
        for combo in _product(*knob_values):
            cfg = _build(combo)
            if cfg is not None:
                valid.append(cfg)
    else:
        seen: set[tuple] = set()
        attempts = 0
        attempt_cap = sample_budget * 4
        while len(valid) < sample_budget and attempts < attempt_cap:
            combo = tuple(rng_enum.choice(v) for v in knob_values)
            attempts += 1
            if combo in seen:
                continue
            seen.add(combo)
            cfg = _build(combo)
            if cfg is not None:
                valid.append(cfg)
        say(f"sampled {len(valid)} valid / {attempts} attempts from {total_combos:,} combos")

    return valid


def _build_initial_population(
    valid_configs: list,
    seeds: list,
    knob_names: list[str],
    pop_size: int,
    rng: random.Random,
) -> list:
    """Seed the population: warm seeds first, random fill after."""
    seed_keys = {tuple(getattr(c, k) for k in knob_names) for c in seeds}
    non_seeds = [
        c for c in valid_configs if tuple(getattr(c, k) for k in knob_names) not in seed_keys
    ]
    seed_slots = min(len(seeds), pop_size)
    rand_slots = pop_size - seed_slots
    population = list(seeds[:seed_slots]) + rng.sample(non_seeds, min(rand_slots, len(non_seeds)))
    while len(population) < min(pop_size, len(valid_configs)):
        population.append(rng.choice(valid_configs))
    return population


def _breed_next_generation(
    elites: list,
    valid_configs: list,
    valid_set: set,
    valid_by_key: dict,
    knob_names: list[str],
    knob_values: list[list],
    fixed: dict,
    config_cls,
    pop_size: int,
    mutate_prob: float,
    rng: random.Random,
) -> list:
    """Crossover+mutate elites to produce ``pop_size`` new configs,
    rejecting any that aren't in the valid set. Pads with random valid
    configs if breeding exhausts its budget."""
    next_pop = list(elites)
    attempts = 0
    while len(next_pop) < pop_size and attempts < pop_size * 20:
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

    while len(next_pop) < pop_size:
        next_pop.append(rng.choice(valid_configs))
    return next_pop


def _evaluate_config(
    cfg,
    kernel_cls,
    spec,
    spec_cls,
    spec_dict: dict,
    compiled,
    tensors,
    pspec,
    out_idx: int,
    reference,
    bench_ms: float,
    say,
) -> tuple[float, str] | None:
    """Correctness-check then time one already-compiled config.

    Returns ``(us, None)`` on success, ``(inf, reason)`` on failure, or
    ``None`` if the kernel's ``prepare_launch_tensors`` raised (caller
    logs and skips). Side effect: logs outcome via ``say``.
    """
    import sys as _sys

    from quark.correctness import check_correctness
    from quark.launcher.launcher import _time_callable
    from quark.runtime.sync import ir_dtype_of, synchronize, zero_buffer

    _is_metal = _sys.platform == "darwin"

    # Drain any async error queued by a prior config. Without the sync
    # a prior launch's misaligned / OOB access surfaces on the NEXT
    # config's ``zero_buffer`` (or the first device call after the
    # faulting kernel) and gets misattributed. Surface here, move on.
    try:
        synchronize()
    except BaseException as _e:
        say(f"  [prior cfg error drained]: {type(_e).__name__}: {_e}")

    # Alt-impl configs (e.g. cuBLAS) can't be compiled through the PTX
    # lowering pipeline — the ``compiled`` kernel threaded in here is
    # either a PTX fallback with default knobs or None. Delegate
    # correctness to the kernel's hook and do timing via its own
    # dispatch under the same _time_callable budget so the comparison
    # vs PTX configs is apples-to-apples.
    impl = getattr(cfg, "impl", "ptx")
    if impl != "ptx":
        check_alt = getattr(kernel_cls, "check_alt_config", None)
        if not callable(check_alt):
            say(f"  [alt err] {cfg}: no check_alt_config hook for impl={impl!r}")
            return float("inf"), "launch"
        try:
            passed, cos_sim, err = check_alt(spec, cfg, tensors=tensors, reference=reference)
        except BaseException as _e:
            say(f"  [alt check err] {cfg}: {type(_e).__name__}: {_e}")
            return float("inf"), "launch"
        if err is not None:
            say(f"  [alt launch err] {cfg}: {err}")
            return float("inf"), "launch"
        if not passed:
            say(f"  [wrong] {cfg}: cos={cos_sim:.4f}")
            return float("inf"), "wrong"

        # Time the alt dispatch under the same bench budget as PTX.
        from quark.kernels.gemm.cublas_dispatch import dispatch_cublas

        A = tensors["A"]
        B = tensors["B"]
        Out = tensors["Out"]
        Bias = tensors.get("Bias") if spec.has_bias else None
        try:
            us = _time_callable(
                lambda: dispatch_cublas(A=A, B=B, Out=Out, Bias=Bias),
                warmup_ms=10.0,
                bench_ms=bench_ms,
            )
        except BaseException as _e:
            say(f"  [alt bench err] {cfg}: {type(_e).__name__}: {_e}")
            return float("inf"), "time"
        say(f"  {us:8.2f} μs  cos={cos_sim:.6f}  {cfg}")
        return us, ""

    cfg_kernel = kernel_cls(spec_cls(**spec_dict), cfg)
    try:
        launch_tensors = cfg_kernel.prepare_launch_tensors(tensors)
    except Exception as _e:
        say(f"  [prep err] {cfg}: {_e}")
        return float("inf"), "prep"

    cfg_buffers = [launch_tensors[b.name] for b in pspec.buffers]
    try:
        cfg_output_buf = zero_buffer(cfg_buffers[out_idx])
        cfg_buffers[out_idx] = cfg_output_buf
        _launch_result = compiled.launch(buffers=cfg_buffers)
        synchronize()
    except BaseException as _e:
        say(f"  [launch err] {cfg}: {type(_e).__name__}: {_e}")
        return float("inf"), "launch"

    if _is_metal and _launch_result:
        out_pos = sum(1 for b in pspec.buffers[:out_idx] if not b.readonly)
        actual_output = _launch_result[out_pos]
    else:
        actual_output = cfg_output_buf

    out_dtype = ir_dtype_of(actual_output)
    cr = check_correctness(
        actual_output,
        reference,
        out_dtype=out_dtype,
        threshold=kernel_cls.correctness_threshold(out_dtype),
    )
    if not cr.passed:
        say(f"  [wrong] {cfg}: cos={cr.cos_sim:.4f}")
        return float("inf"), "wrong"

    try:
        cfg_output_buf = zero_buffer(cfg_buffers[out_idx])
        cfg_buffers[out_idx] = cfg_output_buf
        us = _time_callable(
            lambda _c=compiled, _b=cfg_buffers: _c.launch(buffers=_b),
            warmup_ms=10.0,
            bench_ms=bench_ms,
        )
    except BaseException as _e:
        say(f"  [bench err] {cfg}: {type(_e).__name__}: {_e}")
        return float("inf"), "time"

    say(f"  {us:8.2f} μs  cos={cr.cos_sim:.6f}  {cfg}")
    return us, ""


def run_full_search(
    cache,
    kernel_cls,
    spec,
    seeds: list,
    *,
    tune_space: dict | None = None,
) -> Optional[tuple]:
    """Full genetic search. Returns ``(best_cfg, best_us)`` or ``None``.

    Falls back to the cache's fast search when the launcher or
    make_tensors/reference are unavailable. ``tune_space`` overrides the
    kernel's default space — used by ``tools/autotune.py`` to pin
    problem-specific config overrides.
    """
    name = getattr(kernel_cls, "NAME", kernel_cls.__name__)
    lnch = getattr(cache, "_launcher", None)
    if lnch is None:
        _log(name, "full search: no launcher, falling back to fast")
        fast_cfg = cache._search_fast(kernel_cls, spec, seeds)
        return (fast_cfg, float("inf")) if fast_cfg is not None else None

    from quark.refs import ref_cache
    from quark.runtime.device_tensors import numpy_to_device_dict

    spec_dict = dataclasses.asdict(spec) if dataclasses.is_dataclass(spec) else {}

    try:
        inputs_np, outputs_np = ref_cache().get(kernel_cls, spec_dict)
    except Exception as e:
        _log(name, f"full search: ref_cache.get failed ({e}), falling back to fast")
        fast_cfg = cache._search_fast(kernel_cls, spec, seeds)
        return (fast_cfg, float("inf")) if fast_cfg is not None else None

    try:
        default_kernel = kernel_cls.from_problem(spec_dict)
        pspec = default_kernel.param_spec()
        tensors = numpy_to_device_dict(kernel_cls, default_kernel.spec, inputs_np)
        out_idx = kernel_cls.OUTPUT_IDX
        if out_idx < 0:
            out_idx = len(pspec.buffers) + out_idx
        output_name = pspec.buffers[out_idx].name
        reference = outputs_np[output_name]
    except Exception as e:
        _log(name, f"full search: ref setup failed ({e}), falling back to fast")
        fast_cfg = cache._search_fast(kernel_cls, spec, seeds)
        return (fast_cfg, float("inf")) if fast_cfg is not None else None

    if tune_space is None:
        try:
            tune_space = kernel_cls.tune_space_resolved(spec, lnch.device)
        except Exception:
            try:
                tune_space = kernel_cls.tune_space()
            except Exception:
                fast_cfg = cache._search_fast(kernel_cls, spec, seeds)
                return (fast_cfg, float("inf")) if fast_cfg is not None else None

    if not tune_space:
        fast_cfg = cache._search_fast(kernel_cls, spec, seeds)
        return (fast_cfg, float("inf")) if fast_cfg is not None else None

    spec_cls = kernel_cls.SPEC_CLS
    config_cls = kernel_cls.CONFIG_CLS
    if config_cls is None:
        return None

    knob_names = list(tune_space.keys())
    knob_values = [tune_space[k] for k in knob_names]

    all_cfg_fields = (
        {f.name for f in dataclasses.fields(default_kernel.config)}
        if dataclasses.is_dataclass(default_kernel.config)
        else set()
    )
    fixed = {k: getattr(default_kernel.config, k) for k in all_cfg_fields if k not in tune_space}

    def _say(msg: str) -> None:
        _log(name, msg)

    from quark.autotune.cache import _spec_summary

    _say(f"spec={_spec_summary(spec)}")

    valid_configs = _enumerate_full_valid_configs(
        kernel_cls,
        spec_cls,
        config_cls,
        spec_dict,
        lnch.device.caps,
        knob_names,
        knob_values,
        fixed,
        sample_budget=max(1024, (cache.pop_size or 64) * 8),
        rng_seed=cache.rng_seed,
        say=_say,
    )

    # Alternate-implementation configs (e.g. cuBLAS for GEMM). Mirror
    # the hook in cache._enumerate_valid so full search sees the same
    # candidate set as fast search.
    import contextlib as _contextlib

    alt_fn = getattr(kernel_cls, "alt_configs", None)
    if callable(alt_fn):
        with _contextlib.suppress(TypeError, NotImplementedError):
            valid_configs = list(valid_configs) + list(alt_fn(spec, lnch.device.caps))

    n_valid = len(valid_configs)
    if n_valid == 0:
        _say("(no valid configs)")
        return None

    auto_pop = max(16, int(math.log2(max(n_valid, 2)) * 16))
    _pop_size = cache.pop_size if cache.pop_size is not None else auto_pop
    dense = n_valid <= _pop_size

    if dense:
        _pop_size = n_valid
        _n_gens = 1
        mode = "exhaustive"
    else:
        _n_gens = cache.n_gens
        mode = f"genetic (pop={_pop_size}, gens={_n_gens})"

    _say(f"{n_valid} valid configs, {mode}")

    rng = random.Random(cache.rng_seed)

    if dense:
        population = list(valid_configs)
    else:
        population = _build_initial_population(
            valid_configs, seeds or [], knob_names, _pop_size, rng
        )

    seen: dict[tuple, tuple[float, object]] = {}
    valid_set = {tuple(getattr(c, k) for k in knob_names) for c in valid_configs}
    valid_by_key = {tuple(getattr(c, k) for k in knob_names): c for c in valid_configs}

    best_us = float("inf")
    best_cfg: Any = None
    stale_gens = 0

    for gen in range(_n_gens):
        prev_best_us = best_us
        to_compile = [c for c in population if tuple(getattr(c, k) for k in knob_names) not in seen]

        if to_compile:
            n_threads = max(
                1, min(cache.max_workers or (os.cpu_count() or 2) // 2, len(to_compile))
            )
            _say(f"gen {gen}: compiling {len(to_compile)} new configs ({n_threads} threads)...")

            compile_cache = _parallel_compile(
                kernel_cls,
                spec,
                to_compile,
                lnch,
                knob_names=knob_names,
                max_workers=cache.max_workers,
                on_compile_error=cache._compile_error_hook,
            )

            for cfg in to_compile:
                key = tuple(getattr(cfg, k) for k in knob_names)
                compiled, err = compile_cache.get(key, (None, "not compiled"))
                # Alt-impl configs (e.g. cuBLAS) never get compiled — the
                # parallel-compile worker returns (None, None) for them.
                # Let them fall through to _evaluate_config, which routes
                # them to the kernel's alt hook.
                is_alt = getattr(cfg, "impl", "ptx") != "ptx"
                if compiled is None and err is None and not is_alt:
                    err = "compile returned None"
                if err is not None:
                    _say(f"  [compile err] {cfg}: {err}")
                    seen[key] = (float("inf"), cfg)
                    continue

                result = _evaluate_config(
                    cfg,
                    kernel_cls,
                    spec,
                    spec_cls,
                    spec_dict,
                    compiled,
                    tensors,
                    pspec,
                    out_idx,
                    reference,
                    cache.bench_ms,
                    _say,
                )
                if result is None:
                    seen[key] = (float("inf"), cfg)
                    continue
                us, reason = result
                seen[key] = (us, cfg)

                if us < best_us:
                    best_us = us
                    best_cfg = cfg

        gen_results = [
            (seen[tuple(getattr(c, k) for k in knob_names)], c)
            for c in population
            if tuple(getattr(c, k) for k in knob_names) in seen
        ]
        gen_results.sort(key=lambda x: x[0][0])
        gen_best = gen_results[0][0][0] if gen_results else float("inf")
        n_ok = sum(1 for (us, _), _ in gen_results if us < float("inf"))
        _say(f"gen {gen}: best={gen_best:.2f} μs, valid={n_ok}/{len(population)}")

        if gen > 0:
            if gen_best >= prev_best_us * 0.99:
                stale_gens += 1
            else:
                stale_gens = 0
            if stale_gens >= cache.early_stop_gens:
                _say(f"early stop (no >1% improvement for {cache.early_stop_gens} gens)")
                break

        if dense or gen == _n_gens - 1:
            continue

        n_elite = max(2, _pop_size // 4)
        elites = [c for (_, _), c in gen_results[:n_elite]]
        population = _breed_next_generation(
            elites,
            valid_configs,
            valid_set,
            valid_by_key,
            knob_names,
            knob_values,
            fixed,
            config_cls,
            _pop_size,
            cache.mutate_prob,
            rng,
        )

    if best_cfg is None:
        return None
    return (best_cfg, best_us)
