"""Registry-driven kernel benchmark with pretty-printed tables.

Usage:
  python tools/bench.py                     # all kernels
  python tools/bench.py --kernel gemm       # one kernel
  python tools/bench.py --tag production    # only tagged problems
  python tools/bench.py --bench-ms 100      # longer bench budget
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from popcorn.utils.pretty import Table, print_header, print_kv, style


def _configs_dir() -> Path:
    import os

    env = os.environ.get("POPCORN_CONFIGS_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[1] / "configs"
    return repo if repo.exists() else Path("configs")


def _load_saved_config(kernel_cls, problem):
    name = kernel_cls.NAME
    cdir = _configs_dir()
    fname = cdir / f"{name}_{problem.name}.json"
    if fname.exists():
        with open(fname) as f:
            data = json.load(f)
        cfg_dict = data.get("config", {})
        config_cls = kernel_cls.CONFIG_CLS
        if config_cls is not None:
            try:
                return config_cls(**cfg_dict), True
            except (TypeError, ValueError):
                pass
    return None, False


def _time_callable(fn, *, warmup_ms=10.0, bench_ms=50.0) -> float:
    """Budget-based timing — dispatches through PT."""
    from popcorn.backend import PT

    return PT.time_callable(fn, warmup_ms=warmup_ms, bench_ms=bench_ms)


def _time_kernel(compiled, buffers, *, warmup_ms=10.0, bench_ms=50.0) -> float:
    return _time_callable(
        lambda: compiled.launch(buffers=buffers),
        warmup_ms=warmup_ms,
        bench_ms=bench_ms,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", help="restrict to one kernel name")
    parser.add_argument("--problem", help="restrict to one problem name")
    parser.add_argument("--tag", help="only bench problems with this tag")
    parser.add_argument("--exclude-tag", help="skip problems with this tag")
    parser.add_argument("--bench-ms", type=float, default=50.0)
    parser.add_argument("--warmup-ms", type=float, default=10.0)
    args = parser.parse_args(argv)

    from popcorn.backend import IS_METAL, PT
    from popcorn.correctness import check_correctness
    from popcorn.device import current_device
    from popcorn.kernels import all_kernels, get
    from popcorn.launcher import Launcher

    kernels = [get(args.kernel)] if args.kernel else all_kernels()
    launcher = Launcher(device=current_device())

    print_header(f"Bench — {launcher.device.caps.name}")
    print_kv("Configs", str(_configs_dir()))

    for cls in kernels:
        problems = cls.problems()
        if args.problem:
            problems = [p for p in problems if p.name == args.problem]
        if args.tag:
            problems = [p for p in problems if args.tag in p.tags]
        if args.exclude_tag:
            problems = [p for p in problems if args.exclude_tag not in p.tags]
        if not problems:
            continue

        # First pass: build result records for every problem. Defer table
        # construction until we know the full set of baseline names across
        # this kernel's problems — each baseline gets its own column, so
        # the column set is stable within a kernel.
        records: list[dict] = []
        baseline_names: list[str] = []  # insertion-ordered unique names

        for problem in problems:
            rec: dict = {"problem": problem.name, "baselines": {}}
            records.append(rec)

            saved_cfg, _is_tuned = _load_saved_config(cls, problem)
            if saved_cfg is not None:
                spec = cls.SPEC_CLS(**problem.params)
                kernel = cls(spec, saved_cfg)
                rec["cfg_label"] = style("tuned", "green")
            else:
                kernel = cls.from_problem(problem.params)
                overrides = getattr(problem, "config_overrides", None)
                if overrides:
                    import dataclasses as _dc

                    kernel.config = _dc.replace(kernel.config, **overrides)
                rec["cfg_label"] = style("default", "dim")

            if not kernel.is_valid_for(launcher.device.caps):
                rec["state"] = "invalid"
                continue

            try:
                tensors = cls.make_tensors(problem.params)
                pspec = kernel.param_spec()
                launch_tensors = kernel.prepare_launch_tensors(tensors)
                buffers = [launch_tensors[b.name] for b in pspec.buffers]
                out_idx = cls.OUTPUT_IDX
                if out_idx < 0:
                    out_idx = len(buffers) + out_idx
                output_buf = buffers[out_idx]
                plain_buffers = [tensors[b.name] for b in pspec.buffers]
                inputs = [plain_buffers[i] for i in range(len(plain_buffers)) if i != out_idx]
            except Exception as e:
                rec["state"] = "err"
                full = f"tensors: {type(e).__name__}: {e}"
                rec["err"] = full[:30]
                print(f"    {style('ERR', 'red')} {full}")
                continue

            try:
                compiled = launcher.compile(cls, kernel.spec, kernel.config)
            except Exception as e:
                from popcorn.utils.ptx_dump import classify_compile_error, write_ptx_for

                rec["state"] = "err"
                if classify_compile_error(e) == "ptx":
                    # Save the PTX the driver rejected so a human can
                    # diff it; the message path is too short to be
                    # useful by itself.
                    path = write_ptx_for(cls, kernel.spec, kernel.config, launcher.device.caps)
                    loc = f" (ptx saved: {path})" if path else ""
                    rec["err"] = f"compile: {e}{loc}"
                else:
                    rec["err"] = f"compile: {e}"
                print(f"    {style('ERR', 'red')} {rec['err']}")
                continue

            try:
                launch_result = compiled.launch(buffers=buffers)
                PT.synchronize()
                if IS_METAL and launch_result:
                    # launch_result from MLX has one entry per non-
                    # readonly buffer in pspec order. Translate the
                    # full-buffer-list OUTPUT_IDX into the position
                    # within that subset — kernels with multiple
                    # write targets (kv_cache_update) had picked up
                    # the wrong buffer via ``launch_result[0]``.
                    out_pos = sum(1 for b in pspec.buffers[:out_idx] if not b.readonly)
                    actual_output = launch_result[out_pos]
                else:
                    actual_output = output_buf
                ref = kernel.reference(*inputs)
                out_dtype = PT.backend_dtype_to_ir(actual_output.dtype)
                cr = check_correctness(
                    actual_output,
                    ref,
                    out_dtype=out_dtype,
                    threshold=cls.correctness_threshold(out_dtype),
                )
            except Exception as e:
                import traceback

                rec["state"] = "err"
                full = f"launch: {type(e).__name__}: {e}"
                rec["err"] = full[:30]
                print(f"    {style('ERR', 'red')} {full}")
                # Print the traceback so root cause surfaces — 'list
                # index out of range' truncated to 30 chars tells a
                # debugger nothing.
                traceback.print_exc()
                continue

            output_buf = PT.zero_(output_buf)
            try:
                us = _time_kernel(
                    compiled,
                    buffers,
                    warmup_ms=args.warmup_ms,
                    bench_ms=args.bench_ms,
                )
            except Exception as e:
                rec["state"] = "err"
                full = f"time: {type(e).__name__}: {e}"
                rec["err"] = full[:30]
                print(f"    {style('ERR', 'red')} {full}")
                continue

            rec["state"] = "ok"
            rec["us"] = us
            try:
                rec["tflops"] = kernel.flops() / (us * 1e-6) / 1e12
            except Exception:
                rec["tflops"] = None
            rec["cos_sim"] = cr.cos_sim
            rec["passed"] = cr.passed

            # Baseline timing — each baseline gets its own column in the
            # final table. ``reported_time_scale`` pro-rates the raw time
            # for fused baselines covering multiple logical ops (e.g. MoE
            # passes 0.5 because flashinfer_fused_moe covers in+out in one
            # call). Speedup = baseline_us / kernel_us — coloured green
            # (≥1.0), yellow (≥0.85), red (<0.85).
            try:
                bls = kernel.baselines(tensors)
            except Exception as e:
                rec["baseline_err"] = f"{e}"[:30]
                bls = []
            for bl in bls:
                if bl.name not in baseline_names:
                    baseline_names.append(bl.name)
                try:
                    raw_us = _time_callable(
                        bl.fn,
                        warmup_ms=args.warmup_ms,
                        bench_ms=args.bench_ms,
                    )
                    bl_us = raw_us * bl.reported_time_scale
                except Exception as e:
                    rec["baselines"][bl.name] = {"err": f"{e}"[:20]}
                    continue
                rec["baselines"][bl.name] = {"us": bl_us}

        # Second pass: render one table for this kernel, with per-baseline cols.
        table = Table()
        table.add_column("Problem", min_width=14)
        table.add_column("Config", align="center", min_width=7)
        table.add_column("Time (μs)", align="right", min_width=10)
        table.add_column("TFLOPS", align="right", min_width=8)
        for name in baseline_names:
            table.add_column(f"vs {name} (μs)", align="right", min_width=18)
        table.add_column("cos_sim", align="right", min_width=10)
        table.add_column("Status", align="center", min_width=7)

        for rec in records:
            state = rec.get("state")
            if state == "invalid":
                row = [rec["problem"], rec["cfg_label"], "—", "—"]
                row.extend(["—"] * len(baseline_names))
                row.extend(["—", style("INVALID", "yellow")])
                table.add_row(*row)
                continue
            if state == "err":
                row = [rec["problem"], rec["cfg_label"], "—", "—"]
                row.extend(["—"] * len(baseline_names))
                row.extend(["—", style(f"ERR: {rec['err']}"[:30], "red")])
                table.add_row(*row)
                continue

            us = rec["us"]
            tflops_str = f"{rec['tflops']:.1f}" if rec.get("tflops") is not None else "—"
            cos_str = f"{rec['cos_sim']:.6f}"
            status = style("OK", "green") if rec["passed"] else style("FAIL", "red")

            bl_cells: list[str] = []
            for name in baseline_names:
                bl = rec["baselines"].get(name)
                if bl is None:
                    bl_cells.append(style("—", "dim"))
                elif "err" in bl:
                    bl_cells.append(style(f"ERR: {bl['err']}"[:18], "red"))
                else:
                    bl_us = bl["us"]
                    speedup = bl_us / us if us > 0 else float("nan")
                    if speedup >= 1.0:
                        colour = "green"
                    elif speedup >= 0.85:
                        colour = "yellow"
                    else:
                        colour = "red"
                    bl_cells.append(style(f"{bl_us:.2f} ({speedup:.2f}x)", colour))

            row = [
                rec["problem"],
                rec["cfg_label"],
                f"{us:.2f}",
                tflops_str,
                *bl_cells,
                cos_str,
                status,
            ]
            table.add_row(*row)

        print()
        print(f"  {style(cls.NAME, 'bold', 'cyan')}")
        table.print()
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
