#!/usr/bin/env python
"""Fuzz-style correctness harness over all registered kernels.

For every kernel target known to ``tools/bench.py``, iterates over the
kernel's ``bench_problems()`` and runs the correctness check with two
configs:

* **default** — whatever ``_pick_default_cfg(spec)`` returns
* **tuned**   — the JSON config saved by ``tools/autotune.py --save``,
                if one exists for this exact problem

Each (kernel, problem, config) cell prints ``✓`` or ``✗ <reason>``. This
is the quick "did anything in the repo break" command — distinct from
``pytest tests/`` which is the unit-test layer.

Each TARGET runs in its own subprocess so a CUDA error from one kernel
(e.g. ILLEGAL_ADDRESS that poisons the device context) can't take down
the rest of the harness.

Usage:
    make test-fuzz
    python tools/test.py
    python tools/test.py --target moe_inproj
    python tools/test.py --atol 0.1 --rtol 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _make_cli_for_target(
    target: str, n_iters: int, warmup: int, atol: float, rtol: float
) -> list[str]:
    return [
        sys.executable,
        os.path.abspath(__file__),
        "--_subprocess",
        "--target",
        target,
        "--n-iters",
        str(n_iters),
        "--warmup",
        str(warmup),
        "--atol",
        str(atol),
        "--rtol",
        str(rtol),
    ]


# ════════════════════════════════════════════════════════════════════════════
# Subprocess body — runs ONE target end-to-end and writes a JSON summary to
# stdout. Imports cupy + torch + the kernel registry only inside this branch
# so the parent process never touches CUDA.
# ════════════════════════════════════════════════════════════════════════════


def _subprocess_main(target: str, n_iters: int, warmup: int, atol: float, rtol: float) -> None:
    # IMPORT ORDER MATTERS for older cupy-based kernels: touch cupy first.
    import cupy as cp

    cp.cuda.Device(0).synchronize()
    import torch  # noqa: F401

    from popcorn.kernels import Kernel  # noqa: F401

    # tools/ isn't a package — add it to sys.path so we can import bench.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench import BENCH_TARGETS

    if target not in BENCH_TARGETS:
        json.dump({"error": f"unknown target {target}"}, sys.stdout)
        return

    kernel_cls = BENCH_TARGETS[target]
    problems = kernel_cls.bench_problems()

    rows = []
    for problem in problems:
        row: dict = {
            "label": _problem_label(problem),
            "default": _run_one_cell(
                kernel_cls,
                problem,
                mode="default",
                atol=atol,
                rtol=rtol,
                n_iters=n_iters,
                warmup=warmup,
            ),
            "tuned": _run_one_cell(
                kernel_cls,
                problem,
                mode="tuned",
                atol=atol,
                rtol=rtol,
                n_iters=n_iters,
                warmup=warmup,
            ),
        }
        rows.append(row)

    json.dump({"target": target, "rows": rows}, sys.stdout)


def _problem_label(problem: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in problem.items() if not str(k).startswith("_"))


def _run_one_cell(kernel_cls, problem, *, mode, atol, rtol, n_iters, warmup) -> dict:
    """Build + compile + correctness + time one (kernel, problem, mode) cell.

    mode:
      - "default": force ``_pick_default_cfg`` via POPCORN_FORCE_DEFAULT_CONFIG
      - "tuned":   only if a tuned JSON exists for this problem
    """
    if mode == "tuned":
        tuned = kernel_cls.load_tuned_config(problem)
        if tuned is None:
            return {"status": "skip", "msg": "no tuned JSON"}

    prev_env = os.environ.get("POPCORN_FORCE_DEFAULT_CONFIG")
    if mode == "default":
        os.environ["POPCORN_FORCE_DEFAULT_CONFIG"] = "1"
    try:
        try:
            kernel = kernel_cls.from_problem(problem)
        except Exception as e:
            return {"status": "fail", "stage": "build", "msg": str(e)[:120]}

        try:
            compiled = kernel.compile()
        except Exception as e:
            return {"status": "fail", "stage": "compile", "msg": str(e)[:120]}

        try:
            tensors = kernel.make_tensors()
            launch_list = kernel.launch_tensor_list(tensors)
        except Exception as e:
            return {"status": "fail", "stage": "alloc", "msg": str(e)[:120]}

        try:
            check = kernel.correctness(launch_list, ref=None, atol=atol, rtol=rtol)
            if not check.get("match", False):
                return {
                    "status": "fail",
                    "stage": "check",
                    "msg": (
                        f"max_abs={check.get('max_abs', 0):.2g} "
                        f"norm_err={check.get('norm_err', 0):.2g}"
                    ),
                }
        except Exception as e:
            return {"status": "fail", "stage": "check", "msg": str(e)[:120]}

        try:
            rt = kernel.time(launch_list, n_iters=n_iters, warmup=warmup)
        except Exception as e:
            return {"status": "fail", "stage": "time", "msg": str(e)[:120]}

        return {
            "status": "ok",
            "runtime_us": rt,
            "smem": compiled.smem_bytes,
            "regs": compiled.footprint.reg_total,
        }
    finally:
        if prev_env is None:
            os.environ.pop("POPCORN_FORCE_DEFAULT_CONFIG", None)
        else:
            os.environ["POPCORN_FORCE_DEFAULT_CONFIG"] = prev_env


# ════════════════════════════════════════════════════════════════════════════
# Parent process — fans out subprocesses + renders the rich table
# ════════════════════════════════════════════════════════════════════════════


def _render_target(console, target: str, summary: dict | None) -> tuple[int, int]:
    """Render one target's table and return (n_pass, n_total) for the
    summary line."""
    from rich import box
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        title=f"[bold]{target}[/]",
        title_justify="left",
    )
    table.add_column("Problem", overflow="fold", min_width=24)
    table.add_column("default", justify="left", min_width=24)
    table.add_column("tuned", justify="left", min_width=24)

    if summary is None or "rows" not in summary:
        msg = (summary or {}).get("error", "subprocess crashed before producing output")
        table.add_row("(no rows)", Text(f"✗ {msg}", style="red"), Text("—", style="dim"))
        console.print(
            Panel(
                table, border_style="red", subtitle="[bold]process error[/]", subtitle_align="right"
            )
        )
        return (0, 0)

    n_pass = 0
    n_total = 0

    def _cell(c: dict) -> Text:
        nonlocal n_pass, n_total
        st = c.get("status")
        if st == "ok":
            n_pass += 1
            n_total += 1
            return Text(
                f"✓ {c['runtime_us']:.0f} us  smem={c['smem'] // 1024}KB",
                style="green",
            )
        if st == "skip":
            return Text(f"— {c.get('msg', 'skip')}", style="dim")
        # fail
        n_total += 1
        return Text(f"✗ {c.get('stage', '?')}: {c.get('msg', '')}", style="red")

    for row in summary["rows"]:
        table.add_row(row["label"], _cell(row["default"]), _cell(row["tuned"]))

    border_style = "green" if n_pass == n_total else "red"
    subtitle = f"[bold]{n_pass}/{n_total} cells passed[/]"
    console.print(
        Panel(table, border_style=border_style, subtitle=subtitle, subtitle_align="right")
    )
    return (n_pass, n_total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="all",
        help="Which kernel to fuzz (default: all)",
    )
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    # Internal flag: when set, run as the subprocess body for one target
    # and emit JSON to stdout instead of rich tables.
    parser.add_argument("--_subprocess", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._subprocess:
        _subprocess_main(
            args.target,
            args.n_iters,
            args.warmup,
            args.atol,
            args.rtol,
        )
        return

    # Parent: discover targets without importing CUDA libs into this proc.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # Importing bench drags in cupy/torch — but the parent never compiles or
    # launches anything itself, so the import is harmless. Doing it lazily
    # to keep `--help` cheap.
    from bench import BENCH_TARGETS

    if args.target == "all":
        targets = sorted(BENCH_TARGETS.keys())
    else:
        if args.target not in BENCH_TARGETS:
            print(f"unknown target {args.target}; choose from {sorted(BENCH_TARGETS)}")
            sys.exit(2)
        targets = [args.target]

    from rich.console import Console

    console = Console()

    t0 = time.monotonic()
    grand_pass = 0
    grand_total = 0
    for target in targets:
        cmd = _make_cli_for_target(
            target,
            args.n_iters,
            args.warmup,
            args.atol,
            args.rtol,
        )
        try:
            with console.status(f"[bold]Running[/] {target}...", spinner="dots"):
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            summary: dict | None
            if proc.returncode != 0:
                summary = {
                    "error": f"exit {proc.returncode}: {(proc.stderr or '').splitlines()[-1] if proc.stderr else 'no stderr'}"
                }
            else:
                try:
                    summary = json.loads(proc.stdout)
                except Exception:
                    summary = {"error": f"non-JSON stdout: {proc.stdout[:100]}"}
        except subprocess.TimeoutExpired:
            summary = {"error": "subprocess timeout"}
        except Exception as e:
            summary = {"error": f"subprocess launch: {e}"}

        n_pass, n_total = _render_target(console, target, summary)
        grand_pass += n_pass
        grand_total += n_total

    elapsed = time.monotonic() - t0
    color = "green" if grand_pass == grand_total else "red"
    console.print(f"\n[bold {color}]{grand_pass}/{grand_total} cells passed in {elapsed:.1f}s[/]")
    if grand_pass != grand_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
