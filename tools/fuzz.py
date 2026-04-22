"""Correctness fuzz harness — sweeps every kernel x every problem.

Usage:
  python tools/fuzz.py                       # every kernel, every problem
  python tools/fuzz.py --kernel gemm         # one kernel
  python tools/fuzz.py --kernel gemm --problem small_bf16
  python tools/fuzz.py --tag smoke           # only problems tagged "smoke"

numpy-refs flow: ``RefCache`` returns ``(inputs_np, outputs_np)``
per (kernel, problem) — cached in-memory + on disk at
``~/.cache/popcorn/refs``. Launch consumes numpy-staged device tensors
via ``numpy_to_device_dict``; the output is compared against the numpy
reference by ``check_correctness`` (which normalizes both sides to
f32 numpy internally).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from typing import Optional

from popcorn.correctness import check_correctness
from popcorn.device import current_device
from popcorn.ir import DType
from popcorn.refs import ref_cache
from popcorn.runtime.device_tensors import numpy_to_device_dict
from popcorn.utils.pretty import Table, print_header, style


@dataclass
class _FuzzResult:
    kernel: str
    problem_name: str
    passed: bool
    cos_sim: Optional[float] = None
    reason: Optional[str] = None
    error: Optional[str] = None


def _output_dtype(kernel_cls, spec) -> DType:
    """Look up the declared dtype of the OUTPUT_IDX buffer."""
    # OUTPUT_IDX indexes into ParamSpec.buffers, not TENSORS. For the
    # common case the role="out" TensorDecl at that index is what's
    # compared — which is the last role="out" entry for OUTPUT_IDX=-1.
    out_tensors = [t for t in kernel_cls.TENSORS if getattr(t, "role", None) == "out"]
    decl = out_tensors[kernel_cls.OUTPUT_IDX]
    dt = decl.dtype(spec, None) if callable(decl.dtype) else decl.dtype
    return dt


def _run_one(kernel_cls, problem, launcher) -> _FuzzResult:
    name = kernel_cls.NAME
    try:
        kernel = kernel_cls.from_problem(problem.params)
        # Apply problem config_overrides (e.g. b_shuffle=True for a
        # preshuffle variant).
        overrides = getattr(problem, "config_overrides", None)
        if overrides:
            import dataclasses as _dc

            kernel.config = _dc.replace(kernel.config, **overrides)
        if not kernel.is_valid_for(launcher.device.caps):
            return _FuzzResult(name, problem.name, False, reason="not valid for device")

        inputs_np, outputs_np = ref_cache().get(kernel_cls, problem.params)

        # Convert numpy inputs → device tensors at the launcher boundary.
        tensors = numpy_to_device_dict(kernel_cls, kernel.spec, inputs_np)
        pspec = kernel.param_spec()
        launch_tensors = kernel.prepare_launch_tensors(tensors)
        buffers = [launch_tensors[b.name] for b in pspec.buffers]
        out_idx = kernel_cls.OUTPUT_IDX
        if out_idx < 0:
            out_idx = len(buffers) + out_idx
        output_buf = buffers[out_idx]
        output_name = pspec.buffers[out_idx].name

        compiled = launcher.compile(kernel_cls, kernel.spec, kernel.config)
        compiled.launch(buffers=buffers)

        ref_np = outputs_np[output_name]
        out_dtype = _output_dtype(kernel_cls, kernel.spec)
        result = check_correctness(
            output_buf,
            ref_np,
            out_dtype=out_dtype,
            threshold=kernel_cls.correctness_threshold(out_dtype),
        )
        return _FuzzResult(
            name, problem.name, result.passed, cos_sim=result.cos_sim, reason=result.reason
        )
    except Exception as e:
        return _FuzzResult(
            name, problem.name, False, error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="popcorn correctness fuzzer")
    parser.add_argument("--kernel", help="restrict to one kernel name")
    parser.add_argument("--problem", help="restrict to one problem name")
    parser.add_argument("--tag", help="only run problems with this tag")
    parser.add_argument("--exclude-tag", help="skip problems with this tag")
    args = parser.parse_args(argv)

    from popcorn.kernels import all_kernels, get
    from popcorn.launcher import Launcher

    kernels = [get(args.kernel)] if args.kernel else all_kernels()
    if not kernels:
        print("No kernels registered.")
        return 0

    launcher = Launcher(device=current_device())
    print_header(f"Fuzz — {launcher.device.caps.name}")

    table = Table()
    table.add_column("Kernel", min_width=20)
    table.add_column("Problem", min_width=16)
    table.add_column("cos_sim", align="right", min_width=10)
    table.add_column("Status", align="center", min_width=6)

    results = []
    for cls in kernels:
        problems = cls.problems()
        if args.problem:
            problems = [p for p in problems if p.name == args.problem]
        if args.tag:
            problems = [p for p in problems if args.tag in p.tags]
        if args.exclude_tag:
            problems = [p for p in problems if args.exclude_tag not in p.tags]
        for problem in problems:
            r = _run_one(cls, problem, launcher)
            results.append(r)
            cos_str = f"{r.cos_sim:.6f}" if r.cos_sim is not None else "—"
            if r.passed:
                status = style("OK", "green")
            else:
                status = style("FAIL", "red")
            table.add_row(r.kernel, r.problem_name, cos_str, status)

    table.print()

    n = len(results)
    n_pass = sum(1 for r in results if r.passed)
    print(f"\n  {n_pass}/{n} passed")
    if n_pass < n:
        print()
        for r in results:
            if not r.passed:
                detail = r.reason or (r.error or "").splitlines()[0]
                print(f"  {style('FAIL', 'red')} {r.kernel} / {r.problem_name}: {detail}")
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    sys.exit(main())
