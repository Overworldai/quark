"""Correctness fuzz harness — sweeps every kernel x every problem.

Usage:
  python tools/fuzz.py                       # every kernel, every problem
  python tools/fuzz.py --kernel gemm         # one kernel
  python tools/fuzz.py --kernel gemm --problem small_bf16
  python tools/fuzz.py --tag smoke           # only problems tagged "smoke"
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
from popcorn.utils.pretty import Table, print_header, style

_TORCH_TO_IR_DTYPE = {}


def _build_dtype_map():
    global _TORCH_TO_IR_DTYPE
    if _TORCH_TO_IR_DTYPE:
        return _TORCH_TO_IR_DTYPE
    import torch

    _TORCH_TO_IR_DTYPE = {
        torch.float32: DType.F32,
        torch.float16: DType.F16,
        torch.bfloat16: DType.BF16,
        torch.int8: DType.S8,
        torch.uint8: DType.U8,
        torch.int32: DType.S32,
    }
    # fp8 dtypes — only present when torch was built with fp8 support.
    # Without these, fp8 outputs fall back to the F32 threshold (1-1e-5)
    # which is impossibly tight for e4m3's ~3-bit mantissa, and a working
    # kernel reports false failures.
    fp8_e4m3 = getattr(torch, "float8_e4m3fn", None)
    if fp8_e4m3 is not None:
        _TORCH_TO_IR_DTYPE[fp8_e4m3] = DType.E4M3
    fp8_e5m2 = getattr(torch, "float8_e5m2", None)
    if fp8_e5m2 is not None:
        _TORCH_TO_IR_DTYPE[fp8_e5m2] = DType.E5M2
    return _TORCH_TO_IR_DTYPE


@dataclass
class _FuzzResult:
    kernel: str
    problem_name: str
    passed: bool
    cos_sim: Optional[float] = None
    reason: Optional[str] = None
    error: Optional[str] = None


def _run_one(kernel_cls, problem, launcher) -> _FuzzResult:
    name = kernel_cls.NAME
    try:
        import torch

        kernel = kernel_cls.from_problem(problem.params)
        # Apply the problem's config_overrides (pins specific config knobs
        # for this problem — e.g. b_shuffle=True for a preshuffle variant).
        overrides = getattr(problem, "config_overrides", None)
        if overrides:
            import dataclasses as _dc

            kernel.config = _dc.replace(kernel.config, **overrides)
        if not kernel.is_valid_for(launcher.device.caps):
            return _FuzzResult(name, problem.name, False, reason="not valid for device")
        tensors = kernel_cls.make_tensors(problem.params)
        pspec = kernel.param_spec()
        # Reference uses plain tensors; launch uses whatever the kernel
        # wants (e.g. pre-shuffled W_in when config.b_shuffle=True).
        launch_tensors = kernel.prepare_launch_tensors(tensors)
        buffers = [launch_tensors[b.name] for b in pspec.buffers]
        out_idx = kernel_cls.OUTPUT_IDX
        if out_idx < 0:
            out_idx = len(buffers) + out_idx
        output_buf = buffers[out_idx]
        plain_buffers = [tensors[b.name] for b in pspec.buffers]
        inputs = [plain_buffers[i] for i in range(len(plain_buffers)) if i != out_idx]

        compiled = launcher.compile(kernel_cls, kernel.spec, kernel.config)
        compiled.launch(buffers=buffers)
        torch.cuda.synchronize()

        ref = kernel.reference(*inputs)
        out_dtype = _build_dtype_map().get(output_buf.dtype, DType.F32)
        result = check_correctness(
            output_buf,
            ref,
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
