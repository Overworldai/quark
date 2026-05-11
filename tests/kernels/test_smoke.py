"""Auto-parameterized smoke test over the kernel registry.

Per quark cleanup proposal §5.2. One smoke test per registered
kernel: build it from `problems()[0]`, compile via the Launcher,
launch with `make_tensors(problem)`, and gate the result against
`reference()` via `quark.correctness.check_correctness` (cosine
similarity, the only correctness metric per §7).

This file replaces the per-kernel `tests/test_<kernel>.py` files
that ran shape sweeps and timed assertions. Deep correctness sweeps
live in `tools/fuzz.py`; perf timing lives in `tools/bench.py`.
Tests do "the kernel runs at all on this device, once."

Skips:
- The whole module is skipped when no GPU is present.
- Each parametrized case skips when `is_valid_for(caps)` is False.
- The whole test set is empty (no parametrize) when the registry is
  empty — this is the expected state until the first kernel migrates
  to the new folder layout, so an empty parametrize is fine.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("smoke tests need a GPU", allow_module_level=True)

from quark.correctness import check_correctness  # noqa: E402
from quark.device import current_device  # noqa: E402
from quark.ir import DType  # noqa: E402
from quark.kernels import all_kernels  # noqa: E402
from quark.launcher import Launcher  # noqa: E402

# Module-level singletons so the parametrize fixture and the test
# share the same Device + Launcher (so the autotune cache is shared
# and we don't repeatedly probe libcuda).
DEVICE = current_device()
LAUNCHER = Launcher(device=DEVICE)


_TORCH_TO_IR_DTYPE = {
    torch.float32: DType.F32,
    torch.float16: DType.F16,
    torch.bfloat16: DType.BF16,
    torch.int8: DType.S8,
    torch.uint8: DType.U8,
    torch.int32: DType.S32,
}


def _kernels_to_test() -> list:
    """The registry's contents at collection time. Returns a list
    even when empty so pytest collects the parametrize cleanly."""
    return all_kernels()


def _kernel_id(cls) -> str:
    return getattr(cls, "NAME", cls.__name__)


@pytest.mark.parametrize(
    "kernel_cls",
    _kernels_to_test(),
    ids=_kernel_id,
)
def test_kernel_smoke(kernel_cls):
    """One problem × default config × correctness vs reference.

    The "the kernel compiles, launches, and produces something
    cosine-similar to its reference" gate. Deep sweeps belong in
    the fuzzer."""
    problems = kernel_cls.problems()
    if not problems:
        pytest.skip(f"{_kernel_id(kernel_cls)}: no problems declared")
    problem = problems[0]
    kernel = kernel_cls.from_problem(problem)
    if not kernel.is_valid_for(DEVICE.caps):
        pytest.skip(f"{_kernel_id(kernel_cls)}: not valid on {DEVICE.caps.name}")

    tensors = kernel_cls.make_tensors(problem)
    if not isinstance(tensors, dict):
        pytest.skip(
            f"{_kernel_id(kernel_cls)}: make_tensors returned "
            f"{type(tensors).__name__}, expected dict"
        )

    pspec = kernel.param_spec()
    buffers = [tensors[b.name] for b in pspec.buffers]

    # The reference takes only the *input* tensors. The proposal
    # didn't pin a precise input/output split convention, so we
    # delegate to the kernel: it knows which buffers are inputs
    # and how its reference() expects them. The simplest contract
    # we can enforce here is "reference takes the input buffers
    # in declaration order"; the kernel author either matches
    # that or overrides this smoke test in their own folder.
    n_inputs = max(0, len(buffers) - 1)  # all but the last buffer (output)
    inputs = buffers[:n_inputs]
    output_buf = buffers[-1]

    # Compile + launch via the new Launcher path. config=None makes
    # the launcher consult its AutotuneCache; if no bundled default
    # exists, the cache will run the bounded online search (which
    # itself uses make_tensors / problems()).
    compiled = LAUNCHER.compile(kernel_cls, kernel.spec)
    compiled.launch(buffers=buffers)
    torch.cuda.synchronize()

    ref = kernel.reference(*inputs)
    out_dtype = _TORCH_TO_IR_DTYPE.get(output_buf.dtype, DType.F32)
    result = check_correctness(output_buf, ref, out_dtype=out_dtype)
    assert result.passed, (
        f"{_kernel_id(kernel_cls)}: smoke failed cos_sim="
        f"{result.cos_sim:.6f} threshold={result.threshold:.6f} "
        f"reason={result.reason}"
    )


def test_registry_is_consultable():
    """Sanity: the registry import path works even if no kernels
    are currently registered. Catches the import-loop bug."""
    kernels = all_kernels()
    assert isinstance(kernels, list)
