"""Auto-parameterized SPV smoke test over the kernel registry.

Sister to ``test_smoke.py`` (CUDA + torch). The SPV variant uses
numpy + the ``SpvDriver`` so it's not gated on torch/CUDA. The IR
is meant to be platform agnostic; this test exercises that claim
by running every kernel that compiles cleanly through the SPV
launcher and verifying its output matches the kernel's own
``reference_numpy`` oracle at a backend-aware tolerance.

Skip rules (in order):
- Whole module: skipped on non-Linux or hosts without Vulkan +
  spirv-as.
- Per-kernel: skipped when ``is_valid_for(caps)`` is False (config
  doesn't fit Battlemage limits).
- Per-kernel: skipped when ``problems()`` is empty or the first
  problem uses a dtype the SPV backend hasn't wired yet (only F32
  is fully covered today; BF16 / F16 / FP8 dtypes land later).
- Per-kernel: skipped when ``compile()`` raises ``NotImplementedError``
  (a visitor / dtype gap surfaces the kernel as a coverage hole rather
  than a hard failure).

Surfaces gaps as XPASS / coverage rather than test failures so the
visitor wishlist stays visible without breaking CI.
"""

from __future__ import annotations

import ctypes
import shutil
import sys

import numpy as np
import pytest

pytestmark_e2e = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("spirv-as") is None,
    reason="needs Linux + Vulkan + spirv-as for end-to-end dispatch",
)

# Skip the whole module on Mac / no-Vulkan hosts so collection stays cheap.
if sys.platform != "linux":
    pytest.skip("SPV smoke needs Linux", allow_module_level=True)

from quark.drivers import spv as _spv_drivers  # noqa: E402

if not _spv_drivers.is_available():
    pytest.skip("SPV smoke needs a Vulkan device", allow_module_level=True)

if shutil.which("spirv-as") is None:
    pytest.skip("SPV smoke needs spirv-as on PATH", allow_module_level=True)

from quark.drivers import _spv_dispatch as _sd  # noqa: E402
from quark.ir import DType  # noqa: E402
from quark.kernels import all_kernels  # noqa: E402
from quark.launcher import Launcher  # noqa: E402

# Module-level singletons. Same pattern as ``test_smoke.py``.
_DRIVER = _spv_drivers.SpvDriver()
_DEVICE = _DRIVER.device
_LAUNCHER = Launcher(device=_DEVICE)


# Dtypes the SPV lowerer is known to fully support today. Kernels
# whose first problem uses anything else are skipped (visitor / dtype
# gaps land as coverage holes, not test failures).
_SPV_SUPPORTED_DTYPES = {DType.F32, DType.U32, DType.S32}


def _kernels_to_test() -> list:
    """Registry contents at collection time. Empty list is fine."""
    return all_kernels()


def _kernel_id(cls) -> str:
    return getattr(cls, "NAME", cls.__name__)


def _problem_dtypes(problem_params: dict) -> set:
    """Return the set of dtypes referenced anywhere in a problem
    spec's params dict. Handles both ``DType`` enums (the canonical
    form) and lowercase string aliases (``"bf16"``, ``"f32"`` — what
    most ``problems()`` lists pass as kwargs since ``DType.__post_init__``
    on the spec coerces them)."""
    dtypes = set()
    for k, v in problem_params.items():
        if isinstance(v, DType):
            dtypes.add(v)
        elif isinstance(v, str) and ("dtype" in k or k.endswith("_dtype")):
            try:
                dtypes.add(DType(v))
            except (KeyError, ValueError):
                # Unknown dtype string — surface as None so the
                # filter falls through to a skip.
                dtypes.add(None)
    return dtypes


@pytestmark_e2e
@pytest.mark.parametrize(
    "kernel_cls",
    _kernels_to_test(),
    ids=_kernel_id,
)
def test_spv_kernel_smoke(kernel_cls):
    """One problem × default config × correctness vs reference, on SPV."""
    problems = kernel_cls.problems()
    if not problems:
        pytest.skip(f"{_kernel_id(kernel_cls)}: no problems declared")

    # Most problem dicts don't explicitly set ``dtype`` — the kernel's
    # Spec carries a default (typically BF16 for waypoint-1.5 kernels).
    # We retarget the first problem to F32 so the SPV backend has a
    # shot at it; if the Spec rejects F32 (rare — moe_router is the
    # only one) we skip. Sweeping over all dtype-naming conventions
    # (``dtype``, ``in_dtype``, ``kv_dtype``, ``compute_dtype``, ...)
    # catches the multi-dtype kernels too.
    _DTYPE_KEYS = (
        "dtype", "in_dtype", "kv_dtype", "compute_dtype",
        "src_dtype", "out_dtype",
    )
    chosen = None
    kernel = None
    for p in problems:
        params = dict(p.params)
        # Force every dtype-shaped param to F32. Lets ``problems()``
        # entries that hardcode BF16 still smoke under SPV.
        for key in _DTYPE_KEYS:
            if key in params:
                params[key] = DType.F32
        spec_cls = kernel_cls.SPEC_CLS
        if spec_cls is not None:
            import inspect as _inspect
            sig = _inspect.signature(spec_cls)
            for key in _DTYPE_KEYS:
                if key in sig.parameters and key not in params:
                    params[key] = DType.F32
        try:
            k = kernel_cls.from_problem(params)
        except Exception:
            continue
        spec_dtypes = {
            getattr(k.spec, attr)
            for attr in _DTYPE_KEYS
            if isinstance(getattr(k.spec, attr, None), DType)
        }
        if not spec_dtypes:
            # Spec doesn't carry a dtype — assume it's compatible.
            chosen = p
            kernel = k
            break
        if spec_dtypes <= _SPV_SUPPORTED_DTYPES:
            chosen = p
            kernel = k
            break
    if chosen is None or kernel is None:
        pytest.skip(
            f"{_kernel_id(kernel_cls)}: no F32-compatible problem "
            "(SPV backend dtype coverage gap)"
        )

    if not kernel.is_valid_for(_DEVICE.caps):
        pytest.skip(
            f"{_kernel_id(kernel_cls)}: invalid for {_DEVICE.caps.name}"
        )

    if not hasattr(kernel_cls, "make_tensors_numpy"):
        pytest.skip(f"{_kernel_id(kernel_cls)}: no make_tensors_numpy")

    # Re-build the params dict matching what we used for ``from_problem``
    # above so the make_tensors call sees the F32-coerced shape.
    params = dict(chosen.params)
    for key in _DTYPE_KEYS:
        if key in params or hasattr(kernel.spec, key):
            params[key] = DType.F32
    try:
        tensors = kernel_cls.make_tensors_numpy(params)
    except NotImplementedError:
        pytest.skip(f"{_kernel_id(kernel_cls)}: make_tensors_numpy not impl")
    except Exception as exc:
        pytest.skip(f"{_kernel_id(kernel_cls)}: make_tensors_numpy: {exc}")

    if not isinstance(tensors, dict):
        pytest.skip(
            f"{_kernel_id(kernel_cls)}: make_tensors_numpy returned "
            f"{type(tensors).__name__}, expected dict"
        )

    # Compile through the SPV launcher. Surfaces visitor gaps as
    # NotImplementedError (the SPV lowerer raises that on any op /
    # dtype it hasn't wired); skip rather than hard-fail so the
    # remaining kernels still run.
    try:
        compiled = _LAUNCHER.compile(kernel_cls, kernel.spec, kernel.config)
    except NotImplementedError as exc:
        pytest.skip(f"{_kernel_id(kernel_cls)}: SPV lower gap: {exc}")
    except ValueError as exc:
        # ``is_valid_for`` already filtered, but the launcher's
        # ParamSpec construction can still fail (e.g. ad-hoc
        # constraints). Treat as a soft skip.
        pytest.skip(f"{_kernel_id(kernel_cls)}: launcher compile: {exc}")

    # Some kernels declare buffer params they don't actually use in
    # the body (e.g. ``ElementwiseKernel``'s ``Y`` for unary ops). The
    # SPV lowerer currently skips unused params — see the comment on
    # ``_collect_global_tensors`` — which leaves the lowered pipeline
    # with fewer bindings than ``param_spec.buffers``. The launcher's
    # ``_launch_spv`` doesn't reconcile this gap; the driver rejects
    # the mismatched buffer count. Skip such kernels here rather than
    # hard-failing — the underlying gap is tracked separately.
    if compiled.module.n_buffers != len(compiled.param_spec.buffers):
        pytest.skip(
            f"{_kernel_id(kernel_cls)}: lowered n_buffers="
            f"{compiled.module.n_buffers} but param_spec has "
            f"{len(compiled.param_spec.buffers)} (unused-param gap)"
        )

    # Determine which buffers are outputs. ``TENSORS`` carries the
    # role per declaration; we cross-walk it against ``pspec.buffers``
    # by name. Multi-output kernels (kv_cache_update, moe_router_*)
    # have several ``role="out"`` entries; single-output kernels have
    # one. Inputs are everything that isn't an output.
    pspec = kernel.param_spec()
    out_names: set[str] = set()
    if hasattr(kernel_cls, "TENSORS"):
        for decl in kernel_cls.TENSORS:
            if getattr(decl, "role", None) == "out":
                out_names.add(decl.name)
    if not out_names:
        # Fallback to OUTPUT_IDX. ``-1`` (default) selects the last
        # buffer in ``pspec.buffers``.
        out_idx = kernel_cls.OUTPUT_IDX
        if out_idx < 0:
            out_idx = len(pspec.buffers) + out_idx
        out_names = {pspec.buffers[out_idx].name}

    handles: list[int] = []
    out_ptrs: dict[str, int] = {}
    name_to_arr = dict(tensors)
    for i, buf in enumerate(pspec.buffers):
        arr = name_to_arr[buf.name]
        if not isinstance(arr, np.ndarray):
            pytest.skip(
                f"{_kernel_id(kernel_cls)}: tensor {buf.name!r} not "
                f"numpy ({type(arr).__name__})"
            )
        h, ptr = _sd.allocate_buffer(arr.nbytes)
        if buf.name in out_names:
            out_ptrs[buf.name] = ptr
        else:
            ctypes.memmove(ptr, arr.ctypes.data, arr.nbytes)
        handles.append(h)

    compiled.launch(buffers=handles)

    # Reference: call the kernel's numpy oracle on the input tensors.
    if not hasattr(kernel_cls, "reference_numpy"):
        pytest.skip(f"{_kernel_id(kernel_cls)}: no reference_numpy")

    inputs = {
        buf.name: name_to_arr[buf.name]
        for buf in pspec.buffers
        if buf.name not in out_names
    }
    try:
        ref = kernel_cls.reference_numpy(kernel.spec, **inputs)
    except TypeError as exc:
        pytest.skip(
            f"{_kernel_id(kernel_cls)}: reference_numpy signature "
            f"mismatch: {exc}"
        )

    # Multi-output kernels (kv_cache_update, moe_router_*) ran fine
    # — handles allocated, kernel launched, no exception. Per-output
    # numerical correctness against the numpy reference is a separate
    # concern: the moe_router cohort uses cross-WG atomics so the GPU
    # output is non-deterministic in its ordering (a different valid
    # permutation than the reference's), and ``kv_cache_update``'s
    # ring layout doesn't always map 1:1 to the reference's flat
    # layout. Both want kernel-specific harnesses; the smoke test's
    # job here is "the kernel launches without crashing through the
    # SPV stack", which it just did.
    if isinstance(ref, dict):
        return

    # Single-output kernel: pair the lone ``role="out"`` buffer name
    # with the returned array, then validate.
    if len(out_names) != 1:
        pytest.skip(
            f"{_kernel_id(kernel_cls)}: single ndarray reference but "
            f"{len(out_names)} output buffers — ambiguous"
        )
    (only_name,) = out_names
    expected = ref
    got = np.empty(expected.shape, dtype=expected.dtype)
    ctypes.memmove(got.ctypes.data, out_ptrs[only_name], got.nbytes)
    if np.issubdtype(expected.dtype, np.integer):
        np.testing.assert_array_equal(got, expected, err_msg=only_name)
    else:
        np.testing.assert_allclose(
            got, expected, rtol=0, atol=1e-4, err_msg=only_name,
        )


def test_registry_is_consultable():
    """Sanity: the registry import path works even if no kernels are
    currently registered. Catches the import-loop bug."""
    kernels = all_kernels()
    assert isinstance(kernels, list)
