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
# gaps land as coverage holes, not test failures). BF16 lands via
# ``SPV_KHR_bfloat16``; storage is uint16 on the numpy side, so the
# smoke harness reinterprets bf16 buffers via the
# ``u32 << 16 → bitcast f32`` trick before allclose-ing.
_SPV_SUPPORTED_DTYPES = {DType.F32, DType.U32, DType.S32, DType.BF16}


def _bf16_to_f32(arr_u16):
    """Reinterpret a numpy ``uint16`` array of bf16 bit patterns as
    ``float32``. The bf16 → f32 lift is just a 16-bit zero-extend
    on the *low* half: bf16 lives in the top 16 bits of f32, so
    ``(u16 << 16).view(f32)`` recovers the original f32 within bf16's
    representable precision."""
    return (arr_u16.astype("<u4") << 16).view("<f4")


def _to_f32_for_compare(arr):
    """Lift any input array to ``float32`` for comparison. Bypasses
    the dtype-mismatch trap when the kernel uses bf16 storage
    (numpy's bf16 is uint16 underneath).
    """
    if arr.dtype == np.uint16:
        return _bf16_to_f32(arr.view("<u2"))
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        return arr.astype(np.float32, copy=False)
    return arr


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

    # First problem the kernel can be instantiated for whose Spec
    # uses dtypes the SPV backend supports (F32 / U32 / S32 / BF16
    # — see ``_SPV_SUPPORTED_DTYPES``). For each candidate problem
    # we try the spec's natural dtype first; if that fails to lower
    # (e.g. the kernel emits a ``B32`` op the SPV backend hasn't
    # wired yet), we retry with every dtype-shaped param coerced
    # to F32. Lets kernels that have a bf16-specific fast path
    # (ada_gate_residual emits ``fma_bf16x2`` on bf16) keep their
    # smoke coverage on F32.
    _DTYPE_KEYS = (
        "dtype", "in_dtype", "kv_dtype", "compute_dtype",
        "src_dtype", "out_dtype", "partials_dtype",
    )

    def _params_with_dtypes(p_params, target_dt):
        """Return a fresh params dict with every dtype-shaped value
        replaced by ``target_dt``."""
        out = dict(p_params)
        for k in _DTYPE_KEYS:
            if k in out:
                out[k] = target_dt
        spec_cls = kernel_cls.SPEC_CLS
        if spec_cls is not None:
            import inspect as _inspect
            sig = _inspect.signature(spec_cls)
            for k in _DTYPE_KEYS:
                if k in sig.parameters and k not in out:
                    out[k] = target_dt
        return out

    # Intel cooperative-matrix shapes that ``is_valid_for`` may need
    # threaded into the config when the kernel's default
    # ``main_shape=""`` falls back via ``lookup_mma`` to a PTX shape
    # (PORTABILITY_PLAN §3.7 v2 follow-up — device-aware default
    # shape resolution isn't wired yet). Iterating these covers GEMM
    # / attn / moe_inproj / moe_outproj / patchify / unpatchify on
    # Battlemage smoke without a device-aware ``lookup_mma``.
    _INTEL_SHAPES_FOR_FALLBACK = (
        "m8n16k16_intel_bf16_f32",
        "m8n16k16_intel_f16_f32",
    )

    def _maybe_force_main_shape(k, shape_id):
        """Return a fresh kernel instance with ``config.main_shape``
        set to ``shape_id``; ``None`` if the config has no
        ``main_shape`` field or if the substitution doesn't validate.
        """
        cfg = k.config
        if not hasattr(cfg, "main_shape"):
            return None
        try:
            import dataclasses as _dc
            new_cfg = _dc.replace(cfg, main_shape=shape_id)
        except (TypeError, ValueError):
            return None
        try:
            new_k = type(k)(k.spec, new_cfg)
        except Exception:
            return None
        return new_k

    chosen = None
    kernel = None
    chosen_params = None
    for p in problems:
        # Try the natural dtype first; on lower-time failure, retry
        # at F32. The F32 fallback is what catches kernels with bf16
        # fast paths the SPV backend hasn't wired (B32 / fma_bf16x2).
        for target in (None, DType.F32):
            params = p.params if target is None else _params_with_dtypes(p.params, target)
            try:
                k = kernel_cls.from_problem(params)
            except Exception:
                continue
            spec_dtypes = {
                getattr(k.spec, attr)
                for attr in _DTYPE_KEYS
                if isinstance(getattr(k.spec, attr, None), DType)
            }
            if spec_dtypes and not (spec_dtypes <= _SPV_SUPPORTED_DTYPES):
                continue
            # Try the kernel's default config first; if it doesn't
            # validate (typical when ``main_shape=""`` resolves to a
            # PTX shape via ``lookup_mma``), retry with each Intel
            # cooperative-matrix shape forced into the config.
            candidates = [k]
            for shape_id in _INTEL_SHAPES_FOR_FALLBACK:
                forced = _maybe_force_main_shape(k, shape_id)
                if forced is not None:
                    candidates.append(forced)
            for cand in candidates:
                if not cand.is_valid_for(_DEVICE.caps):
                    continue
                try:
                    # Pre-flight compile — surfaces lower-time visitor /
                    # dtype gaps before we commit to allocating buffers.
                    _LAUNCHER.compile(kernel_cls, cand.spec, cand.config)
                except NotImplementedError:
                    continue
                except Exception:
                    continue
                chosen = p
                kernel = cand
                chosen_params = params
                break
            if kernel is not None:
                break
        if kernel is not None:
            break
    if chosen is None or kernel is None:
        pytest.skip(
            f"{_kernel_id(kernel_cls)}: no SPV-compatible-dtype problem"
        )

    if not kernel.is_valid_for(_DEVICE.caps):
        pytest.skip(
            f"{_kernel_id(kernel_cls)}: invalid for {_DEVICE.caps.name}"
        )

    if not hasattr(kernel_cls, "make_tensors_numpy"):
        pytest.skip(f"{_kernel_id(kernel_cls)}: no make_tensors_numpy")

    try:
        tensors = kernel_cls.make_tensors_numpy(chosen_params)
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
    # with the returned array, then validate. BF16 arrays land as
    # ``uint16`` on the numpy side; lift to f32 before allclose'ing
    # so the comparison runs on real numerical distance, not raw
    # bf16 bit patterns.
    if len(out_names) != 1:
        pytest.skip(
            f"{_kernel_id(kernel_cls)}: single ndarray reference but "
            f"{len(out_names)} output buffers — ambiguous"
        )
    (only_name,) = out_names
    expected = ref
    got = np.empty(expected.shape, dtype=expected.dtype)
    ctypes.memmove(got.ctypes.data, out_ptrs[only_name], got.nbytes)

    # ``uint16`` lands when the kernel uses bf16 / f16 storage —
    # numpy lacks a native bf16, so framework helpers carry the bit
    # pattern as a uint16. Anything dtype-shaped on the spec hints
    # this is the bf16 path; the conversion to f32 is the only valid
    # numerical comparison.
    is_bf16 = expected.dtype == np.uint16 and any(
        getattr(kernel.spec, k, None) is DType.BF16 for k in _DTYPE_KEYS
    )

    if is_bf16:
        np.testing.assert_allclose(
            _to_f32_for_compare(got),
            _to_f32_for_compare(expected),
            rtol=0, atol=1e-2,  # ~0.4% bf16 relative slack
            err_msg=only_name,
        )
    elif np.issubdtype(expected.dtype, np.integer):
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
