"""MetalDriver — Metal backend via metal-cpp C++ extension.

Compiles MSL source and dispatches compute kernels through the
``_metal_dispatch`` C extension, which calls Metal.framework directly
via Apple's metal-cpp headers. No PyObjC, no MLX.

The harness (``metal_harness.py``) wraps the lowered body in a full
``[[kernel]] void`` signature. This module handles device probing,
compilation with pipeline caching, and batched command buffer dispatch.

Dispatch model (matches MLX):
  Multiple dispatches accumulate in one ``MTLComputeCommandEncoder``
  inside a single ``MTLCommandBuffer``. The buffer is committed when
  the ops threshold is reached, when ``sync()`` is called, or when
  output data is read. This amortizes the per-commit overhead across
  many small kernel launches.
"""

from __future__ import annotations

import ctypes
import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from quark.device import (
    Device,
    DeviceCaps,
    DeviceFamily,
    chip_gen_from_metal_info,
)
from quark.ir import DType

from . import _metal_dispatch as _md
from .metal_harness import (
    BindingLayout,
    ScalarParam,
    TensorParam,
    build_kernel_source,
)

if TYPE_CHECKING:
    from quark.lower.msl.lower import LoweredMslKernel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GPU_GEN_RE = re.compile(r"applegpu_g(\d+)")
_MTL_LANGUAGE_VERSION_3_2 = 196610  # 0x30002
_MTL_LANGUAGE_VERSION_4_0 = 262144  # 0x40000


_PC_TO_NP = {
    "f32": np.float32,
    "f16": np.float16,
    "bf16": np.uint16,
    "s32": np.int32,
    "s64": np.int64,
    "u8": np.uint8,
    "s8": np.int8,
    "u32": np.uint32,
    "u16": np.uint16,
    "e4m3": np.uint8,
    "e5m2": np.uint8,
}


# Pre-computed byte-size lookup. Saves a per-call ``np.dtype(...).itemsize``
# (numpy attribute access + type check) inside the hot
# ``_quark_tensor_as_metadata_ndarray`` path. Same coverage as ``_PC_TO_NP``.
_PC_TO_ELEM_BYTES: dict[str, int] = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "s32": 4,
    "s64": 8,
    "u8": 1,
    "s8": 1,
    "u32": 4,
    "u16": 2,
    "e4m3": 1,
    "e5m2": 1,
}


def _quark_tensor_as_metadata_ndarray(t: Any) -> np.ndarray:
    """Build a numpy view of a Metal-backed QuarkTensor's buffer.

    Skips ``to_numpy``'s ``eval_queue`` flush — the C++ side only reads
    shape/strides/ndim from this ndarray; the actual data is fetched via
    the parallel ``input_handles`` path.

    Hot path optimisation: bypass ``np.prod`` and ``np.dtype().itemsize``
    (each ~3 µs of ufunc/attribute overhead per call) for the shape
    multiply — small tuples (1-4 dims) are faster as a Python loop
    against the pre-computed byte-size map.
    """
    np_dt = _PC_TO_NP.get(t.dtype, np.uint8)
    elem_size = _PC_TO_ELEM_BYTES.get(t.dtype, 1)
    n = 1
    for s in t.shape:
        n *= int(s)
    nbytes = n * elem_size if n > 0 else 1
    storage = t._storage
    ptr = storage.ptr + getattr(t, "_offset", 0) * elem_size
    # ctypes char arrays implement the buffer protocol; ty's stubs for
    # np.frombuffer don't list it as a buffer-protocol overload.
    return np.frombuffer(  # ty: ignore[no-matching-overload]
        (ctypes.c_char * nbytes).from_address(ptr), dtype=np_dt
    ).reshape(t.shape)


# ---------------------------------------------------------------------------
# Compiled kernel handle
# ---------------------------------------------------------------------------


@dataclass
class MetalCompiledModule:
    """Compiled Metal kernel — pipeline capsule + binding layout."""

    name: str
    pipeline: Any  # PyCapsule wrapping MTL::ComputePipelineState*
    layout: BindingLayout
    source: str
    inputs: list[TensorParam]
    outputs: list[TensorParam]
    scalars: list[ScalarParam]
    smem_bytes: int
    # Pre-built binding plan: flat list of (slot_kind, slot_name, buffer_index).
    _binding_plan: list[tuple[str, str, int]] | None = None
    # Precomputed (kind_int, param_idx, buffer_idx) tuples ready to hand
    # straight to the C extension. Built at compile time — every name
    # lookup that used to live on the per-launch hot path resolves here.
    _binding_plan_int: list[tuple[int, int, int]] | None = None


# ---------------------------------------------------------------------------
# Architecture / NAX helpers
# ---------------------------------------------------------------------------


def _parse_gpu_gen(arch_name: str) -> int:
    m = _GPU_GEN_RE.search(arch_name)
    return int(m.group(1)) if m else 0


def _detect_nax(arch_name: str, supports_metal4: bool) -> bool:
    if not supports_metal4:
        return False
    gen = _parse_gpu_gen(arch_name)
    is_phone = arch_name.endswith("p")
    return gen >= (18 if is_phone else 17)


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------

_pipeline_cache: dict[tuple[str, str], MetalCompiledModule] = {}

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class MetalDriver:
    """Driver-protocol implementation for Apple Metal via metal-cpp."""

    family = DeviceFamily.METAL

    def __init__(self, device: Device | None = None) -> None:
        if device is None:
            device = Device(
                family=DeviceFamily.METAL,
                index=0,
                caps=self.probe(0),
            )
        self.device = device

    def probe(self, index: int) -> DeviceCaps:
        from quark.ir.mma_registry import shapes_for_chip

        info = _md.probe()
        name = info["name"]
        arch_name = info["architecture"]
        supports_metal4 = info["supports_metal4"]
        supports_nax = _detect_nax(arch_name, supports_metal4)

        chip_info = {"architecture": arch_name, "device_name": name}
        chip_gen = chip_gen_from_metal_info(chip_info)
        arch_tag = "metal4" if supports_metal4 else "metal3"

        # Set auto-commit threshold based on architecture.
        suffix = arch_name[-1] if arch_name else "g"
        max_ops = {"p": 20, "s": 50, "d": 50}.get(suffix, 40)
        _md.set_max_ops(max_ops)

        return DeviceCaps(
            family=DeviceFamily.METAL,
            name=name,
            compute_unit_count=0,
            subgroup_width=32,
            max_threads_per_block=1024,
            max_smem_per_block=info["max_threadgroup_memory"],
            max_regs_per_thread=None,
            max_regs_per_block=None,
            arch_tag=arch_tag,
            compute_capability=None,
            supports_async_copy=False,
            supports_graph_capture=False,
            supports_fp8_e4m3=False,
            supports_bf16_mma=True,
            max_ir_ops=8000,
            max_mma_accumulator_tiles=1024,
            matmul_shapes=shapes_for_chip(chip_gen),
            atomic_add_dtypes=frozenset({DType.F32, DType.S32, DType.U32}),
            atomic_add_vector=frozenset(),
            has_native_subgroup_reduce=True,
            supported_dtypes=frozenset(
                {"f32", "f16", "bf16", "u32", "s32", "u8", "s8", "u16", "s16"}
            ),
            cpu_features=frozenset(),
            chip_gen=chip_gen,
            supports_metal4=supports_metal4,
            supports_nax=supports_nax,
        )

    # -----------------------------------------------------------------
    # Compile
    # -----------------------------------------------------------------

    def compile(
        self, lowered: LoweredMslKernel, entry_name: str, smem_bytes: int
    ) -> MetalCompiledModule:
        scalar_set = set(lowered.scalar_names)
        inputs: list[TensorParam] = []
        for name, dtype in zip(lowered.input_names, lowered.input_dtypes, strict=False):
            if name not in scalar_set:
                inputs.append(TensorParam(name=name, dtype=dtype))
        outputs: list[TensorParam] = []
        for i, name in enumerate(lowered.output_names):
            dt = lowered.output_dtypes[i] if i < len(lowered.output_dtypes) else "float"
            outputs.append(
                TensorParam(name=name, dtype=dt, atomic=name in lowered.atomic_output_names)
            )
        scalars: list[ScalarParam] = []
        for i, name in enumerate(lowered.scalar_names):
            dt = lowered.scalar_dtypes[i] if i < len(lowered.scalar_dtypes) else "int"
            scalars.append(ScalarParam(name=name, dtype=dt))

        full_source, layout = build_kernel_source(
            name=lowered.kernel_name,
            body=lowered.source,
            inputs=inputs,
            outputs=outputs,
            scalars=scalars,
            header=lowered.header or "",
        )

        source_hash = hashlib.sha256(full_source.encode()).hexdigest()
        cache_key = (source_hash, lowered.kernel_name)
        if cache_key in _pipeline_cache:
            return _pipeline_cache[cache_key]

        lang = (
            _MTL_LANGUAGE_VERSION_4_0
            if self.device.caps.supports_metal4
            else _MTL_LANGUAGE_VERSION_3_2
        )
        pipeline = _md.compile(full_source, lowered.kernel_name, lang)

        # Pre-build the binding plan — flat list replayed at launch time.
        # Explicit type so ty doesn't infer the narrower kind-Literal and trip
        # list-invariance on the MetalCompiledModule._binding_plan assignment.
        plan: list[tuple[str, str, int]] = [(s.kind, s.name, s.buffer_index) for s in layout.slots]

        # Resolve every name → param_idx now so the launch path doesn't
        # touch a single dict per dispatch. Mirrors MLX's pattern: the
        # `Primitive` carries everything the eval pass needs.
        input_idx = {inp.name: i for i, inp in enumerate(inputs)}
        output_idx = {out.name: i for i, out in enumerate(outputs)}
        scalar_idx = {sc.name: i for i, sc in enumerate(scalars)}
        all_idx = {
            name: i
            for i, name in enumerate([inp.name for inp in inputs] + [out.name for out in outputs])
        }
        plan_int: list[tuple[int, int, int]] = []
        for kind, name, buf_idx in plan:
            if kind == "input":
                plan_int.append((0, input_idx[name], buf_idx))
            elif kind == "output":
                plan_int.append((1, output_idx[name], buf_idx))
            elif kind == "scalar":
                plan_int.append((2, scalar_idx[name], buf_idx))
            elif kind == "shape":
                plan_int.append((3, all_idx[name], buf_idx))
            elif kind == "strides":
                plan_int.append((4, all_idx[name], buf_idx))
            else:  # ndim
                plan_int.append((5, all_idx[name], buf_idx))

        mod = MetalCompiledModule(
            name=lowered.kernel_name,
            pipeline=pipeline,
            layout=layout,
            source=full_source,
            inputs=inputs,
            outputs=outputs,
            scalars=scalars,
            smem_bytes=smem_bytes,
            _binding_plan=plan,
            _binding_plan_int=plan_int,
        )
        _pipeline_cache[cache_key] = mod
        return mod

    # -----------------------------------------------------------------
    # Launch
    # -----------------------------------------------------------------

    def launch(
        self,
        mod: MetalCompiledModule,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        buffer_ptrs: list[int],
        scalar_args: list[bytes],
        stream: Any,
    ) -> None:
        raise NotImplementedError(
            "MetalDriver.launch: pointer-based launch not supported. Use launch_metal() instead."
        )

    def launch_metal(
        self,
        mod: MetalCompiledModule,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        input_arrays: list[Any],
        output_shapes: list[tuple[int, ...]],
        output_dtypes: list[Any],
        scalar_arrays: list[Any] | None = None,
        input_handles: list[int] | None = None,
        output_handles: list[int] | None = None,
    ) -> list[Any]:
        """Dispatch a compiled kernel with NumPy array / pinned inputs.

        ``input_handles`` is an optional parallel list (one per input):
        ``handle >= 0`` means the input is already a pinned buffer in
        the Metal pool (``QuarkTensor`` with ``_MetalStorage``) and is
        used zero-copy. ``handle == -1`` (the default for missing
        entries) means the corresponding ``input_arrays[i]`` is
        memcpy'd into a fresh Metal buffer via the input cache.

        The entire hot path (buffer cache, output pool, zero-fill,
        binding, encoding) runs in the C extension. Python only
        prepares the numpy metadata and the binding plan.

        Caller must call ``sync()`` before reading output arrays.
        """
        scalar_arrays = scalar_arrays or []
        if input_handles is None:
            input_handles = [-1] * len(input_arrays)

        # ``np.ascontiguousarray`` so the C extension's nb::c_contig
        # constraint is satisfied even if a caller hands us a sliced view.
        # On the hot path callers already pass contiguous arrays so this
        # is a fast no-op. Skip for handle inputs — the ndarray slot is
        # only consumed by C++ for shape/stride metadata, which we still
        # need to pull from a real ndarray. Build a numpy view of the
        # Metal buffer (shape/dtype only, no data copy, no eval_queue).
        np_inputs: list[Any] = []
        for a, h in zip(input_arrays, input_handles, strict=True):
            if h < 0:
                np_inputs.append(np.ascontiguousarray(a))
            else:
                np_inputs.append(_quark_tensor_as_metadata_ndarray(a))
        np_scalars = [np.ascontiguousarray(a) for a in scalar_arrays]

        threads_grid = (grid[0] * block[0], grid[1] * block[1], grid[2] * block[2])

        # Hot path: callers (launcher._launch_metal) already pass
        # tuples-of-ints for shapes and ``np.dtype`` instances (from the
        # cached ``_IR_DT_TO_NP_DT`` map) for dtypes. The defensive
        # ``np.dtype()`` / ``tuple(int(x) ...)`` rebuild on every call
        # was ~1.5 µs/dispatch of ufunc + tuple-construct overhead —
        # ~3 ms per forward at 2k dispatches. Trust the inputs and
        # compute nbytes directly via the byte-size lookup.
        np_out_dtypes = output_dtypes
        out_shapes = output_shapes
        out_nbytes = []
        for s, d in zip(out_shapes, np_out_dtypes, strict=True):
            n = 1
            for x in s:
                n *= x
            # Hot-path callers (launcher._launch_metal) pass ``np.dtype``
            # instances from the cached ``_IR_DT_TO_NP_DT`` map — direct
            # ``.itemsize`` lookup is O(1) field access. Tests + ad-hoc
            # callers may pass numpy scalar types (``np.float32``) or
            # dtype strings; both go through ``np.dtype(...)``'s tiny
            # interning fallback so neither hits the
            # ``getset_descriptor.itemsize`` trap.
            itemsize = d.itemsize if isinstance(d, np.dtype) else np.dtype(d).itemsize
            out_nbytes.append(max(n * itemsize, 1))

        # ndarrays cross the FFI boundary directly; C++ pulls
        # ptr/shape/strides/ndim from each via the buffer protocol.
        # Returns parallel (handles, ptrs) lists — eager outputs are
        # allocated via lazy_alloc_output so they have g_lazy_buffers
        # handles and can be wrapped as QuarkTensors with _MetalStorage.
        # ``output_handles[i] >= 0`` overrides the C++ default-pool
        # alloc — kernel writes directly into the caller-pinned buffer
        # at that handle. Used by KV cache state buffers so writes
        # survive eval_queue's pool recycle.
        provided_outs = output_handles if output_handles is not None else []
        output_handles_out, output_ptrs = _md.launch(
            mod.pipeline,
            np_inputs,
            input_handles,
            out_nbytes,
            out_shapes,
            np_scalars,
            mod._binding_plan_int,
            threads_grid,
            block,
            mod.smem_bytes,
            provided_outs,
        )
        return list(zip(output_handles_out, output_ptrs, out_nbytes, strict=True))

    # -----------------------------------------------------------------
    # Misc protocol methods
    # -----------------------------------------------------------------

    def stream_from_torch(self, s: Any) -> Any:
        return s

    def current_torch_stream(self) -> int:
        return 0

    def sync(self, stream: Any) -> None:
        """Process the queued dispatches and wait for GPU completion.

        Calls into the C extension's eval() which encodes all queued
        dispatches in a tight C++ loop, auto-commits without waiting
        (overlapping GPU execution with subsequent encoding), and waits
        once at the end. Matches MLX's eval architecture.
        """
        if _md.has_pending():
            _md.eval()

    def eval(self) -> None:
        """Commit pending dispatches (alias for sync)."""
        self.sync(None)

    def supports_capture(self) -> bool:
        return False

    # Backward compat alias.
    launch_mlx = launch_metal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_available() -> bool:
    try:
        _md.probe()
        return True
    except RuntimeError:
        return False


def _find_array(name, inputs_by_name, outputs_by_name) -> np.ndarray:
    if name in inputs_by_name:
        return inputs_by_name[name][0]
    if name in outputs_by_name:
        return outputs_by_name[name][0]
    raise KeyError(f"No array for parameter {name!r}")


def _numbered(src: str) -> str:
    return "\n".join(f"{i + 1:3d}  {line}" for i, line in enumerate(src.splitlines()))
