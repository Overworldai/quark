"""CudaDriver — implements the launcher's Driver protocol on top of CudaRuntime.

Per popcorn launcher proposal §5.1 (which is a stub in the proposal —
the design here is derived from §5's Driver protocol contract +
§3.1's CUDA probe field table + §4.1's CudaRuntime API).

Pipeline:
  PtxLowerer → str (PTX text)
        ↓
  CudaDriver.compile(source, entry, smem_bytes)
        ↓ cuModuleLoadData → CUmodule
        ↓ cuModuleGetFunction → CUfunction
        ↓ cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) if smem > 48KB
        ↓
  CudaCompiledModule(module, func, smem_bytes)
        ↓
  CudaDriver.launch(mod, grid, block, buffer_ptrs, scalar_args, stream)
        ↓ pack arg slots into a void** array
        ↓ cuLaunchKernel(...)
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING

from popcorn.device import Device, DeviceCaps, DeviceFamily
from popcorn.runtime.cuda import (
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    CudaRuntime,
)

if TYPE_CHECKING:
    import torch

# 48KB is the static-smem threshold above which a kernel must opt in
# via cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES). Below it, the
# attribute set is harmless but unnecessary.
_OPTIN_SMEM_THRESHOLD = 48 * 1024


# ---------------------------------------------------------------------------
# CudaCompiledModule — the handle returned by compile() and consumed by launch()
# ---------------------------------------------------------------------------


@dataclass
class CudaCompiledModule:
    """Opaque CUDA-side compiled handle.

    `module` is a CUmodule (int handle), `func` is the CUfunction
    pointer for the entry symbol, and `smem_bytes` is what the
    launcher must pass to `cuLaunchKernel` for dynamic shared memory.

    Holds backrefs to the runtime so callers don't have to thread it
    in. The module is unloaded on `close()`.
    """

    module: int
    func: int
    smem_bytes: int
    runtime: CudaRuntime

    def close(self) -> None:
        if self.module:
            self.runtime.module_unload(self.module)
            self.module = 0
            self.func = 0


# ---------------------------------------------------------------------------
# CudaDriver
# ---------------------------------------------------------------------------


class CudaDriver:
    """Driver-protocol implementation for NVIDIA CUDA via libcuda.

    Owns a `CudaRuntime` (a process-wide singleton) and a `Device`
    instance with populated `DeviceCaps`. Stateless beyond that — the
    same driver instance is reused across compile/launch calls.
    """

    def __init__(self, device: Device | None = None) -> None:
        self.runtime = CudaRuntime.instance()
        # Activate the primary context for the requested device so
        # tensor pointers from torch are valid in our launches.
        device_index = device.index if device is not None else 0
        self.runtime.retain_primary_context(device_index)
        if device is None:
            device = self._build_device(device_index)
        self.device = device

    # ---- device probing (CUDA-side construction of DeviceCaps) ----

    def _build_device(self, index: int) -> Device:
        return Device(
            family=DeviceFamily.CUDA,
            index=index,
            caps=self.probe(index),
        )

    def probe(self, index: int) -> DeviceCaps:
        """Populate DeviceCaps via cuDeviceGetAttribute calls.

        Mirrors `popcorn.device._probe_cuda_via_torch` but reads from
        libcuda directly instead of torch.cuda.get_device_properties —
        no torch import in this layer.
        """
        rt = self.runtime
        cc_major = rt.get_device_attribute(index, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        cc_minor = rt.get_device_attribute(index, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        sm_count = rt.get_device_attribute(index, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        max_threads = rt.get_device_attribute(index, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        max_smem_optin = rt.get_device_attribute(
            index, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        )
        max_regs_per_block = rt.get_device_attribute(
            index, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
        )
        try:
            max_regs_per_sm = rt.get_device_attribute(
                index, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
            )
        except Exception:
            max_regs_per_sm = None
        name = rt.get_device_name(index)
        arch_tag = f"sm_{cc_major}{cc_minor}"

        # Pull the per-arch matmul / dtype tables from popcorn.device.
        from popcorn.device import _default_dtypes_for, _default_matmul_shapes_for  # type: ignore

        return DeviceCaps(
            family=DeviceFamily.CUDA,
            name=name,
            compute_unit_count=sm_count,
            warp_size=32,
            max_threads_per_block=max_threads,
            max_smem_per_block=max_smem_optin,
            max_regs_per_thread=255,
            max_regs_per_block=max_regs_per_sm or max_regs_per_block,
            arch_tag=arch_tag,
            compute_capability=(cc_major, cc_minor),
            supports_async_copy=cc_major >= 8,
            supports_graph_capture=True,
            supports_fp8_e4m3=cc_major >= 9 or (cc_major == 8 and cc_minor == 9),
            supports_bf16_mma=cc_major >= 8,
            matmul_shapes=_default_matmul_shapes_for(arch_tag),
            supported_dtypes=_default_dtypes_for(arch_tag),
            cpu_features=frozenset(),
        )

    # ---- compile (PTX text → CUmodule + CUfunction) ----

    def compile(self, source: str, entry_name: str, smem_bytes: int) -> CudaCompiledModule:
        """Pass PTX text directly to libcuda's in-process JIT.

        No subprocess to ptxas — the driver assembles PTX itself when
        we call `cuModuleLoadData`. The compiled CUmodule is held
        until the returned `CudaCompiledModule` is `close()`d.
        """
        ptx_bytes = source.encode("ascii") if isinstance(source, str) else source
        mod_handle = self.runtime.module_load_data(ptx_bytes)
        func = self.runtime.module_get_function(mod_handle, entry_name)
        if smem_bytes > _OPTIN_SMEM_THRESHOLD:
            self.runtime.func_set_attribute(
                func,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem_bytes,
            )
        return CudaCompiledModule(
            module=mod_handle,
            func=func,
            smem_bytes=smem_bytes,
            runtime=self.runtime,
        )

    # ---- launch ----

    def launch(
        self,
        mod: CudaCompiledModule,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        buffer_ptrs: list[int],
        scalar_args: list[bytes],
        stream: int,
    ) -> None:
        """Pack buffer pointers + per-scalar bytes blobs into a
        `void**` array and call cuLaunchKernel.

        cuLaunchKernel's `kernelParams` is a `void**` where each cell
        is a `void*` pointing at the host-side storage of one kernel
        parameter. We make one `c_void_p` cell per buffer pointer and
        one `c_ubyte * N` cell per scalar bytes blob, then pass an
        array of cell addresses. The backing cells are kept alive in
        `_holds` until cuLaunchKernel returns.

        `scalar_args` is one bytes blob per scalar parameter, in the
        kernel's declared order — see `ParamSpec.pack_scalars`. PTX
        expects each `.param .<dtype>` slot to be read from its own
        cell, not from a single packed struct, so we must NOT
        concatenate them.
        """
        _holds: list = []
        arg_addrs: list[int] = []

        for ptr in buffer_ptrs:
            cell = ctypes.c_void_p(int(ptr))
            _holds.append(cell)
            arg_addrs.append(ctypes.addressof(cell))

        for blob in scalar_args:
            if not isinstance(blob, (bytes, bytearray)):
                raise TypeError(
                    f"CudaDriver.launch: scalar_args entries must be "
                    f"bytes-like, got {type(blob).__name__}"
                )
            cell = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
            _holds.append(cell)
            arg_addrs.append(ctypes.addressof(cell))

        self.runtime.launch_kernel(
            func=mod.func,
            grid=grid,
            block=block,
            args=arg_addrs,
            smem=mod.smem_bytes,
            stream=int(stream),
        )

        # cuLaunchKernel is asynchronous on the device but reads the
        # arg cells synchronously before returning, so dropping
        # _holds here is safe — the launch has already been issued.
        del _holds

    # ---- streams ----

    def stream_from_torch(self, torch_stream: torch.cuda.Stream) -> int:
        """A CUstream IS just a void* in the driver API, and torch
        exposes its underlying handle via `.cuda_stream`. Pass it
        through unchanged."""
        return int(torch_stream.cuda_stream)

    def current_torch_stream(self) -> int:
        """Convenience: return the current torch CUDA stream as a raw
        CUstream pointer. Used by `CompiledKernel.launch` when the
        caller doesn't pass an explicit stream."""
        import torch

        return self.stream_from_torch(torch.cuda.current_stream())

    def sync(self, stream: int) -> None:
        self.runtime.stream_synchronize(int(stream))

    # ---- graph capture (stubs for now — Bundle 7 wires these up) ----

    def supports_capture(self) -> bool:
        return True

    def begin_capture(self, stream: int) -> None:  # pragma: no cover
        raise NotImplementedError("graph capture: see Bundle 7 (Launcher §8)")

    def end_capture(self, stream: int):  # pragma: no cover
        raise NotImplementedError("graph capture: see Bundle 7 (Launcher §8)")

    def launch_graph(self, graph, stream: int) -> None:  # pragma: no cover
        raise NotImplementedError("graph capture: see Bundle 7 (Launcher §8)")
