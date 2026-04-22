"""MlxDriver — implements the launcher's Driver protocol on top of MLX.

MLX owns all host-side compile/launch plumbing via
`mx.fast.metal_kernel`. We generate MSL source text and hand it to
MLX for JIT compile + dispatch. No Objective-C, no Metal.framework
ctypes.

Pipeline:
  MslLowerer -> LoweredMslKernel (MSL body + metadata)
        |
  MlxDriver.compile(lowered, entry, smem_bytes)
        |  mx.fast.metal_kernel(name, input_names, output_names, source, ...)
        |
  MlxCompiledModule(kernel_fn, input_names, output_names, ...)
        |
  MlxDriver.launch(mod, grid, block, ...)
        |  kernel_fn(inputs=..., output_shapes=..., output_dtypes=..., grid=..., threadgroup=...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quark.device import Device, DeviceCaps, DeviceFamily
from quark.ir import DType

if TYPE_CHECKING:
    from quark.lower.msl.lower import LoweredMslKernel


@dataclass
class MlxCompiledModule:
    """Opaque MLX-side compiled handle.

    `kernel` is the callable returned by `mx.fast.metal_kernel(...)`.
    """

    kernel: Any  # mx.fast.metal_kernel callable
    input_names: list[str]
    output_names: list[str]
    scalar_names: list[str]
    smem_bytes: int


class MlxDriver:
    """Driver-protocol implementation for Apple Metal via MLX.

    Stateless beyond holding a Device instance. The same driver
    instance is reused across compile/launch calls.
    """

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
        """Populate DeviceCaps via MLX's device_info()."""
        import mlx.core as mx

        from quark.device import chip_gen_from_metal_info
        from quark.ir.mma_registry import shapes_for_chip

        info = mx.device_info()
        chip_gen = chip_gen_from_metal_info(info)
        return DeviceCaps(
            family=DeviceFamily.METAL,
            name=str(info.get("device_name", info.get("architecture", "apple-gpu"))),
            compute_unit_count=0,  # MLX doesn't expose CU count directly
            warp_size=32,  # SIMD width on Apple silicon
            max_threads_per_block=1024,
            max_smem_per_block=32 * 1024,  # 32 KiB on all current Apple silicon
            max_regs_per_thread=None,
            max_regs_per_block=None,
            arch_tag="metal3",
            compute_capability=None,
            supports_async_copy=False,
            supports_graph_capture=False,
            supports_fp8_e4m3=False,  # storage only, no MMA
            supports_bf16_mma=True,
            # MSL JIT compile-time proxy — bounded against empirical
            # "works / doesn't" ops counts per kernel on M3 Ultra:
            # gemm working configs top out ~932 ops; 2874 ops is where
            # MSL surfaces "Compiler encountered an internal error"
            # instead of a correct binary. moe_inproj has a similar
            # boundary around 2k. 1500 is a conservative ceiling that
            # keeps the competitive configs in and cuts the tail where
            # the Apple shader compiler actually chokes.
            max_ir_ops=8000,
            # MSL JIT compile time is dominated by the per-warp
            # simdgroup_matrix accumulator grid. gemm small configs
            # (mt*nt=2) finish in ~100ms; mt*nt=8 takes seconds but
            # works; mt*nt=128 crashes the compiler outright. owl_attn
            # with Python-unrolled Q-chunk loops lands at 30–48 chains
            # depending on KvTile. 32 is the empirical sweet spot —
            # lets owl_attn's smallest KvTile=16 config compile (30
            # chains) while still pruning the gemm-extreme tail.
            max_mma_accumulator_tiles=1024,
            matmul_shapes=shapes_for_chip(chip_gen),
            # Metal's `atomic_float` / `atomic_uint32` / `atomic_int32`
            # are the only atomic types; no sub-4-byte or packed atomics
            # (f16 / bf16 / fp8) on any shipping Apple GPU through M4.
            atomic_add_dtypes=frozenset({DType.F32, DType.S32, DType.U32}),
            # No vector / packed atomic ops on Metal. Kernels wanting
            # packed epilogues must fall back to the scalar atomic path.
            atomic_add_vector=frozenset(),
            supported_dtypes=frozenset(
                {"f32", "f16", "bf16", "u32", "s32", "u8", "s8", "u16", "s16"}
            ),
            cpu_features=frozenset(),
            chip_gen=chip_gen,
        )

    def compile(
        self, lowered: LoweredMslKernel, entry_name: str, smem_bytes: int
    ) -> MlxCompiledModule:
        """Compile MSL source via mx.fast.metal_kernel."""
        import mlx.core as mx

        kernel = mx.fast.metal_kernel(
            name=lowered.kernel_name,
            input_names=lowered.input_names + lowered.scalar_names,
            output_names=lowered.output_names,
            source=lowered.source,
            header=lowered.header or "",
            ensure_row_contiguous=False,
            atomic_outputs=lowered.atomic_outputs,
        )
        return MlxCompiledModule(
            kernel=kernel,
            input_names=lowered.input_names,
            output_names=lowered.output_names,
            scalar_names=lowered.scalar_names,
            smem_bytes=smem_bytes,
        )

    def launch(
        self,
        mod: MlxCompiledModule,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        buffer_ptrs: list[int],
        scalar_args: list[bytes],
        stream: Any,
    ) -> None:
        # This path is for the CUDA-style pointer-based launch. The
        # Metal path uses a different launch signature with mx.array
        # objects directly — see launch_mlx() below.
        raise NotImplementedError(
            "MlxDriver.launch: the pointer-based launch path is not supported "
            "on Metal. Use launch_mlx() with mx.array objects instead."
        )

    def launch_mlx(
        self,
        mod: MlxCompiledModule,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        input_arrays: list[Any],
        output_shapes: list[tuple[int, ...]],
        output_dtypes: list[Any],
        scalar_arrays: list[Any] | None = None,
    ) -> list[Any]:
        """Launch with mx.array objects directly.

        Returns the list of output mx.arrays allocated by MLX.
        """
        import mlx.core as mx

        all_inputs = list(input_arrays)
        if scalar_arrays:
            all_inputs.extend(scalar_arrays)

        # MLX's grid is in threads, not threadgroups. Quark's kernels
        # return grid in threadgroup counts (matching CUDA's gridDim), so
        # we multiply: total_threads = grid * block per axis.
        threads_grid = (grid[0] * block[0], grid[1] * block[1], grid[2] * block[2])
        # init_value=0 forces MLX to zero-init the output buffers it
        # allocates. Without this, kernels that read-modify-write their
        # output (atomic-scatter-add in moe_outproj; any scatter target
        # that isn't written by every thread) operate on uninitialized
        # memory and produce garbage. The overhead is a single host-side
        # fill per launch — negligible next to the kernel itself.
        outputs = mod.kernel(
            inputs=all_inputs,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            grid=threads_grid,
            threadgroup=block,
            init_value=0,
        )
        mx.eval(*outputs)
        return list(outputs)

    def stream_from_torch(self, s: Any) -> Any:
        return s  # MLX manages streams

    def current_torch_stream(self) -> int:
        return 0  # opaque; not used on Metal

    def sync(self, stream: Any) -> None:
        import mlx.core as mx

        mx.synchronize()

    def supports_capture(self) -> bool:
        return False
