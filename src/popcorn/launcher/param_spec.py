"""ParamSpec — typed kernel-parameter description, derived from IR Function.params.

Per popcorn launcher proposal §6.2: ParamSpec replaces the ad-hoc
`extra_args` + `KernelArg` tuple-list scheme. It carries the
expected dtype + role for each kernel parameter and provides
`pack_scalars(values)` which produces the bytes blob the driver's
launch path expects.

The proposal references this class throughout §6 but never defines
it concretely; this is the smallest correct shape that satisfies the
contract.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from popcorn.ir import BufferType, DType, Function, ScalarType

# DType → struct format character used by `pack_scalars`. Mirrors the
# torch↔IR DType table from §6.2 but stays on the integer / pointer
# side that scalar params actually use.
_STRUCT_FMT: dict[DType, str] = {
    DType.U8: "B",
    DType.S8: "b",
    DType.U16: "H",
    DType.S16: "h",
    DType.U32: "I",
    DType.S32: "i",
    DType.U64: "Q",
    DType.S64: "q",
    DType.F32: "f",
    DType.F64: "d",
    DType.B16: "H",
    DType.B32: "I",
    DType.B64: "Q",
}


# ---------------------------------------------------------------------------
# Per-parameter shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferSpec:
    """Description of one buffer (pointer) parameter."""

    name: str
    dtype: DType  # element dtype the kernel expects
    readonly: bool = False
    align: int = 0  # 0 means "no minimum alignment beyond natural"


@dataclass(frozen=True)
class ScalarSpec:
    """Description of one scalar parameter."""

    name: str
    dtype: DType


# ---------------------------------------------------------------------------
# ParamSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """Typed parameter list for a compiled kernel.

    Constructed from an `IR Function.params` list — the launcher
    walks the function once at compile time and produces this. The
    runtime launch path then uses it to validate buffer dtypes and
    pack scalars into the bytes blob the driver expects.
    """

    buffers: tuple[BufferSpec, ...] = ()
    scalars: tuple[ScalarSpec, ...] = ()

    @classmethod
    def from_function(cls, fn: Function) -> ParamSpec:
        """Walk the IR function's parameter list and split into
        buffer / scalar slots in declaration order.

        Order matters: the order this method produces is the same
        order `pack_scalars` expects when given a tuple of values,
        and the same order the driver's launch passes pointers to
        the kernel."""
        buffers: list[BufferSpec] = []
        scalars: list[ScalarSpec] = []
        for p in fn.params:
            if isinstance(p.type, BufferType):
                buffers.append(
                    BufferSpec(
                        name=p.name,
                        dtype=p.type.dtype,
                        readonly=p.attrs.readonly,
                        align=p.attrs.align,
                    )
                )
            elif isinstance(p.type, ScalarType):
                scalars.append(ScalarSpec(name=p.name, dtype=p.type.dtype))
            else:
                raise TypeError(
                    f"ParamSpec.from_function: parameter {p.name!r} has unsupported type {p.type!r}"
                )
        return cls(buffers=tuple(buffers), scalars=tuple(scalars))

    def pack_scalars(self, values: tuple) -> list[bytes]:
        """Pack each Python scalar into its own bytes blob.

        Returns a list of `len(self.scalars)` bytes objects, one per
        scalar parameter, in declaration order. The driver makes one
        `cuLaunchKernel` arg cell per element so the host-side layout
        matches what PTX expects (each `.param .<dtype>` slot is read
        from a separate `void**` cell, not from a single packed blob).

        Native byte order and alignment match what nvcc emits.
        """
        if len(values) != len(self.scalars):
            raise ValueError(
                f"ParamSpec.pack_scalars: expected {len(self.scalars)} values, got {len(values)}"
            )
        out: list[bytes] = []
        for slot, val in zip(self.scalars, values, strict=False):
            fmt = _STRUCT_FMT.get(slot.dtype)
            if fmt is None:
                raise TypeError(
                    f"ParamSpec.pack_scalars: scalar {slot.name!r} has "
                    f"unsupported dtype {slot.dtype}"
                )
            out.append(struct.pack("=" + fmt, val))
        return out

    def n_buffers(self) -> int:
        return len(self.buffers)

    def n_scalars(self) -> int:
        return len(self.scalars)


# ---------------------------------------------------------------------------
# ProgramFootprint — placeholder until autotune lands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramFootprint:
    """Resource snapshot used by autotune. Bundle 3 ships a minimal
    version with just the smem byte total; later bundles expand to
    include reg counts, instruction counts, etc."""

    smem_bytes: int
    instruction_count: int = 0
    reg_total: int = 0
    reg_counts: dict[str, int] = field(default_factory=dict)
