"""Module, Function, Param, and the matmul shape registry.

A Module is one unit of compilation. A single kernel compiles one
Function; a megakernel compiles one Function that internally dispatches
via an IfRegion on BlockIdx.

See QUARK_IR_PROPOSAL.md §2, §5.8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .region import Region
from .types import BufferType, DType, ScalarType, Type, ValueShape
from .value import Value, ValueAllocator

if TYPE_CHECKING:
    from .op import SmemAllocOp


# ---------------------------------------------------------------------------
# Parameter attributes and declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamAttrs:
    """Static attributes on a kernel parameter.

    - `align`: known minimum pointer alignment in bytes (informational;
      lowerers may use it to emit wider vector loads).
    - `restrict`: caller promises no aliasing with other params.
    - `readonly`: buffer is only read from inside the kernel.
    """

    align: int = 0
    restrict: bool = False
    readonly: bool = False


@dataclass(eq=False)
class Param:
    """A kernel parameter. Produces a Value of its type so ops can
    consume it like any other SSA value.

    For buffer parameters the Value's shape is (U64, width=1) — the
    base pointer. GlobalTensors are built on top of this Value, not
    on top of the Param directly, so offsets compose the same way
    they do for other memory.
    """

    name: str
    type: Type
    attrs: ParamAttrs = field(default_factory=ParamAttrs)
    value: Value | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.type, (ScalarType, BufferType)):
            raise TypeError(
                f"Param.type must be ScalarType or BufferType, got {type(self.type).__name__}"
            )


# ---------------------------------------------------------------------------
# Function attributes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionAttrs:
    """Function-level attributes: launch bounds, shared-mem requirement, etc."""

    max_threads_per_block: int | None = None
    min_blocks_per_sm: int | None = None
    shared_mem_bytes: int | None = None  # explicit requirement (optional)


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Function:
    """A single kernel function.

    Owns its Value id allocator, its parameter list, and its body
    Region. The smem_allocs list is the module-scope-for-this-function
    set of SmemAllocOp results (they live at the top of the body but
    are indexed for fast lookup during lowering).
    """

    name: str
    params: list[Param] = field(default_factory=list)
    attrs: FunctionAttrs = field(default_factory=FunctionAttrs)
    body: Region = field(default_factory=Region)
    value_allocator: ValueAllocator = field(default_factory=ValueAllocator)
    smem_allocs: list[SmemAllocOp] = field(default_factory=list)

    def add_param(self, name: str, type: Type, attrs: ParamAttrs | None = None) -> Value:
        """Register a new parameter and return its SSA Value."""
        if attrs is None:
            attrs = ParamAttrs()
        if isinstance(type, ScalarType):
            shape = ValueShape(type.dtype)
        elif isinstance(type, BufferType):
            # Parameters carrying buffers are u64 base pointers at the IR level.
            shape = ValueShape(DType.U64)
        else:
            raise TypeError(f"Unsupported Param type: {type!r}")
        value = self.value_allocator.fresh(shape=shape, producer=None, name=name)
        param = Param(name=name, type=type, attrs=attrs, value=value)
        self.params.append(param)
        return value

    def fresh_value(self, shape: ValueShape, producer=None, name: str = "") -> Value:
        return self.value_allocator.fresh(shape=shape, producer=producer, name=name)


# ---------------------------------------------------------------------------
# MmaShape and the matmul shape registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MmaShape:
    """A logical matmul tile shape — backend-neutral.

    The `*_regs` counts describe per-thread register counts in the
    NVIDIA fragment layout (used by the PTX lowerer). Other backends
    ignore them and pattern-match on `(m, n, k, a_dtype, b_dtype)` via
    their own dispatch table.

    Backend-specific mnemonics / tiling strings used to live on this
    class (``ptx`` / ``hip`` / ``msl`` / ``opencl`` fields). They now
    live on the corresponding ``MmaConfig`` in
    ``quark.ir.mma_registry`` as per-backend fields named after
    ``DeviceFamily.<X>.value`` (``cuda``, ``metal``, …), grouped
    next to each backend's ``min_*`` gate. Use ``payload_for()`` to
    look them up generically. The IR no longer needs to know what
    PTX or MSL look like.
    """

    name: str
    m: int
    n: int
    k: int
    a_dtype: DType
    b_dtype: DType
    acc_dtype: DType
    a_regs: int = 0
    b_regs: int = 0
    c_regs: int = 0


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Module:
    """A compilation unit — the thing a backend lowers end-to-end."""

    name: str = ""
    functions: list[Function] = field(default_factory=list)
    kernel_shapes: dict[str, MmaShape] = field(default_factory=dict)

    def add_function(self, fn: Function) -> Function:
        self.functions.append(fn)
        return fn

    def register_shape(self, shape: MmaShape) -> None:
        if shape.name in self.kernel_shapes:
            raise ValueError(f"MmaShape {shape.name!r} already registered in module")
        self.kernel_shapes[shape.name] = shape

    def get_function(self, name: str) -> Optional[Function]:
        for fn in self.functions:
            if fn.name == name:
                return fn
        return None
