"""Human-readable IR printer.

Used by tests and debugging. The format is not a wire format — it's
for eyeballing IR, writing golden-file tests, and surfacing validator
errors with context. It shows each Op on one line followed by any
nested regions indented two spaces.

Example output for a tiny function:

    func @demo(x: buffer<bf16>) {
      %bid_x:u32 = block_idx {dim=x}
      %0:bf16 = const {dtype=bf16, value=0.0}
      yield
    }

See QUARK_IR_PROPOSAL.md §9.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

from .module import Function, Module, Param
from .op import (
    Op,
)
from .region import Region
from .tensor import FragTensor, GlobalTensor, SharedRegion, Tensor
from .types import DType
from .value import Value

_INDENT = "  "


def print_module(module: Module) -> str:
    p = _Printer()
    p.visit_module(module)
    return p.out.getvalue()


def print_function(fn: Function) -> str:
    p = _Printer()
    p.visit_function(fn, depth=0)
    return p.out.getvalue()


def print_op(op: Op, depth: int = 0) -> str:
    p = _Printer()
    p.visit_op(op, depth)
    return p.out.getvalue().rstrip("\n")


class _Printer:
    def __init__(self) -> None:
        self.out = StringIO()

    # ---- top-level ----

    def visit_module(self, module: Module) -> None:
        if module.name:
            self.out.write(f"module @{module.name} {{\n")
        else:
            self.out.write("module {\n")
        if module.kernel_shapes:
            for name in sorted(module.kernel_shapes):
                s = module.kernel_shapes[name]
                self.out.write(
                    _INDENT + f"mma_shape @{name} "
                    f"{{m={s.m}, n={s.n}, k={s.k}, "
                    f"a={s.a_dtype.value}, b={s.b_dtype.value}, "
                    f"acc={s.acc_dtype.value}}}\n"
                )
        for fn in module.functions:
            self.visit_function(fn, depth=1)
        self.out.write("}\n")

    def visit_function(self, fn: Function, depth: int) -> None:
        params = ", ".join(self._format_param(p) for p in fn.params)
        self.out.write(_INDENT * depth + f"func @{fn.name}({params}) {{\n")
        for op in fn.body.ops:
            self.visit_op(op, depth + 1)
        self.out.write(_INDENT * depth + "}\n")

    # ---- ops ----

    def visit_op(self, op: Op, depth: int) -> None:
        indent = _INDENT * depth
        line = indent + self._format_op_head(op)
        self.out.write(line + "\n")
        for region in op.regions:
            self.visit_region(region, depth + 1)

    def visit_region(self, region: Region, depth: int) -> None:
        indent = _INDENT * depth
        self.out.write(indent + "{\n")
        for op in region.ops:
            self.visit_op(op, depth + 1)
        self.out.write(indent + "}\n")

    # ---- formatting helpers ----

    def _format_param(self, p: Param) -> str:
        return f"{p.name}: {p.type!r}"

    def _format_result_list(self, op: Op) -> str:
        if not op.results:
            return ""
        return ", ".join(self._format_value(v) for v in op.results) + " = "

    def _format_op_head(self, op: Op) -> str:
        results = self._format_result_list(op)
        operands = ", ".join(self._format_value_ref(v) for v in op.operands)
        attrs = self._format_attrs(op.attrs)
        kind = op.KIND
        if operands and attrs:
            body = f"{kind} {operands} {attrs}"
        elif operands:
            body = f"{kind} {operands}"
        elif attrs:
            body = f"{kind} {attrs}"
        else:
            body = kind
        return f"{results}{body}"

    def _format_value(self, v: Value) -> str:
        tag = f"%{v.name}" if v.name else f"%{v.id}"
        return f"{tag}:{v.shape!r}"

    def _format_value_ref(self, v: Value) -> str:
        tag = f"%{v.name}" if v.name else f"%{v.id}"
        return tag

    def _format_attrs(self, attrs: dict[str, Any]) -> str:
        if not attrs:
            return ""
        parts: list[str] = []
        for k in sorted(attrs):
            v = attrs[k]
            parts.append(f"{k}={_format_attr_value(v)}")
        return "{" + ", ".join(parts) + "}"


def _format_attr_value(v: Any) -> str:
    if isinstance(v, DType):
        return v.value
    if isinstance(v, Value):
        return f"%{v.name}" if v.name else f"%{v.id}"
    if isinstance(v, SharedRegion):
        return f"smem<{v.name}:{v.dtype.value}{list(v.shape)}>"
    if isinstance(v, GlobalTensor):
        sro = v.static_row_offset
        sco = v.static_col_offset
        base = f"gmem<{v.name}:{v.dtype.value}{list(v.shape)}"
        if sro or sco:
            base += f"+({sro},{sco})"
        return base + ">"
    if isinstance(v, FragTensor):
        return f"frag<{v.shape_id}:{v.which}>"
    if isinstance(v, Tensor):
        return f"tensor<{v.dtype.value}{list(v.shape)}>"
    if isinstance(v, tuple):
        return "(" + ", ".join(_format_attr_value(x) for x in v) + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_format_attr_value(x) for x in v) + "]"
    if v is None:
        return "none"
    if isinstance(v, str):
        return repr(v)
    return repr(v)
