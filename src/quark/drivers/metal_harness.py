"""MSL kernel-signature generator.

Our MSL lowerer emits a **body string** (statements that belong inside
``[[kernel]] void <name>(...) { ... }``), not a full function. This
module synthesizes the full ``[[kernel]]`` signature by scanning the
body for referenced parameters and Metal built-ins.

This file is pure Python — no pyobjc, no Metal, unit-testable on any
platform. The driver (``metal.py``) consumes its output.

Algorithm (matching Metal's standard kernel signature conventions):

  1. Emit optional ``header`` verbatim (typedefs, helpers).
  2. Emit ``[[kernel]] void <name>(``.
  3. For each input:
     - Emit ``const device T* <name> [[buffer(i)]]`` — T from dtype.
     - If body contains ``<name>_shape``:   inject ``const constant int* <name>_shape [[buffer(i+1)]]``.
     - If body contains ``<name>_strides``: inject ``const constant int64_t* <name>_strides [[buffer(i+2)]]``.
     - If body contains ``<name>_ndim``:    inject ``const constant int& <name>_ndim [[buffer(i+3)]]``.
     - Each injection consumes one buffer slot.
  4. For each output:
     - Emit ``device T* <name>`` (or ``device atomic<T>* <name>`` if atomic).
     - Same shape/strides/ndim injection.
  5. For each scalar:
     - Emit ``const device T* <name> [[buffer(i)]]`` (MLX convention).
  6. Scan body for each of MLX's 20 thread-position built-ins; emit
     ``<dtype> <attr> [[<attr>]]`` for matches.
  7. Close ``) { <body> }``.

The buffer-index assignment scheme is what the driver uses to know
where to bind each argument at launch time — see the
``BindingLayout`` dataclass returned alongside the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# MSL dtype strings (what appears in the generated signature).
MslDType = str


@dataclass(frozen=True)
class TensorParam:
    """One tensor input or output."""

    name: str
    dtype: MslDType
    atomic: bool = False


@dataclass(frozen=True)
class ScalarParam:
    """One scalar (non-tensor) arg."""

    name: str
    dtype: MslDType


# The 20 Metal built-in thread-position attributes, each paired with
# its MSL dtype.
_THREAD_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("dispatch_quadgroups_per_threadgroup", "uint"),
    ("dispatch_simdgroups_per_threadgroup", "uint"),
    ("dispatch_threads_per_threadgroup", "uint3"),
    ("grid_origin", "uint3"),
    ("grid_size", "uint3"),
    ("quadgroup_index_in_threadgroup", "uint"),
    ("quadgroups_per_threadgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"),
    ("simdgroups_per_threadgroup", "uint"),
    ("thread_execution_width", "uint"),
    ("thread_index_in_quadgroup", "uint"),
    ("thread_index_in_simdgroup", "uint"),
    ("thread_index_in_threadgroup", "uint"),
    ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"),
    ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"),
    ("threads_per_grid", "uint3"),
    ("threads_per_simdgroup", "uint"),
    ("threads_per_threadgroup", "uint3"),
)


# ---------------------------------------------------------------------------
# Binding layout — what the driver needs to know to bind args at launch time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BindingSlot:
    """One argument binding slot."""

    buffer_index: int
    kind: Literal["input", "output", "scalar", "shape", "strides", "ndim"]
    name: str


@dataclass
class BindingLayout:
    """Describes where each argument binds at launch time."""

    slots: list[BindingSlot] = field(default_factory=list)

    def append(self, kind: str, name: str) -> int:
        """Allocate the next buffer index and record the slot."""
        idx = len(self.slots)
        self.slots.append(
            BindingSlot(
                buffer_index=idx,
                kind=kind,  # ty: ignore[invalid-argument-type]
                name=name,
            )
        )
        return idx


# ---------------------------------------------------------------------------
# Signature generator
# ---------------------------------------------------------------------------


def build_kernel_source(
    *,
    name: str,
    body: str,
    inputs: list[TensorParam],
    outputs: list[TensorParam],
    scalars: list[ScalarParam] | None = None,
    header: str = "",
    max_threads_per_threadgroup: int | None = None,
) -> tuple[str, BindingLayout]:
    """Wrap ``body`` in a full ``[[kernel]] void <name>(...)`` declaration.

    Replicates MLX's ``write_signature`` exactly — same buffer-index
    assignment, same shape/strides/ndim auto-injection, same thread-
    attribute scanning.

    ``max_threads_per_threadgroup`` adds the
    ``[[max_total_threads_per_threadgroup(N)]]`` attribute. Tells the
    Apple compiler how many threads run per group so it can pick a
    per-thread register allocation matched to that count — without
    the hint it assumes up to 1024 and allocates fewer registers per
    thread, causing spills. Set this for kernels that statically know
    their threadgroup size (e.g. NAX attn = 32 = single simdgroup).

    Returns ``(full_source, layout)``.
    """
    scalars = scalars or []
    layout = BindingLayout()

    parts: list[str] = []

    if header:
        parts.append(header.rstrip())
        parts.append("")

    if max_threads_per_threadgroup is not None:
        kernel_attrs = (
            f"[[kernel, max_total_threads_per_threadgroup({max_threads_per_threadgroup})]]"
        )
    else:
        kernel_attrs = "[[kernel]]"
    sig_lines: list[str] = [f"{kernel_attrs} void {name}("]

    for inp in inputs:
        sig_lines.extend(_emit_tensor_param(inp, body, layout, is_output=False))

    for out in outputs:
        sig_lines.extend(_emit_tensor_param(out, body, layout, is_output=True))

    for sc in scalars:
        idx = layout.append("scalar", sc.name)
        sig_lines.append(f"    const device {sc.dtype}* {sc.name} [[buffer({idx})]],")

    thread_attrs_used = [(attr, dtype) for attr, dtype in _THREAD_ATTRIBUTES if attr in body]
    for attr, dtype in thread_attrs_used:
        sig_lines.append(f"    {dtype} {attr} [[{attr}]],")

    if sig_lines[-1].endswith(","):
        sig_lines[-1] = sig_lines[-1].rstrip(",")
    sig_lines.append(") {")

    parts.append("\n".join(sig_lines))
    parts.append(body.strip("\n"))
    parts.append("}")

    return "\n".join(parts), layout


def _emit_tensor_param(
    p: TensorParam,
    body: str,
    layout: BindingLayout,
    *,
    is_output: bool,
) -> list[str]:
    """Emit a tensor param + its optional shape/strides/ndim injections."""
    lines: list[str] = []
    kind: Literal["input", "output"] = "output" if is_output else "input"

    ptr_qual = "device" if is_output else "const device"
    elem_type = f"atomic<{p.dtype}>" if (is_output and p.atomic) else p.dtype
    idx = layout.append(kind, p.name)
    lines.append(f"    {ptr_qual} {elem_type}* {p.name} [[buffer({idx})]],")

    if f"{p.name}_shape" in body:
        si = layout.append("shape", p.name)
        lines.append(f"    const constant int* {p.name}_shape [[buffer({si})]],")
    if f"{p.name}_strides" in body:
        si = layout.append("strides", p.name)
        lines.append(f"    const constant int64_t* {p.name}_strides [[buffer({si})]],")
    if f"{p.name}_ndim" in body:
        si = layout.append("ndim", p.name)
        lines.append(f"    const constant int& {p.name}_ndim [[buffer({si})]],")
    return lines
