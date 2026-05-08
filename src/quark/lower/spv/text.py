"""SPIR-V text-assembly emitter helpers.

The lowerer composes a SPIR-V module as text (the format
``spirv-as`` consumes) rather than building binary directly. Text is
human-readable, drops out of failed builds at the first error
``spirv-val`` reports, and re-uses an external assembler the SPIR-V
ecosystem already maintains. The performance cost is one
``spirv-as`` shell-out per ``compile()`` — kernel build is a one-time
cost, not on the dispatch path, so this is fine.

Switching to a direct binary emitter (in-process, no external tool
dep) becomes worthwhile when (a) the visitor count grows past ~20
and (b) build latency on a large module starts showing in profiles.
Until then the text path keeps debugging tractable: a failed
``spirv-as`` invocation prints the line number where the lowerer
emitted invalid syntax.

This file owns the bookkeeping every SPIR-V module needs but no
single visitor wants to redo:

  * SSA id allocation (``%1``, ``%2``, ...; one monotonic counter,
    Khronos convention)
  * Type-id caching (one ``%float`` no matter how many times the
    body asks for it)
  * Constant-id caching (one ``%const_uint_0`` for ``0u``, etc.)
  * Section ordering — Capabilities + Extensions before
    ExtInstImport before MemoryModel before EntryPoint before
    decorations before types before function bodies. ``spirv-as``
    enforces this, and authoring code that wants to emit a
    decoration mid-body would otherwise have to walk back to the
    decoration section by hand.

Used only by the lowerer; not part of the public ``quark`` surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpvText:
    """Section-aware buffer for SPIR-V text assembly.

    Each section is a list of lines that get joined in spec-mandated
    order at ``serialize()`` time. The lowerer's visitors call
    ``emit_<section>(...)`` from anywhere — the section bookkeeping
    + de-dupe happens here.
    """

    next_id: int = 1
    capabilities: list[str] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)
    ext_imports: dict[str, str] = field(default_factory=dict)
    memory_model: str = "OpMemoryModel Logical GLSL450"
    entry_points: list[str] = field(default_factory=list)
    execution_modes: list[str] = field(default_factory=list)
    debug: list[str] = field(default_factory=list)
    decorations: list[str] = field(default_factory=list)
    type_lines: list[str] = field(default_factory=list)
    type_cache: dict[str, str] = field(default_factory=dict)
    const_cache: dict[tuple[str, str], str] = field(default_factory=dict)
    function_lines: list[str] = field(default_factory=list)

    def alloc_id(self, hint: str = "") -> str:
        """Allocate a fresh SSA id. ``hint`` is woven into the id name
        for debuggability — ``%a_42`` reads better than ``%42`` when
        ``spirv-val`` complains about a typed mismatch.
        """
        nid = self.next_id
        self.next_id += 1
        if hint:
            # Sanitize for SPIR-V's id syntax: only letters, digits,
            # underscores. Strip everything else and prefix with the
            # cleaned hint.
            clean = "".join(c if (c.isalnum() or c == "_") else "_" for c in hint)
            return f"%{clean}_{nid}"
        return f"%v_{nid}"

    def add_capability(self, cap: str) -> None:
        line = f"OpCapability {cap}"
        if line not in self.capabilities:
            self.capabilities.append(line)

    def add_extension(self, ext: str) -> None:
        line = f'OpExtension "{ext}"'
        if line not in self.extensions:
            self.extensions.append(line)

    def import_ext_inst(self, set_name: str) -> str:
        """Import an extended instruction set (GLSL.std.450, etc.)
        and return the id bound to it. De-duped — same set_name
        always returns the same id."""
        if set_name in self.ext_imports:
            return self.ext_imports[set_name]
        ext_id = self.alloc_id(set_name.replace(".", "_"))
        self.ext_imports[set_name] = ext_id
        return ext_id

    def add_entry_point(self, fn_id: str, name: str, model: str,
                        builtin_inputs: list[str]) -> None:
        ifaces = " ".join(builtin_inputs)
        if ifaces:
            self.entry_points.append(
                f'OpEntryPoint {model} {fn_id} "{name}" {ifaces}'
            )
        else:
            self.entry_points.append(
                f'OpEntryPoint {model} {fn_id} "{name}"'
            )

    def add_execution_mode(self, line: str) -> None:
        self.execution_modes.append(line)

    def add_debug(self, line: str) -> None:
        self.debug.append(line)

    def add_decoration(self, line: str) -> None:
        # De-dupe: identical decoration strings (same target id, same
        # decoration kind, same operands) tolerate duplicates but
        # clutter the disassembly. Common cause: a struct type cached
        # on first emission gets decorated again on every later use.
        if line not in self.decorations:
            self.decorations.append(line)

    def add_type_line(self, line: str) -> None:
        self.type_lines.append(line)

    def emit_function(self, line: str) -> None:
        self.function_lines.append(line)

    # ── Type / constant de-dupe ───────────────────────────────────

    def type_void(self) -> str:
        return self._cached_type("void", "OpTypeVoid")

    def type_int(self, width: int = 32, signed: bool = False) -> str:
        key = f"int_{width}_{int(signed)}"
        return self._cached_type(key, f"OpTypeInt {width} {int(signed)}")

    def type_float(self, width: int = 32, bfloat16: bool = False) -> str:
        """Emit ``OpTypeFloat <width>`` (or ``OpTypeFloat 16 BFloat16KHR``
        when ``bfloat16`` is set). The BFloat16 variant is gated on
        ``SPV_KHR_bfloat16`` + ``BFloat16TypeKHR`` (declared by callers
        before invoking this); spirv-as ``vulkan1.4`` accepts the
        suffixed form, earlier targets reject it."""
        if bfloat16 and width == 16:
            return self._cached_type("bfloat16", "OpTypeFloat 16 BFloat16KHR")
        key = f"float_{width}"
        return self._cached_type(key, f"OpTypeFloat {width}")

    def type_bool(self) -> str:
        return self._cached_type("bool", "OpTypeBool")

    def type_vec(self, elem_id: str, width: int) -> str:
        key = f"vec_{elem_id}_{width}"
        return self._cached_type(key, f"OpTypeVector {elem_id} {width}")

    def type_runtime_array(self, elem_id: str, *, stride_bytes: int) -> str:
        # Runtime arrays must carry an ArrayStride decoration. Re-use
        # the same array id for matching (elem, stride) pairs.
        key = f"rta_{elem_id}_{stride_bytes}"
        if key in self.type_cache:
            return self.type_cache[key]
        arr_id = self.alloc_id(f"rta{stride_bytes}")
        self.type_cache[key] = arr_id
        self.add_type_line(f"{arr_id} = OpTypeRuntimeArray {elem_id}")
        self.add_decoration(f"OpDecorate {arr_id} ArrayStride {stride_bytes}")
        return arr_id

    def type_struct(self, *member_ids: str) -> str:
        key = f"struct_{'_'.join(member_ids)}"
        members = " ".join(member_ids)
        return self._cached_type(key, f"OpTypeStruct {members}")

    def type_pointer(self, storage_class: str, pointee_id: str) -> str:
        key = f"ptr_{storage_class}_{pointee_id}"
        return self._cached_type(key, f"OpTypePointer {storage_class} {pointee_id}")

    def type_function(self, ret_id: str, *param_ids: str) -> str:
        key = f"fn_{ret_id}_" + "_".join(param_ids)
        params = " ".join(param_ids)
        return self._cached_type(key, f"OpTypeFunction {ret_id} {params}".rstrip())

    def _cached_type(self, key: str, op: str) -> str:
        if key in self.type_cache:
            return self.type_cache[key]
        tid = self.alloc_id(key)
        self.type_cache[key] = tid
        self.add_type_line(f"{tid} = {op}")
        return tid

    def const_uint(self, value: int) -> str:
        key = ("uint", str(value))
        if key in self.const_cache:
            return self.const_cache[key]
        u32 = self.type_int(32, False)
        cid = self.alloc_id(f"u_{value}")
        self.const_cache[key] = cid
        self.add_type_line(f"{cid} = OpConstant {u32} {value}")
        return cid

    def const_float(self, value: float) -> str:
        # SPIR-V text constants accept decimal floats directly, but
        # using the bit pattern is safer for round-tripping.
        import struct
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        key = ("float", str(bits))
        if key in self.const_cache:
            return self.const_cache[key]
        f32 = self.type_float(32)
        cid = self.alloc_id(f"f_{bits:08x}")
        self.const_cache[key] = cid
        # `!0xPATTERN` is SPIR-V text's "raw bits" notation — exact
        # bit pattern, no parser-precision drift.
        self.add_type_line(f"{cid} = OpConstant {f32} !0x{bits:08x}")
        return cid

    def const_composite(self, type_id: str, *member_ids: str) -> str:
        members = " ".join(member_ids)
        cid = self.alloc_id("comp")
        self.add_type_line(f"{cid} = OpConstantComposite {type_id} {members}")
        return cid

    def serialize(self) -> str:
        """Render the module to spec-mandated section order."""
        lines: list[str] = []
        lines.extend(self.capabilities)
        lines.extend(self.extensions)
        for set_name, ext_id in self.ext_imports.items():
            lines.append(f'{ext_id} = OpExtInstImport "{set_name}"')
        lines.append(self.memory_model)
        lines.extend(self.entry_points)
        lines.extend(self.execution_modes)
        lines.extend(self.debug)
        lines.extend(self.decorations)
        lines.extend(self.type_lines)
        lines.extend(self.function_lines)
        return "\n".join(lines) + "\n"
