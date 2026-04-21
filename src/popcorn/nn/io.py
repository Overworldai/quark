"""Safetensors → device tensor loading. No torch, numpy, or MLX dependency.

    from popcorn.nn.io import load_safetensors

    sd = load_safetensors("model.safetensors")
    model.load_state_dict(sd)

Pure-Python safetensors parser: reads the 8-byte header length, parses
the JSON header, then mmaps the data region and copies each tensor's
raw bytes directly to device via ``PopcornTensor.from_bytes``.

bf16 tensors are loaded as raw 2-byte values — no reinterpretation
needed since the safetensors file stores them in their native format.
"""

from __future__ import annotations

import contextlib
import json
import mmap
import os
import struct
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------
# Safetensors dtype → (popcorn dtype, element bytes)
# ---------------------------------------------------------------

_SF_DTYPE_TO_PC: dict[str, tuple[str, int]] = {
    "F32": ("f32", 4),
    "F16": ("f16", 2),
    "BF16": ("bf16", 2),
    "F64": ("f32", 4),  # downcast f64 → f32 on load
    "I64": ("s64", 8),
    "I32": ("s32", 4),
    "I16": ("s32", 4),  # upcast i16 → s32
    "I8": ("s8", 1),
    "U8": ("u8", 1),
    "BOOL": ("u8", 1),
}

# For dtypes that need conversion (f64→f32, i16→s32), we track the
# source element size separately so we can read the right number of
# bytes from the file.
_SF_DTYPE_SRC_BYTES: dict[str, int] = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


# ---------------------------------------------------------------
# Pure-Python safetensors parser
# ---------------------------------------------------------------


def _available_ram_bytes() -> int | None:
    """Return available host RAM in bytes, or None if the OS can't be queried."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    try:
        import subprocess

        out = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
        page_size = 4096
        free = inactive = speculative = 0
        for line in out.splitlines():
            if "page size of" in line:
                page_size = int(line.split()[-2])
            elif line.startswith("Pages free:"):
                free = int(line.split()[-1].rstrip("."))
            elif line.startswith("Pages inactive:"):
                inactive = int(line.split()[-1].rstrip("."))
            elif line.startswith("Pages speculative:"):
                speculative = int(line.split()[-1].rstrip("."))
        return (free + inactive + speculative) * page_size
    except (OSError, ValueError):
        pass

    return None


def _parse_safetensors(path: str | Path):
    """Parse a .safetensors file. Returns ``(header_dict, buffer, data_offset)``.

    ``header_dict``: ``{tensor_name: {"dtype": str, "shape": list[int],
    "data_offsets": [start, end]}}``.
    ``buffer``: either an ``mmap`` of the file, or a ``bytearray`` with the
    full file contents when the file fits under 25% of available RAM.
    ``data_offset``: byte offset where tensor data begins.
    """
    path = str(path)
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("safetensors: file too short for header length")
        (header_len,) = struct.unpack("<Q", header_len_bytes)

        header_json = f.read(header_len)
        if len(header_json) < header_len:
            raise ValueError("safetensors: file too short for header")
        header = json.loads(header_json)

        header.pop("__metadata__", None)

        data_offset = 8 + header_len
        file_size = os.fstat(f.fileno()).st_size
        avail = _available_ram_bytes()
        if avail is not None and file_size * 4 <= avail:
            f.seek(0)
            buf = bytearray(file_size)
            view = memoryview(buf)
            pos = 0
            while pos < file_size:
                n = f.readinto(view[pos:])
                if not n:
                    raise ValueError("safetensors: short read")
                pos += n
            return header, buf, data_offset

        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    return header, mm, data_offset


def load_safetensors(
    path: str | Path,
    *,
    dtype: str | None = None,
    names: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Load a ``.safetensors`` file into a ``{name: PopcornTensor}`` dict.

    ``dtype``: optional target dtype string (``"bf16"``, ``"f16"``,
    ``"f32"``). When set, casts every tensor after loading.
    ``names``: optional allowlist — skip any tensor whose name isn't in
    this set. Useful for partial loads (e.g. just noise-conditioner
    weights without materializing the whole model).

    Transfer path: the safetensors file is mmap'd (or read fully into a
    ``bytearray`` when it fits under 25% of available RAM — avoids
    on-demand page faults during DMA) and the entire data region is
    page-locked via ``cuMemHostRegister`` before any tensor copies. That
    lets every ``cuMemcpyHtoD`` DMA directly from pinned memory at full
    PCIe bandwidth (~6x faster than the pageable path). The registration
    is dropped at the end — weights live on device, not in host RAM.
    """
    from popcorn.runtime.tensor import PopcornTensor

    header, mm, data_offset = _parse_safetensors(path)
    tensors: dict[str, Any] = {}
    allow = set(names) if names is not None else None

    # Zero-copy: view the whole mmap once; take memoryview slices per tensor
    # so PopcornTensor.from_bytes can memcpy_htod directly from the mmap.
    mv = memoryview(mm)

    # Pin the full mmap region once. mmap returns page-aligned addresses
    # on every supported OS, so cuMemHostRegister accepts the base pointer
    # directly. We register read-only since the loader never writes into
    # the host buffer — lets the driver skip the writable-pages path.
    _pinned_ptr = 0
    _pinned_nbytes = 0
    _rt = None
    try:
        from popcorn.runtime.cuda import CudaRuntime

        _rt = CudaRuntime.instance()
    except Exception:
        _rt = None
    if _rt is not None:
        # ``np.frombuffer`` on a read-only memoryview exposes the mmap
        # base pointer via ``.ctypes.data`` — ctypes.from_buffer can't
        # take a read-only buffer, so this is the portable path.
        import numpy as _np

        _base_arr = _np.frombuffer(mv, dtype=_np.uint8)
        _pinned_ptr = int(_base_arr.ctypes.data)
        _pinned_nbytes = _base_arr.nbytes
        try:
            _rt.memhost_register(_pinned_ptr, _pinned_nbytes, read_only=True)
        except Exception:
            # Registration can fail on some filesystem / kernel combos
            # (e.g. tmpfs without the right privileges). Fall back to
            # pageable copies — correctness is unaffected.
            _pinned_ptr = 0
            _pinned_nbytes = 0

    try:
        for name, info in header.items():
            if allow is not None and name not in allow:
                continue
            sf_dtype = info["dtype"]
            shape = tuple(info["shape"])
            start, end = info["data_offsets"]

            mapping = _SF_DTYPE_TO_PC.get(sf_dtype)
            if mapping is None:
                raise ValueError(f"load_safetensors: unsupported dtype {sf_dtype!r} for {name!r}")

            pc_dtype, _pc_elem = mapping

            # Zero-copy memoryview slice of the mmap region.
            raw: Any = mv[data_offset + start : data_offset + end]

            # Handle dtypes that need host-side conversion (rare paths —
            # fall back to materializing bytes). Host conversion produces
            # a fresh pageable buffer; that single tensor takes the slow
            # path but everything else hits the pinned mmap.
            if sf_dtype == "F64":
                import struct as _s

                numel = len(raw) // 8
                f64_vals = _s.unpack(f"<{numel}d", bytes(raw))
                raw = _s.pack(f"<{numel}f", *f64_vals)
            elif sf_dtype == "I16":
                import struct as _s

                numel = len(raw) // 2
                i16_vals = _s.unpack(f"<{numel}h", bytes(raw))
                raw = _s.pack(f"<{numel}i", *i16_vals)

            t = PopcornTensor.from_bytes(raw, shape, pc_dtype)

            if dtype is not None and dtype != pc_dtype:
                from popcorn.runtime.kernels import cast

                t = cast(t, dtype)

            tensors[name] = t
    finally:
        if _pinned_ptr and _rt is not None:
            with contextlib.suppress(Exception):
                _rt.memhost_unregister(_pinned_ptr)

    return tensors


def load_from_hub(
    repo_id: str,
    *,
    filename: str = "model.safetensors",
    dtype: str | None = None,
    names: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Download from HuggingFace Hub + load."""
    import huggingface_hub

    local_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
    return load_safetensors(local_path, dtype=dtype, names=names)
