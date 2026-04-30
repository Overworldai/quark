"""Safetensors → device tensor loading. No torch, numpy, or MLX dependency.

    from quark.nn.io import load_safetensors

    sd = load_safetensors("model.safetensors")
    model.load_state_dict(sd)

Parses the safetensors header, then streams the file's data region
through a **small** double-buffered pinned host ring into a single
pooled device allocation via async ``cuMemcpyHtoDAsync`` on a
dedicated loader stream. Disk reads overlap with DMA, pinned host
memory stays tiny (≤ 128 MB regardless of file size), and all
tensors end up as byte-offset views into the shared pool — no
per-tensor ``cuMemAlloc`` or sync memcpy.

F64 and I16 need host-side down/upcast so they fall off the pooled
path and take a per-tensor slow path (rare).

Set ``QUARK_IO_PROFILE=1`` to print a per-phase wall-clock breakdown.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import time
from pathlib import Path
from typing import Any

# Chunk + slot sizing for the pipelined loader. 64 MB slots × 2 slots =
# 128 MB pinned host RAM. Large enough that per-chunk driver overhead
# is negligible vs. bandwidth; small enough that cuMemHostAlloc is a
# tens-of-ms operation, not seconds.
_CHUNK_BYTES = 64 * 1024 * 1024
_N_SLOTS = 2

# Pool-slot alignment. Matches cuMemAlloc's natural alignment so every
# tensor view has a base pointer that any kernel's vector loads
# (up to 16-byte uint4 / 32-byte fp8 packs) can assume is aligned —
# safetensors files don't pad between tensors, so pool offsets
# mirrored from file offsets would otherwise drift off-boundary.
_POOL_ALIGN = 256

# ---------------------------------------------------------------
# Safetensors dtype → (quark dtype, element bytes)
# ---------------------------------------------------------------

_SF_DTYPE_TO_PC: dict[str, tuple[str, int]] = {
    "F32": ("f32", 4),
    "F16": ("f16", 2),
    "BF16": ("bf16", 2),
    "F64": ("f32", 4),  # downcast f64 → f32 on load (host-side)
    "I64": ("s64", 8),
    "I32": ("s32", 4),
    "I16": ("s32", 4),  # upcast i16 → s32 on load (host-side)
    "I8": ("s8", 1),
    "U8": ("u8", 1),
    "BOOL": ("u8", 1),
}

# Dtypes whose on-disk byte count differs from the target quark dtype
# (host-side conversion required — can't live in the shared pool).
_CONVERT_DTYPES = frozenset({"F64", "I16"})


class _PooledSliceStorage:
    """Byte-offset slice of a parent ``_CudaStorage``. Retain/release
    forward 1:1 to the parent so its ``cuMemFree`` fires exactly when
    the last slice is released. Lets many tensors share one pool at
    arbitrary byte offsets (pair with ``QuarkTensor(..., offset=0)``).
    """

    __slots__ = ("_parent", "nbytes", "ptr")

    def __init__(self, parent, byte_offset: int, nbytes: int):
        parent.retain()
        self.ptr = parent.ptr + byte_offset
        self.nbytes = nbytes
        self._parent = parent

    def retain(self):
        self._parent.retain()
        return self

    def release(self) -> None:
        self._parent.release()


# ---------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------


def _parse_header(path: str) -> tuple[dict, int]:
    """Read the 8-byte length prefix + JSON header from ``path``.

    Returns ``(header_dict, data_offset)`` where ``data_offset`` is
    the byte offset in the file where tensor data begins.
    """
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
    return header, 8 + header_len


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------


def load_safetensors(
    path: str | Path,
    *,
    dtype: str | None = None,
    names: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Load a ``.safetensors`` file into a ``{name: QuarkTensor}`` dict.

    ``dtype``: optional target dtype. When set, tensors whose source
    dtype differs are routed through the per-tensor convert path —
    upload to a small per-tensor source buffer, GPU cast into the
    final target buffer, release the source. The shared pool only
    allocates for tensors already at ``dtype``. This is critical for
    e.g. f32 checkpoints loaded as bf16: peak GPU memory stays at
    ``one_source_tensor + running_bf16_total`` instead of a 17 GB f32
    pool plus a parallel 8.5 GB bf16 cast output (which OOM'd on
    32 GB cards).
    ``names``: optional allowlist — tensors whose names aren't in this
    set are skipped.

    Fast path: one ``cuMemAlloc`` for the data region on device + a
    small ``cuMemHostAlloc`` ring (≤ 128 MB) on host. The loader
    double-buffers between two pinned slots, overlapping disk reads
    with async HtoD on a dedicated stream. The pool's ``cuMemFree``
    fires once the last tensor referencing it is dropped.

    Per-tensor convert path: F64 / I16 (on-disk width != any pc-dtype)
    and any source-dtype-mismatched tensor when ``dtype`` is set. Each
    tensor gets its own cuMemAlloc + memcpy_htod, optionally followed
    by a GPU cast into the target dtype.
    """
    from quark.runtime.cuda import CudaRuntime
    from quark.runtime.tensor import (
        QuarkTensor,
        _contiguous_strides,
        _CudaStorage,
    )

    profile = os.environ.get("QUARK_IO_PROFILE") == "1"
    phase_t: dict[str, float] = {}

    def _tick(label: str, t0: float) -> float:
        t1 = time.perf_counter()
        if profile:
            phase_t[label] = phase_t.get(label, 0.0) + (t1 - t0)
        return t1

    t0 = time.perf_counter()
    path_str = str(path)
    header, data_offset = _parse_header(path_str)
    allow = set(names) if names is not None else None
    rt = CudaRuntime.instance()
    t0 = _tick("parse_header", t0)

    # Triage every requested tensor into pooled vs. per-tensor convert.
    # Convert path covers two cases:
    #   1. Source dtype's on-disk byte width != target's (F64, I16) —
    #      always a host-side struct conversion regardless of caller's
    #      ``dtype=``.
    #   2. Caller asked for ``dtype=`` and source pc-dtype doesn't match
    #      (e.g. an F32 checkpoint loaded with ``dtype="bf16"``). For
    #      these we'd otherwise have to allocate the full source pool
    #      AND a separate post-load cast output, doubling peak device
    #      memory. Routing them to the per-tensor convert path keeps
    #      peak at one tensor's source size + the running target-dtype
    #      total.
    pooled: list = []
    convert: list = []
    for name, info in header.items():
        if allow is not None and name not in allow:
            continue
        sf_dtype = info["dtype"]
        mapping = _SF_DTYPE_TO_PC.get(sf_dtype)
        if mapping is None:
            raise ValueError(f"load_safetensors: unsupported dtype {sf_dtype!r} for {name!r}")
        pc_dtype, _ = mapping
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        needs_dtype_change = dtype is not None and pc_dtype != dtype
        if sf_dtype in _CONVERT_DTYPES or needs_dtype_change:
            # Final pc_dtype after host-side conversion is the caller's
            # target when one is set, else the natural pc_dtype.
            target_pc = dtype if needs_dtype_change else pc_dtype
            convert.append((name, sf_dtype, target_pc, shape, start, end - start))
        else:
            pooled.append((name, pc_dtype, shape, start, end - start))

    tensors: dict[str, Any] = {}

    # ── Pooled fast path ────────────────────────────────────────
    if pooled:
        # Compact the pool: each tensor gets an aligned slot. Safetensors
        # files pack tensors back-to-back with no padding, so mirroring
        # file offsets directly would drop tensors off 16-byte boundaries
        # and break vector-load kernels. Sort by file offset so the
        # disk stream hits tensors in order.
        pooled.sort(key=lambda e: e[3])
        pool_entries: list[tuple[str, str, tuple[int, ...], int, int, int]] = []
        pool_cursor = 0
        for name, pc_dtype, shape, file_start, nbytes in pooled:
            pool_entries.append((name, pc_dtype, shape, file_start, nbytes, pool_cursor))
            pool_cursor = (pool_cursor + nbytes + _POOL_ALIGN - 1) & ~(_POOL_ALIGN - 1)
        pool_nbytes = pool_cursor
        pool_min_file = pool_entries[0][3]
        pool_max_file = pool_entries[-1][3] + pool_entries[-1][4]
        file_span = pool_max_file - pool_min_file

        t0 = _tick("triage+layout", t0)

        # Small pinned ring — one alloc regardless of file size.
        ring_nbytes = _CHUNK_BYTES * _N_SLOTS
        host_ring = rt.memhost_alloc(ring_nbytes)
        t0 = _tick("memhost_alloc (ring)", t0)

        # Device pool sized to the aligned layout.
        pool_storage = _CudaStorage.alloc(pool_nbytes)
        t0 = _tick("device_alloc (pool)", t0)

        # Dedicated loader stream + per-slot events (so we only wait
        # on the slot we're about to overwrite, not the whole stream).
        loader_stream = rt.stream_create()
        slot_events = [rt.event_create() for _ in range(_N_SLOTS)]
        t0 = _tick("stream+events create", t0)

        try:
            # View the ring as writable bytes for readinto.
            ring_buf = (ctypes.c_ubyte * ring_nbytes).from_address(host_ring)
            ring_view = memoryview(ring_buf)

            with open(path_str, "rb") as f:
                f.seek(data_offset + pool_min_file)
                cursor = pool_min_file  # current file offset
                tensor_idx = 0  # index into pool_entries
                chunk_idx = 0
                while cursor < pool_max_file:
                    slot = chunk_idx % _N_SLOTS
                    slot_off = slot * _CHUNK_BYTES
                    n = min(_CHUNK_BYTES, pool_max_file - cursor)

                    # Wait for the prior DMA using THIS slot to finish
                    # before we overwrite it.
                    if chunk_idx >= _N_SLOTS:
                        rt.event_synchronize(slot_events[slot])

                    # Fill this slot from disk.
                    read_view = ring_view[slot_off : slot_off + n]
                    got = 0
                    while got < n:
                        r = f.readinto(read_view[got:])
                        if not r:
                            raise ValueError("load_safetensors: short read on data region")
                        got += r

                    # Dispatch async DMAs for every tensor whose file
                    # range intersects this chunk. Tensors that span the
                    # chunk boundary get one DMA per chunk they touch;
                    # the event on each slot covers all DMAs issued
                    # while that slot held its data, so the ring is
                    # safe to overwrite once the event fires.
                    chunk_fstart = cursor
                    chunk_fend = cursor + n
                    while tensor_idx < len(pool_entries):
                        _tn, _td, _ts, tfstart, tnbytes, tpool_off = pool_entries[tensor_idx]
                        tfend = tfstart + tnbytes
                        if tfstart >= chunk_fend:
                            break
                        istart = max(tfstart, chunk_fstart)
                        iend = min(tfend, chunk_fend)
                        inbytes = iend - istart
                        if inbytes > 0:
                            rt.memcpy_htod_async(
                                pool_storage.ptr + tpool_off + (istart - tfstart),
                                host_ring + slot_off + (istart - chunk_fstart),
                                inbytes,
                                loader_stream,
                            )
                        if tfend <= chunk_fend:
                            tensor_idx += 1
                        else:
                            break  # tensor continues into next chunk

                    rt.event_record(slot_events[slot], loader_stream)
                    cursor += n
                    chunk_idx += 1

            rt.stream_synchronize(loader_stream)
        finally:
            for ev in slot_events:
                rt.event_destroy(ev)
            rt.stream_destroy(loader_stream)
            rt.memhost_free(host_ring)

        t0 = _tick(f"stream load ({file_span / 1e9:.2f} GB, pipelined)", t0)

        # Build tensor views from aligned pool offsets.
        for name, pc_dtype, shape, _file_start, nbytes, pool_off in pool_entries:
            slice_storage = _PooledSliceStorage(pool_storage, pool_off, nbytes)
            tensors[name] = QuarkTensor(
                slice_storage, shape, _contiguous_strides(shape), 0, pc_dtype
            )

        # Hand our "creation" refcount on pool_storage over to the
        # slices — each slice retained in its __init__, so refcount is
        # now 1 + N. Release once so the slices own exactly N.
        pool_storage.release()
        t0 = _tick("build views", t0)

    # ── Per-tensor convert path ─────────────────────────────────
    # Two flavors mixed in this loop:
    #
    #   A. Source on-disk byte width != target pc-dtype (F64, I16).
    #      Host-side struct conversion to the natural pc-dtype, then
    #      per-tensor cuMemAlloc + memcpy_htod via from_bytes.
    #
    #   B. Caller asked for ``dtype=`` and source pc-dtype doesn't
    #      match (e.g. F32 checkpoint loaded as bf16). Upload the
    #      *source* tensor to a small per-tensor device buffer, run
    #      the GPU cast kernel into the final target-dtype tensor,
    #      release the source. Peak device memory stays at one
    #      source tensor + running target total — no full-source pool.
    if convert:
        import mmap as _mmap

        from quark.runtime.kernels import cast as _device_cast

        with open(path_str, "rb") as f:
            mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
        mv = memoryview(mm)
        try:
            for name, sf_dtype, pc_dtype, shape, start, src_nbytes in convert:
                src_pc, _ = _SF_DTYPE_TO_PC[sf_dtype]
                if sf_dtype == "F64":
                    raw = bytes(mv[data_offset + start : data_offset + start + src_nbytes])
                    numel = src_nbytes // 8
                    vals = struct.unpack(f"<{numel}d", raw)
                    raw = struct.pack(f"<{numel}f", *vals)
                    tensors[name] = QuarkTensor.from_bytes(raw, shape, src_pc)
                elif sf_dtype == "I16":
                    raw = bytes(mv[data_offset + start : data_offset + start + src_nbytes])
                    numel = src_nbytes // 2
                    vals = struct.unpack(f"<{numel}h", raw)
                    raw = struct.pack(f"<{numel}i", *vals)
                    tensors[name] = QuarkTensor.from_bytes(raw, shape, src_pc)
                else:
                    # Same on-disk width as src_pc; just upload as src_pc.
                    raw = bytes(mv[data_offset + start : data_offset + start + src_nbytes])
                    tensors[name] = QuarkTensor.from_bytes(raw, shape, src_pc)

                # If caller asked for a different pc-dtype than what we
                # just landed, run the GPU cast kernel and drop the
                # source. ``cast`` returns a fresh device tensor of
                # ``pc_dtype``; the source tensor goes out of scope here
                # and its storage is released by QuarkTensor.__del__.
                if tensors[name].dtype != pc_dtype:
                    tensors[name] = _device_cast(tensors[name], pc_dtype)
        finally:
            # ``memoryview`` keeps an exported pointer into ``mm``;
            # mmap.close raises BufferError until every view is
            # explicitly released.
            mv.release()
            mm.close()
        t0 = _tick("convert path", t0)

    if profile:
        total = sum(phase_t.values())
        print("quark.nn.io.load_safetensors phases (QUARK_IO_PROFILE=1):")
        for label, dt in phase_t.items():
            print(f"  {label:<40s} {dt * 1000:>8.1f} ms")
        print(f"  {'TOTAL':<40s} {total * 1000:>8.1f} ms")

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
