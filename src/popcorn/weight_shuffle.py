"""Pre-shuffle weight matrices so each MMA fragment is contiguous in smem.

EXEMPT FROM 500-LINE RULE: the torch + MLX permutation paths share the
per-mma_k fragment-layout derivation + pad-aware stride math. Splitting
by backend duplicates the layout table; splitting by public API
(``shuffle_b`` / ``cached_shuffle_b`` / reverse-map utilities) strands
the helpers their callers need.

This is the OFFLINE Python implementation — runs on CPU/GPU once before
the kernel launches. The shuffled layout matches what
``bfrag_load_shuffled_*_hoisted`` in :mod:`popcorn.mma.frag` expects: each
warp lane reads ``FRAG_BYTES = mma_k * dtype_bytes / 16 * 4`` contiguous
bytes per fragment from a single ``ld.shared.{b32,v2.b32}``, with no
row-stride math.

Two API surfaces:

* :func:`shuffle_weights_e4m3_k16` — the legacy CPU/numpy implementation,
  e4m3-only, with explicit ``bpad`` support (used by older row_stationary
  bring-up tests).
* :func:`shuffle_b_for_frag_load` — generic dtype-agnostic shuffle. Takes
  a numpy / cupy / torch ``[N, K]`` tensor and returns a same-shape
  same-dtype tensor with the bytes permuted into the contiguous-per-lane
  layout. Used by row_stationary and the moe kernels.

The byte permutation is identical for bf16 (k16) and e4m3 (k16/k32) — only
``FRAG_BYTES`` and ``mma_k`` change. The permutation is built once per
``(K_CHUNK, mma_k, dtype_bytes)`` triple and cached.
"""

from __future__ import annotations

import contextlib
import weakref as _weakref
from dataclasses import dataclass
from typing import Any

import numpy as np


def shuffle_weights_e4m3_k16(
    W: np.ndarray,
    *,
    kchunk: int = 64,
    bpad: int = 0,
) -> np.ndarray:
    """Pre-shuffle a weight matrix W: [N, K] uint8 (e4m3) for vectorized
    BFrag loads with m16n8k16 e4m3.

    Returns a uint8 array of shape [N, n_k_chunks * (kchunk + bpad)].
    """
    assert W.dtype == np.uint8, "shuffle expects e4m3 viewed as uint8"
    N, K = W.shape
    assert K % kchunk == 0, f"K={K} not multiple of kchunk={kchunk}"
    assert N % 8 == 0, f"N={N} not multiple of 8"
    assert kchunk % 16 == 0, f"kchunk={kchunk} not multiple of 16 (k step)"

    bstride = kchunk + bpad
    n_k_chunks = K // kchunk
    out_K = n_k_chunks * bstride
    out = np.zeros((N, out_K), dtype=np.uint8)

    n8_blocks = N // 8
    k_steps_per_chunk = kchunk // 16

    for n_block in range(n8_blocks):
        n_base = n_block * 8
        for k_chunk in range(n_k_chunks):
            k_chunk_byte_base = k_chunk * kchunk
            out_chunk_off = k_chunk * bstride
            for k_step in range(k_steps_per_chunk):
                kk = k_step * 16
                for lane in range(32):
                    gid = lane >> 2
                    tid = lane & 3
                    src_row = n_base + gid
                    src_col = k_chunk_byte_base + kk + tid * 4
                    src_bytes = W[src_row, src_col : src_col + 4]
                    for byte in range(4):
                        frag_off = k_step * 32 * 4 + lane * 4 + byte
                        out_row = n_base + frag_off // bstride
                        out_col = out_chunk_off + (frag_off % bstride)
                        out[out_row, out_col] = src_bytes[byte]

    return out


def verify_shuffled_layout(
    W: np.ndarray,
    W_shuf: np.ndarray,
    *,
    kchunk: int = 64,
    bpad: int = 0,
) -> bool:
    """Verify that W_shuf was correctly shuffled from W."""
    N, K = W.shape
    bstride = kchunk + bpad
    n8_blocks = N // 8
    k_steps_per_chunk = kchunk // 16

    for n_block in range(n8_blocks):
        n_base = n_block * 8
        for k_chunk in range(K // kchunk):
            out_chunk_off = k_chunk * bstride
            for k_step in range(k_steps_per_chunk):
                kk = k_step * 16
                for lane in range(32):
                    gid = lane >> 2
                    tid = lane & 3
                    src = W[
                        n_base + gid,
                        k_chunk * kchunk + kk + tid * 4 : k_chunk * kchunk + kk + tid * 4 + 4,
                    ]
                    for byte in range(4):
                        frag_off = k_step * 32 * 4 + lane * 4 + byte
                        out_row = n_base + frag_off // bstride
                        out_col = out_chunk_off + (frag_off % bstride)
                        if W_shuf[out_row, out_col] != src[byte]:
                            return False
    return True


# ─── Generic byte-level shuffle (numpy / cupy / torch compatible) ────


def _build_shuffle_perm(K_CHUNK: int, mma_k: int, dtype_bytes: int) -> np.ndarray:
    """Build the int64 permutation array used by
    :func:`shuffle_b_for_frag_load`.

    For any flat ``[8, K_CHUNK*dtype_bytes]`` tile, the shuffled output is
    ``out_flat[i] = in_flat[perm[i]]``. The permutation reorders bytes so
    that lane ``L``'s ``FRAG_BYTES`` of one fragment land at flat offset
    ``ks * 32 * FRAG_BYTES + L * FRAG_BYTES + r * 4 + b``, with
    ``FRAG_BYTES = mma_k * dtype_bytes // 16 * 4``.
    """
    K_CHUNK_BYTES = K_CHUNK * dtype_bytes
    FRAG_REGS = mma_k * dtype_bytes // 16
    FRAG_BYTES = FRAG_REGS * 4
    K_STEPS = K_CHUNK // mma_k
    mma_k_bytes = mma_k * dtype_bytes

    perm = np.empty(8 * K_CHUNK_BYTES, dtype=np.int64)
    for ks in range(K_STEPS):
        for lane in range(32):
            gid = lane >> 2
            tid = lane & 3
            for r in range(FRAG_REGS):
                src_col = ks * mma_k_bytes + tid * 4 + r * 16
                for b in range(4):
                    src_flat = gid * K_CHUNK_BYTES + src_col + b
                    dst_flat = ks * 32 * FRAG_BYTES + lane * FRAG_BYTES + r * 4 + b
                    perm[dst_flat] = src_flat
    return perm


# Cached numpy perms keyed by (K_CHUNK, mma_k, dtype_bytes).
_perm_cache_np: dict[tuple[int, int, int], np.ndarray] = {}

# Process-wide cache of shuffled B weight tensors. Keyed by
# ``id(W)`` — the Python object identity — so two different tensors
# never collide even if PyTorch's allocator reuses a freed data_ptr.
# A WeakValueDictionary lets entries drop when the plain W or its
# shuffled result are garbage collected.
_shuffled_tensor_cache: _weakref.WeakValueDictionary[tuple[int, int, int, int], object] = (
    _weakref.WeakValueDictionary()
)


def cached_shuffle_b(W, K_CHUNK: int, mma_k: int, bpad: int = 0):
    """Shuffle ``W`` for fragment loads, with process-wide memoization.

    When ``bpad > 0``, the padding is baked into the K dimension of the
    output tensor (physical K = ``(K / K_CHUNK) * (K_CHUNK + bpad)``)
    via :meth:`ShuffledWeight.from_plain`. This is the bank-conflict
    avoidance the non-shuffled path gets via the SmemPlan ``b_pad``
    knob; the shuffled path carries it in the data layout instead.

    The cache key is ``(id(W), K_CHUNK, mma_k, bpad)`` — we key on
    Python object identity rather than ``data_ptr()`` because torch's
    allocator freely reuses data_ptrs across fresh tensors, which would
    otherwise serve stale shuffled data back to a caller that just
    allocated a new W tensor.
    """
    key = (id(W), K_CHUNK, mma_k, bpad)
    cached = _shuffled_tensor_cache.get(key)
    if cached is not None:
        return cached
    if bpad > 0:
        sw = ShuffledWeight.from_plain(W, kchunk=K_CHUNK, bpad=bpad, mma_k=mma_k)
        result = sw.tensor
    else:
        result = shuffle_b_for_frag_load(W, K_CHUNK, mma_k)
    with contextlib.suppress(TypeError):
        _shuffled_tensor_cache[key] = result
    return result


def _cached_perm_np(K_CHUNK: int, mma_k: int, dtype_bytes: int) -> np.ndarray:
    key = (K_CHUNK, mma_k, dtype_bytes)
    perm = _perm_cache_np.get(key)
    if perm is None:
        perm = _build_shuffle_perm(K_CHUNK, mma_k, dtype_bytes)
        _perm_cache_np[key] = perm
    return perm


def shuffle_b_for_frag_load(W, K_CHUNK: int, mma_k: int):
    """Pre-shuffle ``W: [N, K]`` for the contiguous-per-lane bfrag loaders.

    Accepts numpy, torch, or MLX arrays. Returns a same-dtype same-shape
    tensor (matching the input backend) with the bytes permuted.
    Bytes-only operation (handles bf16 via uint16 → uint8 view and e4m3
    via uint8 directly), so works for any 2-byte or 1-byte element type.

    Constraints:
      * ``N`` must be a multiple of 8 (one fragment row block).
      * ``K * dtype.bytes`` must be a multiple of ``K_CHUNK * dtype.bytes``.
      * ``K_CHUNK`` must be a multiple of ``mma_k``.
    """
    # Dispatch by input type:
    #   * PopcornTensor — has ``to_bytes`` + string dtype.
    #   * MLX — has .dtype as mlx.core.Dtype.
    #   * numpy — has .dtype.itemsize.
    if hasattr(W, "to_bytes") and isinstance(getattr(W, "dtype", None), str):
        return _shuffle_popcorn(W, K_CHUNK, mma_k)
    if _is_mlx_array(W):
        return _shuffle_mlx(W, K_CHUNK, mma_k)
    return _shuffle_numpy(W, K_CHUNK, mma_k)


def _shuffle_popcorn(W, K_CHUNK: int, mma_k: int):
    """PopcornTensor path — copy to numpy, shuffle, copy back."""
    from popcorn.runtime.tensor import PopcornTensor

    shape = tuple(W.shape)
    # View as bytes via the raw copy-to-host.
    raw = W.contiguous().to_bytes()
    elem_bytes = W.nbytes // (W.numel() or 1)
    N, K = shape
    W_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(N, K * elem_bytes).copy()
    # Reshape + shuffle as in _shuffle_numpy, byte-level.
    K_CHUNK_BYTES = K_CHUNK * elem_bytes
    K_BYTES = K * elem_bytes
    if K_BYTES % K_CHUNK_BYTES != 0 or N % 8 != 0:
        raise ValueError(
            f"shuffle_b_for_frag_load: shape {shape} not aligned to (8, K_CHUNK={K_CHUNK})"
        )
    n8_tiles = N // 8
    n_k_tiles = K_BYTES // K_CHUNK_BYTES
    W_tiled = W_bytes.reshape(n8_tiles, 8, n_k_tiles, K_CHUNK_BYTES)
    W_tiled = np.ascontiguousarray(W_tiled.transpose(0, 2, 1, 3))
    flat = W_tiled.reshape(n8_tiles * n_k_tiles, 8 * K_CHUNK_BYTES)
    perm = _cached_perm_np(K_CHUNK, mma_k, elem_bytes)
    out_flat = flat[:, perm]
    out_tiled = out_flat.reshape(n8_tiles, n_k_tiles, 8, K_CHUNK_BYTES)
    out_tiled = np.ascontiguousarray(out_tiled.transpose(0, 2, 1, 3))
    out_bytes = out_tiled.reshape(N, K_BYTES)
    # Back to device in the original dtype.
    return PopcornTensor.from_bytes(out_bytes.tobytes(), shape, W.dtype)


def _is_mlx_array(W) -> bool:
    """Detect mlx.core.array without importing mlx unless it's installed."""
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return isinstance(W, mx.array)


def _shuffle_mlx(W, K_CHUNK: int, mma_k: int):
    """MLX path — go byte-level: view the MLX array as uint8, copy to
    numpy, shuffle, then convert back to MLX and re-view as the
    original dtype. The data is byte-permuted so the round-trip is
    correct for any dtype; cached_shuffle_b memoizes the result so
    the host↔device copy happens at most once per W.
    """
    import mlx.core as mx
    import numpy as _np

    orig_mx_dtype = W.dtype
    N, K = W.shape
    # View as uint8 (mx.uint8 == "B") — gives us 1-byte elements with a
    # total row byte-count of K * dtype.bytes. ``np.array(...)``
    # materializes it via the buffer protocol; uint8 has itemsize 1 so
    # the buffer-format assertion passes for any element dtype.
    elem_bytes = orig_mx_dtype.size
    K_bytes = K * elem_bytes
    np_bytes = _np.array(W.view(mx.uint8)).reshape(N, K_bytes)
    # The shuffle is a byte permutation — operate on the uint8 view
    # so _shuffle_numpy's itemsize arithmetic works, but rewrap the
    # logical shape (N, K) with a dummy bf16-equivalent dtype so the
    # shuffle's per-element dtype-aware tiling matches the kernel's
    # expected layout. We skip that by calling the low-level permute
    # directly: reshape into (n8_tiles, 8, n_k_tiles, K_CHUNK_BYTES),
    # apply the perm, reshape back.
    n8_tiles = N // 8
    K_CHUNK_BYTES = K_CHUNK * elem_bytes
    if K_bytes % K_CHUNK_BYTES != 0 or N % 8 != 0:
        raise ValueError(
            f"shuffle_b_for_frag_load: shape ({N}, {K}) not aligned to (8, K_CHUNK={K_CHUNK})"
        )
    n_k_tiles = K_bytes // K_CHUNK_BYTES
    W_tiled = np_bytes.reshape(n8_tiles, 8, n_k_tiles, K_CHUNK_BYTES)
    W_tiled = _np.ascontiguousarray(W_tiled.transpose(0, 2, 1, 3))
    flat = W_tiled.reshape(n8_tiles * n_k_tiles, 8 * K_CHUNK_BYTES)
    perm = _cached_perm_np(K_CHUNK, mma_k, elem_bytes)
    out_flat = flat[:, perm]
    out_tiled = out_flat.reshape(n8_tiles, n_k_tiles, 8, K_CHUNK_BYTES)
    out_tiled = _np.ascontiguousarray(out_tiled.transpose(0, 2, 1, 3))
    out_bytes = out_tiled.reshape(N, K_bytes)
    # Back to MLX uint8, then re-view as the original element dtype.
    return mx.array(out_bytes).reshape(N * K_bytes).view(orig_mx_dtype).reshape(N, K)


def _shuffle_numpy(W: np.ndarray, K_CHUNK: int, mma_k: int) -> np.ndarray:
    elem_bytes = W.dtype.itemsize
    N, K = W.shape
    K_CHUNK_BYTES = K_CHUNK * elem_bytes
    K_BYTES = K * elem_bytes
    if K_BYTES % K_CHUNK_BYTES != 0 or N % 8 != 0:
        raise ValueError(
            f"shuffle_b_for_frag_load: shape ({N}, {K}) not aligned to (8, K_CHUNK={K_CHUNK})"
        )
    W_bytes = W.view(np.uint8).reshape(N, K_BYTES)
    n8_tiles = N // 8
    n_k_tiles = K_BYTES // K_CHUNK_BYTES

    W_tiled = W_bytes.reshape(n8_tiles, 8, n_k_tiles, K_CHUNK_BYTES)
    W_tiled = np.ascontiguousarray(W_tiled.transpose(0, 2, 1, 3))
    flat = W_tiled.reshape(n8_tiles * n_k_tiles, 8 * K_CHUNK_BYTES)

    perm = _cached_perm_np(K_CHUNK, mma_k, elem_bytes)
    out_flat = flat[:, perm]

    out_tiled = out_flat.reshape(n8_tiles, n_k_tiles, 8, K_CHUNK_BYTES)
    out_tiled = np.ascontiguousarray(out_tiled.transpose(0, 2, 1, 3))
    return out_tiled.reshape(N, K_BYTES).view(W.dtype).reshape(N, K)


# ═══════════════════════════════════════════════════════════════════
# ShuffledWeight — typed wrapper for offline-shuffled weight tensors
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ShuffledWeight:
    """A weight matrix pre-shuffled for contiguous-per-lane fragment loads,
    with bank-conflict-avoidance padding baked into the K dimension.

    Logical: ``[N, K]`` — the original weight dimensions (used by the
    reference implementation, the MoE expert-offset math, etc.).

    Physical: ``[N, K_shuffled]`` — the actual bytes in device memory,
    where ``K_shuffled = n_k_tiles * bstride`` and
    ``bstride = kchunk + bpad``. The padding bytes are baked in so the
    smem row stride naturally includes the bank-conflict gap. No extra
    smem padding is needed at runtime — ``SmemPlan`` uses ``b_pad=0``
    and ``b_cols=bstride``.

    At runtime, ``cp.async`` loads ``bstride`` bytes per row from gmem
    directly into smem. The frag loader reads from the ``bstride``-wide
    smem rows using the shuffled per-lane layout. Everything matches.
    """

    tensor: Any  # physical [N, K_shuffled] on device — PopcornTensor / mx.array
    logical_K: int  # original K (before shuffle + pad)
    kchunk: int  # logical K elements per pipeline stage
    bpad: int  # pad elements per k-tile (0 = no pad)
    mma_k: int  # MMA inner-K (16 or 32)
    dtype_bytes: int  # element width (2 for bf16, 1 for e4m3)

    @property
    def bstride(self) -> int:
        """Elements per padded k-tile row (= kchunk + bpad)."""
        return self.kchunk + self.bpad

    @property
    def bstride_bytes(self) -> int:
        return self.bstride * self.dtype_bytes

    @property
    def physical_K(self) -> int:
        """K columns in the physical tensor (> logical_K when bpad > 0)."""
        return self.tensor.shape[1]

    @property
    def n_k_tiles(self) -> int:
        return self.logical_K // self.kchunk

    def data_ptr(self):
        return self.tensor.data_ptr()

    @staticmethod
    def from_plain(W, *, kchunk: int, bpad: int = 0, mma_k: int = 16):
        """Offline shuffle ``W: [N, K]`` → ``ShuffledWeight([N, K_shuffled])``.

        Accepts ``PopcornTensor`` (CUDA) or ``mx.array`` (Metal). Handles
        bf16 and e4m3 via byte-level permutation. The padding is baked
        into the output's K dimension — each ``[8, kchunk]`` source tile
        becomes an ``[8, kchunk + bpad]`` shuffled tile.
        """
        # ─── Normalize input → (byte-view np.ndarray, elem_bytes, dtype handle) ───
        if hasattr(W, "to_bytes") and isinstance(getattr(W, "dtype", None), str):
            # PopcornTensor path
            N, K = tuple(W.shape)
            elem_bytes = W.nbytes // (W.numel() or 1)
            W_np_bytes = (
                np.frombuffer(W.contiguous().to_bytes(), dtype=np.uint8)
                .reshape(N, K * elem_bytes)
                .copy()
            )
            orig_dtype = W.dtype  # short string
            is_popcorn = True
        else:
            # MLX path (import lazily to keep CUDA-only envs free of mlx).
            import mlx.core as mx

            if not isinstance(W, mx.array):
                raise TypeError(f"ShuffledWeight.from_plain: unsupported input {type(W).__name__}")
            N, K = tuple(W.shape)
            elem_bytes = W.dtype.size
            W_np_bytes = np.array(W.view(mx.uint8)).reshape(N, K * elem_bytes).copy()
            orig_dtype = W.dtype  # mx.Dtype
            is_popcorn = False

        bstride = kchunk + bpad
        bstride_bytes = bstride * elem_bytes
        kchunk_bytes = kchunk * elem_bytes

        if K % kchunk != 0:
            raise ValueError(f"K={K} not divisible by kchunk={kchunk}")
        if N % 8 != 0:
            raise ValueError(f"N={N} not divisible by 8")
        if kchunk % mma_k != 0:
            raise ValueError(f"kchunk={kchunk} not divisible by mma_k={mma_k}")

        n8 = N // 8
        n_k = K // kchunk

        # Byte-level tiled shuffle + zero pad to bstride.
        perm = _build_padded_perm(kchunk, bpad, mma_k, elem_bytes)
        W_tiled = W_np_bytes.reshape(n8, 8, n_k, kchunk_bytes)
        if bpad > 0:
            pad_width = bstride_bytes - kchunk_bytes
            W_tiled = np.pad(W_tiled, ((0, 0), (0, 0), (0, 0), (0, pad_width)))
        W_tiled = np.ascontiguousarray(W_tiled.transpose(0, 2, 1, 3))
        flat = W_tiled.reshape(n8 * n_k, 8 * bstride_bytes)
        out_flat = flat[:, perm]
        out_tiled = out_flat.reshape(n8, n_k, 8, bstride_bytes)
        out_tiled = np.ascontiguousarray(out_tiled.transpose(0, 2, 1, 3))
        K_shuffled_bytes = n_k * bstride_bytes
        out_bytes = out_tiled.reshape(N, K_shuffled_bytes)
        K_shuffled = K_shuffled_bytes // elem_bytes

        if is_popcorn:
            from popcorn.runtime.tensor import PopcornTensor

            out = PopcornTensor.from_bytes(out_bytes.tobytes(), (N, K_shuffled), orig_dtype)
        else:
            import mlx.core as mx

            out = (
                mx.array(out_bytes)
                .reshape(N * K_shuffled_bytes)
                .view(orig_dtype)
                .reshape(N, K_shuffled)
            )

        return ShuffledWeight(
            tensor=out,
            logical_K=K,
            kchunk=kchunk,
            bpad=bpad,
            mma_k=mma_k,
            dtype_bytes=elem_bytes,
        )


# ─── Padded permutation builder ──────────────────────────────────────


_padded_perm_cache: dict[tuple, np.ndarray] = {}


def _build_padded_perm(kchunk: int, bpad: int, mma_k: int, dtype_bytes: int) -> np.ndarray:
    """Build the byte permutation for one ``[8, bstride]`` shuffled tile.

    Same structure as :func:`_build_shuffle_perm` but the source/dest
    tiles are ``bstride``-bytes wide (bstride = kchunk + bpad), with the
    source's pad zone filled with zeros by the caller.

    Positions in the output that aren't covered by any fragment byte
    (the "gap" positions between frag lanes that provide bank-conflict
    avoidance) are mapped to byte 0 of the tile (a zero-pad byte)
    so the output is deterministic.
    """
    key = (kchunk, bpad, mma_k, dtype_bytes)
    cached = _padded_perm_cache.get(key)
    if cached is not None:
        return cached

    bstride = kchunk + bpad
    bstride_bytes = bstride * dtype_bytes
    kchunk_bytes = kchunk * dtype_bytes
    FRAG_REGS = mma_k * dtype_bytes // 16
    FRAG_BYTES = FRAG_REGS * 4
    K_STEPS = kchunk // mma_k
    mma_k_bytes = mma_k * dtype_bytes

    tile_bytes = 8 * bstride_bytes
    # Default: map to the first pad byte (= kchunk_bytes in row 0),
    # which is always zero. This makes un-covered positions deterministic.
    first_pad_byte = kchunk_bytes  # first zero-pad byte in the padded tile
    perm = np.full(tile_bytes, first_pad_byte, dtype=np.int64)

    for ks in range(K_STEPS):
        for lane in range(32):
            gid = lane >> 2
            tid = lane & 3
            for r in range(FRAG_REGS):
                src_col = ks * mma_k_bytes + tid * 4 + r * 16
                for b in range(4):
                    src_flat = gid * bstride_bytes + src_col + b
                    dst_flat = ks * 32 * FRAG_BYTES + lane * FRAG_BYTES + r * 4 + b
                    perm[dst_flat] = src_flat

    _padded_perm_cache[key] = perm
    return perm
