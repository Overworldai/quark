"""kv_cache_update numpy reference — ortho-RoPE + ring KV cache write.

Mirrors the kernel's compact ring layout exactly:
  - applies ortho-RoPE to K in fp32 then casts to kv_dtype
  - writes K, V_t into ring + tail slots based on frame_t / pinned_dilation
  - emits (start, length) segments describing the valid attendable ranges

Returns a dict with all four role="out" tensors so the RefCache
picks them up by name.

RoPE convention matches world_engine OrthoRoPE exactly:
  - cos/sin tables: [tpf, Dh//2]
  - input x: [..., Dh]; pairs are (x[2i], x[2i+1])
  - y0 = x0*cos - x1*sin;  y1 = x1*cos + x0*sin
  - output = cat([y0, y1], dim=-1)  (NOT interleave)
"""

from __future__ import annotations

import math

import numpy as np

from quark.runtime.npconv import astype_numpy, to_f32_numpy

# ---------------------------------------------------------------
# ortho-RoPE helpers
# ---------------------------------------------------------------


def _apply_rope_fp32(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """``x[..., Dh]`` + cos/sin ``[..., Dh//2]`` → ``[..., Dh]`` in f32.
    Pairs are (x[2i], x[2i+1]); output layout is cat([y0, y1], dim=-1)."""
    x32 = x.astype(np.float32, copy=False)
    x0 = x32[..., 0::2]
    x1 = x32[..., 1::2]
    c = cos.astype(np.float32, copy=False)
    s = sin.astype(np.float32, copy=False)
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s
    return np.concatenate([y0, y1], axis=-1)


def _rope_positional_freqs(
    t: np.ndarray, dim: int, *, freqs_for: str, max_freq: float = 10.0, theta: float = 10000.0
) -> np.ndarray:
    """Inline port of lucidrains' RotaryEmbedding on position vector
    ``t``. Returns ``[..., dim]`` = ``(t ⊗ freqs)`` repeat-interleaved by 2."""
    if freqs_for == "pixel":
        freqs = np.linspace(1.0, max_freq / 2, dim // 2) * math.pi
    elif freqs_for == "lang":
        exponents = np.arange(0, dim, 2, dtype=np.float32) / dim
        freqs = 1.0 / (theta**exponents)
        freqs = freqs[: dim // 2]
    else:
        raise ValueError(f"freqs_for must be 'pixel' or 'lang', got {freqs_for!r}")
    raw = t.astype(np.float32)[..., None] * freqs  # [..., dim//2]
    return np.repeat(raw, 2, axis=-1)  # [..., dim]


def _make_ortho_rope_freqs(H: int, W: int, T: int, Dh: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (cos, sin) of shape ``[T*H*W, Dh//2]`` matching
    world_engine OrthoRoPE."""
    # The downstream ``np.linspace(-1 + 1/W, 1 - 1/W, W)`` and
    # ``1/H`` scalar divisions raise ``ZeroDivisionError: float
    # division by zero`` if H or W is 0, which the autotune
    # reference-setup path swallows as a generic "reference setup
    # failed" log line. Catch here with a message that names the
    # offending dim so callers don't have to spelunk through the
    # RoPE math to figure out which spec field tripped it.
    if H <= 0 or W <= 0 or Dh <= 0:
        raise ValueError(
            f"_make_ortho_rope_freqs: H_spatial, W_spatial, and Dh must all "
            f"be > 0; got H={H}, W={W}, Dh={Dh}. Check the KVCacheUpdate "
            f"layer's construction — zero spatial dims come from a model "
            f"config that hasn't set height/width (cfg.height / cfg.width)."
        )
    head_dim = Dh
    max_freq = min(H, W) * 0.8

    freqs_x = _rope_positional_freqs(
        np.linspace(-1 + 1 / W, 1 - 1 / W, W),
        head_dim // 8,
        freqs_for="pixel",
        max_freq=max_freq,
    )[None, :, :]  # [1, W, d/8]
    freqs_y = _rope_positional_freqs(
        np.linspace(-1 + 1 / H, 1 - 1 / H, H),
        head_dim // 8,
        freqs_for="pixel",
        max_freq=max_freq,
    )[:, None, :]  # [H, 1, d/8]
    freq_t = _rope_positional_freqs(
        np.arange(T),
        head_dim // 4,
        freqs_for="lang",
    )  # [T, d/4]

    fx = np.broadcast_to(freqs_x, (H, W, freqs_x.shape[-1])).reshape(1, H * W, -1)
    fx = np.broadcast_to(fx, (T, H * W, fx.shape[-1])).reshape(T * H * W, -1)
    fy = np.broadcast_to(freqs_y, (H, W, freqs_y.shape[-1])).reshape(1, H * W, -1)
    fy = np.broadcast_to(fy, (T, H * W, fy.shape[-1])).reshape(T * H * W, -1)
    ft = np.broadcast_to(freq_t[:, None, :], (T, H * W, freq_t.shape[-1])).reshape(T * H * W, -1)

    freqs = np.concatenate([fx, fy, ft], axis=-1).astype(np.float32)
    return np.cos(freqs), np.sin(freqs)


def _compute_segments(
    frame_t: int,
    num_buckets: int,
    tpf: int,
    pinned_dilation: int,
    L: int,
    max_segments: int,
) -> tuple[np.ndarray, int]:
    """Return ``(segments[max_segments, 2] s32, n_segments int)``.

    ``tpf`` here is the *cached* per-frame token count (post-quilt:
    ``H*W // quilt_factor``). ``L`` is ``num_buckets * tpf``. The
    quilt-dense path passes the un-divided values and the math is
    identical."""
    bucket = (frame_t + pinned_dilation - 1) // pinned_dilation
    is_write_frame = (frame_t % pinned_dilation) == 0
    is_post_wrap = bucket >= num_buckets
    slot = bucket % num_buckets

    if is_post_wrap and is_write_frame:
        rows = [
            [0, slot * tpf],
            [(slot + 1) * tpf, (num_buckets - slot - 1) * tpf],
            [L, tpf],
        ]
        n = 3
    else:
        ring_len = min(bucket, num_buckets) * tpf
        rows = [[0, ring_len], [L, tpf]]
        n = 2
    rows.extend([[0, 0]] * (max_segments - len(rows)))
    return np.array(rows, dtype=np.int32), n


# ---------------------------------------------------------------
# Reference entry point
# ---------------------------------------------------------------


def kv_cache_update_reference_numpy(
    spec,
    *,
    K,
    V,
    frame_t,
    frozen,
    K_cache=None,
    Vt_cache=None,
    segments=None,
    n_segments=None,
):
    """Full reference — returns dict with every role='out' tensor
    (K_cache, Vt_cache, segments, n_segments)."""
    del K_cache, Vt_cache, segments, n_segments  # refs produce fresh buffers

    s = spec
    kv_dt = s.kv_dtype
    in_hint = s.in_dtype.value

    # frame_t is an s32 1-elem tensor.
    ft_arr = to_f32_numpy(frame_t, dtype_hint="s32")
    ft = int(ft_arr[0])

    # Extract K/V from packed QKV when needed.
    K_np = to_f32_numpy(K, dtype_hint=in_hint)
    if s.packed_qkv:
        Hk, Dh = s.n_kv_heads, s.Dh
        k_start = s.k_col_offset
        v_start = s.v_col_offset
        # K here is the packed QKV input [B*tpf, qkv_dim].
        K_flat = K_np[:, k_start : k_start + Hk * Dh]
        V_flat = K_np[:, v_start : v_start + Hk * Dh]
        K_4d = K_flat.reshape(s.B, s.tpf, Hk, Dh).transpose(0, 2, 1, 3)
        V_4d = V_flat.reshape(s.B, s.tpf, Hk, Dh).transpose(0, 2, 1, 3)
    else:
        V_np = to_f32_numpy(V, dtype_hint=in_hint)
        K_4d = K_np.reshape(s.B, s.n_kv_heads, s.tpf, s.Dh)
        V_4d = V_np.reshape(s.B, s.n_kv_heads, s.tpf, s.Dh)

    # cos/sin for this frame (fp32).
    cos_full, sin_full = _make_ortho_rope_freqs(s.H_spatial, s.W_spatial, ft + 1, s.Dh)
    cos = cos_full[ft * s.tpf : (ft + 1) * s.tpf, :].reshape(s.tpf, s.Dh // 2)
    sin = sin_full[ft * s.tpf : (ft + 1) * s.tpf, :].reshape(s.tpf, s.Dh // 2)

    # RoPE on K in f32, then narrow through the kv-cache carrier to
    # match the kernel's on-store cast.
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    k_rot = _apply_rope_fp32(K_4d, cos_b, sin_b)
    k_rot_carried = to_f32_numpy(astype_numpy(k_rot, kv_dt), dtype_hint=kv_dt.value)
    v_carried = to_f32_numpy(astype_numpy(V_4d, kv_dt), dtype_hint=kv_dt.value)

    # Quilt: keep only every quilt_factor-th pixel along the per-frame
    # token axis, starting at quilt_offset. Cache slots are contiguous
    # in cached space — slot i holds input pixel quilt_offset + i*quilt.
    quilt, qoff = s.quilt_factor, s.quilt_offset
    if quilt > 1:
        k_rot_carried = k_rot_carried[:, :, qoff::quilt, :]
        v_carried = v_carried[:, :, qoff::quilt, :]

    tpf_cached = s.tpf_cached

    # Allocate fresh cache buffers (f32 internally, narrowed at return).
    L = s.num_buckets * tpf_cached
    K_cache = np.zeros((s.B, s.n_kv_heads, s.capacity, s.Dh), dtype=np.float32)
    Vt_cache = np.zeros((s.B, s.n_kv_heads, s.Dh, s.capacity), dtype=np.float32)

    # Tail-slot write (always).
    K_cache[:, :, L : L + tpf_cached, :] = k_rot_carried
    # Vt_cache expects V transposed last-two.
    Vt_cache[:, :, :, L : L + tpf_cached] = v_carried.transpose(0, 1, 3, 2)

    # Conditional ring write. Honor ``frozen`` the same way the kernel
    # does: skip ring writes entirely when frozen!=0.
    frozen_arr = to_f32_numpy(frozen, dtype_hint="s32")
    is_frozen = int(frozen_arr[0]) != 0
    if not is_frozen and (ft % s.pinned_dilation) == 0:
        bucket = (ft + s.pinned_dilation - 1) // s.pinned_dilation
        slot = bucket % s.num_buckets
        base = slot * tpf_cached
        K_cache[:, :, base : base + tpf_cached, :] = k_rot_carried
        Vt_cache[:, :, :, base : base + tpf_cached] = v_carried.transpose(0, 1, 3, 2)

    # Segments.
    segs, n_segs = _compute_segments(
        frame_t=ft,
        num_buckets=s.num_buckets,
        tpf=tpf_cached,
        pinned_dilation=s.pinned_dilation,
        L=L,
        max_segments=s.max_segments,
    )
    segs_flat = (
        np.broadcast_to(segs[None, :, :], (s.B, s.max_segments, 2)).reshape(-1).astype(np.int32)
    )
    n_segs_arr = np.full((s.B,), n_segs, dtype=np.int32)

    return {
        "K_cache": astype_numpy(K_cache.reshape(s.B * s.n_kv_heads * s.capacity, s.Dh), kv_dt),
        "Vt_cache": astype_numpy(Vt_cache.reshape(s.B * s.n_kv_heads * s.Dh, s.capacity), kv_dt),
        "segments": segs_flat,
        "n_segments": n_segs_arr,
    }
