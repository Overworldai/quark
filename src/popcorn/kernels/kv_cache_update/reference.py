"""Pure-torch reference for kv_cache_update.

Mirrors the popcorn kernel semantics exactly (compact ring layout):
  - applies ortho-RoPE to K in fp32 then casts to kv_dtype
  - writes K, V_t into ring + tail slots based on frame_t / pinned_dilation
  - emits a small `segments` list = valid contiguous ranges in the cache

This is what `KVCacheUpdateKernel.reference` returns AND what the
torch.compile baseline runs against. Both write into pre-allocated cache
buffers in place (mirroring the kernel's destructive write semantics).

RoPE convention matches world_engine OrthoRoPE / RoPE.forward exactly:
  - cos, sin tables have shape [tpf, Dh//2] (one angle per rotation pair)
  - input x: [..., Dh]; pairs are (x[2i], x[2i+1])
  - x0 = x[..., 0::2], x1 = x[..., 1::2]
  - y0 = x0*cos - x1*sin
  - y1 = x1*cos + x0*sin
  - output = cat([y0, y1], dim=-1)  (NOT interleave; this is the
    deliberate world_engine layout — Q and K share it so attention works)
"""

from __future__ import annotations

import math

from popcorn.backend import PT


def apply_rope_fp32(x, cos, sin):
    """Apply ortho-RoPE in fp32 with concat-output layout.

    x:   [..., Dh] (any dtype; computed in fp32)
    cos: [..., Dh//2] f32 (broadcast over leading dims)
    sin: [..., Dh//2] f32
    Returns: [..., Dh] in x.dtype
    """
    orig_dt = x.dtype
    x32 = PT.astype(x, PT.float32)
    x0 = x32[..., 0::2]
    x1 = x32[..., 1::2]
    c = PT.astype(cos, PT.float32)
    s = PT.astype(sin, PT.float32)
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s
    return PT.astype(PT.cat((y0, y1), dim=-1), orig_dt)


def compute_segments(
    frame_t: int,
    num_buckets: int,
    tpf: int,
    pinned_dilation: int,
    L: int,
    capacity: int,
    max_segments: int,
):
    """Compute valid (start, length) segments after this step's write.

    Matches WE's mask semantics:
      - Always attend to tail [L, capacity).
      - Attend to ring slots [0, bucket) when not wrapped.
      - After wrap, attend to all ring slots EXCEPT slot % nb when
        ``frame_t % pd == 0`` (the "about to be overwritten" slot).

    Returns ``(segments[max_segments, 2] int32, n_segments int)``.
    """
    bucket = (frame_t + pinned_dilation - 1) // pinned_dilation
    is_write_frame = (frame_t % pinned_dilation) == 0
    is_post_wrap = bucket >= num_buckets
    slot = bucket % num_buckets

    if is_post_wrap and is_write_frame:
        # Split the full ring around the excluded slot.
        rows = [
            [0, slot * tpf],
            [(slot + 1) * tpf, (num_buckets - slot - 1) * tpf],
            [L, tpf],
        ]
        n = 3
    else:
        # One contiguous ring run + tail.
        ring_len = min(bucket, num_buckets) * tpf
        rows = [[0, ring_len], [L, tpf]]
        n = 2

    rows.extend([[0, 0]] * (max_segments - len(rows)))
    return PT.tensor(rows, dtype=PT.int32), n


def kv_cache_update_reference(
    K,  # [B, Hk, tpf, Dh] in_dtype
    V,  # [B, Hk, tpf, Dh] in_dtype
    cos,  # [tpf, Dh//2] f32
    sin,  # [tpf, Dh//2] f32
    K_cache,  # [B, Hk, capacity, Dh] kv_dtype
    Vt_cache,  # [B, Hk, Dh, capacity] kv_dtype
    segments,  # [B, max_segments, 2] int32
    n_segments,  # [B] int32
    *,
    frame_t: int,
    num_buckets: int,
    pinned_dilation: int,
):
    """Reference for the kv_cache_update kernel.

    Returns ``(K_cache, Vt_cache, segments, n_segments)`` — on torch the
    returned tensors are the same objects as the inputs (mutated in
    place); on mlx they are freshly allocated because mlx arrays are
    immutable. Callers should always rebind: ``K_cache, Vt_cache,
    segments, n_segments = kv_cache_update_reference(...)``.
    """
    B, Hk, tpf, Dh = K.shape
    L = num_buckets * tpf
    capacity = L + tpf
    assert tuple(K_cache.shape) == (B, Hk, capacity, Dh)
    assert tuple(Vt_cache.shape) == (B, Hk, Dh, capacity)
    assert tuple(cos.shape) == (tpf, Dh // 2)
    assert tuple(sin.shape) == (tpf, Dh // 2)

    # 1) RoPE on K in fp32 → cast to cache dtype.
    cos_b = cos[None, None, :, :]  # [1,1,tpf,Dh//2]
    sin_b = sin[None, None, :, :]
    k_rot = apply_rope_fp32(K, cos_b, sin_b)  # [B, Hk, tpf, Dh]

    k_w = PT.astype(k_rot, K_cache.dtype)
    v_w = PT.astype(V, K_cache.dtype)
    v_w_t = PT.transpose(v_w)  # [B, Hk, tpf, Dh] → [B, Hk, Dh, tpf]

    # 2) Tail-slot write.
    K_cache = PT.set_slice(K_cache, axis=2, start=L, stop=capacity, src=k_w)
    Vt_cache = PT.set_slice(Vt_cache, axis=3, start=L, stop=capacity, src=v_w_t)

    # 3) Conditional ring write.
    if (frame_t % pinned_dilation) == 0:
        bucket = (frame_t + pinned_dilation - 1) // pinned_dilation
        slot = bucket % num_buckets
        base = slot * tpf
        K_cache = PT.set_slice(K_cache, axis=2, start=base, stop=base + tpf, src=k_w)
        Vt_cache = PT.set_slice(Vt_cache, axis=3, start=base, stop=base + tpf, src=v_w_t)

    # 4) Segments — replicated across batch.
    segs, n_segs = compute_segments(
        frame_t=frame_t,
        num_buckets=num_buckets,
        tpf=tpf,
        pinned_dilation=pinned_dilation,
        L=L,
        capacity=capacity,
        max_segments=int(segments.shape[1]),
    )
    segs_b = PT.broadcast_to(segs[None, :, :], (B, *segs.shape))
    segments = segs_b
    n_segments = PT.tensor([n_segs] * B, dtype=PT.int32)
    return K_cache, Vt_cache, segments, n_segments


def _rope_positional_freqs(
    t, dim: int, *, freqs_for: str, max_freq: float = 10.0, theta: float = 10000.0
):
    """Inline port of lucidrains' RotaryEmbedding applied to position
    vector ``t``. Returns ``[..., dim]`` = ``(t ⊗ freqs)`` repeated by 2."""
    if freqs_for == "pixel":
        freqs = PT.linspace(1.0, max_freq / 2, dim // 2) * math.pi
    elif freqs_for == "lang":
        # 1 / theta^(arange[0,dim,2][:dim//2] / dim)
        exponents = PT.astype(PT.arange(0, dim, 2), PT.float32) / dim
        freqs = 1.0 / (theta**exponents)
        freqs = freqs[: dim // 2]
    else:
        raise ValueError(f"freqs_for must be 'pixel' or 'lang', got {freqs_for!r}")
    raw = PT.astype(t[..., None], PT.float32) * freqs  # [..., dim//2]
    return PT.repeat_interleave(raw, 2, dim=-1)  # [..., dim]


def make_ortho_rope_freqs(
    H_spatial: int,
    W_spatial: int,
    n_frames_total: int,
    Dh: int,
    dtype=None,
):
    """Build the world_engine OrthoRoPE freq table → ``(cos, sin)`` of
    shape ``[n_frames_total*H*W, Dh//2]``. Backend-polymorphic via ``PT``."""
    H, W, T = H_spatial, W_spatial, n_frames_total
    head_dim = Dh
    max_freq = min(H, W) * 0.8
    dt = dtype if dtype is not None else PT.float32

    freqs_x = _rope_positional_freqs(
        PT.linspace(-1 + 1 / W, 1 - 1 / W, W),
        head_dim // 8,
        freqs_for="pixel",
        max_freq=max_freq,
    )[None, :, :]  # [1, W, d/8]
    freqs_y = _rope_positional_freqs(
        PT.linspace(-1 + 1 / H, 1 - 1 / H, H),
        head_dim // 8,
        freqs_for="pixel",
        max_freq=max_freq,
    )[:, None, :]  # [H, 1, d/8]
    freq_t = _rope_positional_freqs(
        PT.arange(T),
        head_dim // 4,
        freqs_for="lang",
    )  # [T, d/4]

    fx = PT.broadcast_to(freqs_x, (H, W, freqs_x.shape[-1])).reshape(1, H * W, -1)
    fx = PT.broadcast_to(fx, (T, H * W, fx.shape[-1])).reshape(T * H * W, -1)
    fy = PT.broadcast_to(freqs_y, (H, W, freqs_y.shape[-1])).reshape(1, H * W, -1)
    fy = PT.broadcast_to(fy, (T, H * W, fy.shape[-1])).reshape(T * H * W, -1)
    ft = PT.broadcast_to(freq_t[:, None, :], (T, H * W, freq_t.shape[-1])).reshape(T * H * W, -1)

    freqs = PT.astype(PT.cat([fx, fy, ft], dim=-1), dt)
    return PT.cos(freqs), PT.sin(freqs)


def kv_cache_update_reference_for_spec(
    kernel, K, V, frame_t, frozen, Vt_cache, segments, n_segments
):
    """Run the reference and return a fresh K_cache tensor.

    Inline RoPE: cos/sin computed from (H, W, frame_t, Dh).
    Signature matches TENSORS minus output (K_cache): K, V, frame_t,
    Vt_cache, segments, n_segments.
    """
    s = kernel.spec
    kv_dt = s.kv_dtype.backend
    ft = int(frame_t.item() if hasattr(frame_t, "item") else frame_t[0])

    # Compute cos/sin from frame_t via the vectorized ortho-RoPE helper.
    cos_full, sin_full = make_ortho_rope_freqs(s.H_spatial, s.W_spatial, ft + 1, s.Dh)
    cos = cos_full[ft * s.tpf : (ft + 1) * s.tpf, :]
    sin = sin_full[ft * s.tpf : (ft + 1) * s.tpf, :]

    # Extract K/V from packed QKV if needed.
    if s.packed_qkv:
        Hk, Dh = s.n_kv_heads, s.Dh
        k_start = s.k_col_offset
        v_start = s.v_col_offset
        K_flat = K[:, k_start : k_start + Hk * Dh]  # [B*tpf, Hk*Dh]
        V_flat = K[:, v_start : v_start + Hk * Dh]  # [B*tpf, Hk*Dh] (V from same packed tensor)
        K_4d = K_flat.reshape(s.B, s.tpf, Hk, Dh)
        K_4d = PT.permute(K_4d, (0, 2, 1, 3))  # [B, Hk, tpf, Dh]
        V_4d = V_flat.reshape(s.B, s.tpf, Hk, Dh)
        V_4d = PT.permute(V_4d, (0, 2, 1, 3))  # [B, Hk, tpf, Dh]
    else:
        K_4d = K.reshape(s.B, s.n_kv_heads, s.tpf, s.Dh)
        V_4d = V.reshape(s.B, s.n_kv_heads, s.tpf, s.Dh)

    K_cache_init = PT.zeros(s.B, s.n_kv_heads, s.capacity, s.Dh, dtype=kv_dt)
    K_cache_out, _Vt, _segs, _nsegs = kv_cache_update_reference(
        K_4d,
        V_4d,
        cos.reshape(s.tpf, s.Dh // 2),
        sin.reshape(s.tpf, s.Dh // 2),
        K_cache_init,
        Vt_cache.reshape(s.B, s.n_kv_heads, s.Dh, s.capacity),
        segments.reshape(s.B, s.max_segments, 2),
        n_segments.reshape(s.B),
        frame_t=ft,
        num_buckets=s.num_buckets,
        pinned_dilation=s.pinned_dilation,
    )
    return K_cache_out.reshape(s.B * s.n_kv_heads * s.capacity, s.Dh)
