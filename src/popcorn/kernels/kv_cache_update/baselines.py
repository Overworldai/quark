"""Baseline for the kv_cache_update kernel — K-RoPE + dtype cast +
ring-buffer writes, backend-polymorphic through ``PT``."""

from __future__ import annotations

from popcorn.backend import PT
from popcorn.kernels.base import Baseline


def _rope_and_write(K, V, cos, sin, frame_t_t, Kc, Vt, *, num_buckets, pd, L, tpf):
    """K-RoPE in fp32 + cast + dual-slot write (tail + ring)."""
    Kf = PT.astype(K, PT.float32)
    K0 = Kf[..., 0::2]
    K1 = Kf[..., 1::2]
    cf = PT.astype(cos, PT.float32)[None, None, :, :]
    sf = PT.astype(sin, PT.float32)[None, None, :, :]
    y0 = K0 * cf - K1 * sf
    y1 = K1 * cf + K0 * sf
    k_rot = PT.astype(PT.cat([y0, y1], dim=-1), Kc.dtype)
    v_w = PT.astype(V, Kc.dtype)

    # Tail write (always).
    Kc[:, :, L:, :] = k_rot
    Vt[:, :, :, L:] = PT.transpose(v_w)

    # Ring write: conditional target base via where.
    ft = PT.astype(frame_t_t, PT.int64)
    bucket = (ft + (pd - 1)) // pd
    slot = bucket % num_buckets
    base = slot * tpf
    write_step = (ft % pd) == 0
    dst_base = PT.where(write_step, base, PT.tensor(L, dtype=ft.dtype))
    offsets = PT.arange(tpf, dtype=ft.dtype)
    dst_idx = PT.astype(dst_base + offsets, PT.int64)
    Kc = PT.index_copy(Kc, 2, dst_idx, k_rot)
    Vt = PT.index_copy(Vt, 3, dst_idx, PT.transpose(v_w))
    return Kc


def kv_cache_update_baselines(kernel, tensors: dict) -> list[Baseline]:
    s = kernel.spec
    K = tensors["K"]
    V = tensors["V"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    frame_t = tensors["frame_t"]

    K4 = K.reshape(s.B, s.n_kv_heads, s.tpf, s.Dh)
    V4 = V.reshape(s.B, s.n_kv_heads, s.tpf, s.Dh)
    cos2 = cos.reshape(s.tpf, s.Dh // 2)
    sin2 = sin.reshape(s.tpf, s.Dh // 2)
    # Baseline gets its own scratch so the kernel-under-test outputs
    # aren't clobbered between iterations.
    Kc4 = PT.zeros_like(tensors["K_cache"]).reshape(s.B, s.n_kv_heads, s.capacity, s.Dh)
    Vt4 = PT.zeros_like(tensors["Vt_cache"]).reshape(s.B, s.n_kv_heads, s.Dh, s.capacity)

    num_buckets, pd, L, tpf = s.num_buckets, s.pinned_dilation, s.L, s.tpf

    # PT.compile → torch.compile on CUDA, identity on Metal. index_copy_
    # blocks cudagraphs so we opt out via `max-autotune-no-cudagraphs`;
    # on Metal the kwargs are ignored.
    @PT.compile(fullgraph=True, mode="max-autotune-no-cudagraphs", dynamic=False)
    def step(K, V, cos, sin, frame_t_t, Kc, Vt):
        return _rope_and_write(
            K,
            V,
            cos,
            sin,
            frame_t_t,
            Kc,
            Vt,
            num_buckets=num_buckets,
            pd=pd,
            L=L,
            tpf=tpf,
        )

    def run():
        step(K4, V4, cos2, sin2, frame_t, Kc4, Vt4)

    return [Baseline("compile[rope+ringwrite]", run)]
