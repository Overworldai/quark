"""End-to-end baselines for the owl_attn kernel.

Folds kv_cache_update + Q-RoPE + masked SDPA into one call so the
baseline can be compared against world_engine's step. CUDA uses a
torch.compile'd flex_attention graph; Metal uses mx.fast.sdpa with
a precomputed additive mask."""

from __future__ import annotations

import torch as _torch

from popcorn.kernels.base import Baseline

_STR_TO_TORCH: dict[str, _torch.dtype] = {
    "bf16": _torch.bfloat16,
    "fp16": _torch.float16,
    "f16": _torch.float16,
    "f32": _torch.float32,
}
_fp8 = getattr(_torch, "float8_e4m3fn", None)
if _fp8 is not None:
    _STR_TO_TORCH["e4m3"] = _fp8


def _torch_dtype(s: str) -> _torch.dtype:
    return _STR_TO_TORCH.get(s, _torch.bfloat16)


def _block_mask_from_segments_torch(segments, n_segments, *, Q_LEN, KV_LEN, BlockMask, BS):
    """Construct a flex_attention BlockMask from segments."""
    B = segments.shape[0]
    device = segments.device
    KV_blocks = (KV_LEN + BS - 1) // BS
    Q_blocks = (Q_LEN + BS - 1) // BS

    valid = _torch.zeros(B, KV_LEN, dtype=_torch.bool, device=device)
    segs_cpu = segments.cpu()
    nsegs_cpu = n_segments.cpu()
    for b in range(B):
        n = int(nsegs_cpu[b].item())
        for i in range(n):
            st = int(segs_cpu[b, i, 0].item())
            ln = int(segs_cpu[b, i, 1].item())
            if ln > 0:
                valid[b, st : st + ln] = True

    pad_amt = KV_blocks * BS - KV_LEN
    valid_padded = _torch.nn.functional.pad(valid, (0, pad_amt))
    valid_blocks = valid_padded.view(B, KV_blocks, BS).any(-1)
    full_blocks = valid_padded.view(B, KV_blocks, BS).all(-1)
    partial_blocks = valid_blocks & ~full_blocks

    def _ordered(dense):
        num = dense.sum(dim=-1, dtype=_torch.int32)
        idx = dense.argsort(dim=-1, descending=True, stable=True).to(_torch.int32)
        return num.contiguous(), idx.contiguous()

    full_bm = full_blocks[:, None, :].expand(B, Q_blocks, KV_blocks)
    partial_bm = partial_blocks[:, None, :].expand(B, Q_blocks, KV_blocks)
    kv_num_blocks, kv_indices = _ordered(partial_bm)
    full_kv_num_blocks, full_kv_indices = _ordered(full_bm)

    def mask_mod(b, h, q, kv):
        return valid[b, kv]

    return BlockMask.from_kv_blocks(
        kv_num_blocks[:, None],
        kv_indices[:, None],
        full_kv_num_blocks[:, None],
        full_kv_indices[:, None],
        BLOCK_SIZE=BS,
        mask_mod=mask_mod,
        seq_lengths=(Q_LEN, KV_LEN),
        compute_q_blocks=False,
    )


def _baselines_flex_attention_cuda(kernel, tensors: dict) -> list[Baseline]:
    s = kernel.spec
    out_torch = _torch_dtype(s.out_dtype)

    Q = tensors["Q"]
    Kc = tensors["K_cache"]
    Vtc = tensors["Vt_cache"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    segs = tensors["segments"]
    nseg = tensors["n_segments"]

    Q4 = Q.view(s.B, s.n_q_heads, s.tpf, s.Dh)
    Kc4 = Kc.view(s.B, s.n_kv_heads, s.capacity, s.Dh)
    Vtc4 = Vtc.view(s.B, s.n_kv_heads, s.Dh, s.capacity)
    cos2 = cos.view(s.tpf, s.Dh // 2)
    sin2 = sin.view(s.tpf, s.Dh // 2)
    segs3 = segs.view(s.B, s.max_segments, 2)
    nseg1 = nseg.view(s.B)

    from torch.nn.attention.flex_attention import (
        _DEFAULT_SPARSE_BLOCK_SIZE,
        BlockMask,
        flex_attention,
    )

    bm = _block_mask_from_segments_torch(
        segs3,
        nseg1,
        Q_LEN=s.tpf,
        KV_LEN=s.capacity,
        BlockMask=BlockMask,
        BS=_DEFAULT_SPARSE_BLOCK_SIZE,
    )

    kv_compute_dtype = _torch.bfloat16

    a_dt = _torch_dtype(s.a_dtype)
    K_fresh = (_torch.randn(s.B, s.n_kv_heads, s.tpf, s.Dh, device="cuda") * 0.3).to(a_dt)
    V_fresh = (_torch.randn(s.B, s.n_kv_heads, s.tpf, s.Dh, device="cuda") * 0.3).to(a_dt)
    frame_t_t = _torch.tensor(s.num_buckets * s.pinned_dilation, device="cuda", dtype=_torch.int32)

    num_buckets = s.num_buckets
    pd = s.pinned_dilation
    L = s.L
    tpf = s.tpf

    def _full_step(Q, K, V, cos, sin, frame_t_t, Kc, Vt, bm):
        x32 = K.float()
        x0 = x32[..., 0::2]
        x1 = x32[..., 1::2]
        c = cos.float().unsqueeze(0).unsqueeze(0)
        s_ = sin.float().unsqueeze(0).unsqueeze(0)
        y0 = x0 * c - x1 * s_
        y1 = x1 * c + x0 * s_
        k_rot = _torch.cat((y0, y1), dim=-1).to(Kc.dtype)
        v_w = V.to(Kc.dtype)
        Kc[:, :, L:, :] = k_rot
        Vt[:, :, :, L:] = v_w.transpose(-1, -2)
        ft = frame_t_t.to(_torch.int64)
        bucket = (ft + (pd - 1)) // pd
        slot = bucket % num_buckets
        base = slot * tpf
        write_step = (ft % pd) == 0
        dst_base = _torch.where(
            write_step, base, _torch.tensor(L, device=ft.device, dtype=ft.dtype)
        )
        offsets = _torch.arange(tpf, device=ft.device, dtype=ft.dtype)
        dst_idx = (dst_base + offsets).to(_torch.long)
        Kc.index_copy_(2, dst_idx, k_rot)
        Vt.index_copy_(3, dst_idx, v_w.transpose(-1, -2))

        qx32 = Q.float()
        q0 = qx32[..., 0::2]
        q1 = qx32[..., 1::2]
        qy0 = q0 * c - q1 * s_
        qy1 = q1 * c + q0 * s_
        Q_rot = _torch.cat((qy0, qy1), dim=-1).to(Q.dtype)

        Kc_attn = Kc.to(kv_compute_dtype)
        V_attn = Vt.transpose(-1, -2).contiguous().to(kv_compute_dtype)
        Q_attn = Q_rot.to(kv_compute_dtype)
        return flex_attention(
            Q_attn,
            Kc_attn,
            V_attn,
            block_mask=bm,
            enable_gqa=(s.n_q_heads != s.n_kv_heads),
        )

    compiled = _torch.compile(
        _full_step,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",  # index_copy_ blocks cudagraphs
        dynamic=False,
    )

    def run():
        return compiled(Q4, K_fresh, V_fresh, cos2, sin2, frame_t_t, Kc4, Vtc4, bm).to(out_torch)

    return [Baseline("torch.compile[kv_cache+Qrope+flex_attn]", run)]


def _baselines_mlx(kernel, tensors: dict) -> list[Baseline]:
    """MLX flash-attn baseline for Metal. Cache already warmed; this
    measures Q-RoPE + masked SDPA only."""
    import mlx.core as mx

    from popcorn.backend import PT
    from popcorn.kernels.owl_attn.reference import (
        _additive_mask_from_segments,
        apply_rope_fp32_concat,
    )

    s = kernel.spec
    Q = tensors["Q"].reshape(s.B, s.n_q_heads, s.tpf, s.Dh)
    Kc = tensors["K_cache"].reshape(s.B, s.n_kv_heads, s.capacity, s.Dh)
    Vtc = tensors["Vt_cache"].reshape(s.B, s.n_kv_heads, s.Dh, s.capacity)
    cos = tensors["cos"]
    sin = tensors["sin"]
    mask = _additive_mask_from_segments(
        tensors["segments"].reshape(s.B, s.max_segments, 2),
        tensors["n_segments"].reshape(s.B),
        s.capacity,
    )
    V_cache = PT.transpose(Vtc)

    def run():
        cos_b = cos[None, None, :, :]
        sin_b = sin[None, None, :, :]
        Q_rot = apply_rope_fp32_concat(Q, cos_b, sin_b)
        Qf = PT.astype(Q_rot, PT.bfloat16)
        Kf = PT.astype(Kc, PT.bfloat16)
        Vf = PT.astype(V_cache, PT.bfloat16)
        mx.eval(PT.attention(Qf, Kf, Vf, mask=PT.astype(mask, PT.bfloat16)))

    return [Baseline("mx.fast.sdpa[Qrope+masked_sdpa]", run)]


def owl_attn_baselines(kernel, tensors: dict) -> list[Baseline]:
    from popcorn.backend import IS_METAL

    if IS_METAL:
        return _baselines_mlx(kernel, tensors)
    return _baselines_flex_attention_cuda(kernel, tensors)
