"""Full single-frame forward comparison: world_engine.WorldModel vs a
popcorn re-implementation of the same ``forward``. Both use the *same*
weights and the *same* Attn module (flex_attn monkey-patched to plain
SDPA so it runs on CPU). What differs is every other op in the forward:

    patchify             torch Conv2d            pcf.patchify_2x2
    denoise_step_emb     NoiseConditioner        pcf.noise_cond_mlp
    cond_head            silu + Linear×6         silu + pcf.gemm×6
    ada_rmsnorm          torch                   pcf.ada_rmsnorm
    ada_gate + residual  torch                   pcf.ada_gate_residual
    mlp                  silu + Linear×2         pcf.gemm(act="silu") + pcf.gemm
    out_norm             torch AdaLN             pcf.ada_rmsnorm
    unpatchify           torch Linear            pcf.gemm + bias add

The Attn path is identical (world_engine's own module, monkey-patched
flex_attn) so any output difference is attributable to the popcorn
replacements.

Uses a small config (~50M params) so CPU runtime is bearable.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/Users/work/world_engine/src")

import numpy as np
import torch
import torch.nn.attention.flex_attention as _fa_mod
import torch.nn.functional as F

# Ensure any subsequent `from torch.nn.attention.flex_attention import flex_attention`
# also resolves to the patched version. world_engine.attn already did
# the import at module load; we overwrite there too.
from model import attn as _we_attn
from model import kv_cache as _we_kv
from model import world_model as _we_world
from omegaconf import OmegaConf
from tensordict import TensorDict

import popcorn.functional as pcf

# ── Monkey-patch flex_attention → plain SDPA BEFORE importing world_engine ──
# flex_attention is CUDA-only in this torch build; SDPA gives us the
# same mathematical result on CPU when the block_mask would be no-op
# (first frame, empty history → mask reduces to causal-or-full over
# the current frame's tokens, which matches SDPA(is_causal=True) for
# the causal setting).


def _fake_flex_attention(q, k, v, block_mask=None, enable_gqa=False, **kwargs):
    # block_mask is the ring-cache segment mask; on frame 0 with an
    # empty cache, the only valid KV block is the tail (current frame),
    # so the effective mask is "within-frame full attention" (non-causal
    # across the 512 tokens of one frame — this is what waypoint does).
    return F.scaled_dot_product_attention(q, k, v, enable_gqa=enable_gqa, is_causal=False)


_fa_mod.flex_attention = _fake_flex_attention


_we_attn.flex_attention = _fake_flex_attention
# Also neutralize make_block_mask — our fake SDPA ignores the mask.
_we_kv.make_block_mask = lambda T, L, written: None


# ---------------------------------------------------------------
# Small test config (keeps CPU runtime under a minute).
# ---------------------------------------------------------------


def make_config():
    return OmegaConf.create(
        {
            "model_type": "waypoint-1.5",
            "base_fps": 15,
            "inference_fps": 60,
            "temporal_compression": 4,
            "taehv_ae": True,
            "ae_uri": "none",
            "prompt_conditioning": None,
            "channels": 4,  # smaller latent depth for the test
            "n_layers": 2,  # only 2 blocks to keep this fast
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_model": 256,
            "mlp_ratio": 4,
            "mlp_gradient_checkpointing": False,
            "block_gradient_checkpointing": False,
            "causal": True,
            "moe": False,
            "n_buttons": 16,
            # tpf must be a multiple of the flex_attention default sparse
            # block size (128). 8*16 = 128 is the smallest valid size.
            "tokens_per_frame": 128,
            "height": 8,
            "width": 16,
            "conv_kw": {"kernel_size": [2, 2], "stride": [2, 2]},
            "patch": [2, 2],
            "local_window": 4,
            "global_window": 16,
            "global_pinned_dilation": 4,
            "global_attn_period": 2,
            "global_attn_offset": -1,
            "n_frames": 32,
            "rope_impl": "ortho",
            "rope_theta": 10000.0,
            "rope_nyquist_frac": 0.8,
            "value_residual": True,
            "gated_attn": False,
            "noise_conditioning": "wan",
            "ctrl_conditioning": None,  # simplify: skip ctrl_mlpfusion
            "ctrl_cond_dropout": 0.0,
            "ctrl_conditioning_period": None,
            "prompt_cond_dropout": 0.0,
            "scheduler_sigmas": [1.0, 0.9, 0.75, 0.3, 0.0],
            "auto_aspect_ratio": True,
            "prompt_encoder_uri": "none",
        }
    )


# ---------------------------------------------------------------
# popcorn reimplementation of WorldModel.forward
# ---------------------------------------------------------------


def _as_mlx(x: torch.Tensor):
    import mlx.core as mx

    a = x.detach().contiguous().cpu()
    if a.dtype == torch.bfloat16:
        return mx.array(a.to(torch.float32).numpy()).astype(mx.bfloat16)
    return mx.array(a.numpy())


def _as_torch(x) -> torch.Tensor:
    import mlx.core as mx

    if isinstance(x, torch.Tensor):
        return x
    if x.dtype == mx.bfloat16:
        return torch.from_numpy(np.asarray(x.astype(mx.float32))).to(torch.bfloat16)
    return torch.from_numpy(np.asarray(x))


def _pcf_gemm_padded(A_t: torch.Tensor, W_t: torch.Tensor, activation=None) -> torch.Tensor:
    """pcf.gemm with M-pad when M < BM_MIN (the smallest BM in the
    gemm tune space). Autotune handles N divisibility natively now
    that the tune space includes BN=16, so no N-pad is needed.
    """
    M, K = int(A_t.shape[0]), int(A_t.shape[1])
    BM_MIN = 16
    m_pad = (BM_MIN - M) if M < BM_MIN else 0
    if m_pad:
        A_t = torch.cat([A_t, A_t.new_zeros(m_pad, K)], dim=0)
    y = _as_torch(pcf.gemm(_as_mlx(A_t), _as_mlx(W_t), activation=activation))
    return y.to(A_t.dtype)[:M]


def _pcf_gemm_torch(A_t, W_t):
    return _pcf_gemm_padded(A_t, W_t)


def _pcf_gemm_silu_torch(A_t, W_t):
    return _pcf_gemm_padded(A_t, W_t, activation="silu")


def _pcf_rmsnorm_torch(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    y = pcf.rmsnorm(_as_mlx(x.reshape(-1, orig_shape[-1])))
    return _as_torch(y).reshape(orig_shape).to(x.dtype)


def _pcf_ada_rmsnorm_torch(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor):
    # x: [b, n*m, d], scale/bias: [b, n, d]
    b, nm, d = x.shape
    _, n, _ = scale.shape
    x_flat = x.reshape(b * nm, d)
    scale_flat = scale.reshape(b * n, d)
    bias_flat = bias.reshape(b * n, d)
    y = pcf.ada_rmsnorm(_as_mlx(x_flat), _as_mlx(scale_flat), _as_mlx(bias_flat))
    return _as_torch(y).reshape(b, nm, d).to(x.dtype)


def _pcf_ada_gate_residual_torch(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor):
    b, nm, d = x.shape
    _, n, _ = gate.shape
    out = pcf.ada_gate_residual(
        _as_mlx(x.reshape(b * nm, d)),
        _as_mlx(y.reshape(b * nm, d)),
        _as_mlx(gate.reshape(b * n, d)),
    )
    return _as_torch(out).reshape(b, nm, d).to(x.dtype)


def popcorn_forward(we_model, x, sigma, frame_timestamp, mouse, button, scroll):
    """Mirror of WorldModel.forward using popcorn ops."""
    cfg = we_model.config
    B, N, C, H, W = x.shape
    ph, pw = we_model.patch
    Hp, Wp = H // ph, W // pw

    # pos_ids (same as world_model.forward)
    T = cfg.tokens_per_frame
    idx = torch.arange(T, dtype=torch.long)
    t_pos_1f = frame_timestamp[0, 0].expand(T)
    y_pos_1f = idx.div(cfg.width, rounding_mode="floor")
    x_pos_1f = idx.remainder(cfg.width)
    pos_ids = TensorDict(
        {
            "f_pos": t_pos_1f[None],
            "t_pos": t_pos_1f[None],
            "y_pos": y_pos_1f[None],
            "x_pos": x_pos_1f[None],
        },
        batch_size=[1, T],
    )

    # 1) Noise embedding via popcorn (world_engine expects [B, N, D]).
    cond = pcf.noise_cond_mlp(
        _as_mlx(sigma.to(torch.float32)),
        _as_mlx(we_model.denoise_step_emb.freq),
        _as_mlx(we_model.denoise_step_emb.mlp.fc1.weight),
        _as_mlx(we_model.denoise_step_emb.mlp.fc2.weight),
    )
    cond = _as_torch(cond).reshape(B, N, cfg.d_model).to(x.dtype)

    # 2) Patchify via popcorn: [B, N, C, H, W] → [B, N*Hp*Wp, D]
    patched = pcf.patchify_2x2(
        _as_mlx(x.reshape(B * N, C, H, W)),
        _as_mlx(we_model.patchify.weight),
    )
    h = _as_torch(patched).to(x.dtype)  # [B*N*Hp*Wp, D]
    h = h.reshape(B, N * Hp * Wp, cfg.d_model)

    # 3) Precompute RoPE angles (world_engine's OrthoRoPEAngles — untouched).
    rope_angles = we_model.transformer.rope_angles(pos_ids)
    v = None

    # 4) Per-block loop. CondHead is SHARED across layers in waypoint-1.5.
    shared_cond_proj = we_model.transformer.blocks[0].cond_head.cond_proj
    for block in we_model.transformer.blocks:
        # CondHead: (optional bias_in) → silu → 6 linear projections.
        c = cond + block.cond_head.bias_in if block.cond_head.bias_in is not None else cond
        c_silu = F.silu(c)  # [B, N, D] fp32-ish
        s0, b0, g0, s1, b1, g1 = (
            _pcf_gemm_torch(c_silu.reshape(-1, cfg.d_model), p.weight).reshape(B, N, cfg.d_model)
            for p in shared_cond_proj
        )

        # Self-attn sub-block.
        residual = h
        h_norm = _pcf_ada_rmsnorm_torch(h, s0, b0)
        attn_out, v = block.attn(h_norm, pos_ids, rope_angles, v, kv_cache=_kv)
        h = _pcf_ada_gate_residual_torch(residual, attn_out, g0)

        # MLP sub-block.
        h_norm2 = _pcf_ada_rmsnorm_torch(h, s1, b1)
        z = _pcf_gemm_silu_torch(h_norm2.reshape(-1, cfg.d_model), block.mlp.fc1.weight).reshape(
            B, N * Hp * Wp, -1
        )
        h_mlp = _pcf_gemm_torch(z.reshape(-1, z.shape[-1]), block.mlp.fc2.weight).reshape(
            B, N * Hp * Wp, cfg.d_model
        )
        h = _pcf_ada_gate_residual_torch(h, h_mlp, g1)

    # 5) Out norm: silu(AdaLN(x, cond)) — AdaLN = silu+linear on cond → ada_rmsnorm.
    c_silu_on = F.silu(cond)
    ab = F.linear(c_silu_on, we_model.out_norm.fc.weight)
    s_on, b_on = ab.chunk(2, dim=-1)
    h = _pcf_ada_rmsnorm_torch(h, s_on, b_on)
    h = F.silu(h)

    # 6) Unpatchify via popcorn gemm + bias add.
    # we_model.unpatchify is a Linear; after state_dict fixup its weight
    # is [C*ph*pw, D] with bias [C*ph*pw].
    W_up = we_model.unpatchify.weight  # [C*ph*pw, D]
    b_up = we_model.unpatchify.bias  # [C*ph*pw]
    h_flat = _pcf_gemm_torch(h.reshape(-1, cfg.d_model), W_up)
    h_flat = h_flat + b_up  # [M, C*ph*pw]

    # Rearrange to [B, N, C, H, W].
    h_flat = h_flat.reshape(B, N, Hp, Wp, C, ph, pw)
    out = h_flat.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, N, C, H, W)
    return out


# we need a kv_cache ref — kept as a module global so the closure above works
_kv = None


# ---------------------------------------------------------------
# Harness
# ---------------------------------------------------------------


def main():
    torch.manual_seed(7)
    cfg = make_config()

    # Build model in fp32 for max numerical precision on the reference.
    dtype = torch.bfloat16
    model = _we_world.WorldModel(cfg).to(dtype=dtype).eval()

    # KV-cache used by both paths (reset before each run).
    global _kv
    _kv = _we_kv.StaticKVCache(cfg, batch_size=1, dtype=dtype)
    _kv.to(dtype=dtype)

    # Inputs for one-frame forward.
    B, N, C = 1, 1, cfg.channels
    H = cfg.height * cfg.patch[0]
    W = cfg.width * cfg.patch[1]
    x = torch.randn(B, N, C, H, W, dtype=dtype)
    sigma = torch.tensor([[0.7]], dtype=dtype)
    frame_timestamp = torch.tensor([[0]], dtype=torch.long)
    mouse = torch.zeros(B, N, 2, dtype=dtype)
    button = torch.zeros(B, N, cfg.n_buttons, dtype=dtype)
    scroll = torch.zeros(B, N, 1, dtype=dtype)

    # ── Reference ──
    _kv.reset()
    _kv.set_frozen(False)  # first-frame: persist into ring
    with torch.no_grad():
        y_ref = model(
            x,
            sigma,
            frame_timestamp,
            mouse=mouse,
            button=button,
            scroll=scroll,
            kv_cache=_kv,
        )
    print(f"reference output shape: {tuple(y_ref.shape)}")

    # ── popcorn path ──
    # Reset the KV cache for a clean second run (cache state is stateful).
    _kv.reset()
    _kv.set_frozen(False)
    with torch.no_grad():
        y_mine = popcorn_forward(model, x, sigma, frame_timestamp, mouse, button, scroll)
    print(f"popcorn   output shape: {tuple(y_mine.shape)}")

    # Compare.
    a = y_ref.to(torch.float32).flatten()
    b = y_mine.to(torch.float32).flatten()
    cos = float(F.cosine_similarity(a, b, dim=0))
    mae = float((a - b).abs().mean())
    rel = mae / (a.abs().mean().clamp_min(1e-9))
    print(f"cos_sim={cos:.6f}  mae={mae:.4e}  rel_mae={rel:.4e}")
    # With 2 layers of popcorn replacements, expect cos_sim > 0.999.
    threshold = 0.999
    if cos >= threshold:
        print(f"[PASS] full forward matches reference (threshold={threshold})")
    else:
        print(f"[FAIL] cos_sim {cos:.6f} < {threshold}")


if __name__ == "__main__":
    main()
