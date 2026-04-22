"""Cross-check quark.functional kernels against world_engine's CPU torch
reference for each primitive they replace. This is the correctness harness
for the Tier-1 kernel set.

Run with:

    source .venv/bin/activate
    PYTHONPATH=/Users/work/world_engine/src \
        python tests/cross_check_world_engine.py

The script skips any op that requires flex_attn (GPU-only in the current
torch build). For attention + KV-cache we do a standalone single-pass
SDPA check that doesn't touch world_engine's Attn class.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/Users/work/world_engine/src")

import numpy as np
import torch
import torch.nn.functional as F

# world_engine primitives — imported without the top-level package init
# so we don't pull in the engine's entry point.
from model import nn as we_nn

import quark.functional as pcf


def _as_mlx(x: torch.Tensor):
    import mlx.core as mx

    # torch → numpy → mlx, preserving dtype.
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


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    return float(F.cosine_similarity(a, b, dim=0))


def _report(name: str, cos: float, threshold: float = 0.999) -> None:
    status = "PASS" if cos >= threshold else "FAIL"
    print(f"  [{status}] {name:40s}  cos_sim={cos:.6f}")


# ---------------------------------------------------------------
# 1. rms_norm
# ---------------------------------------------------------------


def check_rmsnorm():
    print("rms_norm (pcf.rmsnorm vs F.rms_norm):")
    torch.manual_seed(0)
    x_t = torch.randn(512, 2048, dtype=torch.bfloat16)
    y_ref = we_nn.rms_norm(x_t)  # [B, D]

    x_mx = _as_mlx(x_t)
    y_mx = pcf.rmsnorm(x_mx)
    y_back = _as_torch(y_mx).to(torch.bfloat16)

    _report("single [512, 2048]", _cos_sim(y_ref, y_back))

    # Rank-3 reshape path.
    x_t3 = torch.randn(1, 512, 2048, dtype=torch.bfloat16)
    y3_ref = we_nn.rms_norm(x_t3)
    y3_mx = pcf.rmsnorm(_as_mlx(x_t3))
    _report("rank-3 [1, 512, 2048]", _cos_sim(y3_ref, _as_torch(y3_mx)))


# ---------------------------------------------------------------
# 2. ada_rmsnorm
# ---------------------------------------------------------------


def check_ada_rmsnorm():
    print("ada_rmsnorm (pcf.ada_rmsnorm vs we_nn.ada_rmsnorm):")
    torch.manual_seed(1)
    # world_engine shape: x [b, n*m, d], scale/bias [b, n, d].
    b, n, m, d = 1, 1, 512, 2048
    x_t = torch.randn(b, n * m, d, dtype=torch.bfloat16)
    scale_t = torch.randn(b, n, d, dtype=torch.bfloat16) * 0.1
    bias_t = torch.randn(b, n, d, dtype=torch.bfloat16) * 0.1

    y_ref = we_nn.ada_rmsnorm(x_t, scale_t, bias_t)  # [b, n*m, d]

    # quark accepts [B*M, D] and [G, D] where G=b*n, M=m.
    x_mx = _as_mlx(x_t.reshape(b * n * m, d))
    scale_mx = _as_mlx(scale_t.reshape(b * n, d))
    bias_mx = _as_mlx(bias_t.reshape(b * n, d))
    y_mx = pcf.ada_rmsnorm(x_mx, scale_mx, bias_mx)
    y_back = _as_torch(y_mx).reshape(b, n * m, d)

    _report(f"b={b} n={n} m={m} d={d}", _cos_sim(y_ref, y_back))

    # Multi-group: n=8 frames.
    b, n, m, d = 1, 8, 512, 2048
    x_t = torch.randn(b, n * m, d, dtype=torch.bfloat16)
    scale_t = torch.randn(b, n, d, dtype=torch.bfloat16) * 0.1
    bias_t = torch.randn(b, n, d, dtype=torch.bfloat16) * 0.1
    y_ref = we_nn.ada_rmsnorm(x_t, scale_t, bias_t)
    y_mx = pcf.ada_rmsnorm(
        _as_mlx(x_t.reshape(b * n * m, d)),
        _as_mlx(scale_t.reshape(b * n, d)),
        _as_mlx(bias_t.reshape(b * n, d)),
    )
    y_back = _as_torch(y_mx).reshape(b, n * m, d)
    _report(f"b={b} n={n} m={m} d={d}", _cos_sim(y_ref, y_back))


# ---------------------------------------------------------------
# 3. ada_gate (residual-fused quark form)
# ---------------------------------------------------------------


def check_ada_gate_residual():
    print("ada_gate_residual (x + we_nn.ada_gate(y, gate) vs pcf.ada_gate_residual):")
    torch.manual_seed(2)
    b, n, m, d = 1, 1, 512, 2048
    x_t = torch.randn(b, n * m, d, dtype=torch.bfloat16)
    y_t = torch.randn(b, n * m, d, dtype=torch.bfloat16)
    gate_t = torch.randn(b, n, d, dtype=torch.bfloat16) * 0.5

    # world_engine applies `x + ada_gate(h, gate)` in the block; compose here.
    out_ref = x_t + we_nn.ada_gate(y_t, gate_t)

    out_mx = pcf.ada_gate_residual(
        _as_mlx(x_t.reshape(b * n * m, d)),
        _as_mlx(y_t.reshape(b * n * m, d)),
        _as_mlx(gate_t.reshape(b * n, d)),
    )
    out_back = _as_torch(out_mx).reshape(b, n * m, d)

    _report(f"b={b} n={n} m={m} d={d}", _cos_sim(out_ref, out_back))

    b, n, m, d = 1, 8, 512, 2048
    x_t = torch.randn(b, n * m, d, dtype=torch.bfloat16)
    y_t = torch.randn(b, n * m, d, dtype=torch.bfloat16)
    gate_t = torch.randn(b, n, d, dtype=torch.bfloat16) * 0.5
    out_ref = x_t + we_nn.ada_gate(y_t, gate_t)
    out_mx = pcf.ada_gate_residual(
        _as_mlx(x_t.reshape(b * n * m, d)),
        _as_mlx(y_t.reshape(b * n * m, d)),
        _as_mlx(gate_t.reshape(b * n, d)),
    )
    out_back = _as_torch(out_mx).reshape(b, n * m, d)
    _report(f"b={b} n={n} m={m} d={d}", _cos_sim(out_ref, out_back))


# ---------------------------------------------------------------
# 4. NoiseConditioner
# ---------------------------------------------------------------


def check_noise_conditioner():
    print("noise_cond_mlp (pcf.noise_cond_mlp vs we_nn.NoiseConditioner):")
    torch.manual_seed(3)
    d_model = 2048
    fourier_dim = 512

    we_mod = we_nn.NoiseConditioner(d_model, fourier_dim=fourier_dim).eval()
    # Keep weights in fp32 (module registers no dtype coercion by default).
    with torch.no_grad():
        # Use a deterministic sigma batch big enough to clear the BM=32 gate.
        sigma_t = torch.linspace(0.01, 1.0, 64).to(torch.float32)
        emb_ref = we_mod(sigma_t)  # [64, d_model] fp32

    # Map the trained weights to MLX (bf16 for fc layers — matches quark).
    W1 = we_mod.mlp.fc1.weight.detach().to(torch.bfloat16)  # [4*d, F]
    W2 = we_mod.mlp.fc2.weight.detach().to(torch.bfloat16)  # [d, 4*d]
    freqs = we_mod.freq.detach().to(torch.float32)  # [F/2]

    emb_mx = pcf.noise_cond_mlp(
        _as_mlx(sigma_t),
        _as_mlx(freqs),
        _as_mlx(W1),
        _as_mlx(W2),
    )
    emb_back = _as_torch(emb_mx).to(torch.float32)

    _report(f"sigma_batch=64 d={d_model}", _cos_sim(emb_ref, emb_back))


# ---------------------------------------------------------------
# 5. AdaLN (world_engine's "output norm" module)
# ---------------------------------------------------------------


def check_adaln_out_norm():
    print("AdaLN (output norm: linear+silu on cond → (1+a)·rms(x) + b):")
    torch.manual_seed(4)
    d = 2048
    b, n, m = 1, 1, 512
    mod = we_nn.AdaLN(d).to(torch.bfloat16).eval()
    with torch.no_grad():
        x_t = torch.randn(b, n * m, d, dtype=torch.bfloat16)
        cond_t = torch.randn(b, n, d, dtype=torch.bfloat16)
        y_ref = mod(x_t, cond_t)

    # Reproduce in quark: silu + linear → (scale, bias) = chunk(ab, 2).
    # silu is native; the linear uses the world_engine AdaLN.fc weight.
    W_fc = mod.fc.weight.detach().to(torch.bfloat16)  # [2d, d]
    # silu(cond) @ W_fc^T  — plain gemm.
    y_silu = torch.nn.functional.silu(cond_t)
    ab = torch.nn.functional.linear(y_silu, W_fc)  # [b, n, 2d]
    scale, bias = ab.chunk(2, dim=-1)

    y_mx = pcf.ada_rmsnorm(
        _as_mlx(x_t.reshape(b * n * m, d)),
        _as_mlx(scale.reshape(b * n, d)),
        _as_mlx(bias.reshape(b * n, d)),
    )
    y_back = _as_torch(y_mx).reshape(b, n * m, d)
    _report(f"b={b} n={n} m={m} d={d}", _cos_sim(y_ref, y_back))


# ---------------------------------------------------------------
# 6. gemm + silu fusion (MLP fc1 path)
# ---------------------------------------------------------------


def check_gemm_silu():
    print("gemm[activation=silu] vs F.silu(F.linear):")
    torch.manual_seed(5)
    M, K, N = 512, 2048, 8192
    A_t = torch.randn(M, K, dtype=torch.bfloat16)
    W_t = torch.randn(N, K, dtype=torch.bfloat16)  # torch convention: [out, in]
    # Reference: silu(A @ W^T)
    y_ref = torch.nn.functional.silu(torch.nn.functional.linear(A_t, W_t))

    A_mx = _as_mlx(A_t)
    W_mx = _as_mlx(W_t)
    y_mx = pcf.gemm(A_mx, W_mx, activation="silu")
    y_back = _as_torch(y_mx)
    _report(f"M={M} N={N} K={K}", _cos_sim(y_ref, y_back))


# ---------------------------------------------------------------
# 7. single-frame SDPA (plain causal attention, no KV cache / no flex_attn)
#     This sanity-checks the quark QKV→attention→out_proj composition
#     against a pure-torch SDPA reference. owl_attn itself is exercised
#     in its own kernel-level smoke test; here we just confirm the
#     surrounding plumbing.
# ---------------------------------------------------------------


def check_attn_singlepass():
    print("single-frame SDPA (quark Q/K/V via gemm+rmsnorm vs pure torch):")
    torch.manual_seed(6)
    B, T = 1, 64
    n_heads, n_kv_heads, Dh = 32, 16, 64
    d = n_heads * Dh
    gqa_ratio = n_heads // n_kv_heads

    x_t = torch.randn(B, T, d, dtype=torch.bfloat16)
    W_qkv = torch.randn(n_heads * Dh + 2 * n_kv_heads * Dh, d, dtype=torch.bfloat16) * 0.02
    W_out = torch.randn(d, n_heads * Dh, dtype=torch.bfloat16) * 0.02

    # ── reference: torch linear → split → rms_norm(Q,K) → SDPA → linear ──
    qkv = torch.nn.functional.linear(x_t, W_qkv)  # [B, T, qkv_dim]
    q_end = n_heads * Dh
    k_end = q_end + n_kv_heads * Dh
    Q = qkv[..., :q_end].reshape(B, T, n_heads, Dh)
    K = qkv[..., q_end:k_end].reshape(B, T, n_kv_heads, Dh)
    V = qkv[..., k_end:].reshape(B, T, n_kv_heads, Dh)

    Qn = we_nn.rms_norm(Q)
    Kn = we_nn.rms_norm(K)

    # Repeat K, V for GQA.
    Kn_rep = Kn.repeat_interleave(gqa_ratio, dim=2)  # [B, T, H, Dh]
    V_rep = V.repeat_interleave(gqa_ratio, dim=2)

    Qp = Qn.permute(0, 2, 1, 3)  # [B, H, T, Dh]
    Kp = Kn_rep.permute(0, 2, 1, 3)
    Vp = V_rep.permute(0, 2, 1, 3)

    attn_ref = torch.nn.functional.scaled_dot_product_attention(Qp, Kp, Vp, is_causal=True)
    attn_ref = attn_ref.permute(0, 2, 1, 3).reshape(B, T, d)
    y_ref = torch.nn.functional.linear(attn_ref, W_out)

    # ── quark: gemm → reshape → rmsnorm → torch SDPA (no owl_attn) → gemm ──
    # (We use torch SDPA because owl_attn needs the ring cache structure
    # and won't run in this stripped-down single-pass setup.)
    x_mx = _as_mlx(x_t.reshape(B * T, d))
    qkv_mx = pcf.gemm(x_mx, _as_mlx(W_qkv))
    qkv_back = _as_torch(qkv_mx).reshape(B, T, -1).to(torch.bfloat16)

    Q_p = qkv_back[..., :q_end].reshape(B, T, n_heads, Dh)
    K_p = qkv_back[..., q_end:k_end].reshape(B, T, n_kv_heads, Dh)
    V_p = qkv_back[..., k_end:].reshape(B, T, n_kv_heads, Dh)

    # rmsnorm on Q/K using quark.
    Qn_p_flat = pcf.rmsnorm(_as_mlx(Q_p.reshape(-1, Dh)))
    Kn_p_flat = pcf.rmsnorm(_as_mlx(K_p.reshape(-1, Dh)))
    Qn_p = _as_torch(Qn_p_flat).reshape(B, T, n_heads, Dh)
    Kn_p = _as_torch(Kn_p_flat).reshape(B, T, n_kv_heads, Dh)

    Kn_rep = Kn_p.repeat_interleave(gqa_ratio, dim=2)
    V_rep = V_p.repeat_interleave(gqa_ratio, dim=2)

    Qp_p = Qn_p.permute(0, 2, 1, 3)
    Kp_p = Kn_rep.permute(0, 2, 1, 3)
    Vp_p = V_rep.permute(0, 2, 1, 3)

    attn_p = torch.nn.functional.scaled_dot_product_attention(Qp_p, Kp_p, Vp_p, is_causal=True)
    attn_p = attn_p.permute(0, 2, 1, 3).reshape(B * T, d)

    y_mx = pcf.gemm(_as_mlx(attn_p), _as_mlx(W_out))
    y_back = _as_torch(y_mx).reshape(B, T, d).to(torch.bfloat16)

    _report(f"B={B} T={T} H={n_heads}/{n_kv_heads} Dh={Dh}", _cos_sim(y_ref, y_back))


# ---------------------------------------------------------------
# driver
# ---------------------------------------------------------------


def main():
    checks = [
        check_rmsnorm,
        check_ada_rmsnorm,
        check_ada_gate_residual,
        check_noise_conditioner,
        check_adaln_out_norm,
        check_gemm_silu,
        check_attn_singlepass,
    ]
    for fn in checks:
        fn()
        print()


if __name__ == "__main__":
    main()
