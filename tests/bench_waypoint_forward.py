"""Waypoint-1.5 @ 360p forward-pass benchmark (excl. kv_cache_update).

Config matches the user's waypoint-1.5 yaml:
    d_model = 2048, n_layers = 24, n_heads = 32, n_kv_heads = 16,
    mlp_ratio = 4 → mlp_dim = 8192, Dh = 64, gqa_ratio = 2.

Resolution scaled to 360p: tokens_per_frame = 128 (H_spatial=8, W_spatial=16).
Global-attn pattern: local_window=16, global_window=128, dilation=8,
period=4, offset=-1 → layers 3, 7, 11, 15, 19, 23 are global.

Timed region per forward pass (24 layers):
    patchify
    24× { ada_rmsnorm, qkv, rmsnorm(Q), rmsnorm(K), value_residual,
          owl_attn, out_proj, ada_gate_residual,
          ada_rmsnorm, fc1(silu), fc2, ada_gate_residual }
    out_norm (silu + linear + ada_rmsnorm + silu)
    unpatchify

Excluded from the timing: kv_cache_update (cache is primed in a
warmup pass and held frozen during the timed passes).

Autotune runs under ``with popcorn.max_autotune()`` for every distinct
(spec, config) encountered. First pass includes the autotune cost.
"""

from __future__ import annotations

import time

import mlx.core as mx

import popcorn
import popcorn.functional as pcf

# ---------------------------------------------------------------
# Config — mirrors waypoint-1.5 yaml, tokens_per_frame scaled for 360p.
# ---------------------------------------------------------------


class Cfg:
    d_model = 2048
    n_layers = 24
    n_heads = 32
    n_kv_heads = 16
    mlp_ratio = 4
    Dh = 64
    gqa_ratio = 2  # n_heads // n_kv_heads
    mlp_dim = 8192
    channels = 32  # latent depth
    patch = (2, 2)

    # 360p spatial → tpf = 128
    H_spatial = 8
    W_spatial = 16
    tokens_per_frame = 128  # 8 * 16

    # Attention window pattern.
    local_window = 16
    global_window = 128
    global_pinned_dilation = 8
    global_attn_period = 4
    global_attn_offset = -1

    @classmethod
    def is_global_layer(cls, i: int) -> bool:
        return (i - cls.global_attn_offset) % cls.global_attn_period == 0

    @classmethod
    def num_buckets(cls, i: int) -> int:
        return (
            cls.global_window // cls.global_pinned_dilation
            if cls.is_global_layer(i)
            else cls.local_window
        )

    @classmethod
    def pinned_dilation(cls, i: int) -> int:
        return cls.global_pinned_dilation if cls.is_global_layer(i) else 1


# ---------------------------------------------------------------
# Weights (random, bf16) + per-layer KV cache buffers.
# ---------------------------------------------------------------


def _randn(*shape, scale=0.02):
    return (mx.random.normal(shape) * scale).astype(mx.bfloat16)


def build_weights():
    d = Cfg.d_model
    Dh = Cfg.Dh
    ph, pw = Cfg.patch
    qkv_out = Cfg.n_heads * Dh + 2 * Cfg.n_kv_heads * Dh  # 4096

    layers = []
    for _ in range(Cfg.n_layers):
        layers.append(
            {
                "qkv": _randn(qkv_out, d),
                "out": _randn(d, Cfg.n_heads * Dh),
                "fc1": _randn(Cfg.mlp_dim, d),
                "fc2": _randn(d, Cfg.mlp_dim),
                "value_lamb": mx.array([0.5], dtype=mx.float32),
            }
        )

    cond_proj = [_randn(d, d, scale=0.01) for _ in range(6)]
    return {
        "patchify": _randn(d, Cfg.channels, ph, pw),
        "unpatch_w": _randn(Cfg.channels * ph * pw, d),
        "unpatch_b": _randn(Cfg.channels * ph * pw),
        "out_norm_fc": _randn(2 * d, d, scale=0.01),
        "cond_bias_in": _randn(d, scale=0.01),
        "cond_proj": cond_proj,
        "noise_fc1": _randn(d * 4, 512),
        "noise_fc2": _randn(d, d * 4),
        "fourier_freqs": mx.array([10000.0 ** (-i / 255) for i in range(256)], dtype=mx.float32),
        "layers": layers,
    }


def build_kv_caches(dtype=mx.bfloat16):
    tpf = Cfg.tokens_per_frame
    Hk = Cfg.n_kv_heads
    Dh = Cfg.Dh
    caches = []
    for i in range(Cfg.n_layers):
        nb = Cfg.num_buckets(i)
        cap = nb * tpf + tpf
        caches.append(
            {
                "K_cache": mx.zeros((1 * Hk * cap, Dh), dtype=dtype),
                "Vt_cache": mx.zeros((1 * Hk * Dh, cap), dtype=dtype),
                "segments": mx.zeros((1 * 3 * 2,), dtype=mx.int32),
                "n_segments": mx.zeros((1,), dtype=mx.int32),
                "frame_t": mx.zeros((1,), dtype=mx.int32),
                "frozen": mx.zeros((1,), dtype=mx.int32),
            }
        )
    return caches


# ---------------------------------------------------------------
# Forward pass (one model eval at a given sigma). kv_cache_update is
# called in *warmup* mode but skipped in the timed region (flag).
# ---------------------------------------------------------------


def forward(weights, caches, x_latent, sigma, *, skip_kv_update: bool):
    """One full forward. Returns the unpatchified delta [B, N, C, H, W]."""
    d = Cfg.d_model
    Dh = Cfg.Dh
    tpf = Cfg.tokens_per_frame
    Hk = Cfg.n_kv_heads
    nq = Cfg.n_heads
    ph, pw = Cfg.patch
    Hp, Wp = Cfg.H_spatial, Cfg.W_spatial
    B, N, C, H, W = 1, 1, Cfg.channels, Hp * ph, Wp * pw

    # Noise embed — sigma [1] → [1, 1, d_model] via pcf.noise_cond_mlp.
    cond = pcf.noise_cond_mlp(
        mx.array([sigma], dtype=mx.float32),
        weights["fourier_freqs"],
        weights["noise_fc1"],
        weights["noise_fc2"],
    )  # [1, d_model] after reshape internally

    # Cond head: shared across layers. bias_in + silu + 6 gemms (M=1
    # padded to 16 inside the helper below).
    cond_flat = cond.reshape(-1, d) + weights["cond_bias_in"]  # [1, d]
    # pad M to 16 for autotune validity (no M=1 path)
    cond_pad = mx.concatenate([cond_flat, mx.zeros((15, d), dtype=cond_flat.dtype)], axis=0)
    cond_silu = cond_pad * mx.sigmoid(cond_pad.astype(mx.float32)).astype(cond_pad.dtype)
    s0 = pcf.gemm(cond_silu, weights["cond_proj"][0])[:1]
    b0 = pcf.gemm(cond_silu, weights["cond_proj"][1])[:1]
    g0 = pcf.gemm(cond_silu, weights["cond_proj"][2])[:1]
    s1 = pcf.gemm(cond_silu, weights["cond_proj"][3])[:1]
    b1 = pcf.gemm(cond_silu, weights["cond_proj"][4])[:1]
    g1 = pcf.gemm(cond_silu, weights["cond_proj"][5])[:1]

    # Patchify: conv2d(kernel=stride=patch) → reshape+gemm.
    x = pcf.patchify_2x2(x_latent.reshape(B * N, C, H, W), weights["patchify"])
    # [M, d_model] where M = B*N*Hp*Wp = 128

    v1 = None
    for li, lw in enumerate(weights["layers"]):
        residual = x
        x_norm = pcf.ada_rmsnorm(x, s0, b0)
        qkv = pcf.gemm(x_norm, lw["qkv"])
        # Split.
        q_end = nq * Dh
        k_end = q_end + Hk * Dh
        Q = qkv[:, :q_end].reshape(tpf, nq, Dh)
        K = qkv[:, q_end:k_end].reshape(tpf, Hk, Dh)
        V = qkv[:, k_end:].reshape(tpf, Hk, Dh)

        Q = pcf.rmsnorm(Q.reshape(-1, Dh)).reshape(tpf, nq, Dh)
        K = pcf.rmsnorm(K.reshape(-1, Dh)).reshape(tpf, Hk, Dh)

        # Value residual.
        if li == 0:
            v1 = V
        else:
            V = pcf.value_residual(V, v1, lw["value_lamb"])

        # KV cache update (skipped in the timed region).
        K_flat = mx.transpose(K, (1, 0, 2)).reshape(1 * Hk * tpf, Dh)
        V_flat = mx.transpose(V, (1, 0, 2)).reshape(1 * Hk * tpf, Dh)

        cache = caches[li]
        if not skip_kv_update:
            K_cache, Vt_cache, segments, n_segments = pcf.kv_cache_update(
                K_flat,
                V_flat,
                cache["frame_t"],
                cache["frozen"],
                cache["Vt_cache"],
                cache["segments"],
                cache["n_segments"],
                cache["K_cache"],
                B=1,
                n_kv_heads=Hk,
                H_spatial=Hp,
                W_spatial=Wp,
                num_buckets=Cfg.num_buckets(li),
                pinned_dilation=Cfg.pinned_dilation(li),
            )
            cache["K_cache"] = K_cache
            cache["Vt_cache"] = Vt_cache
            cache["segments"] = segments
            cache["n_segments"] = n_segments

        Q_flat = mx.transpose(Q, (1, 0, 2)).reshape(1 * nq * tpf, Dh)
        attn_out = pcf.owl_attn(
            Q_flat,
            cache["K_cache"],
            cache["Vt_cache"],
            cache["segments"],
            cache["n_segments"],
            B=1,
            n_kv_heads=Hk,
            gqa_ratio=Cfg.gqa_ratio,
            H_spatial=Hp,
            W_spatial=Wp,
            num_buckets=Cfg.num_buckets(li),
            pinned_dilation=Cfg.pinned_dilation(li),
            frame_t=cache["frame_t"],
        )  # [1*nq*tpf, Dh] f32
        attn_out = (
            mx.transpose(attn_out.reshape(1, nq, tpf, Dh), (0, 2, 1, 3))
            .reshape(tpf, nq * Dh)
            .astype(x.dtype)
        )

        h_attn = pcf.gemm(attn_out, lw["out"])
        x = pcf.ada_gate_residual(residual, h_attn, g0)

        # MLP.
        x_norm2 = pcf.ada_rmsnorm(x, s1, b1)
        z = pcf.gemm(x_norm2, lw["fc1"], activation="silu")
        h_mlp = pcf.gemm(z, lw["fc2"])
        x = pcf.ada_gate_residual(x, h_mlp, g1)

    # Out norm: silu(ada_rmsnorm(x, s_on, b_on)) with (s_on, b_on) from
    # linear(silu(cond)).
    ab = pcf.gemm(cond_silu, weights["out_norm_fc"])[:1]
    s_on, b_on = ab[:, :d], ab[:, d:]
    x = pcf.ada_rmsnorm(x, s_on, b_on)
    x = x * mx.sigmoid(x.astype(mx.float32)).astype(x.dtype)

    # Unpatchify: gemm + bias.
    h_flat = pcf.gemm(x, weights["unpatch_w"]) + weights["unpatch_b"]
    h_flat = h_flat.reshape(1, 1, Hp, Wp, C, ph, pw)
    out = mx.transpose(h_flat, (0, 1, 4, 2, 5, 3, 6)).reshape(1, 1, C, H, W)
    return out


# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------


def main():
    print("waypoint-1.5 @ 360p forward benchmark")
    print(
        f"  d_model={Cfg.d_model}, n_layers={Cfg.n_layers}, "
        f"n_heads={Cfg.n_heads}/{Cfg.n_kv_heads}, Dh={Cfg.Dh}"
    )
    print(
        f"  tokens_per_frame={Cfg.tokens_per_frame} "
        f"(H={Cfg.H_spatial}, W={Cfg.W_spatial}), mlp_dim={Cfg.mlp_dim}"
    )
    gl = [i for i in range(Cfg.n_layers) if Cfg.is_global_layer(i)]
    print(f"  global-attn layers: {gl}")

    weights = build_weights()
    caches = build_kv_caches()

    B, N, C = 1, 1, Cfg.channels
    H = Cfg.H_spatial * Cfg.patch[0]
    W = Cfg.W_spatial * Cfg.patch[1]
    x = mx.random.normal((B, N, C, H, W)).astype(mx.bfloat16)

    # ── Autotune every distinct kernel call on this workload ──
    print("\nautotuning (max_autotune, full genetic search) …")
    t0 = time.perf_counter()
    with popcorn.max_autotune():
        out = forward(weights, caches, x, 0.7, skip_kv_update=False)
        mx.eval(out)
    print(f"  first-pass (autotune + prime cache): {time.perf_counter() - t0:.2f}s")

    # Prime-cache pass done. Freeze it and stop calling kv_cache_update
    # in the timed region.
    print("\ntimed passes (kv_cache_update excluded) …")
    N_WARM = 3
    N_ITERS = 10

    # Warmup (no autotune, compiled configs already cached).
    for _ in range(N_WARM):
        out = forward(weights, caches, x, 0.5, skip_kv_update=True)
    mx.eval(out)

    # Timed.
    times = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        out = forward(weights, caches, x, 0.5, skip_kv_update=True)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    mean_ms = sum(times) / len(times)
    median_ms = times[len(times) // 2]
    p90_ms = times[min(len(times) - 1, int(len(times) * 0.9))]
    print(f"  iters={N_ITERS}")
    print(f"  min    : {times[0]:.2f} ms")
    print(f"  median : {median_ms:.2f} ms")
    print(f"  mean   : {mean_ms:.2f} ms")
    print(f"  p90    : {p90_ms:.2f} ms")
    print(f"  max    : {times[-1]:.2f} ms")
    print(f"\n  target fps for 60 fps: {1000 / 60:.1f} ms/frame")
    print(
        f"  achieved fps (median): {1000 / median_ms:.1f} fps "
        f"({median_ms / (1000 / 60):.2f}× target budget)"
    )


if __name__ == "__main__":
    main()
