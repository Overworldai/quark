"""State-dict remap + ctrl buffer builder for ``Waypoint15``.

Kept separate from ``waypoint_15.py`` so the model definition stays
under the 800-line hard cap. Nothing here is meant to be imported
directly — ``Waypoint15`` / ``Waypoint15Config`` re-export the
public names (``remap_world_engine_state_dict``,
``Waypoint15.make_ctrl_buffer``, ``Waypoint15.from_world_engine_hub``).
"""

from __future__ import annotations

import sys

_IS_METAL = sys.platform == "darwin"


# ---------------------------------------------------------------
# Backend-dispatch helpers (CUDA QuarkTensor vs Metal mx.array)
# ---------------------------------------------------------------


def _reshape(x, *shape):
    if _IS_METAL:
        import mlx.core as mx

        return mx.reshape(x, shape)
    return x.reshape(*shape)


def _permute(x, dims):
    if _IS_METAL:
        import mlx.core as mx

        return mx.transpose(x, dims)
    return x.permute(*dims)


def _cat(tensors, dim=0):
    if _IS_METAL:
        import mlx.core as mx

        return mx.concatenate(tensors, axis=dim)
    from quark.runtime.tensor import QuarkTensor

    return QuarkTensor.cat(tensors, dim=dim)


def _astype(x, dtype: str):
    if _IS_METAL:
        import mlx.core as mx

        _dt = {"bf16": mx.bfloat16, "f16": mx.float16, "f32": mx.float32, "s32": mx.int32}
        return x.astype(_dt.get(dtype, mx.bfloat16))
    return x.astype(dtype)


def _from_numpy(arr, dtype: str = "bf16"):
    if _IS_METAL:
        import mlx.core as mx
        import numpy as _np

        if dtype == "bf16" and arr.dtype == _np.float32:
            u16 = (arr.view(_np.uint32) >> 16).astype(_np.uint16)
            return mx.array(u16).view(mx.bfloat16)
        return mx.array(arr)
    from quark.runtime.tensor import QuarkTensor

    return QuarkTensor.from_numpy(arr, dtype=dtype)


def _tile_b_to_patched(b, C: int, ph: int, pw: int):
    """Tile ``[C]`` bias into ``[C * ph * pw]`` by repeating each
    element ``ph*pw`` times. Used by the unpatchify bias remap.
    """
    if _IS_METAL:
        import mlx.core as mx

        return mx.tile(b[:, None, None], (1, ph, pw)).reshape(-1)
    import struct as _struct

    from quark.runtime.tensor import QuarkTensor

    b_f32 = b.astype("f32")
    b_vals = list(_struct.unpack(f"<{C}f", b_f32.to_bytes()))
    tiled = []
    for v in b_vals:
        tiled.extend([v] * (ph * pw))
    target_dtype = b.dtype if hasattr(b, "dtype") else "bf16"
    return QuarkTensor.from_list(tiled, dtype="f32").astype(target_dtype)


# ---------------------------------------------------------------
# State-dict remap (world_engine → quark)
# ---------------------------------------------------------------


def remap_world_engine_state_dict(raw_sd: dict, cfg) -> dict:
    """Translate a world_engine-saved state dict into the layout
    ``Waypoint15`` expects (merged ``qkv_proj``, split ``ctrl_fusion``,
    per-block ``attn_cond`` / ``mlp_cond`` heads, flattened patch /
    unpatch weights, tiled unpatchify bias, padded ctrl-emb fc1, …).

    Accepts ``QuarkTensor`` (CUDA) or ``mx.array`` (Metal) in the
    input dict — both paths share the dispatch helpers above.

    Caller feeds the result directly into
    ``model.load_state_dict(..., strict=False)``.
    """
    sd: dict = {}
    d = cfg.d_model
    ph, pw = cfg.patch
    C = cfg.channels

    w = raw_sd["patchify.weight"]
    sd["patchify.weight"] = _reshape(w, d, C * ph * pw) if w.ndim == 4 else w

    w = raw_sd["unpatchify.weight"]
    if w.ndim == 4:
        w = _reshape(_permute(w, (1, 2, 3, 0)), C * ph * pw, d)
    sd["unpatchify.weight"] = w

    b = raw_sd["unpatchify.bias"]
    if int(b.shape[0]) == C:
        b = _tile_b_to_patched(b, C, ph, pw)
    sd["unpatchify.bias"] = b

    sd["noise_fc1.weight"] = raw_sd["denoise_step_emb.mlp.fc1.weight"]
    sd["noise_fc2.weight"] = raw_sd["denoise_step_emb.mlp.fc2.weight"]
    sd["out_norm_proj.weight"] = raw_sd["out_norm.fc.weight"]

    for i in range(cfg.n_layers):
        p = f"transformer.blocks.{i}."

        # Per-block conditioning: shared cond_head (both attn & mlp
        # paths share bias_in) OR separate attn_cond_head /
        # mlp_cond_head with distinct bias_in.
        if p + "cond_head.bias_in" in raw_sd:
            sd[f"blocks.{i}.attn_cond_bias_in"] = raw_sd[p + "cond_head.bias_in"]
            sd[f"blocks.{i}.mlp_cond_bias_in"] = raw_sd[p + "cond_head.bias_in"]
            for j in range(6):
                sd[f"blocks.{i}.cond_projs.{j}.weight"] = raw_sd[
                    p + f"cond_head.cond_proj.{j}.weight"
                ]
        else:
            sd[f"blocks.{i}.attn_cond_bias_in"] = raw_sd[p + "attn_cond_head.bias_in"]
            sd[f"blocks.{i}.mlp_cond_bias_in"] = raw_sd[p + "mlp_cond_head.bias_in"]
            for j in range(3):
                sd[f"blocks.{i}.cond_projs.{j}.weight"] = raw_sd[
                    p + f"attn_cond_head.cond_proj.{j}.weight"
                ]
                sd[f"blocks.{i}.cond_projs.{j + 3}.weight"] = raw_sd[
                    p + f"mlp_cond_head.cond_proj.{j}.weight"
                ]

        q_w = raw_sd[p + "attn.q_proj.weight"]
        k_w = raw_sd[p + "attn.k_proj.weight"]
        v_w = raw_sd[p + "attn.v_proj.weight"]
        sd[f"blocks.{i}.qkv_proj.weight"] = _cat([q_w, k_w, v_w], dim=0)
        sd[f"blocks.{i}.out_proj.weight"] = raw_sd[p + "attn.out_proj.weight"]

        # MLP block — either dense (mlp.fc1 / mlp.fc2) or MoE
        # (mlp.router.weight + mlp.expert_in + mlp.expert_out). Sniff the
        # checkpoint instead of the cfg so a "dense weights into MoE
        # model" mistake fails loud here rather than at forward time.
        moe_router_key = next(
            (
                p + prefix + "router.weight"
                for prefix in ("mlp.", "dit_mlp.")
                if p + prefix + "router.weight" in raw_sd
            ),
            None,
        )
        if moe_router_key is not None:
            # MoE branch — flatten [E, H, D] / [E, D, H] to the [E*H, D]
            # / [E*D, H] layout the moe_inproj/outproj kernels expect.
            # Same memory, just collapse the leading two dims.
            base = moe_router_key[: -len("router.weight")]
            sd[f"blocks.{i}.mlp.router.weight"] = raw_sd[moe_router_key]
            ein = raw_sd[base + "expert_in"]
            eout = raw_sd[base + "expert_out"]
            E = cfg.moe_n_experts
            H = cfg.mlp_dim // cfg.moe_top_k
            D = cfg.d_model
            if ein.ndim == 3:
                ein = _reshape(ein, E * H, D)
            if eout.ndim == 3:
                eout = _reshape(eout, E * D, H)
            sd[f"blocks.{i}.mlp.expert_in"] = ein
            sd[f"blocks.{i}.mlp.expert_out"] = eout
        else:
            fc1 = p + ("mlp.fc1.weight" if p + "mlp.fc1.weight" in raw_sd else "dit_mlp.fc1.weight")
            fc2 = p + ("mlp.fc2.weight" if p + "mlp.fc2.weight" in raw_sd else "dit_mlp.fc2.weight")
            sd[f"blocks.{i}.mlp.fc1.weight"] = raw_sd[fc1]
            sd[f"blocks.{i}.mlp.fc2.weight"] = raw_sd[fc2]

        if cfg.value_residual and p + "attn.v_lamb" in raw_sd:
            lamb = raw_sd[p + "attn.v_lamb"]
            if lamb.ndim == 0:
                lamb = _reshape(lamb, 1)
            sd[f"blocks.{i}.v_residual.lamb"] = _astype(lamb, "f32")

        if cfg.ctrl_conditioning:
            fc1_x_key = p + "ctrl_mlpfusion.fc1_x.weight"
            fc1_c_key = p + "ctrl_mlpfusion.fc1_c.weight"
            fc2_key = p + "ctrl_mlpfusion.fc2.weight"
            if fc1_x_key in raw_sd:
                sd[f"blocks.{i}.ctrl_fusion.fc1_x.weight"] = raw_sd[fc1_x_key]
                sd[f"blocks.{i}.ctrl_fusion.fc1_c.weight"] = raw_sd[fc1_c_key]
                sd[f"blocks.{i}.ctrl_fusion.fc2.weight"] = raw_sd[fc2_key]
            elif p + "ctrl_mlpfusion.mlp.fc1.weight" in raw_sd:
                fc1_cat = raw_sd[p + "ctrl_mlpfusion.mlp.fc1.weight"]
                d_model = int(fc1_cat.shape[1]) // 2
                sd[f"blocks.{i}.ctrl_fusion.fc1_x.weight"] = fc1_cat[:, :d_model]
                sd[f"blocks.{i}.ctrl_fusion.fc1_c.weight"] = fc1_cat[:, d_model:]
                sd[f"blocks.{i}.ctrl_fusion.fc2.weight"] = raw_sd[
                    p + "ctrl_mlpfusion.mlp.fc2.weight"
                ]

    ctrl_fc1_key = "ctrl_emb.mlp.fc1.weight"
    if cfg.ctrl_conditioning and ctrl_fc1_key in raw_sd:
        import numpy as _np

        fc1_w = raw_sd[ctrl_fc1_key]  # [d_mid, n_buttons+3]
        # Pad columns to a multiple of 16 for GEMM tile alignment.
        raw_k = int(fc1_w.shape[1])
        padded_k = ((raw_k + 15) // 16) * 16
        if padded_k > raw_k:
            half_dt = "f16" if cfg.use_f16 else "bf16"
            if _IS_METAL:
                import mlx.core as mx

                w_np = _np.array(fc1_w.astype(mx.float32))
            else:
                w_np = (fc1_w.astype("f32") if fc1_w.dtype != "f32" else fc1_w).to_numpy()
            padded = _np.zeros((w_np.shape[0], padded_k), dtype=_np.float32)
            padded[:, :raw_k] = w_np
            fc1_w = _from_numpy(padded, dtype=half_dt)
        sd["ctrl_emb.fc1.weight"] = fc1_w
        sd["ctrl_emb.fc2.weight"] = raw_sd["ctrl_emb.mlp.fc2.weight"]

    return sd


# ---------------------------------------------------------------
# Persistent ctrl input buffer (host packer + stable device tensor)
# ---------------------------------------------------------------


def build_ctrl_buffer(model):
    """Back the ``Waypoint15.make_ctrl_buffer`` method.

    Returns ``(dev_tensor, fill_fn)`` for a model with ``ctrl_emb``, or
    ``(None, None)`` otherwise. See the method's docstring for the
    full contract.
    """
    if not hasattr(model, "ctrl_emb"):
        return None, None

    import numpy as _np

    shape = model.ctrl_input_shape
    dtype = model.ctrl_input_dtype()
    n_buttons = model.cfg.n_buttons

    if _IS_METAL:
        import mlx.core as mx

        mx_dt = mx.bfloat16 if dtype == "bf16" else mx.float16
        dev = mx.zeros(shape, dtype=mx_dt)
        host = _np.zeros(shape, dtype=_np.float32)

        def fill(ctrl):
            nonlocal dev
            host.fill(0.0)
            host[0, 0] = float(ctrl.mouse[0])
            host[0, 1] = float(ctrl.mouse[1])
            for b in ctrl.button:
                if 0 <= b < n_buttons:
                    host[0, 2 + b] = 1.0
            host[0, 2 + n_buttons] = float(ctrl.scroll_wheel)
            if dtype == "bf16":
                u16 = (host.view(_np.uint32) >> 16).astype(_np.uint16)
                dev = mx.array(u16).view(mx.bfloat16).reshape(*shape)
            else:
                dev = mx.array(host.astype(_np.float16))
            return dev

        return dev, fill

    from quark.runtime.cuda import CudaRuntime
    from quark.runtime.tensor import QuarkTensor

    dev = QuarkTensor.zeros(*shape, dtype=dtype)
    host = _np.zeros(shape, dtype=_np.float32)
    # Pre-allocate the packed-bits staging buffer once; per-call is an
    # in-place f32→target cast + a single memcpy_htod into the stable
    # device pointer (graph-safe).
    if dtype == "bf16":
        staged = _np.zeros(shape, dtype=_np.uint16)
    else:  # f16
        staged = _np.zeros(shape, dtype=_np.float16)
    rt = CudaRuntime.instance()

    def fill(ctrl):
        host.fill(0.0)
        host[0, 0] = float(ctrl.mouse[0])
        host[0, 1] = float(ctrl.mouse[1])
        for b in ctrl.button:
            if 0 <= b < n_buttons:
                host[0, 2 + b] = 1.0
        host[0, 2 + n_buttons] = float(ctrl.scroll_wheel)
        if dtype == "bf16":
            staged[:] = (host.view(_np.uint32) >> 16).astype(_np.uint16)
        else:
            staged[:] = host.astype(_np.float16)
        rt.memcpy_htod(dev.data_ptr(), staged.ctypes.data, staged.nbytes)
        return dev

    return dev, fill


# ---------------------------------------------------------------
# Noise embedding + per-block cond LUT precompute (numpy f32)
# ---------------------------------------------------------------


def _to_f32_np(t):
    """Pull a backend-native tensor down to a numpy f32 array."""
    import numpy as _np

    if hasattr(t, "to_numpy"):
        return t.astype("f32").to_numpy()
    if _IS_METAL:
        import mlx.core as mx

        return _np.array(t.astype(mx.float32)) if hasattr(t, "astype") else _np.array(t)
    return _np.array(t, dtype=_np.float32)


def _np_to_device(arr_f32, dt: str):
    """Move an f32 numpy array to the active backend at ``dt``."""
    import numpy as _np

    if _IS_METAL:
        import mlx.core as mx

        if dt == "bf16":
            u16 = (arr_f32.view(_np.uint32) >> 16).astype(_np.uint16)
            return mx.array(u16).view(mx.bfloat16)
        return mx.array(arr_f32.astype(_np.float16 if dt == "f16" else _np.float32))
    from quark.runtime.tensor import QuarkTensor

    return QuarkTensor.from_numpy(arr_f32, dtype=dt)


def compute_noise_emb_numpy(model, cfg) -> None:
    """Populate ``model._noise_emb_np`` from already-loaded bf16 weights.

    Replays world_engine's ``NoiseConditioner.forward`` in numpy f32 —
    bf16 → f32 is lossless so there's no precision win from re-loading
    f32 weights. Stays in f32 (not f64): every per-block matmul in
    ``prepare_cond_luts`` mixes this against f32 projection weights;
    promoting to f64 here would force a 16MB scratch alloc per matmul
    × 30 matmuls/block × 24 blocks ≈ 60s of pure-numpy upcast cost.
    """
    import math

    import numpy as np

    from quark.runtime.sync import synchronize

    synchronize()

    half = cfg.fourier_dim // 2
    freqs = np.array(
        [10000.0 ** (-i / max(half - 1, 1)) for i in range(half)],
        dtype=np.float32,
    )

    w1 = _to_f32_np(model.noise_fc1.weight.data)
    w2 = _to_f32_np(model.noise_fc2.weight.data)

    sqrt2 = math.sqrt(2.0)
    sigmas = list(cfg.scheduler_sigmas)
    n = len(sigmas)
    fourier_dim = len(freqs) * 2

    fourier = np.zeros((n, fourier_dim), dtype=np.float32)
    for i, s in enumerate(sigmas):
        phase = np.array([float(s) * 1000.0 * f for f in freqs], dtype=np.float32)
        fourier[i, : len(freqs)] = sqrt2 * np.sin(phase)
        fourier[i, len(freqs) :] = sqrt2 * np.cos(phase)

    h = fourier @ w1.T
    h = h * (1.0 / (1.0 + np.exp(-h)))  # silu
    emb = h @ w2.T  # [n, d] f32

    model._noise_emb_np = [emb[i].copy() for i in range(n)]  # f32 [d] each


def prepare_cond_luts(model, cfg, d: int) -> None:
    """Precompute per-block cond LUTs and the out-norm LUT in numpy f32.

    Each TransformerBlock gets ``block._cond_lut[si]`` = a 6-tuple of
    half-precision device tensors (the projected (silu(emb + bias)) for
    sigmas[si]). The out-norm path emits ``model._out_norm_luts[si]``
    = (s_on, b_on) for the AdaRMSNorm at the model output.
    """
    import numpy as np

    from quark.runtime.sync import synchronize

    out_dt = "f16" if cfg.use_f16 else "bf16"

    # Per-block: separate attn and mlp cond heads with different bias_in.
    # Use f32 noise embeddings from compute_noise_emb_numpy().
    for block in model.blocks:
        attn_bias = _to_f32_np(block.attn_cond_bias_in.data)
        mlp_bias = _to_f32_np(block.mlp_cond_bias_in.data)
        proj_ws = [_to_f32_np(block.cond_projs[j].weight.data) for j in range(6)]
        block_lut = []
        for si in range(len(cfg.scheduler_sigmas)):
            emb = model._noise_emb_np[si]
            # attn path: emb + attn_bias → silu → projs 0-2
            h_attn = emb + attn_bias
            h_attn = h_attn * (1.0 / (1.0 + np.exp(-h_attn)))
            # mlp path: emb + mlp_bias → silu → projs 3-5
            h_mlp = emb + mlp_bias
            h_mlp = h_mlp * (1.0 / (1.0 + np.exp(-h_mlp)))
            projs = [(h_attn if j < 3 else h_mlp) @ proj_ws[j].T for j in range(6)]
            block_lut.append(
                tuple(_np_to_device(p.reshape(1, d).astype(np.float32), out_dt) for p in projs)
            )
        block._cond_lut = block_lut

    # Out-norm: silu(noise_emb) @ fc.weight.T — NO per-block bias_in.
    on_w = _to_f32_np(model.out_norm_proj.weight.data)  # [2*d, d]
    model._out_norm_luts = []
    for si in range(len(cfg.scheduler_sigmas)):
        emb = model._noise_emb_np[si]
        h = emb * (1.0 / (1.0 + np.exp(-emb)))  # silu(emb), no bias
        ab = h @ on_w.T  # [2*d]
        model._out_norm_luts.append(
            (
                _np_to_device(ab[:d].reshape(1, d).astype(np.float32), out_dt),
                _np_to_device(ab[d:].reshape(1, d).astype(np.float32), out_dt),
            )
        )
    synchronize()
