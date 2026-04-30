"""Waypoint-1.5 inference as a ``quark.nn.Module``.

EXEMPT FROM 500-LINE RULE: model + config + generate-one-frame all
tightly coupled; splitting would strand the per-block conditioning
and euler ODE logic that references model internals.

**Zero non-kernel ops in forward().** The forward path is entirely
``nn.Module.__call__`` dispatches — same syntax as PyTorch, but every
leaf module lowers to a ``pcf.*`` kernel.

Pre-computed at init (``prepare()``):
  - Noise embedding LUT
  - Cond projections per sigma (6 linears + out-norm linear)
  - RoPE cos/sin tables
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import cast

import quark.functional as pcf
import quark.nn as nn

# RoPE cos/sin computed inline in kernels — no precomputed tables.
from quark.nn.module import _tensor, _zeros

_IS_METAL = sys.platform == "darwin"


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------


@dataclass
class CtrlInput:
    """Controller input for one frame."""

    button: set[int] = field(default_factory=set)
    mouse: tuple[float, float] = (0.0, 0.0)
    scroll_wheel: int = 0


@dataclass(frozen=True)
class Waypoint15Config:
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 32
    n_kv_heads: int = 16
    mlp_ratio: int = 4
    channels: int = 32
    patch: tuple[int, int] = (2, 2)
    height: int = 16
    width: int = 32
    local_window: int = 16
    global_window: int = 128
    global_pinned_dilation: int = 8
    global_attn_period: int = 4
    global_attn_offset: int = -1
    value_residual: bool = True
    fourier_dim: int = 512
    scheduler_sigmas: tuple[float, ...] = (1.0, 0.9, 0.75, 0.3, 0.0)
    n_buttons: int = 256
    ctrl_conditioning: bool = True
    ctrl_conditioning_period: int = 3
    use_f16: bool = True  # False → bf16 weights + f32 output (safe baseline)
    # End-to-end fp8 mode. When True (default on sm_89+):
    #   - Linear weights are quantized to e4m3 by ``prepare()``
    #   - KV cache is stored as e4m3
    #   - OwlAttn runs both MMAs in e4m3 (Q/K/V all cast to e4m3
    #     internally)
    # When False, everything stays in the "half" dtype (bf16 by default)
    # from weights through KV cache through attention MMAs — the safe
    # fallback for devices without fp8 support or for precision debugging.
    # ``QUARK_MOE_NO_FP8=1`` separately opts the MoE block out of fp8
    # while leaving everything else fp8.
    use_fp8: bool = True

    # Spatial-quilting factor for the kv_cache + owl_attn path. ``1``
    # is the no-op default (every token participates in attention every
    # layer); higher values stagger which token positions write/read the
    # KV ring per layer (per-layer offset = layer_idx % quilt_factor).
    quilt_factor: int = 1

    # Per-block MoE — swap dense MLP for ``nn.MoE``. Per-expert hidden
    # = ``mlp_dim / moe_top_k`` so active params match dense. bf16-only
    # regardless of ``use_fp8`` (set ``QUARK_MOE_NO_FP8=1`` to opt out
    # explicitly). ``moe_routing`` picks the routing kernel — see
    # ``nn.MoE`` for the matrix of behaviors per mode.
    moe: bool = False
    moe_n_experts: int = 8
    moe_top_k: int = 2
    moe_routing: str = "correct"

    @property
    def Dh(self):
        return self.d_model // self.n_heads

    @property
    def mlp_dim(self):
        return self.d_model * self.mlp_ratio

    @property
    def gqa_ratio(self):
        return self.n_heads // self.n_kv_heads

    @property
    def tpf(self):
        return self.height * self.width

    @property
    def qkv_dim(self):
        return (self.n_heads + 2 * self.n_kv_heads) * self.Dh

    @property
    def v_col_offset(self):
        return (self.n_heads + self.n_kv_heads) * self.Dh

    @property
    def v_width(self):
        return self.n_kv_heads * self.Dh

    def is_global(self, layer):
        return (layer - self.global_attn_offset) % self.global_attn_period == 0

    def num_buckets(self, layer):
        return (
            self.global_window // self.global_pinned_dilation
            if self.is_global(layer)
            else self.local_window
        )

    def pinned_dilation(self, layer):
        return self.global_pinned_dilation if self.is_global(layer) else 1

    @classmethod
    def from_world_engine(cls, we_cfg, **overrides) -> Waypoint15Config:
        """Build a ``Waypoint15Config`` from a ``world_engine`` model config.

        ``we_cfg`` is either an OmegaConf node or a plain dict (or any
        mapping that supports ``in`` + ``__getitem__``). Only the fields
        that ``Waypoint15Config`` knows about are copied — everything
        else on the source config (``ae_uri``, ``prompt_conditioning``,
        ``base_fps``, …) is the caller's concern.

        ``we_cfg.ctrl_conditioning`` is the *controller spec* on the
        world_engine side and truthiness decides whether quark should
        wire up the ctrl path; ``cfg.ctrl_conditioning`` on this side is
        a plain bool. ``overrides`` wins over the mapped values so
        callers can tweak e.g. ``use_f16`` without rebuilding the dict.
        """
        _SCALAR_FIELDS = (
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "mlp_ratio",
            "channels",
            "height",
            "width",
            "local_window",
            "global_window",
            "global_pinned_dilation",
            "global_attn_period",
            "global_attn_offset",
            "value_residual",
            "fourier_dim",
            "n_buttons",
            "ctrl_conditioning_period",
            "moe",
            "moe_n_experts",
            "moe_top_k",
            "moe_routing",
            "quilt_factor",
        )
        kwargs = {name: we_cfg[name] for name in _SCALAR_FIELDS if name in we_cfg}
        if "patch" in we_cfg:
            kwargs["patch"] = tuple(we_cfg["patch"])
        if "scheduler_sigmas" in we_cfg:
            kwargs["scheduler_sigmas"] = tuple(we_cfg["scheduler_sigmas"])
        if "ctrl_conditioning" in we_cfg:
            kwargs["ctrl_conditioning"] = we_cfg["ctrl_conditioning"] is not None
        kwargs.update(overrides)
        return cls(**kwargs)


def _check_dtype(t, expected: str, where: str) -> None:
    """Raise if ``t``'s dtype doesn't match ``expected``.

    Surfaces dtype-flow bugs loudly rather than papering over them
    with silent casts. ``where`` is an operator-level label that names
    the point in the graph so the stack trace + message identify
    exactly which edge went wrong.
    """
    actual = t.dtype if isinstance(t.dtype, str) else str(t.dtype)
    if actual != expected:
        raise TypeError(
            f"Waypoint15: dtype mismatch at {where!r}: expected {expected!r}, got {actual!r}. "
            f"Fix the upstream module's out_dtype so the graph's dtypes align end-to-end."
        )


# Public re-export — see ``quark.models._waypoint_15_io``. Kept in the
# model module's namespace so ``from quark.models.waypoint_15 import
# remap_world_engine_state_dict`` keeps working.
from quark.models._waypoint_15_io import remap_world_engine_state_dict  # noqa: E402

# ---------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Waypoint15Config, layer_idx: int, rope_n_frames: int):
        d = cfg.d_model
        half_dt = "f16" if cfg.use_f16 else "bf16"
        out_dt = "f16" if cfg.use_f16 else "bf16"

        # Per-block conditioning: separate attn and mlp cond heads, each with
        # their own bias_in. WE has attn_cond_head and mlp_cond_head.
        self.attn_cond_bias_in = nn.Parameter(_zeros(d, dtype="bf16"))
        self.mlp_cond_bias_in = nn.Parameter(_zeros(d, dtype="bf16"))
        self.cond_projs = nn.ModuleList([nn.Linear(d, d, out_dtype=out_dt) for _ in range(6)])
        # Precomputed per-sigma cond LUT — filled in by Waypoint15.prepare().
        self._cond_lut: list[tuple] = []

        # Expected half-precision dtype for every tensor flowing through
        # this block (qkv, activations, ctrl emb, …). Captured once and
        # referenced by the dtype checks in forward().
        self._half_dt = half_dt
        # Intermediate tensors (qkv_proj output, MLP fc1 output) stay in
        # half_dt. The fp8 compute path is driven by Linear.forward
        # pre-casting ``x`` to e4m3 when the weight is fp8 (cuBLAS takes
        # it from there) — the upstream norm/silu kernels don't yet
        # support fp8 stores (their store path uses scalar
        # ``cvt → e4m3``, which PTX can't emit; a packed_convert refactor
        # would unlock norm → e4m3 → Linear without the extra cast).
        self.pre_attn_norm = nn.AdaRMSNorm()
        self.qkv_proj = nn.Linear(d, cfg.qkv_dim, out_dtype=out_dt)
        self.head_norm = nn.HeadRMSNorm(cfg.n_heads, cfg.n_kv_heads, cfg.Dh)
        if cfg.value_residual:
            self.v_residual = nn.ValueResidualPacked(cfg.v_col_offset, cfg.v_width)
        # KV cache + owl_attn MMAs are driven by ``cfg.use_fp8``:
        #   fp8 path: e4m3 cache + e4m3 MMAs; Q is cast to e4m3 during
        #     the Q-RoPE pass (packed_convert) inside owl_attn. Cache
        #     bandwidth halves + sm_89+ fp8 MMA throughput win.
        #   bf16 path: kv cache, Q, K, V all in half_dt; MMAs inherit
        #     that from the tensor dtypes (compute_dtype=None). The
        #     safe fallback when fp8 is misbehaving or the device
        #     doesn't support it.
        kv_cache_dt = "e4m3" if cfg.use_fp8 else half_dt
        owl_compute_dt = "e4m3" if cfg.use_fp8 else None
        # Quilt offset alternates by layer so the model in aggregate
        # sees every pixel. layer_idx 0 keeps residue 0, layer 1 keeps
        # residue 1, …, wrapping every quilt_factor blocks.
        quilt_offset = layer_idx % cfg.quilt_factor
        self.kv_cache = nn.KVCacheUpdate(
            B=1,
            n_kv_heads=cfg.n_kv_heads,
            n_q_heads=cfg.n_heads,
            H_spatial=cfg.height,
            W_spatial=cfg.width,
            Dh=cfg.Dh,
            num_buckets=cfg.num_buckets(layer_idx),
            pinned_dilation=cfg.pinned_dilation(layer_idx),
            packed_qkv=True,
            rope_n_frames=rope_n_frames,
            dtype=kv_cache_dt,
            quilt_factor=cfg.quilt_factor,
            quilt_offset=quilt_offset,
        )
        self.attn = nn.OwlAttn(
            B=1,
            n_kv_heads=cfg.n_kv_heads,
            gqa_ratio=cfg.gqa_ratio,
            H_spatial=cfg.height,
            W_spatial=cfg.width,
            num_buckets=cfg.num_buckets(layer_idx),
            pinned_dilation=cfg.pinned_dilation(layer_idx),
            packed_qkv=True,
            rope_n_frames=rope_n_frames,
            compute_dtype=owl_compute_dt,
            quilt_factor=cfg.quilt_factor,
            quilt_offset=quilt_offset,
        )
        out_dt = "f16" if cfg.use_f16 else "bf16"
        self.out_proj = nn.Linear(cfg.n_heads * cfg.Dh, d, out_dtype=out_dt)
        self.attn_gate = nn.AdaGateResidual()

        if cfg.ctrl_conditioning and layer_idx % cfg.ctrl_conditioning_period == 0:
            # ctrl_fusion stays bf16 end-to-end regardless of ``cfg.use_f16`` —
            # weights are bf16, so an f16 out_dtype would force a f16→bf16
            # cast of fc1_c's cond input and fc2's h input on every call (3
            # extra elementwise kernel launches per block × 5 NFE × 24
            # blocks). Keeping it all bf16 matches the weights and removes
            # those casts. Caller rmsnorm preserves dtype so cond stays bf16
            # from ctrl_emb all the way through.
            self.ctrl_fusion = nn.MLPFusion(d, out_dtype="bf16")
            # Two RMSNorms per ctrl_fusion: one on the (per-block) x input,
            # one on the (shared) ctrl_emb. Each owns a cached output
            # buffer — a bare ``pcf.rmsnorm`` would auto-alloc ~150 µs
            # per call.
            self.ctrl_x_norm = nn.RMSNorm()
            self.ctrl_emb_norm = nn.RMSNorm()
            # Residual add for ``x + ctrl_fusion(...)`` — without this
            # the ``+`` goes through ``QuarkTensor.__add__`` which
            # allocates a fresh 2 MB output every call.
            self.ctrl_residual = nn.Add()

        self.pre_mlp_norm = nn.AdaRMSNorm()
        # fc1 keeps its fused-silu path on the custom kernel with a
        # half_dt store (the custom epilogue's ``cvt → e4m3`` is scalar-
        # only and PTX has no scalar fp8 cvt). fc2 now runs whichever
        # config wins the autotune race for its spec — cuBLAS applies
        # the bf16→e4m3 cast on-stream when it's picked; PTX uses its
        # own compute_dtype smem down-cast when it wins.
        if cfg.moe:
            self.mlp = nn.MoE(
                M=cfg.tpf,
                d_model=d,
                d_intermediate=cfg.mlp_dim // cfg.moe_top_k,
                n_experts=cfg.moe_n_experts,
                top_k=cfg.moe_top_k,
                routing=cfg.moe_routing,
                out_dtype=half_dt,
            )
        else:
            self.mlp = nn.MLP(d, cfg.mlp_dim, d, out_dtype=out_dt)
        self.mlp_gate = nn.AdaGateResidual()

    def forward(
        self, x, sigma_idx: int, frame_t, *, qkv_first, ctrl_emb=None, frozen: bool = False
    ):
        s0, b0, g0, s1, b1, g1 = self._cond_lut[sigma_idx]
        half_dt = self._half_dt

        _check_dtype(x, half_dt, "block input x")

        # ── Attention ──
        qkv = self.head_norm(self.qkv_proj(self.pre_attn_norm(x, s0, b0)))
        _check_dtype(qkv, half_dt, "qkv (after head_norm)")

        if hasattr(self, "v_residual") and qkv_first is not None:
            qkv = self.v_residual(qkv, qkv_first)
            _check_dtype(qkv, half_dt, "qkv (after v_residual)")

        self.kv_cache(qkv, frame_t, frozen=frozen)
        attn_out = self.attn(qkv, self.kv_cache, frame_t)
        _check_dtype(attn_out, half_dt, "attn_out")
        x = self.attn_gate(x, self.out_proj(attn_out), g0)
        _check_dtype(x, half_dt, "x (after attn_gate)")

        # ── Controller conditioning ──
        if ctrl_emb is not None and hasattr(self, "ctrl_fusion"):
            # ctrl_fusion is pinned to bf16 end-to-end regardless of
            # cfg.use_f16 — see TransformerBlock.__init__ comment. So
            # both rmsnorm inputs AND the ctrl_emb must be bf16 here.
            _check_dtype(ctrl_emb, "bf16", "ctrl_emb (ctrl_fusion path)")
            _check_dtype(x, "bf16", "x (ctrl_fusion path)")
            x = self.ctrl_residual(
                x, self.ctrl_fusion(self.ctrl_x_norm(x), self.ctrl_emb_norm(ctrl_emb))
            )

        # ── MLP ──
        x = self.mlp_gate(x, self.mlp(self.pre_mlp_norm(x, s1, b1)), g1)
        _check_dtype(x, half_dt, "x (after mlp_gate)")

        return x, qkv


# ---------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------


class Waypoint15(nn.Module):
    def __init__(self, cfg: Waypoint15Config | None = None):
        if cfg is None:
            cfg = Waypoint15Config()
        self.cfg = cfg
        d = cfg.d_model
        ph, pw = cfg.patch

        n_frames = max(cfg.num_buckets(i) * cfg.pinned_dilation(i) + 4 for i in range(cfg.n_layers))

        self.patchify = nn.Patchify(cfg.channels, d, cfg.height * ph, cfg.width * pw, ph, pw)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, i, n_frames) for i in range(cfg.n_layers)]
        )
        self.out_norm = nn.AdaRMSNorm()
        self.unpatchify = nn.Unpatchify(d, cfg.channels, cfg.height * ph, cfg.width * pw, ph, pw)

        # Output norm projection (not per-block).
        out_dt = "f16" if cfg.use_f16 else "bf16"
        self.out_norm_proj = nn.Linear(d, 2 * d, out_dtype=out_dt)

        # Controller input embedding (only when ctrl_conditioning enabled).
        # Pin out_dtype=bf16 so the downstream ctrl_fusion.fc1_c (bf16
        # weights) reads its cond input without a cast — see the
        # matching MLPFusion(out_dtype="bf16") in TransformerBlock.
        if cfg.ctrl_conditioning:
            # fp8_skip=True: ctrl_emb.fc1's K is the padded controller
            # input (n_buttons + 3 rounded up to 16), typically not a
            # multiple of 32 — incompatible with Ada's fp8 MMA
            # (m16n8k32_e4m3). The layer is also small and
            # precision-sensitive (one row per frame), so keeping it
            # bf16 under --fp8 is the right call anyway.
            self.ctrl_emb = nn.ControllerInputEmbedding(
                cfg.n_buttons, d, cfg.mlp_ratio, out_dtype="bf16", fp8_skip=True
            )

        # Noise conditioner weights.
        half = cfg.fourier_dim // 2
        self.noise_freq = nn.Parameter(
            _tensor(
                [10000.0 ** (-i / max(half - 1, 1)) for i in range(half)],
                dtype="f32",
            )
        )
        self.noise_fc1 = nn.Linear(cfg.fourier_dim, d * 4, out_dtype=out_dt)
        self.noise_fc2 = nn.Linear(d * 4, d, out_dtype=out_dt)

        # RoPE cos/sin computed inline in kernels — no precomputed tables.
        self._cond_luts: list[tuple] = []

        # ``prepare()`` is deliberately NOT called here — it needs real
        # weights, not the zero-init placeholders. Callers must run
        # ``load_state_dict(...)`` then ``prepare()`` themselves.

    def prepare(self, **kwargs):
        """Pre-compute per-sigma cond LUTs. Call after loading weights.

        ``fp8`` defaults to ``cfg.use_fp8`` so Linear weight quantization
        stays consistent with the KV cache / OwlAttn paths.
        """
        from quark.models._waypoint_15_io import (
            compute_noise_emb_numpy,
            prepare_cond_luts,
        )

        cfg = self.cfg
        kwargs.setdefault("fp8", cfg.use_fp8)

        pcf.precompute_noise_lut(
            list(cfg.scheduler_sigmas),
            self.noise_freq.data,
            self.noise_fc1.weight.data,
            self.noise_fc2.weight.data,
        )
        _sync()

        # Noise embeddings in numpy f32 from the already-loaded bf16
        # weights. bf16 → f32 is lossless so there's no precision win
        # from re-downloading f32 weights from the hub.
        compute_noise_emb_numpy(self, cfg)
        prepare_cond_luts(self, cfg, cfg.d_model)

        # Recurse into children (Linear.prepare handles fp8 + shuffle).
        super().prepare(**kwargs)

    def set_frame_t(self, value: int, stream: int = 0) -> None:
        """Update frame_t on all KV caches (async memset on stream)."""
        for block in cast("list[TransformerBlock]", self.blocks):
            block.kv_cache.set_frame_t(value, stream=stream)

    def reset(self, stream: int = 0) -> None:
        """Clear per-block KV ring buffers and rewind frame_t to zero.

        Callers swap generations (new prompt, new seed, pause/resume)
        by calling this between runs. Not safe while a capture/replay
        is in flight on the same stream.
        """
        for block in cast("list[TransformerBlock]", self.blocks):
            block.kv_cache.reset(stream=stream)

    def make_ctrl_buffer(self):
        """Allocate the persistent ``[1, padded_in]`` ctrl MLP input and
        return ``(dev_tensor, fill_fn)`` where ``fill_fn(ctrl)`` updates
        the device tensor in place from a ``CtrlInput`` and returns it.

        Shape / dtype come from the model so callers never hard-code
        them. Returns ``(None, None)`` when the model has no ``ctrl_emb``.

        On CUDA the returned tensor is a stable device pointer — safe
        to capture inside a CUDA graph. ``fill_fn`` packs host f32 →
        bf16 packed uint16 via ``(u32 >> 16)`` once per frame and does
        a single ``memcpy_htod`` into that stable buffer.
        """
        from quark.models._waypoint_15_io import build_ctrl_buffer

        return build_ctrl_buffer(self)

    @classmethod
    def from_world_engine_hub(
        cls,
        repo_id: str,
        *,
        cfg: Waypoint15Config | None = None,
        we_cfg=None,
        dtype: str = "bf16",
        filename: str = "model.safetensors",
    ) -> Waypoint15:
        """Download a world_engine-format checkpoint, remap, and load.

        Either ``cfg`` (a ``Waypoint15Config`` already built by the
        caller) or ``we_cfg`` (a world_engine OmegaConf/dict to map
        through ``Waypoint15Config.from_world_engine``) must be given.

        The returned model has weights loaded but has NOT been
        ``prepare()``'d — callers choose their own ``fp8`` / ``shuffle``
        kwargs. Wrap in ``GenerateFrame(model)`` for graph inference.
        """
        if cfg is None:
            if we_cfg is None:
                raise ValueError(
                    "from_world_engine_hub: pass either cfg=Waypoint15Config(...) "
                    "or we_cfg=<world_engine model config>"
                )
            cfg = Waypoint15Config.from_world_engine(we_cfg)

        from quark.nn.io import load_from_hub

        raw_sd = load_from_hub(repo_id, filename=filename, dtype=dtype)
        model = cls(cfg)
        model.load_state_dict(remap_world_engine_state_dict(raw_sd, cfg), strict=False)
        return model

    def encode_ctrl(self, ctrl_input):
        """Run the controller MLP on a pre-built ``[1, padded_in]`` tensor.

        The caller owns the input tensor (allocate once at startup,
        update in place per frame). ``ctrl_input`` may be ``None``
        when the model has no ``ctrl_emb``; returns ``None`` in that
        case so the forward path can short-circuit.
        """
        if ctrl_input is None or not hasattr(self, "ctrl_emb"):
            return None
        _check_dtype(ctrl_input, "bf16", "encode_ctrl input")
        out = self.ctrl_emb(ctrl_input)
        _check_dtype(out, "bf16", "encode_ctrl output")
        return out

    @property
    def ctrl_input_shape(self) -> tuple[int, int]:
        """``(1, padded_in)`` — the shape ``encode_ctrl`` expects."""
        return (1, self.ctrl_emb._padded_in)

    def ctrl_input_dtype(self) -> str:
        """The dtype ``encode_ctrl`` expects for its input tensor.

        Pinned to ``bf16`` to match the ``ctrl_emb.fc1`` weight dtype —
        an f16 input would force a per-call f16→bf16 cast inside the
        Linear and defeat the rest of the bf16-end-to-end ctrl path.
        """
        return "bf16"

    def forward(self, latent, sigma_idx: int, frame_t=None, ctrl_emb=None, frozen: bool = False):
        """One forward pass — denoise or commit."""
        half_dt = "f16" if self.cfg.use_f16 else "bf16"
        s_on, b_on = self._out_norm_luts[sigma_idx]

        x = self.patchify(latent)
        _check_dtype(x, half_dt, "patchify output")
        qkv_first = None
        for li, block in enumerate(self.blocks):
            x, qkv = block(
                x,
                sigma_idx,
                frame_t,
                qkv_first=qkv_first,
                ctrl_emb=ctrl_emb,
                frozen=frozen,
            )
            if li == 0 and self.cfg.value_residual:
                qkv_first = qkv

        x = self.out_norm(x, s_on, b_on, activation="silu")
        _check_dtype(x, half_dt, "out_norm output")
        return self.unpatchify(x)


class GenerateFrame(nn.Module):
    """One self-contained frame: encode_ctrl → denoise → KV commit → frame_t++.

    Owns every piece of per-frame state so the whole thing captures as
    one CUDA graph and replays as a single opaque op. The caller just
    does:

        gen_frame = GenerateFrame(model)
        gen_frame.set_frame_t(seed_frame_count)
        gen_frame.graph(example_latent, example_ctrl_input)  # capture once

        for fi in range(n_frames):
            ctrl_fill(ctrl_sequence[fi])       # host → ctrl_input device tensor
            latent = gen_frame(noise_pool[fi], ctrl_input)

    ``Module.__call__`` dispatches through ``_graph_replay`` once
    ``graph(...)`` has been called, so inputs are D2D-copied into the
    captured-stable buffers and the output is returned as a fresh clone.

    ``frame_t`` is internal state — if we took it as an arg, the
    ``increment`` inside the graph would bump the captured clone, not
    the caller's tensor. Keeping it on ``self`` means the single
    captured counter gets incremented every replay. Call
    ``set_frame_t`` (outside capture) to reposition the counter.
    """

    def __init__(self, model: Waypoint15):
        self.model = model
        cfg = model.cfg
        sigmas = cfg.scheduler_sigmas
        self.n_denoise = len(sigmas) - 1
        # Per-step dsig as a single-element f32 tensor on device.
        self.dsig_tensors = [
            _tensor([float(sigmas[i + 1] - sigmas[i])], dtype="f32") for i in range(self.n_denoise)
        ]
        # One EulerStep module per denoise step — each owns a cached
        # output buffer keyed on ``(latent.shape, latent.dtype)`` so the
        # hot loop doesn't ``cuMemAllocAsync`` every step.
        self.euler_steps = nn.ModuleList([nn.EulerStep() for _ in range(self.n_denoise)])
        # Frame counter owned by this module — lives across replays so
        # the internal ``increment`` kernel captured in the graph
        # advances the *same* counter every frame (the pre/post-capture
        # value is set via ``set_frame_t`` from host).
        self.frame_t = _zeros(1, dtype="s32")

    def set_frame_t(self, value: int, stream: int = 0) -> None:
        """Reposition the internal frame_t counter (async memset).

        Call before the first frame and between runs. Not safe to call
        while a replay is in flight on the same stream.
        """
        from quark.runtime.cuda import CudaRuntime

        CudaRuntime.instance().memset_d32(
            self.frame_t.data_ptr(), int(value) & 0xFFFFFFFF, 1, stream=stream
        )

    def reset(self, stream: int = 0) -> None:
        """Reset the wrapped model's KV state and rewind this module's
        frame_t counter. Intended for between-generation cleanup; not
        safe during capture/replay on the same stream.
        """
        self.model.reset(stream=stream)
        self.set_frame_t(0, stream=stream)

    def prepare_graph(self, example_latent, example_ctrl_input, *, start_frame_t: int = 0) -> None:
        """Warmup + capture the one-frame graph.

        Runs one eager forward so every cached output buffer inside the
        model (EulerStep, AdaGateResidual, ctrl path, RMSNorm) is
        allocated BEFORE capture — otherwise the graph bakes in
        ``cuMemAllocAsync`` nodes whose replay addresses desync the
        cached-ptr state on host. Then captures via ``Module.graph(...)``
        so subsequent ``__call__``s replay.

        Both the warmup and the capture forward bump ``frame_t`` once
        each; this helper rewinds it to ``start_frame_t`` before
        returning so the next real frame starts from the right count.
        """
        from quark.runtime.cuda import CudaRuntime

        self.set_frame_t(start_frame_t)
        _ = self(example_latent, example_ctrl_input)  # eager warmup
        CudaRuntime.instance().stream_synchronize(0)

        self.set_frame_t(start_frame_t)
        self.graph(example_latent, example_ctrl_input)  # capture

        self.set_frame_t(start_frame_t)
        CudaRuntime.instance().stream_synchronize(0)

    def forward(self, latent, ctrl_input):
        """Encode controls → denoise + commit one frame. Returns the denoised latent."""
        ctrl_emb = self.model.encode_ctrl(ctrl_input)
        ft = self.frame_t
        x = latent
        for si in range(self.n_denoise):
            v = self.model(x, sigma_idx=si, frame_t=ft, ctrl_emb=ctrl_emb, frozen=True)
            x = self.euler_steps[si](x, v, self.dsig_tensors[si])
        # Commit pass (frozen=False writes this frame's KV to the ring).
        self.model(x, sigma_idx=self.n_denoise, frame_t=ft, ctrl_emb=ctrl_emb)
        ft.increment()
        return x


# Back-compat alias — old scripts import ``GenerateOneFrame``.
GenerateOneFrame = GenerateFrame


def _sync():
    """Synchronize the active backend."""
    from quark.runtime.sync import synchronize

    synchronize()
