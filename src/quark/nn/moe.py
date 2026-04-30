"""``nn.MoE`` — capacity-bounded top-K Mixture-of-Experts block.

Lives in its own module to keep ``nn/layers.py`` under the 800-line
hard cap.

Reproduces the world_engine ``MoE`` formulation
(router → topk softmax → expert_in[expert] @ x → silu →
expert_out[expert] @ h → weighted sum) but with a fixed-capacity
routing structure: each expert always processes exactly ``capacity``
slots, which makes ``moe_inproj`` / ``moe_outproj`` see static shapes
(autotune cache hits, no per-call sort/cumsum).

Capacity ``C = M * top_k / E`` rounded up to ``BM = 32`` so the tile
layout divides cleanly. The router falls through to *any* non-claimed
expert with room when a top-K choice is full — no token is dropped
(structurally guaranteed for K << E with E*C >= M*top_k).

Routing modes (``routing=`` constructor kwarg):
  * ``"balanced"`` (default) — per-token top-K with full-fallback. Each
    token picks its own K experts (substituting if its top choice is
    full). Maximum routing quality; capacity is divided across all E
    experts.
  * ``"shared_experts"`` — every token in the frame routes to the
    *same* K experts, picked globally by cumulative softmax preference.
    Total compute = K full-batch GEMMs against the chosen experts'
    weights = exactly the cost of a dense MLP. Routing quality is
    coarser since per-token preference is averaged out at the K
    selection.
  * ``"correct"`` — each token routes to its actual top-K experts (no
    capacity bound, no substitution). Slots are sorted by expert with
    BM-padding per expert; chunks past the active region get
    ``expert=-1`` and the inproj/outproj kernels skip them. Buffer
    budget is ``2× M*top_k`` (worst-case BM-padded layout).

Weight layout matches world_engine bit-for-bit:
  * ``router_weight``: ``[E, D]``
  * ``expert_in``:     ``[E*H, D]``  (== world_engine's ``expert_in.view(E*H, D)``)
  * ``expert_out``:    ``[E*D, H]``  (== world_engine's ``expert_out.view(E*D, H)``)

``prepare(fp8=True)`` quantizes ``expert_in`` / ``expert_out`` to e4m3
and routes the inproj / outproj calls through the fp8 compute path
(``compute_dtype="e4m3"``). The router stays bf16 (precision-sensitive,
small E). The inproj output ``_buf_h`` stays in ``out_dtype`` (bf16/f16)
because the fused-silu epilogue can't yet store fp8 (PTX has no scalar
fp8 cvt — pending the ``packed_convert`` refactor); outproj reads it
back and casts on smem-load via ``compute_dtype``.

**Env var ``QUARK_MOE_NO_FP8``**: opt the MoE block out of fp8 even
when the surrounding model is configured for fp8 (``cfg.use_fp8=True``
in Waypoint15). Set this before calling ``model.prepare()`` — once the
expert weights are quantized to e4m3 they can't be un-quantized without
reloading the state dict. Useful when the runtime bf16→e4m3 smem cast
on the input tile dominates the kernel time and the cuBLAS gemms are
the only thing that actually wins from fp8.
"""

from __future__ import annotations

import os
import sys

from quark.nn.layers import Cast, Linear
from quark.nn.module import Module, Parameter, _quantize_to_e4m3, _tensor, _zeros


def _moe_fp8_disabled() -> bool:
    """``QUARK_MOE_NO_FP8`` opt-out — read at prepare-time so callers
    can flip it without restarting the process. Truthy = disable fp8
    quantization for MoE blocks even when the outer model passes
    ``fp8=True``."""
    return os.environ.get("QUARK_MOE_NO_FP8", "").strip().lower() in ("1", "true", "yes")


_IS_METAL = sys.platform == "darwin"


def _empty(*shape, dtype: str):
    """Backend-native uninitialized tensor — mx.array on Metal,
    QuarkTensor on CUDA. Mirrors ``module._zeros`` minus the zero-fill
    so the cached output buffers don't pay for a memset they'll
    immediately overwrite."""
    if _IS_METAL:
        import mlx.core as mx

        from quark.nn.module import _mx_dt

        return mx.zeros(shape, dtype=_mx_dt(dtype))
    from quark.runtime.tensor import QuarkTensor

    return QuarkTensor.empty(*shape, dtype=dtype)


# moe_inproj/outproj's work_list contract: each (grp_start, expert)
# entry covers BM contiguous output slots under one expert. With
# capacity routing each expert owns exactly ``capacity`` slots, so
# the work_list is fully determined by (E, capacity, BM) — built once
# in MoE.__init__ and reused every call.
_MOE_BM = 32


class MoE(Module):
    def __init__(
        self,
        M: int,
        d_model: int,
        d_intermediate: int,
        n_experts: int,
        top_k: int,
        *,
        routing: str = "balanced",
        out_dtype: str = "bf16",
    ):
        if routing not in ("balanced", "shared_experts", "correct"):
            raise ValueError(
                f"MoE: routing must be 'balanced' / 'shared_experts' / 'correct'; got {routing!r}"
            )

        # Capacity sizing per routing mode:
        #   * balanced       → M*K/E rounded to BM. Full-fallback router
        #                      prevents drops at this exact-fit budget for
        #                      K << E (the typical regime).
        #   * shared_experts → M*K/E rounded to BM, but must equal M*K
        #                      exactly (validated below) — the layout is
        #                      "K blocks of M slots each".
        #   * correct        → worst-case BM-padded: M*K + E*(BM-1) so any
        #                      per-expert count fits with room to round up.
        if routing == "correct":
            worst_case = M * top_k + n_experts * (_MOE_BM - 1)
            cap_per_expert = (worst_case + n_experts - 1) // n_experts
            capacity = ((cap_per_expert + _MOE_BM - 1) // _MOE_BM) * _MOE_BM
        else:
            min_capacity = (M * top_k + n_experts - 1) // n_experts
            capacity = ((min_capacity + _MOE_BM - 1) // _MOE_BM) * _MOE_BM

        if capacity * n_experts < M * top_k:
            raise ValueError(
                f"MoE: rounded capacity {capacity} insufficient for "
                f"M*top_k={M * top_k}, E={n_experts}"
            )
        if routing == "shared_experts" and capacity * n_experts != M * top_k:
            # Shared-experts spec validates exact equality — round-up
            # to BM created headroom we can't use. Configs landing here
            # are uncommon (would need M*top_k % (E*32) != 0) but error
            # explicitly so the user picks a different M.
            raise ValueError(
                f"MoE: shared_experts requires E*capacity ({n_experts * capacity}) "
                f"== M*top_k ({M * top_k}); pick M so M*top_k is a multiple of E*32"
            )

        self._M = M
        self._D = d_model
        self._H = d_intermediate
        self._E = n_experts
        self._K = top_k
        self._C = capacity
        self._routing = routing
        self._out_dtype = out_dtype

        # f32 router output → pcf.moe_router consumes directly. fp8_skip:
        # E is tiny and routing is precision-sensitive.
        self.router = Linear(d_model, n_experts, out_dtype="f32", fp8_skip=True)

        # Same layout as world_engine's ``.view()``'d expert_in/out so
        # the state-dict remap is one ``.reshape`` per layer.
        self.expert_in = Parameter(_zeros(n_experts * d_intermediate, d_model, dtype="bf16"))
        self.expert_out = Parameter(_zeros(n_experts * d_model, d_intermediate, dtype="bf16"))

        total_slots = n_experts * capacity
        wl_size = 2 * total_slots // _MOE_BM

        if routing == "balanced":
            # Deterministic work_list — built once.
            wl: list[int] = []
            for i in range(total_slots // _MOE_BM):
                grp_start = i * _MOE_BM
                wl.append(grp_start)
                wl.append(grp_start // capacity)
            self.work_list = _tensor(wl, dtype="s32")
        else:
            # shared_experts / correct: work_list is rewritten by the
            # router each call (chosen experts / per-expert counts vary
            # per input). Allocate the buffer; contents start zeroed but
            # are overwritten before being read.
            self.work_list = _empty(wl_size, dtype="s32")

        # Pre-allocate every output buffer the kernels write into. Reusing
        # them across calls saves ~5 ``cuMemAllocAsync`` per forward (~150 µs
        # each on Ada/Blackwell), which is most of the gap vs ``F.grouped_mm``
        # for a 24-layer × 5-NFE inference loop. Routing outputs are zeroed
        # by the router kernel itself; the outproj output is atomic-add'd
        # into so it gets a per-call ``zero_()`` in forward.
        self._buf_token_ids = _empty(total_slots, dtype="s32")
        self._buf_slot_weights = _empty(total_slots, dtype="f32")
        self._buf_counts = _empty(n_experts, dtype="s32")
        self._buf_h = _empty(total_slots, d_intermediate, dtype=out_dtype)
        self._buf_out_f32 = _empty(M, d_model, dtype="f32")
        # Workspace buffers used only by shared_experts routing.
        if routing == "shared_experts":
            self._buf_cum_probs = _empty(n_experts, dtype="f32")
            self._buf_chosen_experts = _empty(top_k, dtype="s32")
        # Workspace used only by correct routing.
        if routing == "correct":
            self._buf_offsets = _empty(n_experts + 1, dtype="s32")

        self._cast_out = Cast(out_dtype)

    def prepare(self, *, fp8: bool = False, **kwargs) -> None:
        """Optionally quantize expert weights to e4m3 for fp8 MMA compute.

        Mirrors ``Linear.prepare(fp8=True)``: idempotent, keeps the
        router in bf16 (its inner Linear has ``fp8_skip=True``), and
        the inproj output buffer stays in its allocated half dtype
        because the fused-silu epilogue can't store fp8 yet.

        Honors ``QUARK_MOE_NO_FP8`` — when set, the entire MoE block
        stays bf16 even if the parent model passes ``fp8=True``. The
        opt-out also propagates to the recursive ``super().prepare()``
        call so any current/future Linear children inside the block
        skip their fp8 quantization too. The router stays bf16 either
        way (its inner Linear has ``fp8_skip=True``).
        """
        moe_fp8 = fp8 and not _moe_fp8_disabled()
        if moe_fp8 and not getattr(self, "_fp8", False):
            self.expert_in.data = _quantize_to_e4m3(self.expert_in.data)
            self.expert_out.data = _quantize_to_e4m3(self.expert_out.data)
            self._fp8 = True
        super().prepare(fp8=moe_fp8, **kwargs)

    def forward(self, x):
        import quark.functional as pcf

        orig_shape = tuple(x.shape)
        if len(orig_shape) > 2:
            n_total = 1
            for d in orig_shape[:-1]:
                n_total *= int(d)
            x = x.reshape(n_total, orig_shape[-1])

        logits = self.router(x)  # [M, E] f32

        # The router kernel zero-inits its three outputs at entry — we
        # only need to ``zero_()`` the outproj output (atomic-add target)
        # before each invocation.
        if not _IS_METAL:
            self._buf_out_f32.zero_()

        if self._routing == "balanced":
            pcf.moe_router(
                logits,
                E=self._E,
                top_k=self._K,
                capacity=self._C,
                token_ids_out=self._buf_token_ids,
                slot_weights_out=self._buf_slot_weights,
                counts_out=self._buf_counts,
            )
        elif self._routing == "shared_experts":
            pcf.moe_router_shared(
                logits,
                E=self._E,
                top_k=self._K,
                capacity=self._C,
                token_ids_out=self._buf_token_ids,
                slot_weights_out=self._buf_slot_weights,
                counts_out=self._buf_counts,
                work_list_out=self.work_list,
                cum_probs_out=self._buf_cum_probs,
                chosen_experts_out=self._buf_chosen_experts,
            )
        else:  # correct
            pcf.moe_router_correct(
                logits,
                E=self._E,
                top_k=self._K,
                capacity=self._C,
                token_ids_out=self._buf_token_ids,
                slot_weights_out=self._buf_slot_weights,
                counts_out=self._buf_counts,
                work_list_out=self.work_list,
                offsets_out=self._buf_offsets,
            )

        # Fp8 compute: when expert weights have been quantized to e4m3
        # via ``prepare(fp8=True)``, both kernels need ``compute_dtype="e4m3"``
        # so the bf16-side input gets cast on the smem load and the MMA
        # runs in fp8. b_dtype is inferred from the weight tensor dtype.
        compute_dtype = "e4m3" if getattr(self, "_fp8", False) else None

        pcf.moe_inproj(
            x,
            self.expert_in.data,
            self._buf_token_ids,
            self.work_list,
            n_experts=self._E,
            top_k=self._K,
            out_dtype=self._out_dtype,
            compute_dtype=compute_dtype,
            out=self._buf_h,
        )

        y_f32 = pcf.moe_outproj(
            self._buf_h,
            self.expert_out.data,
            self._buf_token_ids,
            self._buf_slot_weights,
            self.work_list,
            M=self._M,
            n_experts=self._E,
            top_k=self._K,
            compute_dtype=compute_dtype,
            out=self._buf_out_f32,
        )

        y = self._cast_out(y_f32)
        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape)
        return y
