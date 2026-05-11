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
small E). The inproj output ``_buf_h`` stays in ``out_dtype`` (bf16)
because the fused-silu epilogue can't yet store fp8 (PTX has no scalar
fp8 cvt — pending the ``packed_convert`` refactor); outproj reads it
back and casts on smem-load via ``compute_dtype``.

Outproj writes per-slot **f32** partials ``[total_slots, D]`` (not an
atomic scatter into ``[M, D]``). The router emits a
``token_slot_table[M, top_k]`` inverse-index, and ``moe_reduce``
gathers + weighted-sums the partials into the final ``[M, D]`` bf16
output. The f32 partial keeps the bf16 quantization confined to a
single cast at the end of the per-token sum (matching the precision
of the previous f32-atomic-add design); compared to bf16 partials,
this cuts per-call MoE error by ~3-4×, which matters because each
frame's MoE error gets baked into K/V via the commit pass and
compounds across frames.

Replaces the previous ``atomic_fetch_add`` design, which on Metal
was racy against the host-memset ``zero_()`` of its F32 accumulator
buffer and produced gibberish from frame 2+.

**``moe_fp8`` kwarg**: opt the MoE block out of fp8 independently of
the surrounding model (``Waypoint15Config.quant.moe = "bf16"``).
Propagated through ``Waypoint15.prepare`` → ``MoE.prepare`` so the
caller flips it on the config and the env doesn't need touching.
Useful when the runtime bf16→e4m3 smem cast on the input tile
dominates and only the cuBLAS gemms actually win from fp8.
"""

from __future__ import annotations

import sys

from quark.nn.layers import Linear
from quark.nn.module import Module, Parameter, _quantize_to_e4m3, _tensor, _zeros

_IS_METAL = sys.platform == "darwin"


def _empty(*shape, dtype: str):
    """Backend-native pinned device tensor.

    Returns a real ``QuarkTensor`` so the buffer carries a
    ``metal_handle`` (Metal) / ``data_ptr`` (CUDA). The launcher's
    auto-route picks those up and writes the kernel's output INTO the
    cached buffer; downstream readers (next kernel in the MoE chain)
    read real data, not a host-side scratch.
    """
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

        # Router output: bf16 (the well-tested NAX-GEMM dtype on
        # Metal). f32 output of the GEMM has known correctness bugs
        # at small N — the audit caught NaN/Inf in some output rows
        # for the small-E router shape. ``forward`` casts the bf16
        # logits up to f32 before handing them to
        # ``moe_router_correct``, which expects f32 for softmax
        # precision. ``fp8_skip``: E is tiny and routing is
        # precision-sensitive.
        self.router = Linear(d_model, n_experts, out_dtype="bf16", fp8_skip=True)

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

        # Pre-allocate every output buffer the kernels write into.
        # Reusing them across calls saves ~5 ``cuMemAllocAsync`` per
        # forward. Routing outputs are zeroed by the router kernel
        # itself; per-slot partials and the final ``[M, D]`` output
        # are fully overwritten before being read, so they don't need
        # per-call zero_().
        self._buf_token_ids = _empty(total_slots, dtype="s32")
        self._buf_slot_weights = _empty(total_slots, dtype="f32")
        self._buf_counts = _empty(n_experts, dtype="s32")
        self._buf_h = _empty(total_slots, d_intermediate, dtype=out_dtype)
        # Per-slot f32 partials from moe_outproj, gathered + reduced
        # into per-token bf16 output by moe_reduce. f32 (vs bf16)
        # doubles the buffer but keeps the bf16 quantization confined
        # to a single cast at the very end of the per-token sum,
        # matching the precision of the previous f32-atomic-add
        # design. Per-call MoE error otherwise compounds via the KV
        # cache (each frame's residual gets baked into K/V, the next
        # frame reads it back through attention).
        self._buf_partials = _empty(total_slots, d_model, dtype="f32")
        # Per-token reduce output [M, D] in the residual-stream dtype.
        # The reduce kernel writes every element it owns so no
        # zero-init is needed.
        self._buf_out = _empty(M, d_model, dtype=out_dtype)
        # Workspace buffers used only by shared_experts routing.
        if routing == "shared_experts":
            self._buf_cum_probs = _empty(n_experts, dtype="f32")
            self._buf_chosen_experts = _empty(top_k, dtype="s32")
        # Workspace used only by correct routing.
        if routing == "correct":
            self._buf_offsets = _empty(n_experts + 1, dtype="s32")
            # Inverse of token_ids: ``token_slot_table[m, k]`` is the
            # slot in token_ids/slot_weights that received token m's
            # k-th expert assignment. Emitted by moe_router_correct,
            # consumed by moe_reduce.
            self._buf_token_slot_table = _empty(M, top_k, dtype="s32")

    def prepare(self, *, fp8: bool = False, moe_fp8: bool | None = None, **kwargs) -> None:
        """Optionally quantize expert weights to e4m3 for fp8 MMA compute.

        Mirrors ``Linear.prepare(fp8=True)``: idempotent, keeps the
        router in bf16 (its inner Linear has ``fp8_skip=True``), and
        the inproj output buffer stays in bf16 because the fused-silu
        epilogue can't store fp8 yet.

        ``moe_fp8`` overrides ``fp8`` for this block specifically —
        when ``False`` the expert weights stay bf16 even if the parent
        model passed ``fp8=True``. Defaults to ``fp8`` when unset.
        The override also propagates to the recursive
        ``super().prepare()`` call so Linear children inside the block
        skip their fp8 quantization too.
        """
        effective_fp8 = fp8 if moe_fp8 is None else moe_fp8
        if effective_fp8 and not getattr(self, "_fp8", False):
            self.expert_in.data = _quantize_to_e4m3(self.expert_in.data)
            self.expert_out.data = _quantize_to_e4m3(self.expert_out.data)
            self._fp8 = True
        super().prepare(fp8=effective_fp8, **kwargs)

    def forward(self, x):
        import quark.functional as qf

        orig_shape = tuple(x.shape)
        if len(orig_shape) > 2:
            n_total = 1
            for d in orig_shape[:-1]:
                n_total *= int(d)
            x = x.reshape(n_total, orig_shape[-1])

        # Router GEMM (bf16 output) → cast to f32 for the softmax.
        # See the ``self.router`` construction comment for why we
        # don't run the GEMM at f32 directly.
        logits = self.router(x)  # [M, E] bf16
        if logits.dtype != "f32":
            logits = logits.astype("f32")

        if self._routing == "balanced":
            qf.moe_router(
                logits,
                E=self._E,
                top_k=self._K,
                capacity=self._C,
                token_ids_out=self._buf_token_ids,
                slot_weights_out=self._buf_slot_weights,
                counts_out=self._buf_counts,
            )
        elif self._routing == "shared_experts":
            qf.moe_router_shared(
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
            qf.moe_router_correct(
                logits,
                E=self._E,
                top_k=self._K,
                capacity=self._C,
                token_ids_out=self._buf_token_ids,
                slot_weights_out=self._buf_slot_weights,
                counts_out=self._buf_counts,
                work_list_out=self.work_list,
                offsets_out=self._buf_offsets,
                token_slot_table_out=self._buf_token_slot_table,
            )

        # Fp8 compute: when expert weights have been quantized to e4m3
        # via ``prepare(fp8=True)``, both kernels need
        # ``compute_dtype="e4m3"`` so the bf16-side input gets cast on
        # the smem load and the MMA runs in fp8. b_dtype is inferred
        # from the weight tensor dtype.
        compute_dtype = "e4m3" if getattr(self, "_fp8", False) else None

        qf.moe_inproj(
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

        # outproj: per-slot f32 partials [total_slots, D]. Each block
        # writes its BM rows in full; sentinel-skipped chunks leave
        # their slice untouched (moe_reduce only reads slots indexed
        # by token_slot_table, which never points at sentinel chunks).
        # f32 (not bf16) so the bf16 quantization happens once on the
        # final per-token sum — see ``_buf_partials`` comment above.
        qf.moe_outproj(
            self._buf_h,
            self.expert_out.data,
            self.work_list,
            M=self._M,
            n_experts=self._E,
            top_k=self._K,
            out_dtype="f32",
            compute_dtype=compute_dtype,
            out=self._buf_partials,
        )

        # reduce: gather + weighted sum → [M, D] bf16. Only the
        # ``correct`` routing emits ``token_slot_table``; the other
        # routing modes still need updating to use this kernel.
        if self._routing != "correct":
            raise NotImplementedError(
                "MoE.forward: routing modes other than 'correct' need "
                "their routers updated to emit token_slot_table for "
                "the new outproj/reduce pipeline. Use routing='correct'."
            )
        y = qf.moe_reduce(
            self._buf_partials,
            self._buf_slot_weights,
            self._buf_token_slot_table,
            n_experts=self._E,
            out_dtype=self._out_dtype,
            out=self._buf_out,
        )

        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape)
        return y
