"""kv_cache_update — ortho-RoPE on K + ring KV-cache write.

EXEMPT FROM 500-LINE RULE: RoPE + ring + tail + segments + partial-
write contract + spec_from_tensors validation all live together in
the kernel to keep the per-frame write semantics legible in one
place. Splitting would strand the multi-buffer role model.

ONE call per transformer layer per video frame. Per-layer specialization:
`pinned_dilation` is baked into the spec (1=dense, 8=dilated). Cache layout
is compact: capacity = num_buckets*tpf + tpf  (ring + tail current frame).

Buffers (declaration order = ParamSpec order):
    K        [B*Hk*tpf, Dh]            in_dtype          read
    V        [B*Hk*tpf, Dh]            in_dtype          read
    cos      [tpf, Dh//2]              f32               read
    sin      [tpf, Dh//2]              f32               read
    frame_t  [1]                       s32               read   (ring address state)
    Vt_cache [B*Hk*Dh, capacity]       kv_dtype          mutate (not OUTPUT_IDX)
    segments [B*max_segments*2]        s32               mutate (not OUTPUT_IDX)
    n_segs   [B]                       s32               mutate (not OUTPUT_IDX)
    K_cache  [B*Hk*capacity, Dh]       kv_dtype          OUTPUT_IDX = -1

Output policy: only K_cache is auto-checked against `reference()` by the
fuzz harness (single-output contract). Vt_cache, segments, n_segments
correctness lives in `tests/kernels/test_kv_cache_update.py` (TODO).

Why frame_t is a buffer (not a scalar param): the fuzz/bench harnesses
launch via `compiled.launch(buffers=...)` only — no scalar plumbing. A
1-elem device tensor is graph-capturable and trivially small.

EMIT NOTES (TODO — not yet implemented):
  Grid: (tpf // tile_T, B*Hk, 1). One block per (B, head, T-tile).
  Per-block flow:
    1. cp.async K, V, cos, sin → smem (16B granularity).
    2. async_wait + barrier.
    3. Per-thread RoPE on K in fp32 (read smem K + cos/sin → fp32 → mul/add),
       cast to kv_dtype in registers, scalar-store into K_out smem (16B
       contiguous in the inner dim → bank-friendly stage for vec store).
    4. Per-thread cast V to kv_dtype in registers, scalar-store TRANSPOSED
       into Vt_out smem ([Dh, tile_T] layout).
    5. Barrier.
    6. Cooperative `vec_store` (width=8 bf16 / width=16 e4m3 = 16B) from
       K_out smem into K_cache at TAIL slot (always) and at RING slot
       (predicated on write_step = (frame_t % pd == 0)).
    7. Same for Vt_out → Vt_cache.
    8. Single-thread per (batch=0 block) writes segments[b] + n_segs[b].

  Constraints: ALL gmem stores must be 16B-vectorized (per user). Smem
  scalar stores are fine (staging).
"""

from __future__ import annotations

from typing import ClassVar

import quark.lang as qk
from quark.blocks import TensorDecl
from quark.ir import DType
from quark.kernels.base import Kernel
from quark.kernels.decorator import kernel
from quark.kernels.kv_cache_update.baselines import kv_cache_update_baselines
from quark.kernels.kv_cache_update.config import KVCacheUpdateConfig
from quark.kernels.kv_cache_update.problems import kv_cache_update_problems
from quark.kernels.kv_cache_update.reference import (
    kv_cache_update_reference_numpy,
)
from quark.kernels.kv_cache_update.spec import KVCacheUpdateSpec


@kernel(
    "kv_cache_update",
    spec=KVCacheUpdateSpec,
    config=KVCacheUpdateConfig,
    output_idx=-1,
    problems=kv_cache_update_problems,
    baselines=kv_cache_update_baselines,
    reference=kv_cache_update_reference_numpy,
)
class KVCacheUpdateKernel(Kernel):
    # Parameter manifest — all tensor shapes are pure functions of spec.
    TENSORS: ClassVar[list[TensorDecl]] = [
        TensorDecl(
            "K",
            dtype=lambda s, c: s.in_dtype,
            shape=lambda s, c: (s.k_input_rows, s.k_input_cols),
        ),
        TensorDecl(
            "V",
            dtype=lambda s, c: s.in_dtype,
            shape=lambda s, c: (
                (s.k_input_rows, s.k_input_cols)
                if s.packed_qkv
                else (s.B * s.n_kv_heads * s.tpf, s.Dh)
            ),
        ),
        # cos/sin computed inline from (h, w, frame_t) — no table tensors.
        TensorDecl("frame_t", dtype=DType.S32, shape=lambda s, c: (1,)),
        # frozen=1: skip ring writes, segments reflect only committed entries.
        TensorDecl("frozen", dtype=DType.S32, shape=lambda s, c: (1,)),
        TensorDecl(
            "Vt_cache",
            dtype=lambda s, c: s.kv_dtype,
            shape=lambda s, c: (s.B * s.n_kv_heads * s.Dh, s.capacity),
            role="out",
        ),
        # segments + n_segments are write targets on every launch — the
        # kernel's single-thread tail writes the new segment entry. They
        # must be ``role="out"`` so the launcher's readonly gate doesn't
        # mark their buffer parameters const (Metal then rejects the
        # store with "read-only variable is not assignable"). The
        # launcher's OUTPUT_IDX convention still treats only the last
        # role="out" tensor (K_cache) as the correctness-check output.
        TensorDecl(
            "segments",
            dtype=DType.S32,
            shape=lambda s, c: (s.B * s.max_segments * 2,),
            role="out",
        ),
        TensorDecl(
            "n_segments",
            dtype=DType.S32,
            shape=lambda s, c: (s.B,),
            role="out",
        ),
        TensorDecl(
            "K_cache",
            dtype=lambda s, c: s.kv_dtype,
            shape=lambda s, c: (s.B * s.n_kv_heads * s.capacity, s.Dh),
            role="out",
        ),
    ]

    spec: KVCacheUpdateSpec
    config: KVCacheUpdateConfig

    # No MMA — override the default mma_sites to return []. _mma_cfg
    # never gets called because no TENSORS entry needs fragment loads.

    @classmethod
    def mma_sites(cls, spec) -> list:
        return []

    # ── Validity ──

    def is_valid(self) -> bool:
        s, c = self.spec, self.config
        if s.Dh % 16 != 0:
            return False
        if s.tpf % c.tile_T != 0:
            return False
        # tile_T must be a multiple of 16B / elem_bytes for the vec-store path
        # to land on aligned chunks of the inner Dh dim.
        elem_bytes = s.kv_dtype.bytes
        per_vec_elems = 16 // elem_bytes
        if s.Dh % per_vec_elems != 0:
            return False
        # Vec-store alignment: the smem row stride (in bytes) must be a
        # multiple of 16 so every `vec_load.b32.v4` address is 16-aligned.
        # - K_out:  row = (Dh + smem_pad) * kv_bytes
        # - Vt_out: row = (tile_T + smem_pad) * kv_bytes
        # `smem_pad=8` with fp8 (1B) bumps row stride by 8B → unaligned →
        # misaligned-address CUDA fault at launch. Reject those combos.
        if ((s.Dh + c.smem_pad) * elem_bytes) % 16 != 0:
            return False
        if ((c.tile_T + c.smem_pad) * elem_bytes) % 16 != 0:
            return False
        # Ring math sanity: dilation must divide num_buckets*pd evenly so the
        # bucket index space is unambiguous (matches world_engine assertion).
        if s.num_buckets <= 0 or s.pinned_dilation < 1:
            return False
        # cos/sin tables are Dh//2 wide — Dh must be even.
        if s.Dh % 2 != 0:
            return False
        # Worst-case segments fit.
        return s.max_segments >= 3

    def grid(self) -> tuple[int, int, int]:
        s, c = self.spec, self.config
        return (s.tpf // c.tile_T, s.B * s.n_kv_heads, 1)

    def flops(self) -> int:
        s = self.spec
        # RoPE: per-pair = 4 muls + 2 adds = 6 flops. tpf * Dh/2 pairs per (B, head).
        rope = s.B * s.n_kv_heads * s.tpf * (s.Dh // 2) * 6
        return rope  # cache write is memory-bound; this is just the math floor.

    @classmethod
    def make_tensors_numpy(cls, problem: dict, *, seed: int = 0x5A1E_5EED) -> dict:
        import numpy as np

        from quark.runtime.npconv import astype_numpy, zeros_for_dtype

        spec = KVCacheUpdateSpec(**problem)
        B, Hk, tpf, Dh = spec.B, spec.n_kv_heads, spec.tpf, spec.Dh
        cap = spec.capacity
        rng = np.random.default_rng(seed)

        if spec.packed_qkv:
            qkv_dim = spec.qkv_dim
            K_arr = astype_numpy(
                (rng.standard_normal((B * tpf, qkv_dim)) * 0.3).astype(np.float32), spec.in_dtype
            )
            # V is the same tensor as K when packed; share the buffer.
            V_arr = K_arr
        else:
            K_arr = astype_numpy(
                (rng.standard_normal((B * Hk * tpf, Dh)) * 0.3).astype(np.float32), spec.in_dtype
            )
            V_arr = astype_numpy(
                (rng.standard_normal((B * Hk * tpf, Dh)) * 0.3).astype(np.float32), spec.in_dtype
            )

        test_frame_t = spec.num_buckets * spec.pinned_dilation
        return {
            "K": K_arr,
            "V": V_arr,
            "frame_t": np.array([test_frame_t], dtype=np.int32),
            "frozen": np.zeros(1, dtype=np.int32),
            "Vt_cache": zeros_for_dtype((B * Hk * Dh, cap), spec.kv_dtype),
            "segments": np.zeros(B * spec.max_segments * 2, dtype=np.int32),
            "n_segments": np.zeros(B, dtype=np.int32),
            "K_cache": zeros_for_dtype((B * Hk * cap, Dh), spec.kv_dtype),
        }

    @classmethod
    def spec_from_tensors(
        cls,
        K,
        V,
        frame_t,
        frozen,
        Vt_cache,
        segments,
        n_segments,
        K_cache,
        *,
        B: int,
        n_kv_heads: int,
        H_spatial: int,
        W_spatial: int,
        num_buckets: int,
        pinned_dilation: int,
        kv_dtype: DType | str | None = None,
        max_segments: int = 3,
        packed_qkv: bool = False,
        n_q_heads: int = 0,
        rope_n_frames: int = 1,
    ) -> KVCacheUpdateSpec:
        """Derive a ``KVCacheUpdateSpec`` from the nine kernel tensors.

        The ring / tail layout is determined entirely by (B, n_kv_heads,
        H_spatial, W_spatial, num_buckets, pinned_dilation) — all of
        which the caller supplies as kwargs. ``Dh`` is read from K.
        """

        if K.ndim != 2:
            raise ValueError(f"pcf.kv_cache_update: K must be rank-2 (flat), got {K.shape}")
        if packed_qkv:
            qkv_dim = int(K.shape[1])
            Dh = qkv_dim // (n_q_heads + 2 * n_kv_heads)
        else:
            Dh = int(K.shape[1])
        tpf = H_spatial * W_spatial
        capacity = num_buckets * tpf + tpf
        if not packed_qkv and tuple(V.shape) != tuple(K.shape):
            raise ValueError(
                f"pcf.kv_cache_update: V shape {tuple(V.shape)} != K shape {tuple(K.shape)}"
            )
        if tuple(K_cache.shape) != (B * n_kv_heads * capacity, Dh):
            raise ValueError(
                f"pcf.kv_cache_update: K_cache shape {tuple(K_cache.shape)} != "
                f"({B * n_kv_heads * capacity}, {Dh})"
            )
        if tuple(Vt_cache.shape) != (B * n_kv_heads * Dh, capacity):
            raise ValueError(
                f"pcf.kv_cache_update: Vt_cache shape {tuple(Vt_cache.shape)} != "
                f"({B * n_kv_heads * Dh}, {capacity})"
            )
        if int(segments.shape[0]) != B * max_segments * 2:
            raise ValueError(
                f"pcf.kv_cache_update: segments shape {tuple(segments.shape)} != "
                f"({B * max_segments * 2},)"
            )
        if int(n_segments.shape[0]) != B:
            raise ValueError(
                f"pcf.kv_cache_update: n_segments shape {tuple(n_segments.shape)} != ({B},)"
            )
        return KVCacheUpdateSpec(
            B=B,
            n_kv_heads=n_kv_heads,
            Dh=Dh,
            H_spatial=H_spatial,
            W_spatial=W_spatial,
            num_buckets=num_buckets,
            pinned_dilation=pinned_dilation,
            in_dtype=DType.from_backend(K.dtype),
            kv_dtype=DType.coerce(kv_dtype) or DType.from_backend(K_cache.dtype),
            max_segments=max_segments,
            packed_qkv=packed_qkv,
            n_q_heads=n_q_heads,
            rope_n_frames=rope_n_frames,
        )

    @classmethod
    def tune_space(cls) -> dict[str, list]:
        # Restricted: autotune sweeping the full cross product of
        # tile_T × n_warps × smem_pad occasionally produced ILLEGAL_ADDRESS
        # from device-side OOB in one of the failing configs. Until that's
        # tracked down, we ship a single known-good point and let
        # ``default_for`` pick it.
        return {"tile_T": [64], "n_warps": [4], "smem_pad": [0]}

    # ── emit() ──

    def build(self) -> None:
        s, c = self.spec, self.config
        g = self.g
        Dh = s.Dh
        half_Dh = Dh // 2
        tile_T = c.tile_T
        tpf = s.tpf
        n_warps = c.n_warps
        n_threads = n_warps * 32
        L = s.L
        capacity = s.capacity
        num_buckets = s.num_buckets
        pd = s.pinned_dilation
        max_segs = s.max_segments
        Hk = s.n_kv_heads

        in_dt = s.in_dtype
        kv_dt = s.kv_dtype
        kv_b = kv_dt.bytes  # 1 (e4m3) or 2 (bf16)
        vec_elems = 16 // kv_b  # elements per 16B vec store (8 bf16 / 16 e4m3)

        bctx = self.bctx  # no MMA — decorator passes mma_cfg=None
        ctx = self.ctx
        g_K, g_V = g.K, g.V
        g_ft, g_Vt = g.frame_t, g.Vt_cache
        g_sg, g_ns, g_Kc = g.segments, g.n_segments, g.K_cache

        tid = bctx.tid

        t_tile_idx = ctx.block_idx("x")
        bh_idx = ctx.block_idx("y")

        # ── Per-block addresses (U32). Auto-CSE dedupes repeated muls. ──
        t_start = t_tile_idx * tile_T
        # Decompose grid Y index into (batch, kv_head).
        kv_h_u = bh_idx % Hk
        b_idx_u = bh_idx // Hk

        if s.packed_qkv:
            # Packed QKV: K/V are columns in [B*tpf, qkv_dim].
            k_in_row_base = b_idx_u * tpf + t_start
            k_in_col_base = bctx.c(s.k_col_offset) + kv_h_u * Dh
            v_in_col_base = bctx.c(s.v_col_offset) + kv_h_u * Dh
        else:
            # Legacy: K/V are [B*Hk*tpf, Dh], col always 0.
            k_in_row_base = bh_idx * tpf + t_start
            k_in_col_base = bctx.c(0)
            v_in_col_base = bctx.c(0)

        bh_x_cap = bh_idx * capacity
        k_cache_tail_row_base = bh_x_cap + (L + t_start)
        vt_d_base = bh_idx * Dh
        vt_col_tail = L + t_start

        # ── frame_t-derived control state (S32) ──
        frame_t_v = qk.load(g_ft, bctx.c(0, dtype=DType.S32), name="frame_t")
        frozen_v = qk.load(g.frozen, bctx.c(0, dtype=DType.S32))
        pd_c = bctx.c(pd, dtype=DType.S32)
        nb_c = bctx.c(num_buckets, dtype=DType.S32)
        zero_s = bctx.c(0, dtype=DType.S32)
        one_s = bctx.c(1, dtype=DType.S32)
        rem_v = frame_t_v % pd_c
        is_write_frame = qk.cmp("eq", rem_v, zero_s)  # PRED
        is_not_frozen = qk.cmp("eq", frozen_v, zero_s)  # PRED: frozen==0
        write_step = qk.and_(is_write_frame, is_not_frozen)  # ring write only when not frozen
        bucket = (frame_t_v + bctx.c(pd - 1, dtype=DType.S32)) // pd_c
        slot = bucket % nb_c
        base_s = slot * bctx.c(tpf, dtype=DType.S32)
        # Convert ring base to U32 for index arithmetic with block bases.
        base_u = qk.convert(base_s, DType.U32)
        k_cache_ring_row_base = bh_x_cap + base_u + t_start
        vt_col_ring = base_u + t_start

        # ── Smem allocs ──
        smem_pad = c.smem_pad
        K_in = qk.smem_alloc("K_in", in_dt, (tile_T, Dh), pad=smem_pad)
        V_in = qk.smem_alloc("V_in", in_dt, (tile_T, Dh), pad=smem_pad)
        # cos/sin computed inline — no smem needed.
        K_out = qk.smem_alloc("K_out", kv_dt, (tile_T, Dh), pad=smem_pad)
        Vt_out = qk.smem_alloc("Vt_out", kv_dt, (Dh, tile_T), pad=smem_pad)

        # ── cp.async loads ──
        K_in.copy_from(
            g_K.tile(row=k_in_row_base, col=k_in_col_base, shape=(tile_T, Dh)),
            tid=tid,
            n_threads=n_threads,
            async_load=True,
        )
        # V source: same tensor as K when packed (QKV), separate tensor otherwise.
        v_src = g_K if s.packed_qkv else g_V
        V_in.copy_from(
            v_src.tile(row=k_in_row_base, col=v_in_col_base, shape=(tile_T, Dh)),
            tid=tid,
            n_threads=n_threads,
            async_load=True,
        )
        # (cos/sin computed inline — no table load needed)
        qk.async_commit()
        qk.async_wait(0)
        qk.barrier("block")

        # ── RoPE pass: K_in + cos/sin (fp32) → K_out (kv_dtype) ──
        # Per thread iterates over (row, c) pair indices in [0, tile_T*half_Dh).
        # For fp8 kv_dtype, PTX has no scalar fp8 store — only the packed
        # `cvt.<fp8>x2.<src>x2` form. We pack two consecutive RoPE outputs
        # (y0[c], y0[c+1]) and (y1[c], y1[c+1]) into one b16 each per
        # iteration, then b16-store covers the 2 fp8 bytes at K_out[r, c]
        # and K_out[r, c+half_Dh].
        n_pairs = tile_T * half_Dh
        if n_pairs % n_threads != 0:
            raise ValueError(f"RoPE: n_pairs={n_pairs} not divisible by n_threads={n_threads}")
        pairs_per_thread = n_pairs // n_threads
        half_Dh_c = bctx.c(half_Dh)
        is_fp8 = kv_dt in (DType.E4M3, DType.E5M2)
        if is_fp8 and pairs_per_thread % 2 != 0:
            raise ValueError(
                f"fp8 RoPE: pairs_per_thread={pairs_per_thread} must be even (need pair grouping)"
            )

        # Inline RoPE: compute cos/sin from (h, w, frame_t) per element.
        from quark.lang.rope import emit_rope_cos_sin

        W_c = bctx.c(s.W_spatial)

        def _rope_one_pair(c_idx_v):
            """Compute (y0_f32, y1_f32) for one (r, c_idx_v) pair. r is closed
            over by the caller via ``r_v`` capture."""
            # Spatial position: absolute token index = t_start + r_v.
            spatial_idx = t_start + r_v
            h_idx = spatial_idx // W_c
            w_idx = spatial_idx % W_c
            c2 = c_idx_v * 2
            c2p1 = c2 + 1
            x0 = qk.convert(K_in[r_v, c2], DType.F32)
            x1 = qk.convert(K_in[r_v, c2p1], DType.F32)
            cv, sv = emit_rope_cos_sin(
                bctx,
                h_idx=h_idx,
                w_idx=w_idx,
                frame_t=qk.convert(frame_t_v, DType.U32),
                c_idx=c_idx_v,
                H=s.H_spatial,
                W=s.W_spatial,
                Dh=Dh,
            )
            y0 = x0 * cv - x1 * sv
            y1 = x1 * cv + x0 * sv
            return y0, y1

        if is_fp8:
            for i in range(0, pairs_per_thread, 2):
                # Two adjacent c_idx values (c_a, c_a+1) on the same row.
                flat_a = tid * pairs_per_thread + i
                r_v = flat_a // half_Dh_c
                c_a = flat_a % half_Dh_c
                c_b = c_a + 1
                y0_a, y1_a = _rope_one_pair(c_a)
                y0_b, y1_b = _rope_one_pair(c_b)
                # Packed convert: b16 holds (lo=y0_a_fp8, hi=y0_b_fp8). When
                # stored as b16 to byte-addressed e4m3 smem, the two fp8
                # bytes land at consecutive (r, c_a) and (r, c_a+1).
                y0_pk = qk.packed_convert(y0_a, y0_b, kv_dt)
                y1_pk = qk.packed_convert(y1_a, y1_b, kv_dt)
                K_out[r_v, c_a] = y0_pk
                K_out[r_v, c_a + half_Dh_c] = y1_pk
        else:
            for i in range(pairs_per_thread):
                flat = tid * pairs_per_thread + i
                r_v = flat // half_Dh_c
                c_idx = flat % half_Dh_c
                y0, y1 = _rope_one_pair(c_idx)
                K_out[r_v, c_idx] = qk.convert(y0, kv_dt)
                K_out[r_v, c_idx + half_Dh_c] = qk.convert(y1, kv_dt)

        # ── V transpose pass: V_in → Vt_out (transposed, kv_dtype) ──
        # Vt_out is [Dh, tile_T] row-major; storing transposed means
        # Vt_out[c, r] = V_in[r, c]. For fp8 we pack pairs along the
        # *inner* dim of Vt_out (= consecutive r's at fixed c) into one
        # b16 store per packed pair.
        if is_fp8:
            n_packed = (tile_T * Dh) // 2
            if n_packed % n_threads != 0:
                raise ValueError(
                    f"fp8 V xpose: n_packed={n_packed} not divisible by n_threads={n_threads}"
                )
            packed_per_thread = n_packed // n_threads
            half_tile_T_c = bctx.c(tile_T // 2)
            for i in range(packed_per_thread):
                pp = tid * packed_per_thread + i
                cc = pp // half_tile_T_c
                r_pair = pp % half_tile_T_c
                r_lo = r_pair * 2
                r_hi = r_lo + 1
                v_lo = V_in[r_lo, cc]
                v_hi = V_in[r_hi, cc]
                # bf16 → e4m3 packed; store as b16 covering (cc, r_lo) and (cc, r_lo+1).
                v_pk = qk.packed_convert(v_lo, v_hi, kv_dt)
                Vt_out[cc, r_lo] = v_pk
        else:
            n_v = tile_T * Dh
            if n_v % n_threads != 0:
                raise ValueError(f"V xpose: n_v={n_v} not divisible by n_threads={n_threads}")
            v_per_thread = n_v // n_threads
            Dh_c = bctx.c(Dh)
            for i in range(v_per_thread):
                flat = tid * v_per_thread + i
                r = flat // Dh_c
                cc = flat % Dh_c
                v_in = V_in[r, cc]
                v_out = v_in if kv_dt is in_dt else qk.convert(v_in, kv_dt)
                Vt_out[cc, r] = v_out  # transposed write

        qk.barrier("block")

        # ── Cooperative 16B vec_store: K_out → K_cache (tail always; ring conditional) ──
        total_K_vecs = (tile_T * Dh) // vec_elems
        if total_K_vecs % n_threads != 0:
            raise ValueError(
                f"K vec_store: total_K_vecs={total_K_vecs} not divisible by n_threads={n_threads}"
            )
        K_vecs_per_thread = total_K_vecs // n_threads
        cols_per_row_K = Dh // vec_elems
        cpr_K_c = bctx.c(cols_per_row_K)
        for i in range(K_vecs_per_thread):
            vid = tid * K_vecs_per_thread + i
            r = vid // cpr_K_c
            col_chunk = vid % cpr_K_c
            col_elem = col_chunk * vec_elems
            v = qk.vec_load(K_out, r, col_elem, width=4, dtype=DType.B32)
            # Tail (always)
            qk.vec_store(g_Kc, v, k_cache_tail_row_base + r, col_elem)
            # Ring (predicated)
            qk.vec_store(g_Kc, v, k_cache_ring_row_base + r, col_elem, pred=write_step)

        # ── Cooperative 16B vec_store: Vt_out → Vt_cache ──
        total_Vt_vecs = (Dh * tile_T) // vec_elems
        if total_Vt_vecs % n_threads != 0:
            raise ValueError(
                f"Vt vec_store: total_Vt_vecs={total_Vt_vecs} not divisible by n_threads={n_threads}"
            )
        Vt_vecs_per_thread = total_Vt_vecs // n_threads
        cols_per_row_Vt = tile_T // vec_elems
        cpr_Vt_c = bctx.c(cols_per_row_Vt)
        for i in range(Vt_vecs_per_thread):
            vid = tid * Vt_vecs_per_thread + i
            d = vid // cpr_Vt_c
            t_chunk = vid % cpr_Vt_c
            t_elem = t_chunk * vec_elems
            v = qk.vec_load(Vt_out, d, t_elem, width=4, dtype=DType.B32)
            gmem_row = vt_d_base + d
            qk.vec_store(g_Vt, v, gmem_row, vt_col_tail + t_elem)
            qk.vec_store(g_Vt, v, gmem_row, vt_col_ring + t_elem, pred=write_step)

        # ── Per-batch segments + n_segments write ──
        # Designated writer: block (t_tile_idx==0, bh_idx == b*Hk), tid==0.
        # Matches WE's mask semantics: the mask excludes the slot being
        # about to be overwritten whenever ``frame_idx % pd == 0``
        # (``is_write_frame``), INDEPENDENT of ``is_frozen``. The ring
        # write itself is still gated on both. Segments layout:
        #
        #   pre-wrap OR post-wrap-no-mask:
        #     (0, bucket*tpf or L), (L, tpf), (0, 0)
        #   post-wrap AND is_write_frame:
        #     (0, slot*tpf), ((slot+1)*tpf, (nb-slot-1)*tpf), (L, tpf)
        is_first_tile = qk.cmp("eq", t_tile_idx, bctx.c(0))
        is_first_head = qk.cmp("eq", kv_h_u, bctx.c(0))
        is_first_thread = qk.cmp("eq", tid, bctx.c(0))
        seg_pred = qk.and_(is_first_tile, is_first_head)
        seg_pred = qk.and_(seg_pred, is_first_thread)

        tpf_s = bctx.c(tpf, dtype=DType.S32)
        L_s = bctx.c(L, dtype=DType.S32)
        two_s = bctx.c(2, dtype=DType.S32)
        three_s = bctx.c(3, dtype=DType.S32)

        # slot = bucket % num_buckets (in S32, non-negative).
        slot = bucket % nb_c
        # Pre-wrap means the ring hasn't filled yet (bucket < num_buckets).
        # Post-wrap (ge num_buckets) and is_write_frame triggers the mask/split.
        is_pre_wrap = qk.cmp("lt", bucket, nb_c)
        is_post_wrap = qk.cmp("ge", bucket, nb_c)
        needs_mask = qk.and_(is_write_frame, is_post_wrap)

        # Pre-wrap "used ring" length: bucket*tpf. Post-wrap full L.
        pre_ring_len = bucket * tpf_s
        full_ring_len = L_s
        no_mask_ring_len = qk.select(is_pre_wrap, pre_ring_len, full_ring_len)

        # When needs_mask:
        #   seg0 = (0, slot*tpf)
        #   seg1 = ((slot+1)*tpf, (nb-slot-1)*tpf)
        #   seg2 = (L, tpf)
        #   n = 3
        # Else:
        #   seg0 = (0, no_mask_ring_len)
        #   seg1 = (L, tpf)
        #   seg2 = (0, 0)
        #   n = 2
        slot_tpf = slot * tpf_s
        slot_plus1_tpf = (slot + one_s) * tpf_s
        after_slot_len = (nb_c - slot - one_s) * tpf_s

        # Segment fields (per needs_mask).
        seg0_start_masked = zero_s
        seg0_len_masked = slot_tpf
        seg1_start_masked = slot_plus1_tpf
        seg1_len_masked = after_slot_len
        seg2_start_masked = L_s
        seg2_len_masked = tpf_s

        seg0_start_unmasked = zero_s
        seg0_len_unmasked = no_mask_ring_len
        seg1_start_unmasked = L_s
        seg1_len_unmasked = tpf_s
        seg2_start_unmasked = zero_s
        seg2_len_unmasked = zero_s

        seg0_start = qk.select(needs_mask, seg0_start_masked, seg0_start_unmasked)
        seg0_len = qk.select(needs_mask, seg0_len_masked, seg0_len_unmasked)
        seg1_start = qk.select(needs_mask, seg1_start_masked, seg1_start_unmasked)
        seg1_len = qk.select(needs_mask, seg1_len_masked, seg1_len_unmasked)
        seg2_start = qk.select(needs_mask, seg2_start_masked, seg2_start_unmasked)
        seg2_len = qk.select(needs_mask, seg2_len_masked, seg2_len_unmasked)
        n_seg = qk.select(needs_mask, three_s, two_s)

        seg_base = b_idx_u * (max_segs * 2)
        qk.store(g_sg, seg0_start, seg_base + 0, pred=seg_pred)
        qk.store(g_sg, seg0_len, seg_base + 1, pred=seg_pred)
        qk.store(g_sg, seg1_start, seg_base + 2, pred=seg_pred)
        qk.store(g_sg, seg1_len, seg_base + 3, pred=seg_pred)
        qk.store(g_sg, seg2_start, seg_base + 4, pred=seg_pred)
        qk.store(g_sg, seg2_len, seg_base + 5, pred=seg_pred)
        qk.store(g_ns, n_seg, b_idx_u, pred=seg_pred)
