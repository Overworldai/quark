"""OwlAttnInt problems — Phase 3.2 smoke shape + production target."""

from __future__ import annotations

from quark.kernels.base import Problem


def owl_attn_int8_problems() -> list[Problem]:
    return [
        # Phase 3.4a smoke: capacity = 2 * KvTile = 64 exercises the
        # online softmax carry across multiple K iterations. Smallest
        # shape that still tests the m_new=max(m_old, m_iter) +
        # rescale = exp(m_old - m_new) update logic between iters.
        Problem(
            "owl_int8_phase4_smoke",
            {
                "B": 1, "n_kv_heads": 1, "gqa_ratio": 1, "Dh": 64,
                "H_spatial": 2, "W_spatial": 4,
                "num_buckets": 7, "pinned_dilation": 1,
                "packed_qkv": False,
            },
            tags={"phase4", "smoke", "int8"},
        ),
        # Phase 3.6a smoke: NCW=2 multi-warp. Needs tpf >= BlockQRows
        # = NCW*MTiles*m_tile = 16. H=2, W=8 → tpf=16. capacity=64
        # (num_buckets=3 → tpf_cached=16, L=48, cap=64) gives 2 K-iters.
        Problem(
            "owl_int8_phase6_smoke",
            {
                "B": 1, "n_kv_heads": 1, "gqa_ratio": 1, "Dh": 64,
                "H_spatial": 2, "W_spatial": 8,
                "num_buckets": 3, "pinned_dilation": 1,
                "packed_qkv": False,
            },
            tags={"phase6", "smoke", "int8"},
        ),
        # Phase 3.2 smoke: smallest shape that still exercises one
        # full m8n16k32 MMA. capacity=32 = one KvTile iteration —
        # the degenerate single-iter case of the carry logic.
        Problem(
            "owl_int8_phase2_smoke",
            {
                "B": 1, "n_kv_heads": 1, "gqa_ratio": 1, "Dh": 64,
                "H_spatial": 2, "W_spatial": 4,
                "num_buckets": 3, "pinned_dilation": 1,
                "packed_qkv": False,
            },
            tags={"phase2", "smoke", "int8"},
        ),
        # Waypoint-1.5-1B 360p dense — the production target.
        # Currently gated by ``is_valid`` (packed_qkv=True, gqa=2,
        # multi-head) until Phase 3.4b lifts the smoke pattern.
        Problem(
            "owl_int8_360p_dense",
            {
                "B": 1, "n_kv_heads": 16, "gqa_ratio": 2, "Dh": 64,
                "H_spatial": 16, "W_spatial": 32,
                "num_buckets": 16, "pinned_dilation": 1,
                "packed_qkv": True,
            },
            tags={"production", "int8"},
        ),
    ]
