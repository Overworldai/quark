"""Platform-dispatched baselines for the moe_outproj kernel."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def moe_outproj_baselines(kernel, tensors: dict) -> list[Baseline]:
    """CUDA: flashinfer's fused MoE. Metal: matmul + weighted scatter."""
    from popcorn.backend import IS_METAL

    s = kernel.spec
    if not IS_METAL:
        from popcorn.kernels.moe_common import flashinfer_fused_moe_baselines

        h_in = tensors["h_in"]
        return flashinfer_fused_moe_baselines(
            M=s.M,
            D=s.D,
            H=s.H,
            n_experts=s.n_experts,
            top_k=s.top_k,
            dtype=h_in.dtype,
            W_out_source=tensors["W_out"],
        )

    import mlx.core as mx

    from popcorn.backend import PT

    h_in = tensors["h_in"]
    W_out = tensors["W_out"].reshape(s.n_experts, s.D, s.H)
    token_ids = PT.to_cpu_numpy(tensors["token_ids"]).tolist()
    slot_weights = PT.to_cpu_numpy(tensors["slot_weights"]).tolist()
    wl = PT.to_cpu_numpy(tensors["work_list"]).reshape(-1, 2).tolist()
    M = s.M

    def run():
        out = mx.zeros((M, s.D), dtype=mx.float32)
        for grp_start, expert in wl:
            gs = int(grp_start)
            e = int(expert)
            h_g = h_in[gs : gs + 32]
            w_slot = mx.array([slot_weights[gs + i] for i in range(32)], dtype=mx.float32)
            proj = (h_g @ PT.transpose(W_out[e])).astype(mx.float32) * w_slot[:, None]
            tids = mx.array([token_ids[gs + i] for i in range(32)], dtype=mx.int32)
            out = out.at[tids].add(proj)
        mx.eval(out)

    return [Baseline("mx.matmul+scatter_add[bf16]", run)]
