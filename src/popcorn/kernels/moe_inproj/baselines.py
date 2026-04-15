"""Platform-dispatched baselines for the moe_inproj kernel."""

from __future__ import annotations

from popcorn.kernels.base import Baseline


def moe_inproj_baselines(kernel, tensors: dict) -> list[Baseline]:
    """CUDA: flashinfer's fused MoE. Metal: per-work-item matmul+SiLU."""
    from popcorn.backend import IS_METAL

    s = kernel.spec
    if not IS_METAL:
        from popcorn.kernels.moe_common import flashinfer_fused_moe_baselines

        X = tensors["X"]
        return flashinfer_fused_moe_baselines(
            M=s.M,
            D=s.D,
            H=s.H,
            n_experts=s.n_experts,
            top_k=s.top_k,
            dtype=X.dtype,
            W_in_source=tensors["W_in"],
            x_source=X,
        )

    import mlx.core as mx

    from popcorn.backend import PT

    X = tensors["X"]
    W_in = tensors["W_in"].reshape(s.n_experts, s.H, s.D)
    token_ids = PT.to_cpu_numpy(tensors["token_ids"]).tolist()
    wl = PT.to_cpu_numpy(tensors["work_list"]).reshape(-1, 2).tolist()

    def run():
        outs = []
        for grp_start, expert in wl:
            gs = int(grp_start)
            e = int(expert)
            tids_arr = mx.array([token_ids[gs + i] for i in range(32)], dtype=mx.int32)
            x_g = X[tids_arr]
            z = x_g @ PT.transpose(W_in[e])
            outs.append(z / (1.0 + mx.exp(-z)))
        mx.eval(mx.concatenate(outs, axis=0))

    return [Baseline("mx.matmul+silu[bf16]", run)]
