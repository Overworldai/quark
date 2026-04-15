"""Backend-agnostic reference for the MoE out-projection."""

from __future__ import annotations

from popcorn.backend import PT


def moe_outproj_reference_for_spec(kernel, h_in, W_out, token_ids, slot_weights, work_list):
    s, c = kernel.spec, kernel.config
    D, H = s.D, s.H
    n_experts = s.n_experts
    BM = c.BM

    # Simulate the kernel's compute-dtype cast on load.
    ct = s.compute_dtype_resolved.backend
    if h_in.dtype != ct:
        h_in = PT.astype(h_in, ct)
    if W_out.dtype != ct:
        W_out = PT.astype(W_out, ct)

    h_f32 = PT.astype(h_in, PT.float32)
    W_3d = PT.astype(W_out, PT.float32).reshape(n_experts, D, H)
    output = PT.zeros(s.M, D, dtype=PT.float32)

    wl = PT.to_cpu_numpy(work_list).reshape(-1, 2).tolist()
    tok = PT.to_cpu_numpy(token_ids).tolist()
    weights_all = PT.to_cpu_numpy(slot_weights).tolist()
    # work_list uses fixed stride BM=32 in make_tensors; larger BM
    # configs alias adjacent entries into one BM block. Dedupe by
    # stepping in BM-stride chunks so bounds stay aligned with
    # len(weights_all) (= total_slots).
    make_tensors_bm = 32
    assert BM % make_tensors_bm == 0, (
        f"reference: BM={BM} must be a multiple of make_tensors stride "
        f"{make_tensors_bm}; update make_tensors or lower BM lower bound."
    )
    wl_step = BM // make_tensors_bm

    for grp_start, expert in wl[::wl_step]:
        gs = int(grp_start)
        e = int(expert)
        if gs + BM > len(weights_all):
            # Tail group that doesn't fit a full BM block — the kernel
            # skips these; skip the reference too.
            continue
        h_g = h_f32[gs : gs + BM]
        w_slot = PT.tensor([weights_all[gs + i] for i in range(BM)], dtype=PT.float32)
        proj = PT.matmul(h_g, PT.transpose(W_3d[e])) * w_slot[:, None]
        # Scatter-add into `output[tok[gs + i]]` for each i. MLX has no
        # index_add_; accumulate row-by-row to stay backend-portable.
        for i in range(BM):
            row = int(tok[gs + i])
            existing = output[row : row + 1]
            output = PT.set_slice(
                output, axis=0, start=row, stop=row + 1, src=existing + proj[i : i + 1]
            )
    return output
