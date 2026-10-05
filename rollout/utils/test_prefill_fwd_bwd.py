import math

import torch

from rollout.attention import flash_prefill_func
from rollout.constant import PAGE_SIZE


def _rope(x, pos, cos, sin, rot):

    half = rot // 2
    c = cos[pos].unsqueeze(1)
    s = sin[pos].unsqueeze(1)
    x1, x2, xp = x[..., :half], x[..., half:rot], x[..., rot:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, xp], -1)


def ref_forward(q, k_flat, v_flat, seqs, cos, sin, group, rot, scale):

    outs = []
    for q_sl, tok_rows, qpos in seqs:
        qq = q[q_sl].float()
        K = k_flat[tok_rows].float()
        V = v_flat[tok_rows].float()
        S = K.shape[0]
        pos_k = torch.arange(S, device=q.device)
        pos_q = qpos + torch.arange(qq.shape[0], device=q.device)
        qo = _rope(qq, pos_q, cos, sin, rot)
        ko = _rope(K, pos_k, cos, sin, rot)
        ko = ko.repeat_interleave(group, 1)
        vv = V.repeat_interleave(group, 1)
        att = torch.einsum("qhd,khd->hqk", qo, ko) * scale
        mask = pos_q[:, None] >= pos_k[None, :]
        att = att.masked_fill(~mask[None], float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", att.softmax(-1), vv))
    return torch.cat(outs)


def run_case(name, head_dim, rot, num_heads, num_kv_heads, lens, qpos_list, seed):
    dev = "cuda"
    P = PAGE_SIZE
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    group = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    half = rot // 2

    max_pos = max(lens) + 8
    inv = 1.0 / (10000.0 ** (torch.arange(0, half, device=dev, dtype=torch.float32) / half))
    freqs = torch.outer(torch.arange(max_pos, device=dev, dtype=torch.float32), inv)
    cos_bf, sin_bf = freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


    npages = [(s + P - 1) // P for s in lens]
    perm = torch.randperm(sum(npages) + 3).tolist()
    page_ids, cu_p, masks, ppos = [], [0], [], []
    seqs_meta = []
    for s, np_ in zip(lens, npages):
        pgs = [perm.pop() for _ in range(np_)]
        page_ids += pgs
        cu_p.append(len(page_ids))
        acc = 0
        for i in range(np_):
            valid = min(P, s - i * P)
            masks.append(valid)
            ppos.append(acc)
            acc += valid
        seqs_meta.append(pgs)
    total_pages = sum(npages) + 3


    lq = [s - qp for s, qp in zip(lens, qpos_list)]
    cu_q = [0]
    for l in lq:
        cu_q.append(cu_q[-1] + l)
    q = torch.randn(cu_q[-1], num_heads, head_dim, device=dev, dtype=dtype, requires_grad=True)
    k_pool = torch.randn(total_pages, P, num_kv_heads, head_dim, device=dev, dtype=dtype,
                         requires_grad=True)
    v_pool = torch.randn_like(k_pool, requires_grad=True)

    t = lambda x, dt: torch.tensor(x, dtype=dt, device=dev)
    page_table = t(page_ids, torch.int32)
    mask_table = t(masks, torch.uint8)
    page_pos = t(ppos, torch.int32)
    cu_pages = t(cu_p, torch.int32)
    cu_q_t = t(cu_q, torch.int32)
    qpos_t = t(qpos_list, torch.int32)
    n = len(lens)

    o = flash_prefill_func(
        q, k_pool, v_pool, page_table, mask_table, cu_pages, cu_q_t, qpos_t,
        page_pos, cos_bf, sin_bf, num_kv_heads, n, scale)
    do = torch.randn_like(o)
    o.backward(do)
    dq, dk, dv = q.grad.clone(), k_pool.grad.clone(), v_pool.grad.clone()


    k_flat = k_pool.view(-1, num_kv_heads, head_dim)
    v_flat = v_pool.view(-1, num_kv_heads, head_dim)
    seqs = []
    for r, (s, qp, pgs) in enumerate(zip(lens, qpos_list, seqs_meta)):
        rows = torch.cat([
            torch.arange(pg * P, pg * P + min(P, s - i * P), device=dev)
            for i, pg in enumerate(pgs)])
        seqs.append((slice(cu_q[r], cu_q[r + 1]), rows, qp))
    q.grad = k_pool.grad = v_pool.grad = None
    o_ref = ref_forward(q, k_flat, v_flat, seqs, cos_bf.float(), sin_bf.float(),
                        group, rot, scale)
    o_ref.backward(do.float())

    def rel(a, b):
        return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-8)).item()

    errs = {"o": rel(o, o_ref), "dq": rel(dq, q.grad),
            "dk": rel(dk, k_pool.grad), "dv": rel(dv, v_pool.grad)}
    print(f"[{name}] " + "  ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    tol = {"o": 2e-2, "dq": 3e-2, "dk": 3e-2, "dv": 3e-2}
    for k, v in errs.items():
        assert v < tol[k], (name, k, v)


def main():
    torch.cuda.init()

    run_case("full-rope varlen", head_dim=128, rot=128, num_heads=8, num_kv_heads=2,
             lens=[190, 77, 260], qpos_list=[0, 30, 128], seed=0)

    run_case("single fresh", head_dim=128, rot=128, num_heads=4, num_kv_heads=4,
             lens=[333], qpos_list=[0], seed=1)

    run_case("partial rotary", head_dim=128, rot=64, num_heads=8, num_kv_heads=2,
             lens=[145, 210], qpos_list=[64, 0], seed=2)

    run_case("prod geometry", head_dim=256, rot=64, num_heads=16, num_kv_heads=4,
             lens=[520, 96], qpos_list=[200, 0], seed=4)

    run_case("long", head_dim=128, rot=128, num_heads=8, num_kv_heads=2,
             lens=[2048 + 17], qpos_list=[1024], seed=3)
    print("flash_prefill_func  forward / backward matches  fp32  matches reference  ✓")


if __name__ == "__main__":
    main()
