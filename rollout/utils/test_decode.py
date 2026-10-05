import torch

from rollout.cache import KVCache
from rollout.attention import flash_prefill, flash_decode
from rollout.constant import PAGE_SIZE

DEV = "cuda"
Hq, Hkv, D, HALF = 16, 4, 256, 32
ROT = 2 * HALF
GROUP = Hq // Hkv
SCALE = 1.0 / (D ** 0.5)


def rope_table(mp):
    inv = 1.0 / (10000.0 ** (torch.arange(0, HALF, device=DEV).float() / HALF))
    f = torch.outer(torch.arange(mp, device=DEV).float(), inv)
    return f.cos().to(torch.bfloat16), f.sin().to(torch.bfloat16)


COS, SIN = rope_table(8192)


def apply_rope(x, pos):
    c = COS[pos].float()[:, None, :]
    s = SIN[pos].float()[:, None, :]
    x1, x2 = x[..., :HALF].float(), x[..., HALF:ROT].float()
    out = x.float().clone()
    out[..., :HALF] = x1 * c - x2 * s
    out[..., HALF:ROT] = x2 * c + x1 * s
    return out


def logical_kv(cache, sid):
    ks, vs = [], []
    for pg, nt in zip(cache.pages[sid], cache.ntok[sid]):
        ks.append(cache.k_pool[0][pg, :nt])
        vs.append(cache.v_pool[0][pg, :nt])
    return torch.cat(ks), torch.cat(vs)


def ref_attn(q, qpos, k, v, causal):
    Lk = k.shape[0]
    kpos = torch.arange(Lk, device=DEV)
    qr = apply_rope(q, qpos)
    kr = apply_rope(k, kpos).repeat_interleave(GROUP, 1)
    vv = v.repeat_interleave(GROUP, 1).float()
    qk = torch.einsum("qhd,khd->hqk", qr, kr) * SCALE
    if causal:
        qk = qk.masked_fill(~(qpos[:, None] >= kpos[None, :])[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", qk.softmax(-1), vv)


def new_cache():
    return KVCache(1, PAGE_SIZE * 512, Hkv, D, max_reside=4, device=DEV)


def prefill(cache, sid, paras):
    cache.alloc(sid)
    cache.plan_reset()
    Lq = 0
    for k, _ in paras:
        cache.plan_append(sid, k.shape[0], new_para=True)
        Lq += k.shape[0]
    T = cache.flush_wpos()
    cache.write(0, torch.cat([k for k, _ in paras]), torch.cat([v for _, v in paras]), T)
    cache.set_req_q({sid: Lq})
    return Lq


def decode_once(cache, sid, q, nk, nv):
    cache.plan_reset()
    cache.plan_append(sid, 1, new_para=False)
    T = cache.flush_wpos()
    cache.write(0, nk, nv, T)
    cache.build_tables([sid], prefill=False)
    return flash_decode(q, cache.page_table, cache.mask_table, cache.cu_pages, cache.qpos, cache.page_pos,
                        cache.k_pool[0], cache.v_pool[0], COS, SIN, Hkv, 4, SCALE)


def main():
    torch.manual_seed(0)
    rnd = lambda L: torch.randn(L, Hkv, D, device=DEV, dtype=torch.bfloat16)
    X, JUNK, Y = rnd(70), rnd(50), rnd(33)


    c = new_cache()
    Lq = prefill(c, 0, [(X, X), (Y, Y)])
    n, _, _ = c.build_tables([0], prefill=True)
    q = torch.randn(Lq, Hq, D, device=DEV, dtype=torch.bfloat16)
    o = flash_prefill(q, c.page_table, c.mask_table, c.cu_pages, c.cu_q, c.qpos, c.page_pos,
                      c.k_pool[0], c.v_pool[0], COS, SIN, Hkv, n, SCALE)
    k, v = logical_kv(c, 0)
    qpos = torch.arange(Lq, device=DEV)
    err = (o.float() - ref_attn(q, qpos, k, v, True)).abs().max().item()
    print(f"[prefill] err={err:.2e}")
    assert err < 3e-2


    c2 = new_cache()
    c2.plan_reset()
    KV = {}
    for sid, L in ((0, 70), (1, 33)):
        c2.alloc(sid)
        c2.plan_append(sid, L, new_para=True)
        KV[sid] = (rnd(L), rnd(L))
    T = c2.flush_wpos()
    c2.write(0, torch.cat([KV[0][0], KV[1][0]]), torch.cat([KV[0][1], KV[1][1]]), T)
    c2.set_req_q({0: 70, 1: 33})
    n, _, _ = c2.build_tables([0, 1], prefill=True)
    qb = torch.randn(103, Hq, D, device=DEV, dtype=torch.bfloat16)
    ob = flash_prefill(qb, c2.page_table, c2.mask_table, c2.cu_pages, c2.cu_q, c2.qpos, c2.page_pos,
                       c2.k_pool[0], c2.v_pool[0], COS, SIN, Hkv, n, SCALE)
    for sid, lo, hi in ((0, 0, 70), (1, 70, 103)):
        k, v = logical_kv(c2, sid)
        r = ref_attn(qb[lo:hi], torch.arange(hi - lo, device=DEV), k, v, True)
        e = (ob[lo:hi].float() - r).abs().max().item()
        print(f"[varlen prefill req{sid}] err={e:.2e}")
        assert e < 3e-2


    cd = new_cache(); prefill(cd, 0, [(X, X), (Y, Y), (JUNK, JUNK)]); cd.build_tables([0], True)
    fb = len(cd.free_pages)
    drop = cd.pop_paragraph(0)
    fa = len(cd.free_pages)
    assert drop == JUNK.shape[0], f"pop  returned  token  count must be  {JUNK.shape[0]}， actual  {drop}"
    assert cd._pid[0] == 2, "pop  must restore  pid  counter back ， otherwise the next paragraph ID will be skipped "
    cc = new_cache(); prefill(cc, 0, [(X, X), (Y, Y)]); cc.build_tables([0], True)
    assert cd.seqlen(0) == cc.seqlen(0) and cd.npage(0) == cc.npage(0)
    assert cd.para_start[0] == cc.para_start[0] and cd.para_id[0] == cc.para_id[0]


    cc.closed[0] = True
    qd = torch.randn(1, Hq, D, device=DEV, dtype=torch.bfloat16)
    nk, nv = rnd(1), rnd(1)
    e = (decode_once(cd, 0, qd, nk, nv).float() - decode_once(cc, 0, qd, nk, nv).float()).abs().max().item()
    print(f"[pop] err={e:.2e}  (pages freed={fa-fb})")
    assert e == 0.0, "pop_paragraph  must be  bit-identical  to the state without this paragraph "
    print("OK")


if __name__ == "__main__":
    main()
