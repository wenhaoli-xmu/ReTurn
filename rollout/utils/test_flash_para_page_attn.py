import torch

from rollout.cache import ParaPageCache
from rollout.attention import flash_para_page_attn


class CausalKV(ParaPageCache):
    pass


class DiagonalKV(ParaPageCache):
    def _vis(self, n):
        return torch.eye(n, dtype=torch.int8, device=self.device)


class SinkLocalKV(ParaPageCache):
    def _vis(self, n):
        i = torch.arange(n, device=self.device)
        m = torch.zeros(n, n, dtype=torch.int8, device=self.device)
        m[i, i] = 1
        m[i[1:], i[1:] - 1] = 1
        m[:, 0] = 1
        return m


def ref_last_para(q_r, paras, vis, scale):
    dev = q_r.device
    n = len(paras); i = n - 1; Lq = paras[i][0].shape[0]
    Ks, Vs, col_para, col_pos = [], [], [], []
    for j in range(n):
        k, v = paras[j]; Ks.append(k); Vs.append(v)
        col_para += [j] * k.shape[0]; col_pos += list(range(k.shape[0]))
    K, V = torch.cat(Ks).float(), torch.cat(Vs).float()
    col_para = torch.tensor(col_para, device=dev); col_pos = torch.tensor(col_pos, device=dev)
    H = q_r.shape[1]; Hkv = K.shape[1]; h2kv = torch.arange(H, device=dev) // (H // Hkv)
    qe = q_r.permute(1, 0, 2).float()
    ke = K[:, h2kv, :].permute(1, 0, 2); ve = V[:, h2kv, :].permute(1, 0, 2)
    s = torch.einsum('hqd,hkd->hqk', qe, ke) * scale
    vmask = vis[i, col_para].bool(); is_last = col_para == i
    pr = torch.arange(Lq, device=dev)[:, None]; causal = pr >= col_pos[None, :]
    mask = vmask[None, :] & (~is_last[None, :] | causal)
    s = s.masked_fill(~mask[None], float('-inf'))
    return torch.einsum('hqk,hkd->hqd', s.softmax(-1), ve).permute(1, 0, 2)


def main():
    torch.manual_seed(0)
    dev = 'cuda'
    P, H, Hkv, D = 16, 8, 2, 64
    scale = 1.0 / (D ** 0.5)
    samples = [[20, 7, 33], [16, 16], [5, 40, 3, 18]]

    for cls in (CausalKV, DiagonalKV, SinkLocalKV):
        cache = cls(max_tokens=P * 512, num_heads=Hkv, head_dim=D, device=dev,
                    max_slots=8, max_para=8)
        req_ids = list(range(len(samples))); kv = {}
        for req, lens in zip(req_ids, samples):
            cache.alloc(req); kv[req] = []
            for L in lens:
                k = torch.randn(L, Hkv, D, device=dev, dtype=torch.bfloat16)
                v = torch.randn(L, Hkv, D, device=dev, dtype=torch.bfloat16)
                cache.insert_paragraph(req, k, v); kv[req].append([k, v])
        cache.activate(req_ids)

        q_list = [torch.randn(kv[req][-1][0].shape[0], H, D, device=dev, dtype=torch.bfloat16) for req in req_ids]
        q = torch.cat(q_list)
        o = flash_para_page_attn(q, cache, scale)

        outs, off = [], 0
        for req in req_ids:
            Lq = kv[req][-1][0].shape[0]
            outs.append(ref_last_para(q[off:off + Lq], kv[req], cache._vis(len(kv[req])), scale))
            off += Lq
        o_ref = torch.cat(outs)
        err = (o.float() - o_ref.float()).abs().max().item()
        print(f"{cls.__name__:12s} o={err:.3e}")
        assert err < 5e-2, f"{cls.__name__}  reference comparison failed "
    print("OK")


if __name__ == "__main__":
    main()
