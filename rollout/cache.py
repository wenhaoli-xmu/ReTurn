import numpy as np
import torch

from rollout.constant import PAGE_SIZE
from rollout.engine import KVCapacityError


class KVCache:
    def __init__(self, num_kv_layers, max_tokens, num_kv_heads, head_dim, max_reside, device="cuda"):
        P = PAGE_SIZE
        self.page_size = P
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.max_reside = max_reside
        self.max_pages = max_tokens // P


        self.k_pool = [torch.empty(self.max_pages, P, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
                       for _ in range(num_kv_layers)]
        self.v_pool = [torch.empty(self.max_pages, P, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
                       for _ in range(num_kv_layers)]
        self.k_flat = [t.view(-1, num_kv_heads, head_dim) for t in self.k_pool]
        self.v_flat = [t.view(-1, num_kv_heads, head_dim) for t in self.v_pool]
        self.free_pages = list(range(self.max_pages))
        self.offloader = None


        self.pages = {}
        self.ntok = {}
        self.para_start = {}
        self.seqlen_map = {}
        self.closed = {}


        self.para_id = {}
        self.alt = {}
        self._pid = {}


        d = device
        self.page_table = torch.zeros(self.max_pages, dtype=torch.int32, device=d)
        self.mask_table = torch.zeros(self.max_pages, dtype=torch.uint8, device=d)


        self.page_pos = torch.zeros(self.max_pages, dtype=torch.int32, device=d)
        self.cu_pages = torch.zeros(max_reside + 1, dtype=torch.int32, device=d)
        self.cu_q = torch.zeros(max_reside + 1, dtype=torch.int32, device=d)
        self.qpos = torch.zeros(max_reside, dtype=torch.int32, device=d)
        self.wpos = torch.zeros(max_tokens, dtype=torch.long, device=d)

        self._plan_T = None
        self._req_q = {}


        h = lambda k, t: torch.empty(k, dtype=t).pin_memory()
        self._h_pg, self._h_mk, self._h_pp = (h(self.max_pages, torch.int32),
                                              h(self.max_pages, torch.uint8),
                                              h(self.max_pages, torch.int32))
        self._h_cp, self._h_cq, self._h_qp = (h(max_reside + 1, torch.int32),
                                              h(max_reside + 1, torch.int32),
                                              h(max_reside, torch.int32))
        self._h_ix, self._h_vl = h(max_reside, torch.int64), h(max_reside, torch.uint8)
        self._h_wp = h(max_tokens, torch.int64)
        (self._a_pg, self._a_mk, self._a_pp, self._a_cp, self._a_cq,
         self._a_qp, self._a_ix, self._a_vl, self._a_wp) = (
            t.numpy() for t in (self._h_pg, self._h_mk, self._h_pp, self._h_cp,
                                self._h_cq, self._h_qp, self._h_ix, self._h_vl, self._h_wp))
        self._d_ix = torch.zeros(max_reside, dtype=torch.int64, device=d)
        self._d_vl = torch.zeros(max_reside, dtype=torch.uint8, device=d)


        self._tv, self._tkey, self._tp = 0, None, 0


    def configure_offload(self, max_cpu_bytes=16 << 30,
                          transfer_bytes=64 << 20):
        if self.offloader is not None or self.pages:
            raise RuntimeError('Configure offload once, before admitting sessions')
        from rollout.offload import KVOffload
        self.offloader = KVOffload(self, max_cpu_bytes, transfer_bytes)

    def alloc(self, sid):
        self._tv += 1
        self.pages[sid] = []
        self.ntok[sid] = []
        self.para_start[sid] = []
        self.seqlen_map[sid] = 0
        self.closed[sid] = True
        self.para_id[sid] = []
        self.alt[sid] = {}
        self._pid[sid] = 0

    def free(self, sid):
        if self.offloader is not None:
            self.offloader.free(sid)
        self._tv += 1
        self.free_pages.extend(self.pages.pop(sid))
        for pg, _ in self.alt.pop(sid).values():
            self.free_pages.extend(p for p in pg if p >= 0)
        self.ntok.pop(sid)
        self.para_start.pop(sid)
        self.seqlen_map.pop(sid)
        self.closed.pop(sid)
        self.para_id.pop(sid)
        self._pid.pop(sid)

    def npage(self, sid):
        return len(self.pages[sid])


    def _bounds(self, sid, i):

        ps = self.para_start[sid]
        return ps[i], ps[i + 1] if i + 1 < len(ps) else len(self.pages[sid])

    def new_paragraph(self, sid):

        self.para_start[sid].append(len(self.pages[sid]))
        self.para_id[sid].append(self._pid[sid])
        self._pid[sid] += 1
        self.closed[sid] = True

    def _write_pos(self, w, base, take):


        if take == 1:
            self._a_wp[w] = base
        else:
            self._a_wp[w:w + take] = np.arange(base, base + take)

    def _append(self, sid, L, w):


        P = self.page_size
        pages, ntok = self.pages[sid], self.ntok[sid]
        spare = P - ntok[-1] if pages and not self.closed[sid] else 0
        needed = (max(0, L - spare) + P - 1) // P
        if needed > len(self.free_pages):
            raise KVCapacityError('KV allocation exceeds capacity outside a probe boundary')
        np0, end = len(pages), w + L

        if pages and not self.closed[sid] and ntok[-1] < P:
            take = min(P - ntok[-1], L)
            self._write_pos(w, pages[-1] * P + ntok[-1], take)
            ntok[-1] += take
            w += take

        while w < end:
            if not self.free_pages:

                raise RuntimeError
            pg = self.free_pages.pop()
            pages.append(pg)
            take = min(P, end - w)
            ntok.append(take)
            self._write_pos(w, pg * P, take)
            w += take
        self.closed[sid] = False
        self.seqlen_map[sid] += L
        self._tv += len(pages) != np0
        return w

    def swap(self, sid, pid):
        if self.offloader is not None:
            self.offloader.swap_many([(sid, pid)])
        else:
            self._swap_resident(sid, pid)

    def swap_many(self, pairs, *, required_pages=0, boundary_sids=(), start_cycle=()):
        if self.offloader is not None:
            self.offloader.swap_many(pairs, required_pages=required_pages,
                                     boundary_sids=boundary_sids, start_cycle=start_cycle)
        else:
            for sid, pid in pairs:
                self.swap(sid, pid)

    def register_alternative(self, sid, pid, variant):
        assert pid in self.para_id[sid] and pid not in self.alt[sid]
        self.alt[sid][pid] = variant
        if self.offloader is not None:
            self.offloader.touch(sid, pid)

    def _validate_swap(self, sid, pid):
        assert pid in self.alt[sid], f"swap: pid {pid}  has no stored alternative variant "
        assert pid != self.para_id[sid][-1], f"swap: pid {pid}  is the last paragraph ， cannot be swapped "

    def _swap_resident(self, sid, pid):


        self._validate_swap(sid, pid)


        self._tv += 1
        i = self.para_id[sid].index(pid)
        a, b = self._bounds(sid, i)
        pages, ntok, ps = self.pages[sid], self.ntok[sid], self.para_start[sid]
        old, (npg, nnt) = (pages[a:b], ntok[a:b]), self.alt[sid][pid]
        assert not npg or npg[0] >= 0, 'CPU KV must be restored before entering the active page table'
        pages[a:b], ntok[a:b] = npg, nnt
        for j in range(i + 1, len(ps)):
            ps[j] += len(npg) - (b - a)
        self.seqlen_map[sid] += sum(nnt) - sum(old[1])
        assert self.seqlen_map[sid] == sum(ntok), "swap  after  KV seqlen/ntok  mismatch "
        self.alt[sid][pid], self.closed[sid] = old, True

    def _truncate_at(self, sid, i, k=None):


        self._tv += 1
        p, n, s, d = self.pages[sid], self.ntok[sid], self.para_start[sid], self.para_id[sid]
        if k is None:
            k = s[i]
        tail = (k, p[k:], n[k:], s[i:], d[i:])
        del p[k:], n[k:], s[i:], d[i:]
        self.closed[sid] = True
        return tail

    def cut(self, sid, pid):


        i = self.para_id[sid].index(pid)
        k, tp, tn, ts, td = self._truncate_at(sid, i)
        self.seqlen_map[sid] = sum(self.ntok[sid])
        return (i, k, tp, tn, ts, td, self._pid[sid])

    def paste(self, sid, tail):


        i, k, tp, tn, ts, td, nxt = tail
        _, new_pg, new_nt, _, _ = self._truncate_at(sid, i, k)
        p, n, s, d = self.pages[sid], self.ntok[sid], self.para_start[sid], self.para_id[sid]
        p += tp
        n += tn
        s += ts
        d += td
        self.seqlen_map[sid] = sum(n)
        self._pid[sid] = nxt
        return (new_pg, new_nt)

    def pop_paragraph(self, sid):


        i = len(self.para_start[sid]) - 1
        pid = self.para_id[sid][i]
        assert pid not in self.alt[sid], \
            f"pop_paragraph  can only pop a newly created paragraph （ without variants ），pid={pid}  has a variant "
        _, tp, tn, _, _ = self._truncate_at(sid, i)
        drop = sum(tn)
        self.free_pages.extend(tp)
        self.seqlen_map[sid] -= drop
        assert self.seqlen_map[sid] == sum(self.ntok[sid]), \
            "pop_paragraph  after  KV seqlen/ntok  mismatch "
        self._pid[sid] = pid
        return drop

    def seqlen(self, sid):
        return self.seqlen_map[sid]

    def paragraph_lens(self, sid):

        ntok = self.ntok[sid]
        return [sum(ntok[slice(*self._bounds(sid, i))]) for i in range(len(self.para_start[sid]))]


    def plan_reset(self):
        self._plan_T = 0

    def plan_append(self, sid, L, new_para=False):
        if new_para:
            self.new_paragraph(sid)
        self._plan_T = self._append(sid, L, self._plan_T)

    def flush_wpos(self):


        T, self._plan_T = self._plan_T, None
        if T:
            self.wpos[:T].copy_(self._h_wp[:T])
        return T

    def build_tables(self, sids, prefill):


        n = len(sids)
        key = (self._tv, tuple(sids))
        hit = key == self._tkey


        if prefill:
            q = [self._req_q[s] for s in sids]
            tot_q = sum(q)
            self._a_qp[:n] = [self.seqlen_map[s] - k for s, k in zip(sids, q)]
            self._a_cq[0], self._a_cq[1:n + 1] = 0, np.cumsum(q)
            self.cu_q[:n + 1].copy_(self._h_cq[:n + 1])
        else:
            tot_q = n
            self._a_qp[:n] = [self.seqlen_map[s] - 1 for s in sids]
        self.qpos[:n].copy_(self._h_qp[:n])
        if hit:


            self._a_vl[:n] = [self.ntok[s][-1] for s in sids]
            self._d_vl[:n].copy_(self._h_vl[:n])
            self.mask_table.index_copy_(0, self._d_ix[:n], self._d_vl[:n])
            return n, self._tp, tot_q
        lens = [len(self.pages[s]) for s in sids]
        tp = sum(lens)
        w = 0
        for s, L in zip(sids, lens):
            if not L:
                continue
            nt = np.asarray(self.ntok[s], dtype=np.int32)
            self._a_pg[w:w + L] = self.pages[s]
            self._a_mk[w:w + L] = nt
            self._a_pp[w] = 0
            self._a_pp[w + 1:w + L] = np.cumsum(nt[:-1])
            w += L
        self._a_cp[0], self._a_cp[1:n + 1] = 0, np.cumsum(lens)
        self._a_ix[:n] = np.cumsum(lens) - 1
        if tp:
            self.page_table[:tp].copy_(self._h_pg[:tp])
            self.mask_table[:tp].copy_(self._h_mk[:tp])
            self.page_pos[:tp].copy_(self._h_pp[:tp])
        self.cu_pages[:n + 1].copy_(self._h_cp[:n + 1])
        self._d_ix[:n].copy_(self._h_ix[:n])
        self._tkey, self._tp = key, tp
        return n, tp, tot_q

    def set_req_q(self, req_q):
        self._req_q = req_q


    def write(self, layer_idx, k, v, T):

        self.k_flat[layer_idx].index_copy_(0, self.wpos[:T], k)
        self.v_flat[layer_idx].index_copy_(0, self.wpos[:T], v)


class LinearStateCache:


    def __init__(self, device, max_reside, dims):
        self.device = device
        Hv, dk, dv, C, K = dims
        self.C, self.K, self.Hv, self.dk, self.dv = C, K, Hv, dk, dv
        self.conv = {}
        self.rec = {}
        self.snap = {}

        self.g_conv = torch.zeros(max_reside, C, K, dtype=torch.bfloat16, device=device)
        self.g_rec = torch.zeros(max_reside, Hv, dk, dv, dtype=torch.float32, device=device)
        self._sids = []
        self.gids = []
        self._dirty = True

    def _moot(self, sids):


        self.gids = [None if g in sids else g for g in self.gids]
        self._dirty = True

    def alloc(self, sid):
        self.conv[sid] = None
        self.rec[sid] = None
        self._dirty = True

    def free(self, sid):
        self.conv.pop(sid, None)
        self.rec.pop(sid, None)
        self.snap.pop(sid, None)
        self._moot({sid})


    def snapshot(self, sid):


        if sid in self.gids:
            i = self.gids.index(sid)
            self.snap[sid] = (self.g_conv[i].clone(), self.g_rec[i].clone())
        else:
            self.snap[sid] = (self.conv[sid], self.rec[sid])

    def restore(self, sid, state):


        self.conv[sid], self.rec[sid] = state
        self._dirty = True

    def rollback(self, sid):


        s = self.snap.pop(sid, None)
        if s is None:
            return
        self.conv[sid], self.rec[sid] = s
        self._moot({sid})


    def fill(self, sids):


        if not self._dirty and self.gids == list(sids):
            self._sids = list(sids)
            return


        prev = {s: i for i, s in enumerate(self.gids) if s is not None}
        n = len(sids)

        def remap(buf, src):
            rows = []
            for s in sids:
                if s in prev:
                    rows.append(buf[prev[s]])
                elif src[s] is not None:
                    rows.append(src[s])
                else:
                    rows.append(torch.zeros_like(buf[0]))
            return torch.stack(rows)

        new_conv = remap(self.g_conv, self.conv)
        new_rec = remap(self.g_rec, self.rec)
        self.g_conv[:n].copy_(new_conv)
        self.g_rec[:n].copy_(new_rec)
        self._sids = self.gids = list(sids)
        self._dirty = False

    def gather(self):
        n = len(self._sids)
        return self.g_conv[:n], self.g_rec[:n]

    def scatter(self, conv, rec):

        n = len(self._sids)
        if conv is not None:
            self.g_conv[:n].copy_(conv)
        if rec is not None:
            self.g_rec[:n].copy_(rec)

    def persist(self, sid):

        if sid in self._sids:
            i = self._sids.index(sid)
            self.conv[sid] = self.g_conv[i].clone()
            self.rec[sid] = self.g_rec[i].clone()


        self._moot({sid})


    def prefill_state(self, sids, ref):

        hist = torch.stack([
            self.conv[s][:, 1:] if self.conv[s] is not None else ref.new_zeros(self.C, self.K - 1)
            for s in sids])
        recs = [self.rec[s] for s in sids]
        if not any(r is not None for r in recs):
            return hist, None
        z = torch.zeros(self.Hv, self.dk, self.dv, device=self.device, dtype=torch.float32)
        init = torch.stack([r if r is not None else z for r in recs])
        return hist, init

    def scatter_prefill(self, sids, conv, rec):


        for i, s in enumerate(sids):
            if conv is not None:
                self.conv[s] = conv[i]
            if rec is not None:
                self.rec[s] = rec[i]
        self._moot(set(sids))
