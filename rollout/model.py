import threading

import numpy as np
import torch
from tqdm import tqdm

from rollout.constant import USE_CUDA_GRAPH, PREFILL_CHUNK
from rollout.prefill import ForwardContext, DECODE_CONTEXT
from rollout.attention import _pick_num_splits


_CAPTURE_LOCK = threading.Lock()


def sample(logits, reqs):
    raw = logits.float()
    dev = logits.device
    temps = [r.temperature for r in reqs]
    tps = [r.top_p for r in reqs]
    tks = [r.top_k for r in reqs]
    logits = raw

    if all(t <= 0 for t in temps):
        tok = logits.argmax(-1)
        _save_logp(raw, tok, reqs)
        return tok.tolist()

    temperature = torch.tensor(temps, device=dev, dtype=torch.float32)
    top_p = torch.tensor(tps, device=dev, dtype=torch.float32)
    top_k = torch.tensor(tks, device=dev, dtype=torch.long)

    greedy = temperature <= 0
    temp = torch.where(greedy, torch.ones_like(temperature), temperature)
    logits = logits / temp[:, None]
    V = logits.shape[-1]

    if not any(k > 0 for k in tks) and not any(p < 1.0 for p in tps):
        probs = logits.softmax(-1)
        tok = torch.multinomial(probs, 1)[:, 0]
        if bool(greedy.any()):
            tok = torch.where(greedy, logits.argmax(-1), tok)
        _save_logp(raw, tok, reqs)
        return tok.tolist()

    eff_k = torch.where(top_k > 0, top_k, torch.full_like(top_k, V))
    kc = int(eff_k.max().clamp(max=V))
    vals, idx = logits.topk(kc, dim=-1)
    k = eff_k.clamp(max=kc)
    col = torch.arange(kc, device=logits.device)
    vals = vals.masked_fill(col[None, :] >= k[:, None], float("-inf"))
    cum = vals.softmax(-1).cumsum(-1)
    remove = cum > top_p[:, None]
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    vals = vals.masked_fill(remove, float("-inf"))
    probs = vals.softmax(-1)
    local = torch.multinomial(probs, 1)[:, 0]
    local = torch.where(greedy, torch.zeros_like(local), local)
    tok = idx.gather(1, local[:, None])[:, 0]
    _save_logp(raw, tok, reqs)
    return tok.tolist()


def _save_logp(logits, tokens, reqs):
    logp = logits.log_softmax(-1).gather(1, tokens[:, None])[:, 0].tolist()
    for req, value in zip(reqs, logp):
        req._next_logprob = value


class _Dummy:
    temperature, top_p, top_k = 0.0, 1.0, 0
    subs, swap = (), ()
    ephem_at = swap_at = None

    def __init__(self, rid, ids=(1, 2, 3, 4)):
        self.id, self.prompt_ids, self.output, self.split_ids = rid, list(ids), [], []

    @property
    def last_token(self):
        return self.output[-1]


class Model:
    def __init__(self, hf_model, max_token, max_reside, device="cuda"):
        self.max_token = max_token
        self.max_reside = max_reside
        self.device = device
        self.graphs = {}
        self._g_in = None


    def get_kv_cache(self):
        raise NotImplementedError


    def get_lin_caches(self):
        raise NotImplementedError


    def forward_logits(self, input_ids, ctx=DECODE_CONTEXT, req_cu=None):
        raise NotImplementedError


    def _conv_index(self, req_cu):
        raise NotImplementedError


    def _split_bounds(self, ids, split_ids):
        if not split_ids:
            return [(0, len(ids), True)]
        pos = [i for i, t in enumerate(ids) if t in split_ids]
        if not pos:
            return [(0, len(ids), False)]
        bounds = []
        if pos[0] > 0:
            bounds.append((0, pos[0], False))
        for k, a in enumerate(pos):
            b = pos[k + 1] if k + 1 < len(pos) else len(ids)
            bounds.append((a, b, True))
        return bounds

    def _decode_splits(self, n):
        return _pick_num_splits(n, self.num_kv_heads, self.device)


    def _plan_prefill_steps(self, reqs, chunk):


        pieces = []
        for ridx, r in enumerate(reqs):
            ids, split = r.prompt_ids, set(r.split_ids or [])
            ephem_at = getattr(r, "ephem_at", None)
            swap_at = getattr(r, "swap_at", None)

            if ephem_at is None:
                bounds = self._split_bounds(ids, split)
            else:
                assert 0 <= ephem_at < len(ids), (ephem_at, len(ids))
                bounds = self._split_bounds(ids[:ephem_at], split) if ephem_at else []
                bounds.append((ephem_at, len(ids), True))


            if swap_at is not None:
                assert 0 <= swap_at < len(ids), (swap_at, len(ids))
                cut = []
                for a, b, is_new in bounds:
                    if a < swap_at < b:
                        cut.extend(((a, swap_at, is_new), (swap_at, b, False)))
                    else:
                        cut.append((a, b, is_new))
                bounds = cut

            for a, b, is_new in bounds:
                seg = r.prompt_ids[a:b]
                off, first = 0, True
                while off < len(seg):
                    sub = seg[off:off + chunk]
                    new_para = is_new and first
                    swap_here = first and swap_at is not None and a == swap_at
                    snap_here = first and ephem_at is not None and a == ephem_at
                    pieces.append((ridx, new_para, sub, swap_here, snap_here))
                    off += len(sub)
                    first = False


        steps, cur, cur_n = [], [], 0
        snap_step = {}
        for ridx, new_para, toks, swap_here, snap_here in pieces:
            if cur and (cur_n + len(toks) > chunk or swap_here or snap_here):
                steps.append(cur)
                cur, cur_n = [], 0
            if snap_here:
                snap_step[ridx] = len(steps)
            cur.append((ridx, new_para, toks, swap_here, snap_here))
            cur_n += len(toks)
        if cur:
            steps.append(cur)
        finish_step = {}
        for si, step in enumerate(steps):
            for ridx, *_ in step:
                finish_step[ridx] = si
        return steps, finish_step, snap_step

    @torch.no_grad()
    def prefill(self, reqs):

        dev = self.device
        with torch.cuda.device(dev):
            kv, lin = self.get_kv_cache(), self.get_lin_caches()


            for r in reqs:
                if r.subs:
                    for pid, ids in r.subs:
                        self.prefill_sub(r.id, pid, ids)
                    r.subs = []

            steps, finish_step, snap_step = self._plan_prefill_steps(reqs, PREFILL_CHUNK)
            out = {}
            for si, step in enumerate(steps):

                swaps, lengths = [], []
                for ridx, _, _, swap_here, _ in step:
                    r = reqs[ridx]
                    if not swap_here:
                        continue


                    kv_start = getattr(r, "_kv_start", None)
                    if kv_start is not None and r.swap_at is not None:
                        expected = kv_start + r.swap_at
                        assert kv.seqlen(r.id) == expected, (
                            f"swap_at KV  boundary mismatch : {kv.seqlen(r.id)} != {expected}")
                    before = kv.seqlen(r.id)
                    lengths.append((r, before))
                    swaps.extend((r.id, pid) for pid in r.swap)
                boundary = [reqs[ridx] for ridx, _, _, swap_here, snap_here in step
                            if swap_here or snap_here]
                if boundary:
                    spares = {}
                    required = 0
                    for remaining_step in steps[si:]:
                        for ridx, new_para, tokens_left, _, _ in remaining_step:
                            r = reqs[ridx]
                            if r.id not in spares:
                                pages, ntok = kv.pages[r.id], kv.ntok[r.id]
                                spares[r.id] = (kv.page_size - ntok[-1]
                                                if pages and not kv.closed[r.id]
                                                and not any(sid == r.id for sid, _ in swaps) else 0)
                            if new_para:
                                spares[r.id] = 0
                            count = max(0, len(tokens_left) - spares[r.id])
                            needed = (count + kv.page_size - 1) // kv.page_size
                            required += needed
                            spares[r.id] = (spares[r.id] + needed * kv.page_size
                                            - len(tokens_left))
                    required += sum(1 for r in reqs if r.id in spares
                                    and r.max_new_tokens > 1 and spares[r.id] == 0)
                    required += max((getattr(r, '_decode_reserve', 0) for r in boundary), default=0)
                    kv.swap_many(swaps, required_pages=required,
                                 boundary_sids={r.id for r in boundary},
                                 start_cycle={reqs[ridx].id for ridx, _, _, _, snap_here in step
                                              if snap_here})
                for r, before in lengths:

                    if getattr(r, "_ephem_kvlen", None) is not None:
                        r._ephem_kvlen += kv.seqlen(r.id) - before
                    r.swap = []


                for ridx, s in snap_step.items():
                    if s == si:
                        r = reqs[ridx]
                        r._ephem_kvlen = kv.seqlen(r.id)
                        for c in lin:
                            c.snapshot(r.id)


                kv.plan_reset()
                packed, req_q, req_cu, step_sids, step_ridx = [], {}, [0], [], []
                for ridx, new_para, toks, _, _ in step:
                    r = reqs[ridx]
                    kv.plan_append(r.id, len(toks), new_para=new_para)
                    packed.extend(toks)
                    if step_sids and step_sids[-1] == r.id:
                        req_q[r.id] += len(toks)
                        req_cu[-1] += len(toks)
                    else:
                        step_sids.append(r.id)
                        step_ridx.append(ridx)
                        req_q[r.id] = len(toks)
                        req_cu.append(req_cu[-1] + len(toks))

                T = kv.flush_wpos()
                kv.set_req_q(req_q)
                n, _, _ = kv.build_tables(step_sids, prefill=True)
                conv = self._conv_index(req_cu) if lin else None
                ctx = ForwardContext(is_prefill=True, n=n, T=T, conv=conv, sids=tuple(step_sids))
                tokens = torch.tensor([packed], device=dev)

                h = self.model(
                    input_ids=tokens, kv=self.kv, lin_caches=self.lin_caches,
                    forward_ctx=ctx).last_hidden_state


                fin_pos = [req_cu[k + 1] - 1 for k, ridx in enumerate(step_ridx) if finish_step[ridx] == si]
                fin_reqs = [reqs[ridx] for ridx in step_ridx if finish_step[ridx] == si]
                if fin_reqs:
                    hs = h[0].index_select(0, torch.tensor(fin_pos, device=dev))
                    toks_out = sample(self.model.lm_head(hs), fin_reqs)
                    for r, t in zip(fin_reqs, toks_out):
                        out[r.id] = t

            return [out[r.id] for r in reqs]


    @torch.no_grad()
    def prefill_sub(self, sid, pid, ids):


        kv, lin = self.get_kv_cache(), self.get_lin_caches()

        assert pid in kv.para_id[sid], f"prefill_sub: pid {pid}  is not in the active paragraph list "
        assert pid not in kv.alt[sid], f"prefill_sub: pid {pid}  already has a variant ， existing variants cannot be overwritten "

        if not ids:
            kv.register_alternative(sid, pid, ([], []))
            return

        keep = [(c.conv[sid], c.rec[sid]) for c in lin]
        tail = kv.cut(sid, pid)
        self.prefill([_Dummy(sid, ids)])
        kv.register_alternative(sid, pid, kv.paste(sid, tail))
        for c, s in zip(lin, keep):
            c.restore(sid, s)


    @torch.no_grad()
    def decode(self, reqs):
        dev = self.device
        n = len(reqs)
        sids = [r.id for r in reqs]
        with torch.cuda.device(dev):
            kv, lin = self.get_kv_cache(), self.get_lin_caches()
            kv.plan_reset()
            for r in reqs:
                kv.plan_append(r.id, 1, new_para=False)
            kv.flush_wpos()
            kv.build_tables(sids, prefill=False)

            for c in lin:
                c.fill(sids)

            if self.graphs:
                self._g_in[:n].copy_(torch.tensor([[r.last_token] for r in reqs], device=dev))
                g, logits = self.graphs[n]
                g.replay()
                return sample(logits[:n], reqs)

            ctx = ForwardContext(is_prefill=False, n=n, T=n, num_splits=self._decode_splits(n))
            input_ids = torch.tensor([[r.last_token] for r in reqs], device=dev)
            logits = self.forward_logits(input_ids, ctx)
            return sample(logits, reqs)


    @torch.no_grad()
    def _capture(self, reqs, pool, warmup):
        dev = self.device
        n = len(reqs)
        sids = [r.id for r in reqs]
        kv, lin = self.get_kv_cache(), self.get_lin_caches()
        ctx = ForwardContext(is_prefill=False, n=n, T=n, num_splits=self._decode_splits(n))

        def step():
            kv.plan_reset()
            for r in reqs:
                kv.plan_append(r.id, 1, new_para=False)
            kv.flush_wpos()
            kv.build_tables(sids, prefill=False)
            for c in lin:
                c.fill(sids)
            self._g_in[:n].copy_(torch.tensor([[r.last_token] for r in reqs], device=dev))

        def run():
            return self.forward_logits(self._g_in[:n], ctx)

        s = torch.cuda.Stream(dev)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                step()
                run()
        torch.cuda.current_stream().wait_stream(s)

        step()
        g = torch.cuda.CUDAGraph()
        with _CAPTURE_LOCK:
            with torch.cuda.graph(g, pool=pool, stream=s, capture_error_mode="thread_local"):
                logits = run()
        self.graphs[n] = (g, logits)


    @torch.no_grad()
    def build_graph(self, warmup=3):
        if not USE_CUDA_GRAPH:
            return
        dev = self.device
        with torch.cuda.device(dev):
            kv, lin = self.get_kv_cache(), self.get_lin_caches()
            dummies = [_Dummy(("_cap", i)) for i in range(self.max_reside)]
            for r in dummies:
                kv.alloc(r.id)
                for c in lin:
                    c.alloc(r.id)
            for r, t in zip(dummies, self.prefill(dummies)):
                r.output.append(t)
            self._g_in = torch.zeros(self.max_reside, 1, dtype=torch.long, device=dev)
            self.graphs = {}
            pool = torch.cuda.graph_pool_handle()
            pos = int(str(dev).rsplit(":", 1)[-1]) if ":" in str(dev) else 0
            for n in tqdm(range(self.max_reside, 0, -1), desc=f"capture {dev}", position=pos):
                self._capture(dummies[:n], pool, warmup)
            for r in dummies:
                kv.free(r.id)
                for c in lin:
                    c.free(r.id)
            torch.cuda.empty_cache()
