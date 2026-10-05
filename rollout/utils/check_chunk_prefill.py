import argparse

import torch
from transformers import AutoModelForCausalLM

import rollout.model as rmodel
from rollout.constant import PAGE_SIZE


class FakeReq:
    temperature, top_p, top_k = 0.0, 1.0, 0
    subs, swap = (), ()

    def __init__(self, rid, prompt_ids, split_ids=None):
        self.id, self.prompt_ids, self.split_ids, self.output = rid, prompt_ids, split_ids or [], []

    @property
    def last_token(self):
        return self.output[-1]


def snapshot(model, sid):

    kv = model.get_kv_cache()
    pages, ntok = kv.pages[sid], kv.ntok[sid]
    kvs = []
    for li in range(len(kv.k_pool)):
        k = torch.cat([kv.k_pool[li][p, :n] for p, n in zip(pages, ntok)]).cpu()
        v = torch.cat([kv.v_pool[li][p, :n] for p, n in zip(pages, ntok)]).cpu()
        kvs.append((k, v))
    lin = [(None if c.conv[sid] is None else c.conv[sid].cpu(),
            None if c.rec[sid] is None else c.rec[sid].cpu())
           for c in model.get_lin_caches()]
    meta = {"ntok": list(ntok), "para_start": list(kv.para_start[sid]),
            "seqlen": kv.seqlen(sid), "para_lens": kv.paragraph_lens(sid)}
    return {"meta": meta, "kv": kvs, "lin": lin}


def run_scenario(model, chunk, D):

    rmodel.PREFILL_CHUNK = chunk
    sid = "s"
    kv, lin = model.get_kv_cache(), model.get_lin_caches()
    kv.alloc(sid)
    for c in lin:
        c.alloc(sid)

    records = []
    orig_sample = rmodel.sample

    def rec_sample(logits, reqs):
        toks = orig_sample(logits, reqs)
        for r, row, t in zip(reqs, logits, toks):
            records.append((r.id, row.detach().cpu().clone(), t))
        return toks

    rmodel.sample = rec_sample
    snaps, outs = [], []
    try:
        def turn(prompt, ndec):
            r = FakeReq(sid, prompt, split_ids=[D["M"]])
            r.output.append(model.prefill([r])[0])
            for _ in range(ndec):
                r.output.append(model.decode([r])[0])
            for c in lin:
                c.persist(sid)
            snaps.append(snapshot(model, sid))
            outs.append(list(r.output))
            return r.output

        o1 = turn(D["p1"], D["n1"])
        o2 = turn([o1[-1]] + D["suf2"], D["n2"])
        turn([o2[-1]] + D["suf3"], D["n3"])
    finally:
        rmodel.sample = orig_sample
        kv.free(sid)
        for c in lin:
            c.free(sid)
    return {"snaps": snaps, "records": records, "outs": outs}


def _cmp(name, a, b, bad):
    if a is None and b is None:
        return
    if not torch.equal(a, b):
        d = (a.float() - b.float()).abs()
        bad.append(f"  {name}: {int((a != b).sum())}/{a.numel()} elems difference , max|Δ|={d.max().item():.3e}")


def compare(ref, run, tag, strict, logp_atol):
    bad = []
    if ref["outs"] != run["outs"]:
        bad.append(f"  greedy tokens  mismatch : ref={ref['outs']} got={run['outs']}")
    assert len(ref["records"]) == len(run["records"])
    max_dlogit = max_dlogp = 0.0
    for i, ((ra, la, ta), (rb, lb, tb)) in enumerate(zip(ref["records"], run["records"])):
        assert ra == rb
        if ta != tb:
            bad.append(f"   sampling event #{i}: token {ta} vs {tb}")
            continue
        if strict:
            _cmp(f" sampling event #{i} logits", la, lb, bad)
        else:
            max_dlogit = max(max_dlogit, (la.float() - lb.float()).abs().max().item())
            dlogp = abs(la.float().log_softmax(-1)[ta].item() - lb.float().log_softmax(-1)[tb].item())
            max_dlogp = max(max_dlogp, dlogp)
            if dlogp > logp_atol:
                bad.append(f"   sampling event #{i}:  sampled  token logp  difference  {dlogp:.3e} > {logp_atol:g}")
    for t, (sa, sb) in enumerate(zip(ref["snaps"], run["snaps"])):
        if sa["meta"] != sb["meta"]:
            bad.append(f"  turn{t + 1}  bookkeeping mismatch : {sa['meta']} vs {sb['meta']}")
        if strict:
            for li, ((ka, va), (kb, vb)) in enumerate(zip(sa["kv"], sb["kv"])):
                _cmp(f"turn{t + 1} KV[{li}].k", ka, kb, bad)
                _cmp(f"turn{t + 1} KV[{li}].v", va, vb, bad)
            for li, ((ca, ra_), (cb, rb_)) in enumerate(zip(sa["lin"], sb["lin"])):
                _cmp(f"turn{t + 1} GDN[{li}].conv", ca, cb, bad)
                _cmp(f"turn{t + 1} GDN[{li}].rec", ra_, rb_, bad)
    extra = "" if strict else f"  max|Δlogit|={max_dlogit:.3e} max|Δlogp(tok)|={max_dlogp:.3e}"
    print(f"[{tag}] {'OK' + ('（bit  level equality ）' if strict else '') if not bad else 'MISMATCH'}{extra}")
    for line in bad[:40]:
        print(line)
    if len(bad) > 40:
        print(f"  ...  total  {len(bad)}  mismatches ")
    return not bad


def _count_steps(model, chunk, D):

    reqs1 = [FakeReq("s", D["p1"], split_ids=[D["M"]])]
    r2 = FakeReq("s", [0] + D["suf2"], split_ids=[D["M"]])
    r3 = FakeReq("s", [0] + D["suf3"], split_ids=[D["M"]])
    out = []
    for reqs in (reqs1, [r2], [r3]):
        steps, _, _ = model._plan_prefill_steps(reqs, chunk)
        out.append(len(steps))
    return out


def _data(aligned):
    torch.manual_seed(0)
    M = 10007
    R = lambda n: torch.randint(0, 10000, (n,)).tolist()
    if aligned:
        p1 = [M] + R(63) + [M] + R(127) + [M] + R(191)
        suf2, suf3 = R(63) + [M] + R(127) + [M] + R(63), R(63)
    else:
        p1 = [M] + R(137) + [M] + R(201) + [M] + R(169)
        suf2, suf3 = R(7) + [M] + R(181) + [M] + R(150), R(9)
    return {"M": M, "p1": p1, "suf2": suf2, "suf3": suf3, "n1": 5, "n2": 4, "n3": 3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--logp-atol", type=float, default=0.25, help=" general-mode sampled  token logp  difference limit （ measured noise  ~0.1）")
    args = p.parse_args()

    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    mt = hf.config.model_type.lower()
    if "qwen3_5" in mt:
        from rollout.monkey_patch.qwen35 import QwenModel
    elif "qwen3" in mt:
        from rollout.monkey_patch.qwen3 import QwenModel
    else:
        raise ValueError(f"no model_type={mt!r}")
    model = QwenModel(hf, max_token=PAGE_SIZE * 256, max_reside=4, device=args.device)

    D = _data(aligned=True)
    print(f"==  strict mode （ aligned  64）: chunk 64 ( steps  {_count_steps(model, 64, D)}) vs 128 ( steps  {_count_steps(model, 128, D)}) ==")
    ok = compare(run_scenario(model, 64, D), run_scenario(model, 128, D),
                 "aligned 64 vs 128", strict=True, logp_atol=0)

    D = _data(aligned=False)
    big = 1 << 30
    print(f"==  general mode :  single step  chunk {big} ( steps  {_count_steps(model, big, D)}) vs  multiple steps  48 ( steps  {_count_steps(model, 48, D)}) ==")
    ok &= compare(run_scenario(model, big, D), run_scenario(model, 48, D),
                  "single vs chunk=48", strict=False, logp_atol=args.logp_atol)

    assert ok, "chunk prefill  reference comparison failed "
    print("OK")


if __name__ == "__main__":
    main()
