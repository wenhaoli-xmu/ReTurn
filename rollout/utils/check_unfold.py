import argparse

import torch
from transformers import AutoModelForCausalLM

import rollout.model as rmodel
from rollout.constant import PAGE_SIZE


SPLIT = 7


class FakeReq:
    temperature, top_p, top_k = 0.0, 1.0, 0
    subs, swap = (), ()
    ephem_at = swap_at = None

    def __init__(self, rid, ids, **kwargs):
        self.id, self.prompt_ids = rid, list(ids)
        self.split_ids, self.output = [SPLIT], []
        self.__dict__.update(kwargs)

    @property
    def last_token(self):
        return self.output[-1]


def para(seed, n):
    return [SPLIT] + [11 + (seed * 31 + i * 17) % 900 for i in range(n - 1)]


def alloc(model, sid):
    kv, lin = model.get_kv_cache(), model.get_lin_caches()
    if sid in kv.pages:
        kv.free(sid)
        for cache in lin:
            cache.free(sid)
    kv.alloc(sid)
    for cache in lin:
        cache.alloc(sid)


def tables_ok(model, sid):

    kv = model.get_kv_cache()
    _, tp, _ = kv.build_tables([sid], prefill=False)
    mask = kv.mask_table[:tp].tolist()
    pos = kv.page_pos[:tp].tolist()
    acc, expected = 0, []
    for size in mask:
        expected.append(acc)
        acc += size
    return pos == expected and acc == kv.seqlen(sid) == sum(kv.ntok[sid])


class Trace:


    def __enter__(self):
        self.rows, self._orig = {}, rmodel.sample

        def patched(logits, reqs):
            tokens = self._orig(logits, reqs)
            for req, row, token in zip(reqs, logits, tokens):
                self.rows.setdefault(req.id, []).append(
                    (row.detach().float().cpu().clone(), token))
            return tokens

        rmodel.sample = patched
        return self

    def __exit__(self, *args):
        rmodel.sample = self._orig


def step(model, reqs, decode_steps):

    for req, token in zip(reqs, model.prefill(reqs)):
        req.output.append(token)
    for _ in range(decode_steps):
        for req, token in zip(reqs, model.decode(reqs)):
            req.output.append(token)


def logical_kv(model, sid):

    kv = model.get_kv_cache()
    return [
        torch.cat([pool[layer][page, :size]
                   for page, size in zip(kv.pages[sid], kv.ntok[sid])]).cpu()
        for pool in (kv.k_pool, kv.v_pool)
        for layer in range(len(kv.k_pool))
    ]


def test_swap(model, data):

    kv, lin = model.get_kv_cache(), model.get_lin_caches()
    alloc(model, "s")
    free_before = sorted(kv.free_pages)
    step(model, [FakeReq("s", data["prompt"])], 0)

    pid = kv.para_id["s"][1]
    bookkeeping = (
        list(kv.pages["s"]),
        list(kv.ntok["s"]),
        list(kv.para_start["s"]),
        kv.seqlen("s"),
    )
    kv_before = logical_kv(model, "s")

    model.prefill_sub("s", pid, data["summary"])
    assert pid in kv.alt["s"] and kv.alt["s"][pid][0], " variant was not stored in  kv.alt"
    assert (
        list(kv.pages["s"]),
        list(kv.ntok["s"]),
        list(kv.para_start["s"]),
        kv.seqlen("s"),
    ) == bookkeeping, "prefill_sub  of  cut/paste  did not restore the active view "

    kv.swap("s", pid)
    assert tables_ok(model, "s")
    assert kv.seqlen("s") < bookkeeping[3], " a shorter variant must reduce total length "

    alt_pages = list(kv.alt["s"][pid][0])
    try:
        model.prefill_sub("s", pid, data["summary"])
    except AssertionError:
        pass
    else:
        raise AssertionError(" same  pid  duplicate variant write was not rejected ")
    assert list(kv.alt["s"][pid][0]) == alt_pages, " rejected writes must not modify  kv.alt"

    kv.swap("s", pid)
    assert (
        list(kv.pages["s"]),
        list(kv.ntok["s"]),
        list(kv.para_start["s"]),
        kv.seqlen("s"),
    ) == bookkeeping, "swap  round trip did not restore bookkeeping "
    assert tables_ok(model, "s")
    assert all(torch.equal(got, expected)
               for got, expected in zip(logical_kv(model, "s"), kv_before)), \
        "swap  after round trip  KV  not  bit  equal "

    kv.free("s")
    for cache in lin:
        cache.free("s")
    assert sorted(kv.free_pages) == free_before, " page leak or duplicate release "
    print("[1] alt/swap  round trip  +  page reclamation  +  duplicate write rejection  OK")


def scenario(model, data, probe):

    kv, lin = model.get_kv_cache(), model.get_lin_caches()
    for sid in ("x", "y"):
        alloc(model, sid)

    with Trace() as trace:
        step(model, [FakeReq("x", data["prompt"]), FakeReq("y", data["other"])], 3)
        for cache in lin:
            for sid in ("x", "y"):
                cache.persist(sid)

        mid = [FakeReq("x", data["guide_p"], ephem_at=0)] if probe else []
        step(model, mid + [FakeReq("y", data["pad"])], 4)
        for cache in lin:
            for sid in (("x", "y") if probe else ("y",)):
                cache.persist(sid)

        if probe:
            assert kv.seqlen("x") > 0
            kv.pop_paragraph("x")
            for cache in lin:
                cache.rollback("x")
            assert tables_ok(model, "x")

        trace.rows.clear()
        step(model, [FakeReq("x", data["guide_a"]), FakeReq("y", data["pad2"])], 8)
        return trace.rows["x"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    from rollout.monkey_patch.qwen35 import QwenModel
    model = QwenModel(hf, max_token=PAGE_SIZE * 512, max_reside=4, device=args.device)

    data = {
        "prompt": para(1, 200) + para(2, 300) + para(3, 150),
        "other": para(4, 180) + para(5, 120),
        "summary": para(6, 40),
        "guide_p": para(7, 5),
        "guide_a": para(8, 5),
        "pad": para(9, 5),
        "pad2": para(9, 5),
    }

    test_swap(model, data)

    got, expected = scenario(model, data, probe=True), scenario(model, data, probe=False)
    assert len(got) == len(expected)
    different = [
        i for i, (left, right) in enumerate(zip(got, expected))
        if not torch.equal(left[0], right[0])
    ]
    max_delta = max(
        abs(left[0] - right[0]).max().item()
        for left, right in zip(got, expected)
    )
    assert [row[1] for row in got] == [row[1] for row in expected], " after probe  token  diverge "
    assert not different, \
        f" after probe  logits  not  bit  equal ， step  {different[:5]}，max|Δ|={max_delta:.3e}"
    print("[2] ephem_at  probe paragraph  + pop + GDN rollback（ multiple sessions per batch ）OK")
    print("OK")


if __name__ == "__main__":
    main()
