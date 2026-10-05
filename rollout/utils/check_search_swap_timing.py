import argparse
import asyncio

import torch
from transformers import AutoModelForCausalLM

import rollout.model as rmodel
from rollout.constant import PAGE_SIZE
from rollout.engine import Engine, Request


SPLIT = 7


def _tokens(seed, n):
    return [101 + (seed * 131 + i * 37) % 9000 for i in range(n)]


def _para(seed, n=64):
    return [SPLIT] + _tokens(seed, n - 1)


async def _batch(eng, reqs):

    eng.pending.extend(reqs)
    eng._wake.set()
    await asyncio.gather(*(r.done for r in reqs))
    return reqs


def _request(sid, prompt, *, split=True, **kw):
    return Request(
        id=sid,
        prompt_ids=list(prompt),
        max_new_tokens=kw.pop("max_new_tokens", 1),
        temperature=0.0,
        split_ids=[SPLIT] if split else [],
        **kw,
    )


class _LogitTrace:
    def __enter__(self):
        self.rows = {}
        self._orig = rmodel.sample

        def patched(logits, reqs):
            toks = self._orig(logits, reqs)
            for req, row, tok in zip(reqs, logits, toks):
                self.rows[req.id] = (row.detach().float().cpu().clone(), tok)
            return toks

        rmodel.sample = patched
        return self

    def __exit__(self, *args):
        rmodel.sample = self._orig


def _logical(pool, kv, layer, sid):
    return torch.cat([
        pool[layer][page, :ntok]
        for page, ntok in zip(kv.pages[sid], kv.ntok[sid])
    ])


def _state_diff(model, left, right):
    kv = model.get_kv_cache()
    maximum = 0.0
    exact = True
    for pool in (kv.k_pool, kv.v_pool):
        for layer in range(len(pool)):
            a = _logical(pool, kv, layer, left)
            b = _logical(pool, kv, layer, right)
            exact &= torch.equal(a, b)
            maximum = max(maximum, (a.float() - b.float()).abs().max().item())
    for cache in model.get_lin_caches():
        for attr in ("conv", "rec"):
            a, b = getattr(cache, attr)[left], getattr(cache, attr)[right]
            if a is None or b is None:
                exact &= a is b
                continue
            exact &= torch.equal(a, b)
            maximum = max(maximum, (a.float() - b.float()).abs().max().item())
    return exact, maximum


def _meta(model, sid):
    kv = model.get_kv_cache()
    return {
        "para_id": list(kv.para_id[sid]),
        "para_lens": kv.paragraph_lens(sid),
        "seqlen": kv.seqlen(sid),
        "alt_pids": sorted(kv.alt[sid]),
    }


class _KVLengthAudit:


    def __init__(self, model):
        self.kv = model.get_kv_cache()
        self.phase = ""
        self.events = []
        self._orig = {}

    def _layout(self, sid):
        kv = self.kv
        pages, ntok = kv.pages[sid], kv.ntok[sid]
        starts, pids = kv.para_start[sid], kv.para_id[sid]
        assert len(pages) == len(ntok)
        assert all(0 < n <= kv.page_size for n in ntok)
        assert kv.seqlen(sid) == sum(ntok)
        assert len(starts) == len(pids)
        assert starts == sorted(starts)
        assert all(0 <= x <= len(pages) for x in starts)
        if starts:
            assert starts[0] == 0
        assert sum(kv.paragraph_lens(sid)) == kv.seqlen(sid)
        assert len(pages) == len(set(pages))
        for alt_pages, alt_ntok in kv.alt[sid].values():
            assert len(alt_pages) == len(alt_ntok)
            assert all(0 < n <= kv.page_size for n in alt_ntok)
            assert len(alt_pages) == len(set(alt_pages))

    def check_resting_state(self):

        kv = self.kv
        owned = []
        for sid in kv.pages:
            self._layout(sid)
            owned.extend(kv.pages[sid])
            for alt_pages, _ in kv.alt[sid].values():
                owned.extend(alt_pages)
        assert len(owned) == len(set(owned)), "active/alternative KV pages  overlap "
        assert len(kv.free_pages) == len(set(kv.free_pages)), "free_pages  contains duplicate pages "
        assert set(owned).isdisjoint(kv.free_pages)
        assert set(owned) | set(kv.free_pages) == set(range(kv.max_pages))

    def one(self, phase, op, sid):
        found = [e for e in self.events
                 if e["phase"] == phase and e["op"] == op and e["sid"] == sid]
        assert len(found) == 1, (phase, op, sid, found)
        return found[0]

    def __enter__(self):
        kv = self.kv
        self._orig = {
            "plan_append": kv.plan_append,
            "swap": kv.swap,
            "pop_paragraph": kv.pop_paragraph,
            "build_tables": kv.build_tables,
        }

        def plan_append(sid, length, new_para=False):
            before = kv.seqlen(sid)
            npara = len(kv.para_id[sid])
            out = self._orig["plan_append"](sid, length, new_para=new_para)
            after = kv.seqlen(sid)
            assert after == before + length
            assert len(kv.para_id[sid]) == npara + int(new_para)
            self._layout(sid)
            self.events.append({
                "phase": self.phase, "op": "append", "sid": sid,
                "before": before, "delta": length, "after": after,
                "new_para": new_para,
            })
            return out

        def swap(sid, pid):
            before = kv.seqlen(sid)
            i = kv.para_id[sid].index(pid)
            active_len = kv.paragraph_lens(sid)[i]
            alt_len = sum(kv.alt[sid][pid][1])
            out = self._orig["swap"](sid, pid)
            after = kv.seqlen(sid)
            assert after == before - active_len + alt_len
            self._layout(sid)
            self.events.append({
                "phase": self.phase, "op": "swap", "sid": sid, "pid": pid,
                "before": before, "active_len": active_len, "alt_len": alt_len,
                "delta": alt_len - active_len, "after": after,
            })
            return out

        def pop_paragraph(sid):
            before = kv.seqlen(sid)
            last_len = kv.paragraph_lens(sid)[-1]
            out = self._orig["pop_paragraph"](sid)
            after = kv.seqlen(sid)
            assert out == last_len
            assert after == before - out
            self._layout(sid)
            self.events.append({
                "phase": self.phase, "op": "pop", "sid": sid,
                "before": before, "drop": out, "after": after,
            })
            return out

        def build_tables(sids, prefill):
            out = self._orig["build_tables"](sids, prefill)
            n, total_pages, total_q = out
            pages = sum((kv.pages[s] for s in sids), [])
            masks = sum((kv.ntok[s] for s in sids), [])
            positions = []
            cu_pages = [0]
            for sid in sids:
                pos = 0
                for length in kv.ntok[sid]:
                    positions.append(pos)
                    pos += length
                cu_pages.append(cu_pages[-1] + len(kv.pages[sid]))
            qlens = [kv._req_q[s] for s in sids] if prefill else [1] * len(sids)
            qpos = [kv.seqlen(s) - q for s, q in zip(sids, qlens)]

            assert n == len(sids)
            assert total_pages == len(pages)
            assert total_q == sum(qlens)
            assert kv.page_table[:total_pages].tolist() == pages
            assert kv.mask_table[:total_pages].tolist() == masks
            assert kv.page_pos[:total_pages].tolist() == positions
            assert kv.cu_pages[:n + 1].tolist() == cu_pages
            assert kv.qpos[:n].tolist() == qpos
            if prefill:
                cu_q = [0]
                for q in qlens:
                    cu_q.append(cu_q[-1] + q)
                assert kv.cu_q[:n + 1].tolist() == cu_q
            for sid in sids:
                self._layout(sid)
            return out

        kv.plan_append = plan_append
        kv.swap = swap
        kv.pop_paragraph = pop_paragraph
        kv.build_tables = build_tables
        return self

    def __exit__(self, *args):
        for name, fn in self._orig.items():
            setattr(self.kv, name, fn)


async def _run(model):
    eng = Engine(0, model)
    eng.start()
    sids = ("fixed", "reference")
    audit = _KVLengthAudit(model)

    try:
        with audit:

            prompt = sum((_para(i) for i in range(4)), [])
            audit.phase = "initial"
            first = await _batch(eng, [_request(sid, prompt) for sid in sids])
            assert len({tuple(r.output) for r in first}) == 1, " initial outputs of both paths  greedy  output mismatch "
            audit.check_resting_state()

            tail = _tokens(20, 7)
            observation = _para(21)
            views = {r.id: prompt + r.output + tail + observation for r in first}


            audit.phase = "reference-flush"
            flush = [
                _request("reference", views["reference"], max_new_tokens=0)
            ]
            await _batch(eng, flush)
            assert eng.reside["reference"] == len(views["reference"])
            audit.check_resting_state()

            probe = _para(22)


            audit.phase = "probe"
            await _batch(eng, [
                _request("fixed", views["fixed"] + probe,
                         subs=[[1, []]], swap=[1],
                         ephem_at=len(views["fixed"]), swap_at=len(views["fixed"])),
                _request("reference", views["reference"] + probe,
                         subs=[[1, []]], ephem_at=len(views["reference"])),
            ])
            audit.check_resting_state()

            boundary = len(views["fixed"])
            folded = len(_para(1))
            fixed_swap = audit.one("probe", "swap", "fixed")
            fixed_pop = audit.one("probe", "pop", "fixed")
            ref_pop = audit.one("probe", "pop", "reference")
            assert (fixed_swap["before"], fixed_swap["active_len"],
                    fixed_swap["alt_len"], fixed_swap["after"]) == (
                        boundary, folded, 0, boundary - folded)
            assert (fixed_pop["before"], fixed_pop["drop"], fixed_pop["after"]) == (
                boundary, len(probe), boundary - folded)
            assert (ref_pop["before"], ref_pop["drop"], ref_pop["after"]) == (
                boundary + len(probe), len(probe), boundary)

            assert eng.reside["fixed"] == boundary
            assert eng.reside["reference"] == boundary

            assert model.get_kv_cache().seqlen("fixed") == boundary - folded
            assert model.get_kv_cache().seqlen("reference") == boundary


            guide = _para(23)
            final = [
                _request("fixed", views["fixed"] + guide),
                _request("reference", views["reference"] + guide,
                         swap=[1], swap_at=len(views["reference"])),
            ]
            audit.phase = "formal-fold"
            with _LogitTrace() as trace:
                await _batch(eng, final)
            audit.check_resting_state()

            formal_swap = audit.one("formal-fold", "swap", "reference")
            assert (formal_swap["before"], formal_swap["active_len"],
                    formal_swap["alt_len"], formal_swap["after"]) == (
                        boundary, folded, 0, boundary - folded)

            meta = {sid: _meta(model, sid) for sid in sids}
            assert meta["fixed"] == meta["reference"], (
                " final states of both paths  paragraph/swap  topology must match ", meta)

            fixed_logits, fixed_tok = trace.rows["fixed"]
            ref_logits, ref_tok = trace.rows["reference"]
            fixed_ref_logit_diff = (fixed_logits - ref_logits).abs().max().item()

            ref_exact, ref_state_diff = _state_diff(model, "fixed", "reference")

            print("final metadata:", meta["fixed"])
            print(
                "fixed vs reference:",
                f"tokens={fixed_tok}/{ref_tok}",
                f"max|Δlogit|={fixed_ref_logit_diff:.3e}",
                f"state_exact={ref_exact}",
                f"max|Δstate|={ref_state_diff:.3e}",
            )
            assert fixed_tok == ref_tok
            assert fixed_ref_logit_diff == 0.0
            assert ref_exact and ref_state_diff == 0.0


            final_by_sid = {r.id: r for r in final}
            unfold_prefix = (
                views["fixed"] + guide + final_by_sid["fixed"].output
                + _tokens(24, 7) + _para(25))
            unfold_at = len(unfold_prefix)
            unfold_guide = _para(26)
            audit.phase = "formal-unfold"
            unfolded = await _batch(eng, [
                _request("fixed", unfold_prefix + unfold_guide,
                         swap=[1], swap_at=unfold_at),
            ])
            audit.check_resting_state()

            unfold_swap = audit.one("formal-unfold", "swap", "fixed")

            assert (unfold_swap["before"], unfold_swap["active_len"],
                    unfold_swap["alt_len"], unfold_swap["after"]) == (
                        unfold_at - folded, 0, folded, unfold_at)
            assert eng.reside["fixed"] == len(unfold_prefix + unfold_guide)
            assert model.get_kv_cache().seqlen("fixed") == len(unfold_prefix + unfold_guide)


            refold_prefix = (
                unfold_prefix + unfold_guide + unfolded[0].output
                + _tokens(27, 7) + _para(28))
            refold_at = len(refold_prefix)
            refold_probe = _para(29)
            audit.phase = "probe-refold"
            await _batch(eng, [
                _request("fixed", refold_prefix + refold_probe,
                         swap=[1], ephem_at=refold_at, swap_at=refold_at),
            ])
            audit.check_resting_state()

            refold_swap = audit.one("probe-refold", "swap", "fixed")
            refold_pop = audit.one("probe-refold", "pop", "fixed")
            assert (refold_swap["before"], refold_swap["active_len"],
                    refold_swap["alt_len"], refold_swap["after"]) == (
                        refold_at, folded, 0, refold_at - folded)
            assert (refold_pop["before"], refold_pop["drop"], refold_pop["after"]) == (
                refold_at - folded + len(refold_probe),
                len(refold_probe), refold_at - folded)
            assert eng.reside["fixed"] == refold_at
            assert model.get_kv_cache().seqlen("fixed") == refold_at - folded


            delayed_probe = _para(30)
            delayed_swap_at = refold_at + 8
            audit.phase = "ephemeral-then-unfold"
            await _batch(eng, [
                _request("fixed", refold_prefix + delayed_probe,
                         swap=[1], ephem_at=refold_at,
                         swap_at=delayed_swap_at),
            ])
            audit.check_resting_state()

            delayed_swap = audit.one("ephemeral-then-unfold", "swap", "fixed")
            delayed_pop = audit.one("ephemeral-then-unfold", "pop", "fixed")
            assert (delayed_swap["before"], delayed_swap["active_len"],
                    delayed_swap["alt_len"], delayed_swap["after"]) == (
                        delayed_swap_at - folded, 0, folded, delayed_swap_at)
            assert (delayed_pop["before"], delayed_pop["drop"], delayed_pop["after"]) == (
                refold_at + len(delayed_probe), len(delayed_probe), refold_at)
            assert eng.reside["fixed"] == refold_at
            assert model.get_kv_cache().seqlen("fixed") == refold_at

            print("KV length audit:")
            for event in (fixed_swap, fixed_pop, ref_pop, formal_swap,
                          unfold_swap, refold_swap, refold_pop,
                          delayed_swap, delayed_pop):
                fields = " ".join(
                    f"{k}={v}" for k, v in event.items()
                    if k not in {"phase", "op", "sid"})
                print(f"  {event['phase']} {event['sid']} {event['op']}: {fields}")
            print("Search observation / swap timing + KV length invariants: OK")
    finally:
        for sid in sids:
            if sid in eng.reside:
                eng.release(sid)
        await asyncio.sleep(0)
        await eng.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model_type = hf.config.model_type.lower()
    if "qwen3_5" in model_type:
        from rollout.monkey_patch.qwen35 import QwenModel
    elif "qwen3" in model_type:
        from rollout.monkey_patch.qwen3 import QwenModel
    else:
        raise ValueError(f"unsupported model_type={model_type!r}")
    model = QwenModel(
        hf,
        max_token=PAGE_SIZE * 64,
        max_reside=3,
        device=args.device,
    )
    asyncio.run(_run(model))


if __name__ == "__main__":
    main()
