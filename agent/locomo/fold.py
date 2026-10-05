import re

from agent.data import Para
from agent.probe import alphabetic_id, parse_ids


def pick_unfold(text, uid2pid):

    return [uid2pid[u] for u in parse_ids(text) if u in uid2pid]


_SPECIAL_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")


def sanitize(text):

    return _SPECIAL_RE.sub(lambda m: m.group(0).replace("<|", "<").replace("|>", ">"), text)


class Ledger:


    def __init__(self, prompt_ids, im_start_id):
        assert prompt_ids and prompt_ids[0] == im_start_id
        self.base = len(prompt_ids)
        self.tokens = list(prompt_ids)
        self.view = list(prompt_ids)
        self.loss_mask = [0] * len(prompt_ids)
        self.logprobs = [0.0] * len(prompt_ids)
        self.paras = [Para(start=s, kind="prompt", pid=n)
                      for n, s in enumerate(i for i, t in enumerate(prompt_ids) if t == im_start_id)]
        self.probes = []
        self.uid2pid = {}
        self.size = {}

    def add_para(self, kind, ids, mask, logp, to_view=True, as_para=True):


        para = Para(start=len(self.tokens), kind=kind)
        self.tokens.extend(ids)
        self.loss_mask.extend(mask)
        self.logprobs.extend(logp)
        if to_view:
            self.view.extend(ids)
        if not as_para:
            self.probes.append(para)
            return para

        para.pid = len(self.paras)
        uid = alphabetic_id(len(self.uid2pid))
        para.uid = uid
        self.uid2pid[uid] = para.pid
        self.size[para.pid] = len(ids)
        self.paras.append(para)
        return para

    def apply(self, swap):
        for p in swap:
            self.paras[p].folded = not self.paras[p].folded

    def context_tokens(self):

        return self.base + sum(
            self.size[p.pid] for p in self.paras if p.kind != "prompt" and not p.folded)

    def full_tokens(self):

        return self.base + sum(self.size[p.pid] for p in self.paras if p.kind != "prompt")

    def merge(self):
        return sorted(self.paras + self.probes, key=lambda p: p.start)


def window_fold(ledger, keep_last_k):

    keep = {p.pid for p in ledger.paras[-keep_last_k:]} if keep_last_k > 0 else set()
    return [p.pid for p in ledger.paras[:-1] if p.kind != "prompt" and p.pid not in keep]


def sanity_check(tokens, paras, table):

    ends = [p.start for p in paras[1:]] + [len(tokens)]
    for i, (p, b) in enumerate(zip(paras, ends)):
        if p.kind == "prompt":
            continue
        pre, suf = table[p.kind]
        seg = tokens[p.start:b]
        assert seg[:len(pre)] == pre, f" paragraph  {i}({p.kind})  prefix mismatch "
        if i == len(paras) - 1:
            continue
        assert seg[len(seg) - len(suf):] == suf, f" paragraph  {i}({p.kind})  suffix mismatch "
