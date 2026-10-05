import argparse

import torch
from transformers import AutoModelForCausalLM

from rollout.monkey_patch.qwen35 import QwenModel
from rollout.constant import PAGE_SIZE


class FakeReq:
    temperature, top_p, top_k = 0.0, 1.0, 0
    subs, swap = (), ()

    def __init__(self, rid, prompt_ids, split_ids=None):
        self.id, self.prompt_ids, self.split_ids, self.output = rid, prompt_ids, split_ids or [], []

    @property
    def last_token(self):
        return self.output[-1]


def hf_greedy(m, prompt, new, dev):
    ids = torch.tensor([prompt], device=dev)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=new, do_sample=False, use_cache=True,
                         attention_mask=torch.ones_like(ids))
    return out[0, len(prompt):].tolist()


def run_seq(model, req, new):
    model.get_kv_cache().alloc(req.id)
    for c in model.get_lin_caches():
        c.alloc(req.id)
    req.output.append(model.prefill([req])[0])
    for _ in range(new - 1):
        req.output.append(model.decode([req])[0])
    for c in model.get_lin_caches():
        c.persist(req.id)
    return req.output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-len", type=int, default=45)
    p.add_argument("--new", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--graph", action="store_true")
    p.add_argument("--split", action="store_true", help=" Force a boundary in  prompt  start a new  paragraph（ page aligned ）， output must remain unchanged ")
    args = p.parse_args()
    dev = args.device
    torch.manual_seed(0)
    p1 = torch.randint(0, 10000, (args.prompt_len,)).tolist()
    pc = torch.randint(0, 10000, (args.prompt_len // 2,)).tolist()

    ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    g1 = hf_greedy(ref, p1, args.new, dev)
    g2 = hf_greedy(ref, p1 + g1 + pc, args.new, dev)
    del ref
    torch.cuda.empty_cache()

    our = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = QwenModel(our, max_token=PAGE_SIZE * 256, max_reside=4, device=dev)
    if args.graph:
        model.build_graph(warmup=3)

    split = [p1[args.prompt_len // 3]] if args.split else []
    A = FakeReq("s", p1, split_ids=split)
    o1 = run_seq(model, A, args.new)
    C = FakeReq("s", [o1[-1]] + pc)
    o2 = run_seq_cont(model, C, args.new)

    ok1, ok2 = o1 == g1, o2 == g2
    path = ("graph" if args.graph else "eager") + (" +split" if args.split else "")
    print(f"[{path}]  turn1: {sum(x==y for x,y in zip(o1,g1))}/{args.new} {'OK' if ok1 else 'MISMATCH'}"
          f"   turn2(cont): {sum(x==y for x,y in zip(o2,g2))}/{args.new} {'OK' if ok2 else 'MISMATCH'}")
    if not ok1:
        print(" ref1", g1, "\n our1", o1)
    if not ok2:
        print(" ref2", g2, "\n our2", o2)
    assert ok1 and ok2, "greedy  not per  token  aligned  HF"
    print("OK")


def run_seq_cont(model, req, new):

    req.output.append(model.prefill([req])[0])
    for _ in range(new - 1):
        req.output.append(model.decode([req])[0])
    for c in model.get_lin_caches():
        c.persist(req.id)
    return req.output


if __name__ == "__main__":
    main()
