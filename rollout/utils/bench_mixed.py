import argparse
import time

import torch
from transformers import AutoModelForCausalLM

from rollout.monkey_patch.qwen35 import QwenModel
from rollout.constant import PAGE_SIZE


class FakeReq:
    temperature, top_p, top_k = 0.0, 1.0, 0

    def __init__(self, rid, prompt_ids):
        self.id = rid
        self.prompt_ids = prompt_ids
        self.output = []

    @property
    def last_token(self):
        return self.output[-1]


def bench_group(model, V, n, short_len, long_len, steps, warmup, dev):
    caches = model.get_kv_cache() + model.get_lin_cache()
    lens = [short_len] * n + [long_len]
    reqs = [FakeReq(("mix", i), torch.randint(0, V, (L,)).tolist()) for i, L in enumerate(lens)]
    bs = len(reqs)
    for r in reqs:
        for c in caches:
            c.alloc(r.id)

    for r, t in zip(reqs, model.prefill(reqs)):
        r.output.append(t)
    for _ in range(warmup):
        for r, t in zip(reqs, model.decode(reqs)):
            r.output.append(t)

    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(steps):
        for r, t in zip(reqs, model.decode(reqs)):
            r.output.append(t)
    torch.cuda.synchronize(dev)
    wall = time.perf_counter() - t0
    ms_step = wall / steps * 1e3
    tps = bs * steps / wall

    for r in reqs:
        for c in caches:
            c.free(r.id)
    torch.cuda.empty_cache()
    print(f" N={n:>3} (bs={bs:>3}: {n}x{short_len}+1x{long_len})   {ms_step:8.3f} ms/step   {tps:9.1f} tok/s", flush=True)
    return n, bs, ms_step, tps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", default="1,2,4,8,16", help=" short  prompt  count list ； per group  batch=N+1")
    p.add_argument("--short-len", type=int, default=4096)
    p.add_argument("--long-len", type=int, default=131072)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(0)
    dev = args.device
    ns = [int(x) for x in args.n.split(",")]
    mb = max(ns) + 1
    mt = max(ns) * args.short_len + args.long_len + mb * (args.warmup + args.steps + 8 + 2 * PAGE_SIZE)
    print(f"[load] {args.model}  N={ns} short={args.short_len} long={args.long_len} (max_tokens={mt}, max_reside={mb})", flush=True)
    model_hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = QwenModel(model_hf, max_token=mt, max_paragraph=8, max_reside=mb, device=dev)
    model.build_graph()
    V = model.vocab_size

    results = [bench_group(model, V, n, args.short_len, args.long_len, args.steps, args.warmup, dev) for n in ns]

    print(f"\n##### summary  {args.short_len}x N + {args.long_len}x 1 #####")
    print("N," + ",".join(str(n) for n, _, _, _ in results))
    print("batch," + ",".join(str(bs) for _, bs, _, _ in results))
    print("tok/s," + ",".join(f"{tps:.0f}" for _, _, _, tps in results))


if __name__ == "__main__":
    main()
