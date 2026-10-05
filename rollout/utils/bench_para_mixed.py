import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from rollout.monkey_patch.qwen35 import QwenModel
from rollout.constant import PAGE_SIZE


class FakeReq:
    temperature, top_p, top_k = 0.0, 1.0, 0

    def __init__(self, rid):
        self.id, self.prompt_ids, self.output = rid, [], []

    @property
    def last_token(self):
        return self.output[-1]


def short_lens(n, lo, hi, seed=0):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(lo, hi + 1, n)]


def bench(model, V, n, long_len, long_paras, lo, hi, steps, warmup, dev):
    caches = model.get_kv_cache() + model.get_lin_cache()
    slens = short_lens(n, lo, hi)
    shorts = [FakeReq(("s", i)) for i in range(n)]
    longr = FakeReq(("long",))
    allr = shorts + [longr]
    for r in allr:
        for c in caches:
            c.alloc(r.id)

    for r, L in zip(shorts, slens):
        r.prompt_ids = torch.randint(0, V, (L,)).tolist()
    if shorts:
        for r, t in zip(shorts, model.prefill(shorts)):
            r.output.append(t)

    Lp = long_len // long_paras
    out = None
    for _ in range(long_paras):
        longr.prompt_ids = torch.randint(0, V, (Lp,)).tolist()
        out = model.prefill([longr])
    longr.output.append(out[0])

    bs = len(allr)
    for _ in range(warmup):
        for r, t in zip(allr, model.decode(allr)):
            r.output.append(t)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(steps):
        for r, t in zip(allr, model.decode(allr)):
            r.output.append(t)
    torch.cuda.synchronize(dev)
    wall = time.perf_counter() - t0
    ms, tps = wall / steps * 1e3, bs * steps / wall
    for r in allr:
        for c in caches:
            c.free(r.id)
    torch.cuda.empty_cache()
    print(f" N={n:>3} (bs={bs:>3}: {n}x short[{min(slens) if slens else 0}-{max(slens) if slens else 0}] + 1x{long_len}@{long_paras}p)"
          f"   {ms:8.3f} ms/step   {tps:9.1f} tok/s", flush=True)
    return n, bs, ms, tps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", default="1,2,4,8,16")
    p.add_argument("--long-len", type=int, default=131072)
    p.add_argument("--long-paras", type=int, default=32)
    p.add_argument("--short-lo", type=int, default=4096)
    p.add_argument("--short-hi", type=int, default=16384)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(0)
    dev = args.device
    ns = [int(x) for x in args.n.split(",")]
    mb = max(ns) + 1
    mp = args.long_paras + 1
    mt = max(ns) * args.short_hi + args.long_len + mb * (args.warmup + args.steps + 8 + args.long_paras + 2 * PAGE_SIZE)
    print(f"[load] {args.model}  n={ns} long={args.long_len}@{args.long_paras}p short~U[{args.short_lo},{args.short_hi}] "
          f"(max_paragraph={mp}, max_reside={mb}, max_tokens={mt})", flush=True)
    model_hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = QwenModel(model_hf, max_token=mt, max_paragraph=mp, max_reside=mb, device=dev)
    model.build_graph()
    V = model.vocab_size
    res = [bench(model, V, n, args.long_len, args.long_paras, args.short_lo, args.short_hi,
                 args.steps, args.warmup, dev) for n in ns]
    print(f"\n##### para-mixed summary  long={args.long_len}@{args.long_paras}p short~U[{args.short_lo},{args.short_hi}] #####")
    print("N," + ",".join(str(n) for n, _, _, _ in res))
    print("batch," + ",".join(str(b) for _, b, _, _ in res))
    print("tok/s," + ",".join(f"{t:.0f}" for _, _, _, t in res))


if __name__ == "__main__":
    main()
