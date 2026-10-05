import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from rollout.monkey_patch.qwen35 import QwenModel
from rollout.constant import PAGE_SIZE


class FakeReq:
    temperature, top_p, top_k = 0.0, 1.0, 0

    def __init__(self, rid, prompt_ids):
        self.id, self.prompt_ids, self.output = rid, prompt_ids, []

    @property
    def last_token(self):
        return self.output[-1]


def draw_lens(bs, hi, seed=0):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(1, hi + 1, bs)]


def bench_bs(model, V, bs, hi, steps, warmup, dev):
    caches = model.get_kv_cache() + model.get_lin_cache()
    lens = draw_lens(bs, hi)
    reqs = [FakeReq(("rag", i), torch.randint(0, V, (L,)).tolist()) for i, L in enumerate(lens)]
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
    tps = bs * steps / wall
    for r in reqs:
        for c in caches:
            c.free(r.id)
    torch.cuda.empty_cache()
    print(f" bs={bs:>3}  lens(min/mean/max)={min(lens)}/{int(np.mean(lens))}/{max(lens)}   "
          f"{wall/steps*1e3:8.3f} ms/step   {tps:9.1f} tok/s", flush=True)
    return bs, tps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--batch", default="1,2,4,8,16,32")
    p.add_argument("--hi", type=int, default=65536)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(0)
    dev = args.device
    batches = [int(x) for x in args.batch.split(",")]
    mb = max(batches)
    mt = sum(draw_lens(mb, args.hi)) + mb * (args.warmup + args.steps + 8 + 2 * PAGE_SIZE)
    print(f"[load] {args.model}  batches={batches} hi={args.hi} (max_tokens={mt})", flush=True)
    model_hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = QwenModel(model_hf, max_token=mt, max_paragraph=8, max_reside=mb, device=dev)
    model.build_graph()
    V = model.vocab_size
    res = [bench_bs(model, V, bs, args.hi, args.steps, args.warmup, dev) for bs in batches]
    print("\n##### ragged summary #####")
    print("bs," + ",".join(str(b) for b, _ in res))
    print("tok/s," + ",".join(f"{t:.0f}" for _, t in res))


if __name__ == "__main__":
    main()
