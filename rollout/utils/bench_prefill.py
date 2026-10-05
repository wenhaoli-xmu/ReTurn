import argparse
import time

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


def bench(model, V, ctx, iters, warmup, dev):
    caches = model.get_kv_cache() + model.get_lin_cache()
    def one():
        r = FakeReq(("pf", time.perf_counter_ns()), torch.randint(0, V, (ctx,)).tolist())
        for c in caches:
            c.alloc(r.id)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        model.prefill([r])
        torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        for c in caches:
            c.free(r.id)
        return dt
    for _ in range(warmup):
        one()
    ts = [one() for _ in range(iters)]
    torch.cuda.empty_cache()
    ms = sum(ts) / len(ts) * 1e3
    tps = ctx / (sum(ts) / len(ts))
    print(f" ctx={ctx:>6}   {ms:9.2f} ms   {tps:10.1f} tok/s", flush=True)
    return ctx, ms, tps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ctx", default="4096,16384,65536")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.manual_seed(0)
    dev = args.device
    ctxs = [int(x) for x in args.ctx.split(",")]
    mt = max(ctxs) + 4 * PAGE_SIZE
    print(f"[load] {args.model}  ctx={ctxs} (max_tokens={mt})", flush=True)
    model_hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = QwenModel(model_hf, max_token=mt, max_paragraph=8, max_reside=1, device=dev)
    model.build_graph()
    V = model.vocab_size
    res = [bench(model, V, c, args.iters, args.warmup, dev) for c in ctxs]
    print("\n##### prefill summary #####")
    print("ctx," + ",".join(str(c) for c, _, _ in res))
    print("tok/s," + ",".join(f"{t:.0f}" for _, _, t in res))


if __name__ == "__main__":
    main()
