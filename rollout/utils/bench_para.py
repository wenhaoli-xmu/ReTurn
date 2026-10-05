import argparse
import time

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


def bench(model, V, M, ctx, bs, steps, warmup, dev):
    caches = model.get_kv_cache() + model.get_lin_cache()
    L = ctx // M
    reqs = [FakeReq(("para", i)) for i in range(bs)]
    for r in reqs:
        for c in caches:
            c.alloc(r.id)
    outs = None
    for _ in range(M):
        for r in reqs:
            r.prompt_ids = torch.randint(0, V, (L,)).tolist()
        outs = model.prefill(reqs)
    for r, t in zip(reqs, outs):
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
    ms, tps = wall / steps * 1e3, bs * steps / wall
    for r in reqs:
        for c in caches:
            c.free(r.id)
    torch.cuda.empty_cache()
    print(f" M={M:>4} paras (L={L}/para, ctx≈{M*L})  bs={bs}   {ms:8.3f} ms/step   {tps:9.1f} tok/s", flush=True)
    return M, ms, tps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ctx", type=int, default=65536)
    p.add_argument("--paras", default="1,2,4,8,16,32")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(0)
    dev = args.device
    Ms = [int(x) for x in args.paras.split(",")]
    mp = max(Ms) + 1

    pages = max(M * ((args.ctx // M + PAGE_SIZE - 1) // PAGE_SIZE) for M in Ms)
    mt = args.batch * (pages * PAGE_SIZE + args.warmup + args.steps + 8 + 2 * PAGE_SIZE)
    print(f"[load] {args.model}  ctx={args.ctx} paras={Ms} bs={args.batch} (max_paragraph={mp}, max_tokens={mt})", flush=True)
    model_hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = QwenModel(model_hf, max_token=mt, max_paragraph=mp, max_reside=args.batch, device=dev)
    model.build_graph()
    V = model.vocab_size
    res = [bench(model, V, M, args.ctx, args.batch, args.steps, args.warmup, dev) for M in Ms]
    print(f"\n##### paragraph-overhead summary  ctx={args.ctx} bs={args.batch} #####")
    print("M," + ",".join(str(m) for m, _, _ in res))
    print("ms/step," + ",".join(f"{ms:.3f}" for _, ms, _ in res))
    print("tok/s," + ",".join(f"{t:.0f}" for _, _, t in res))


if __name__ == "__main__":
    main()
