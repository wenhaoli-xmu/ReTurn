import argparse
import time

import torch
from transformers import AutoModelForCausalLM

from rollout.prof import prof
from rollout.monkey_patch.qwen35 import QwenRunner


class FakeReq:

    def __init__(self, rid, prompt_ids):
        self.id = rid
        self.prompt_ids = prompt_ids
        self.temperature = 1.0
        self.top_p = 1.0
        self.top_k = 0
        self.output = []

    @property
    def last_token(self):
        return self.output[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(0)
    dev = args.device

    print(f"[load] {args.model} -> {dev}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    runner = QwenRunner(model, device=dev)
    V = runner.vocab_size

    reqs = [FakeReq(i, torch.randint(0, V, (args.prompt_len,)).tolist())
            for i in range(args.batch)]
    for r in reqs:
        runner.add(r)


    prof.enable(dev)
    prof.reset()
    toks = runner.prefill(reqs)
    for r, t in zip(reqs, toks):
        r.output.append(t)
    print("\n=== prefill  Summary  (batch=%d, prompt_len=%d) ===" % (args.batch, args.prompt_len))
    print(prof.summary())


    prof.disable()
    for _ in range(args.warmup):
        for r, t in zip(reqs, runner.decode(reqs)):
            r.output.append(t)


    prof.enable(dev)
    prof.reset()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for r, t in zip(reqs, runner.decode(reqs)):
            r.output.append(t)
    torch.cuda.synchronize(dev)
    wall = time.perf_counter() - t0

    n_tok = args.batch * args.steps
    print(f"\n=== decode  Summary  (batch={args.batch}, steps={args.steps}) ===")
    print(prof.summary())
    print(f"\nwall={wall*1e3:.1f}ms  {n_tok} tokens  "
          f"{n_tok/wall:.1f} tok/s  {wall/args.steps*1e3:.2f} ms/step")


if __name__ == "__main__":
    main()
