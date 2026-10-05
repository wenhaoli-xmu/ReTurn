import argparse
import re
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM

from rollout.monkey_patch.qwen35 import QwenRunner
from rollout.constant import PAGE_SIZE


class FakeReq:
    def __init__(self, rid, prompt_ids):
        self.id = rid
        self.prompt_ids = prompt_ids
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 0
        self.output = []

    @property
    def last_token(self):
        return self.output[-1]


BUCKETS = [
    ("linear_attn(GDN)", ["gated_delta", "delta_rule", "chunk_", "recurrent", "causal_conv", "conv1d", "l2norm", "softplus"]),
    ("decode_attn",       ["decode", "split", "combine", "flash", "para_page"]),
    ("gemm/matmul",       ["gemm", "cutlass", "matmul", "cublas", "cublaslt", "nvjet", "jet_", "ampere", "sm80", "sm90", "wgmma"]),
    ("rmsnorm/ln",        ["norm", "rms", "layer_norm"]),
    ("rope",              ["rope", "rotary"]),
    ("elementwise",       ["elementwise", "vectorized", "mul", "add", "sigmoid", "silu", "copy", "cat", "index", "fill", "arange"]),
    ("softmax/sample",    ["softmax", "sort", "topk", "multinomial", "cumsum", "gather"]),
    ("reduce",            ["reduce", "sum", "mean"]),
]


def bucket_of(name):
    low = name.lower()
    for label, keys in BUCKETS:
        if any(k in low for k in keys):
            return label
    return "other"


def profile_bs(runner, V, bs, args, dev):
    reqs = [FakeReq(i, torch.randint(0, V, (args.prompt_len,)).tolist()) for i in range(bs)]
    for r in reqs:
        runner.add(r)
    for r, t in zip(reqs, runner.prefill(reqs)):
        r.output.append(t)
    runner.reseed(reqs)
    for _ in range(args.warmup):
        for r, t in zip(reqs, runner.graph_decode(reqs)):
            r.output.append(t)


    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for r, t in zip(reqs, runner.graph_decode(reqs)):
            r.output.append(t)
    torch.cuda.synchronize(dev)
    wall = time.perf_counter() - t0
    ms_step = wall / args.steps * 1e3
    tps = bs * args.steps / wall

    if getattr(args, "no_prof", False):
        print(f"\n bs={bs}  ctx={args.prompt_len}   wall {ms_step:.3f} ms/step   {tps:.1f} tok/s", flush=True)
        for r in reqs:
            runner.free(r)
        return bs, ms_step, tps, 0.0, {}


    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(args.steps):
            for r, t in zip(reqs, runner.graph_decode(reqs)):
                r.output.append(t)
        torch.cuda.synchronize(dev)

    bucket = defaultdict(float)
    bucket_n = defaultdict(int)
    per_kernel = defaultdict(float)
    for e in prof.key_averages():
        cuda_us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        if cuda_us <= 0:
            continue
        b = bucket_of(e.key)
        bucket[b] += cuda_us
        bucket_n[b] += e.count
        per_kernel[e.key] += cuda_us

    total = sum(bucket.values())
    gpu_ms_step = total / 1e3 / args.steps
    print(f"\n{'='*60}\n bs={bs}  ctx={args.prompt_len}   "
          f"wall {ms_step:.3f} ms/step   {tps:.1f} tok/s   GPU busy {gpu_ms_step:.3f} ms/step "
          f"( utilization  {gpu_ms_step/ms_step*100:.0f}%)\n{'='*60}")
    print(f"{'bucket':<18}{'ms/step':>10}{'%GPU':>8}{'kern/step':>11}")
    print("-" * 47)
    for b, us in sorted(bucket.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{b:<18}{us/1e3/args.steps:>10.3f}{us/total*100:>7.1f}%{bucket_n[b]/args.steps:>11.1f}")
    print("--- top 12 kernel (self device time) ---")
    for name, us in sorted(per_kernel.items(), key=lambda kv: kv[1], reverse=True)[:12]:
        short = re.sub(r"<.*?>", "<..>", name)[:64]
        print(f"{us/1e3/args.steps:>8.3f} ms  {us/total*100:>5.1f}%  {short}")

    for r in reqs:
        runner.free(r)
    return bs, ms_step, tps, gpu_ms_step, dict(bucket)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--batch", default="1,2,4,8", help=" Comma-separated  batch size  list ")
    p.add_argument("--prompt-len", type=int, default=16384)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=0, help="KV  pool  token  capacity ；0= by  max(batch)*(ctx+64)  automatic ")
    p.add_argument("--no-prof", action="store_true", help=" Skip  kineto  bucketing ， measure only  tok/s（sweep  use ）")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(0)
    dev = args.device
    batches = [int(x) for x in args.batch.split(",")]


    mt = args.max_tokens or max(batches) * (args.prompt_len + args.warmup + args.steps + 8 + 2 * PAGE_SIZE)
    print(f"[load] {args.model}  (max_tokens={mt})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    runner = QwenRunner(model, max_tokens=mt, device=dev, max_batch=max(batches))
    runner.capture_all()
    V = runner.vocab_size

    results = []
    for bs in batches:
        print(f"\n[run] bs={bs} len={args.prompt_len} ...", flush=True)
        results.append(profile_bs(runner, V, bs, args, dev))

    print(f"\n\n##########  Summary  (ctx={args.prompt_len}) ##########")
    print(f"{'bs':>4}{'wall ms':>10}{'tok/s':>10}{'GPU ms':>10}{'util':>7}")
    for bs, ms, tps, gms, _ in results:
        print(f"{bs:>4}{ms:>10.2f}{tps:>10.1f}{gms:>10.2f}{gms/ms*100:>6.0f}%")


if __name__ == "__main__":
    main()
