import argparse
import time

import torch
from transformers import AutoModelForCausalLM


def _no_mask_for_generate(*args, **kwargs):
    return None


def timed_generate(model, input_ids, n_new, device):


    if hasattr(model, "_cache"):
        del model._cache
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    model.generate(
        input_ids,
        do_sample=False,
        num_beams=1,
        min_new_tokens=n_new,
        max_new_tokens=n_new,
        cache_implementation="static",
        use_cache=True,
    )
    torch.cuda.synchronize(device)
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=16384)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--attn", default="flash_attention_2")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(0)
    dev = args.device

    print(f"[load] {args.model} attn={args.attn} -> {dev}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn,
    ).to(dev).eval()
    model.create_masks_for_generate = _no_mask_for_generate
    model.generation_config.pad_token_id = model.config.get_text_config().eos_token_id
    V = model.config.get_text_config().vocab_size

    input_ids = torch.randint(0, V, (args.batch, args.prompt_len), device=dev)


    print(f"[warmup] {args.warmup} steps", flush=True)
    for _ in range(args.warmup):
        timed_generate(model, input_ids, 2, dev)


    print("[measure] prefill (+1 decode)", flush=True)
    t1 = timed_generate(model, input_ids, 1, dev)
    print("[measure] prefill + %d decode" % (args.steps + 1), flush=True)
    tN = timed_generate(model, input_ids, args.steps + 1, dev)

    prefill_ms = t1 * 1e3
    decode_ms = (tN - t1) / args.steps * 1e3
    tps = 1e3 / decode_ms * args.batch

    print("\n=== HF baseline (static cache + %s) ===" % args.attn)
    print(f"batch={args.batch}  prompt_len={args.prompt_len}  steps={args.steps}")
    print(f"prefill(+1decode): {prefill_ms:.1f} ms")
    print(f"decode:            {decode_ms:.2f} ms/step   {tps:.1f} tok/s")


if __name__ == "__main__":
    main()
