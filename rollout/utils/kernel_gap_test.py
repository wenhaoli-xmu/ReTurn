import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("FLA_DISABLE_TENSOR_CACHE", "1")

import argparse
import asyncio
import json
import math

MODEL = os.environ.get("MODEL", "/mnt/rl-train/wenhaoli/gdrive/model/Qwen3.5-4B")
DATA = os.environ.get(
    "DATA", "/mnt/rl-train/wenhaoli/gdrive/data/search_train/train_reformulate.jsonl")


def load_prompts(n):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    with open(DATA) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    out = []
    for r in rows:
        text = tok.apply_chat_template(
            r["prompt"], tools=r.get("tools"),
            add_generation_prompt=True, tokenize=False)
        out.append(tok(text, add_special_tokens=False)["input_ids"])
    return out


def phase_rollout(args):
    import torch
    from transformers import AutoModelForCausalLM
    from rollout.engine import Engine, Request
    from rollout.monkey_patch.qwen35 import QwenModel

    prompts = load_prompts(args.num_prompts)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    hf.tie_weights()
    n_req = args.num_prompts * 2
    model = QwenModel(hf, max_token=args.max_token, max_reside=n_req, device="cuda:0")
    model.build_graph()

    async def main():
        engine = Engine(engine_id=0, model=model)
        engine.start()
        reqs = []
        for i, p in enumerate(prompts):
            for temp in (1.0, 0.0):
                reqs.append(Request(
                    id=f"req-{i}-t{temp}", prompt_ids=list(p),
                    max_new_tokens=args.max_new, temperature=temp))
        done = await asyncio.gather(*[engine.submit(r) for r in reqs])
        await engine.stop()
        return done

    done = asyncio.run(main())
    rows = []
    for r in done:
        rows.append({
            "id": r.id,
            "temperature": 0.0 if r.id.endswith("t0.0") else 1.0,
            "prompt_ids": r._ctx,
            "output_ids": r.output,
            "rollout_logprobs": r.output_logprobs,
        })
    with open(args.out, "w") as f:
        json.dump(rows, f)
    print(f"[rollout] wrote {len(rows)} trajectories -> {args.out}", flush=True)


def phase_recompute(args):
    import torch
    from transformers import AutoModelForCausalLM
    from opd.forward import build_rope, body_forward
    from opd.utils import training_linear_fallback

    training_linear_fallback()
    with open(args.out) as f:
        rows = json.load(f)
    dev = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    body = getattr(model, model.base_model_prefix)
    head = model.get_output_embeddings()
    nkv = model.config.get_text_config().num_key_value_heads
    max_pos = max(len(r["prompt_ids"]) + len(r["output_ids"]) for r in rows) + 8
    cos, sin = build_rope(model, max_pos, dev)

    results = []
    with torch.no_grad():
        for r in rows:
            ids = r["prompt_ids"] + r["output_ids"]
            plen = len(r["prompt_ids"])
            tokens = torch.tensor(ids, device=dev)[None]
            hidden, _, _ = body_forward(
                body, tokens, None, None, cos, sin, nkv)

            logp, amax = [], []
            for s in range(plen - 1, len(ids) - 1, 2048):
                e = min(s + 2048, len(ids) - 1)
                logits = head(hidden[s:e]).float()
                lsm = logits.log_softmax(-1)
                tgt = torch.tensor(ids[s + 1:e + 1], device=dev)
                logp += lsm.gather(1, tgt[:, None])[:, 0].tolist()
                amax += logits.argmax(-1).tolist()
            results.append({**{k: r[k] for k in ("id", "temperature")},
                            "rollout_logprobs": r["rollout_logprobs"],
                            "recompute_logprobs": logp,
                            "recompute_argmax": amax,
                            "output_ids": r["output_ids"]})
    with open(args.report, "w") as f:
        json.dump(results, f)
    print(f"[recompute] wrote -> {args.report}", flush=True)
    analyze(results)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def analyze(results):
    print("\n================  Analysis  ================")
    deltas_all, pos_all = [], []
    for r in results:
        if r["temperature"] != 1.0:
            continue
        for j, (a, b) in enumerate(zip(r["recompute_logprobs"], r["rollout_logprobs"])):
            deltas_all.append(a - b)
            pos_all.append(j)
    n = len(deltas_all)
    absd = [abs(d) for d in deltas_all]
    ratios = [math.exp(d) for d in deltas_all]
    kl_mc = -sum(deltas_all) / n
    clip_lo = sum(1 for x in ratios if x < 0.8) / n
    clip_hi = sum(1 for x in ratios if x > 1.28) / n
    print(f"[temp=1.0  sampled  token, n={n}]")
    print(f"  |Δlogp|  mean={sum(absd)/n:.4g}  p50={pct(absd,.5):.4g}  "
          f"p90={pct(absd,.9):.4g}  p99={pct(absd,.99):.4g}  max={max(absd):.4g}")
    print(f"  Δlogp bias(mean)={sum(deltas_all)/n:+.4g}   KL(rollout||recompute) MC≈{kl_mc:+.4g}")
    print(f"  ratio  p1={pct(ratios,.01):.4f}  p50={pct(ratios,.5):.4f}  p99={pct(ratios,.99):.4f}")
    print(f"  step-0  triggers immediately  PPO clip  of  token: ratio<0.8: {clip_lo:.3%}   ratio>1.28: {clip_hi:.3%}")

    print("\n   by  output  bucket by position （ observe amplified numerical differences ）")
    print(f"  {'pos bucket':>14} {'n':>7} {'mean|Δ|':>10} {'p99|Δ|':>10} {'clip%':>8}")
    B = 128
    nb = (max(pos_all) // B) + 1
    for b in range(nb):
        d = [deltas_all[i] for i in range(n) if pos_all[i] // B == b]
        if not d:
            continue
        ad = [abs(x) for x in d]
        cl = sum(1 for x in d if math.exp(x) < 0.8 or math.exp(x) > 1.28) / len(d)
        print(f"  [{b*B:>5},{(b+1)*B:>5}) {len(d):>7} {sum(ad)/len(ad):>10.4g} "
              f"{pct(ad,.99):>10.4g} {cl:>7.2%}")

    print("\n[greedy  trajectory : decode  sampled  token vs  recomputed  argmax  argmax changed ]")
    for r in results:
        if r["temperature"] != 0.0:
            continue
        flips = [j for j, (t, a) in enumerate(zip(r["output_ids"], r["recompute_argmax"]))
                 if t != a]
        L = len(r["output_ids"])
        first = flips[0] if flips else None
        print(f"  {r['id']}: len={L} flips={len(flips)} ({len(flips)/L:.2%}) first_flip@{first}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True, choices=["rollout", "recompute", "analyze"])
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--max-token", type=int, default=65536)
    p.add_argument("--out", default="traj.json")
    p.add_argument("--report", default="report.json")
    args = p.parse_args()
    if args.phase == "rollout":
        phase_rollout(args)
    elif args.phase == "recompute":
        phase_recompute(args)
    else:
        with open(args.report) as f:
            analyze(json.load(f))


if __name__ == "__main__":
    main()
