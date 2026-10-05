import argparse
import asyncio
import os
import time
import uuid

import httpx
from transformers import AutoTokenizer


TRACE_DIR = "trace"


PROMPT = (
    "Please provide a detailed and rigorous derivation of the Theory of Relativity. "
    "Begin with the historical and experimental motivations—in particular the "
    "Michelson–Morley experiment and the failure of the luminiferous aether "
    "hypothesis. Then state Einstein's two postulates of special relativity: the "
    "principle of relativity and the constancy of the speed of light. From these "
    "postulates, derive the Lorentz transformations step by step, and use them to "
    "explain time dilation, length contraction, the relativity of simultaneity, and "
    "the relativistic addition of velocities. Continue with the derivation of "
    "relativistic momentum and energy, culminating in the mass–energy equivalence "
    "relation E = mc^2. Finally, extend the discussion to general relativity: explain "
    "the equivalence principle, the curvature of spacetime, the Einstein field "
    "equations, and their experimental confirmations such as the perihelion "
    "precession of Mercury, gravitational lensing, and gravitational time dilation. "
    "Please be as thorough, precise, and mathematically explicit as possible, showing "
    "every intermediate step."
)


async def one_request(client, base_url, idx, prompt_ids, stop_ids, max_new_tokens, tok):

    session_id = str(uuid.uuid4())
    payload = {
        "session_id": session_id,
        "prompt_ids": prompt_ids,
        "stop_ids": stop_ids,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{base_url}/generate", json=payload, timeout=3600)
        resp.raise_for_status()
        data = resp.json()
    finally:

        try:
            await client.post(f"{base_url}/release", json={"session_id": session_id}, timeout=60)
        except Exception:
            pass
    dt = time.perf_counter() - t0
    n = data["num_tokens"]
    stopped = data.get("finish_reason") == "stop"
    with open(os.path.join(TRACE_DIR, f"{idx}.txt"), "w") as f:
        f.write(tok.decode(data["output_ids"]))
    print(f"[req {idx:>3}]  completed   tokens={n:<6}  elapsed ={dt:6.2f}s  {n/dt:6.1f} tok/s  "
          f"({' stop condition reached ' if stopped else ' length limit reached '})", flush=True)
    return {"idx": idx, "n": n, "dt": dt, "stopped": stopped}


async def main_async(args):

    tok = AutoTokenizer.from_pretrained(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [[eos_id]]

    os.makedirs(TRACE_DIR, exist_ok=True)

    print("===  Templated  prompt ===")
    print(text)
    print(f"(prompt len = {len(prompt_ids)} tokens, eos_id={eos_id})")
    print(f"===  Single-turn load test ： every  {args.interval}s  submitted  1  items ， total  {args.num}  items  ===\n", flush=True)


    tasks = []
    async with httpx.AsyncClient(http2=False) as client:
        wall0 = time.perf_counter()
        for i in range(args.num):
            tasks.append(asyncio.create_task(one_request(
                client, args.url, i, prompt_ids, stop_ids, args.max_new_tokens, tok)))
            print(f"[req {i:>3}]  submitted   (t+{time.perf_counter()-wall0:6.1f}s)", flush=True)
            if i < args.num - 1:
                await asyncio.sleep(args.interval)
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - wall0


    total_tok = sum(r["n"] for r in results)
    n_stop = sum(r["stopped"] for r in results)
    avg_tps = sum(r["n"] / r["dt"] for r in results) / len(results)
    print(f"\n===  Summary  ===")
    print(f" Request count         : {len(results)}   ( stop condition reached  {n_stop}， length limit reached  {len(results)-n_stop})")
    print(f" Total generated  token  : {total_tok}")
    print(f" Total wall time     : {wall:.1f} s")
    print(f" Mean throughput per request : {avg_tps:.1f} tok/s")
    print(f" Aggregate throughput       : {total_tok / wall:.1f} tok/s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="")
    p.add_argument("--model", default=os.path.expanduser(
        os.environ.get("GDRIVE_LOCAL", "/mnt/rl-train/wenhaoli/gdrive")) + "/model/Qwen3.5-4B")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=131072)
    p.add_argument("--num", type=int, default=100, help=" Number of consecutive requests ")
    p.add_argument("--interval", type=float, default=1.0, help=" Interval between requests （ seconds ）")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
