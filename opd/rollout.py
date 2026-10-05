import argparse
import asyncio
import json
import random
from dataclasses import asdict

from transformers import AutoTokenizer

from agent.data import Request
from agent.locomo.config import LOCOMO_CONFIGS, UNFOLD_CONFIGS
from agent.locomo.generate import generate as generate_long
from agent.search.config import UNFOLD_CONFIGS as SEARCH_UNFOLD_CONFIGS
from agent.search.generate import generate as generate_search
from opd.data import assistant_tokens, write_jsonl
from opd.longdata import load, prompt_ids


async def long_worker(doc, index, tok, url, temperature, top_p, top_k, sem):
    async with sem:
        request = Request(
            input_ids=prompt_ids(doc, tok),
            url=url,
            tokenizer=tok,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k)
        traj, _, stats = await generate_long(request, doc)
    row = asdict(traj)
    row.pop("text")
    row.pop("judge")
    return {**row, "group": doc.doc_id, "sample": index, **stats}


async def search_worker(item, index, tok, url, temperature, top_p, top_k, sem):
    doc_id, doc = item
    async with sem:
        text = tok.apply_chat_template(
            doc["prompt"], tools=doc["tools"], add_generation_prompt=True,
            tokenize=False)
        request = Request(
            input_ids=tok(text, add_special_tokens=False)["input_ids"],
            url=url,
            tokenizer=tok,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k)
        traj = await generate_search(request)
    row = asdict(traj)
    row.pop("text")
    row.pop("judge")


    if not assistant_tokens(row):
        return None
    return {**row, "group": f"search-{doc_id}", "sample": index}


def sample_docs(docs, num_prompts, offset, seed):
    if not docs:
        raise ValueError("training data is empty")
    picked, size = [], len(docs)
    while len(picked) < num_prompts:
        epoch, position = divmod(offset, size)
        order = list(range(size))
        random.Random(f"opd-sample:{seed}:epoch:{epoch}").shuffle(order)
        take = min(num_prompts - len(picked), size - position)
        picked.extend(docs[i] for i in order[position:position + take])
        offset += take
    return picked


async def rollout(model, data, output, url, num_prompts=16, samples_per_prompt=4,
                  concurrency=64, temperature=1.0, top_p=1.0, top_k=0,
                  offset=0, sampling_seed=20260822, keep_last_k=3,
                  max_answer_tokens=128):


    UNFOLD_CONFIGS["enable"] = True
    UNFOLD_CONFIGS["keep_last_k"] = keep_last_k
    UNFOLD_CONFIGS["selection_mode"] = "model"
    UNFOLD_CONFIGS["bm25_union_top_k"] = 0
    LOCOMO_CONFIGS["max_answer_tokens"] = max_answer_tokens

    tok = AutoTokenizer.from_pretrained(model)
    is_search = str(data).endswith(".jsonl")
    if is_search:
        with open(data) as f:
            docs = [(i, json.loads(line)) for i, line in enumerate(f) if line.strip()]
        SEARCH_UNFOLD_CONFIGS["enable"] = True
        SEARCH_UNFOLD_CONFIGS["keep_last_k"] = keep_last_k
        SEARCH_UNFOLD_CONFIGS["selection_mode"] = "model"
        SEARCH_UNFOLD_CONFIGS["bm25_union_top_k"] = 0
        worker = search_worker
    else:
        docs = load(data)
        worker = long_worker
    picked = sample_docs(docs, num_prompts, offset, sampling_seed)


    sem = asyncio.Semaphore(concurrency)
    rows = await asyncio.gather(*[
        worker(doc, i, tok, url, temperature, top_p, top_k, sem)
        for doc in picked for i in range(samples_per_prompt)])
    if is_search:
        skipped = sum(row is None for row in rows)
        rows = [row for row in rows if row is not None]
        if skipped:
            print(f"search rollout skipped {skipped} trajectories without assistant sampled tokens")
        if not rows:
            raise RuntimeError("search rollout produced no trajectories with assistant sampled tokens")

    write_jsonl(output, rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--url', required=True)
    p.add_argument('--num-prompts', type=int, default=16)
    p.add_argument('--samples-per-prompt', type=int, default=4)
    p.add_argument('--concurrency', type=int, default=64)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=0)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--sampling-seed', type=int, default=20260822)
    p.add_argument('--keep-last-k', type=int, default=3)
    p.add_argument('--max-answer-tokens', type=int, default=128)
    asyncio.run(rollout(**vars(p.parse_args())))


if __name__ == '__main__':
    main()
