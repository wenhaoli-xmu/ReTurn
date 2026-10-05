import argparse
import os

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from opd.data import (
    assistant_tokens_by_begin, max_pos, predictor_rows, read_jsonl, units,
    validate, write_jsonl)
from opd.forward import Runtime
from opd.utils import setup_distributed, training_linear_fallback


def targets(head, hidden, sampled, top_k, chunk_size):
    ids, logps, at_sampled = [], [], []
    for start in range(0, hidden.size(0), chunk_size):
        chunk = slice(start, start + chunk_size)
        logp = head(hidden[chunk]).float().log_softmax(-1)
        value, index = logp.topk(min(top_k, logp.size(-1)), dim=-1)
        ids.extend(index.cpu().tolist())
        logps.extend(value.cpu().tolist())
        at_sampled.extend(
            logp.gather(-1, sampled[chunk, None]).squeeze(-1).cpu().tolist())
    return ids, logps, at_sampled


@torch.inference_mode()
def annotate_row(row, run, top_k=10, chunk_size=4096):
    validate(row)
    target_by_begin = assistant_tokens_by_begin(row)
    n = len(row["tokens"])
    row["teacher_topk_ids"] = [[] for _ in range(n)]
    row["teacher_topk_logprobs"] = [[] for _ in range(n)]
    row["teacher_token_logprobs"] = [0.0] * n


    table = units(row, fold=False)
    ctx = gdn = None
    for unit in table:
        hidden, kv, gdn = run.forward(unit.ids, ctx, gdn)
        if tokens := target_by_begin.get(unit.begin):
            selected = hidden.index_select(0, torch.tensor(
                predictor_rows(tokens, unit.begin), device=run.device))
            sampled = torch.tensor(
                [row["tokens"][position] for position in tokens], device=run.device)
            ids, logps, at_sampled = targets(
                run.head, selected, sampled, top_k, chunk_size)
            for i, position in enumerate(tokens):
                row["teacher_topk_ids"][position] = ids[i]
                row["teacher_topk_logprobs"][position] = logps[i]
                row["teacher_token_logprobs"][position] = at_sampled[i]
        ctx = kv if ctx is None else [
            (torch.cat([old[0], new[0]], 0),
             torch.cat([old[1], new[1]], 0))
            for old, new in zip(ctx, kv)]
    return validate(row, teacher=True)


def annotate(model_path, input_path, output, top_k=10, chunk_size=4096):
    rank, world, device = setup_distributed()
    training_linear_fallback()
    rows = read_jsonl(input_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to(device).eval()
    run = Runtime.of(model, device, max_pos(rows))

    local = []
    for index, row in list(enumerate(rows))[rank::world]:
        row = annotate_row(row, run, top_k, chunk_size)
        row["_index"] = index
        local.append(row)

    if world == 1:
        for row in local:
            row.pop("_index")
        write_jsonl(output, local)
        return

    shard = f"{output}.rank-{rank}"
    write_jsonl(shard, local)
    dist.barrier()
    if rank == 0:
        merged = []
        for worker in range(world):
            merged.extend(read_jsonl(f"{output}.rank-{worker}"))
        merged.sort(key=lambda row: row.pop("_index"))
        write_jsonl(output, merged)
        for worker in range(world):
            os.remove(f"{output}.rank-{worker}")
    dist.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--input-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=4096)
    annotate(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
