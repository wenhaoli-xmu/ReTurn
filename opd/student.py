import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("FLA_DISABLE_TENSOR_CACHE", "1")

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from opd.data import (
    assistant_tokens, assistant_tokens_by_begin, max_pos, predictor_rows,
    read_jsonl, units, validate)
from opd.forward import Runtime
from opd.lora import inject_lora, load_lora, save_lora
from opd.loss import METRIC_KEYS, OPD, chunked_opd_loss
from opd.replay import replay
from opd.utils import (
    latest_full, latest_lora, setup_distributed, training_linear_fallback)


def load_student(model_path, lora_path, rank=8, alpha=16, full_finetune=1):
    training_linear_fallback()
    if full_finetune:

        checkpoint = latest_full(lora_path)
        return AutoModelForCausalLM.from_pretrained(
            checkpoint or model_path, dtype=torch.bfloat16).requires_grad_(True)
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    checkpoint = latest_lora(lora_path)
    if checkpoint:
        return load_lora(base, checkpoint, trainable=True)
    return inject_lora(base, rank, alpha)


class OffloadedAdamW:

    def __init__(self, parameters, lr, **kwargs):
        self.device_params = list(parameters)
        self.master = [
            parameter.detach().float().cpu().requires_grad_(True)
            for parameter in self.device_params]
        self.optimizer = torch.optim.AdamW(self.master, lr=lr, **kwargs)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        for parameter in self.device_params:
            parameter.grad = None

    def step(self):
        for master, parameter in zip(self.master, self.device_params):
            master.grad = (torch.zeros_like(master) if parameter.grad is None
                           else parameter.grad.detach().to("cpu", torch.float32))
        self.optimizer.step()
        with torch.no_grad():
            for master, parameter in zip(self.master, self.device_params):
                parameter.copy_(master.to(parameter.dtype))

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(),
                "master": [master.detach() for master in self.master]}

    def load_state_dict(self, state):
        for master, saved in zip(self.master, state["master"]):
            master.data.copy_(saved)
        self.optimizer.load_state_dict(state["optimizer"])


def trajectory_backward(row, run, opd, norm, stats, chunk_size=4096):
    validate(row, teacher=True)
    table = units(row)
    target_by_begin = assistant_tokens_by_begin(row)
    ints = lambda x: torch.tensor(x, device=run.device, dtype=torch.long)

    def loss_fn(hidden, index):
        tokens = target_by_begin.get(table[index].begin)
        if not tokens:
            return None
        pick = lambda key: torch.tensor(
            [row[key][p] for p in tokens], device=run.device)
        selected = hidden.index_select(
            0, ints(predictor_rows(tokens, table[index].begin)))
        raw, metrics = chunked_opd_loss(
            selected, run.head, ints([row["tokens"][p] for p in tokens]),
            pick("rollout_logprobs"), pick("teacher_topk_ids"),
            pick("teacher_topk_logprobs"), pick("teacher_token_logprobs"),
            chunk_size=chunk_size, **vars(opd))
        loss = raw * len(tokens) * norm
        stats["loss"] += float(loss.detach())
        stats["tokens"] += len(tokens)
        for key, value in metrics.items():
            stats[key] += value
        return loss

    replay(run, table, loss_fn)


def sync_gradients(parameters):
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if dist.is_initialized():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def reduce_stats(stats, device):
    keys = sorted(stats)
    packed = torch.tensor([stats[key] for key in keys], device=device)
    if dist.is_initialized():
        dist.all_reduce(packed)
    return dict(zip(keys, packed.tolist()))


def report(stats, step):
    tokens = max(stats["tokens"], 1)
    return {"step": step, "loss": stats["loss"], "tokens": int(stats["tokens"]),
            **{key: stats[key] / tokens for key in METRIC_KEYS}}


def prune(root, keep):
    steps = sorted(
        (item for item in Path(root).glob("step-*")
         if item.is_dir() and (item / "config.json").exists()),
        key=lambda item: int(item.name.rsplit("-", 1)[-1]))
    for old in steps[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)


def save(student, optimizer, lora_path, step, rank, alpha,
         full_finetune=0, keep_checkpoints=2):
    root = Path(lora_path)
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"step-{step:06d}"
    tmp = root / f".step-{step:06d}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    if full_finetune:
        tmp.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(tmp, safe_serialization=True)
    else:
        save_lora(student, tmp, rank, alpha)
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    shutil.rmtree(output, ignore_errors=True)
    tmp.replace(output)
    if full_finetune:
        prune(root, keep_checkpoints)
    return output


def train(args):
    rank, world, device = setup_distributed()
    if world > 1:
        dist.all_reduce(torch.zeros(1, device=device))
    torch.manual_seed(0)

    student = load_student(
        args.model_path,
        args.lora_path,
        args.lora_rank,
        args.lora_alpha,
        args.full_finetune).to(device).train()

    parameters = [
        parameter for parameter in student.parameters()
        if parameter.requires_grad]


    optimizer = (
        OffloadedAdamW(parameters, lr=args.lr) if args.full_finetune
        else torch.optim.AdamW(parameters, lr=args.lr))


    checkpoint = (latest_full if args.full_finetune else latest_lora)(args.lora_path)
    if (args.load_optimizer_state and checkpoint and (checkpoint / "optimizer.pt").exists()):
        optimizer.load_state_dict(torch.load(
            checkpoint / "optimizer.pt",
            map_location="cpu" if args.full_finetune else device,
            weights_only=True))
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    rows = read_jsonl(args.input_path)
    optimizer.zero_grad(set_to_none=True)

    run = Runtime.of(student, device, max_pos(rows))

    norm = 1 / max(sum(len(assistant_tokens(r)) for r in rows), 1)
    opd, stats = OPD.from_args(args), dict.fromkeys(
        ("loss", "tokens") + METRIC_KEYS, 0.0)
    for row in rows[rank::world]:
        trajectory_backward(row, run, opd, norm, stats, args.chunk_size)

    torch.cuda.empty_cache()
    sync_gradients(parameters)
    torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
    optimizer.step()
    stats = reduce_stats(stats, device)
    if rank == 0:
        output = save(student, optimizer, args.lora_path, args.step,
                      args.lora_rank, args.lora_alpha, args.full_finetune,
                      args.keep_checkpoints)
        log = report(stats, args.step)

        (Path(args.lora_path) / ".metrics.json").write_text(json.dumps(log))
        print({**log, "ckpt": str(output)}, flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--input-path", required=True)
    p.add_argument("--lora-path", required=True)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--load-optimizer-state", type=int, choices=(0, 1), default=1)
    p.add_argument("--full-finetune", type=int, choices=(0, 1), default=1)
    p.add_argument("--keep-checkpoints", type=int, default=2)
    return OPD.add_arguments(p)


def main():
    train(parser().parse_args())


if __name__ == "__main__":
    main()
