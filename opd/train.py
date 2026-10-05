import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import wandb

from opd.loss import OPD
from opd.rollout import rollout
from opd.utils import latest_full, latest_lora, launch_server, validate_tokenizers


def student_checkpoint(args):
    if not args.full_finetune:
        return args.lora_path, args.model
    checkpoint = latest_full(args.lora_path)
    return None, str(checkpoint) if checkpoint else args.model


def next_step(lora_path):
    ckpt = latest_lora(lora_path) or latest_full(lora_path)
    if ckpt is None or not ckpt.name.startswith("step-"):
        return 1
    return int(ckpt.name.rsplit("-", 1)[-1]) + 1


def distributed(module, nproc, args):
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nproc-per-node", str(nproc),
        "-m", module,
    ]
    for name, value in vars(args).items():
        if value is None:
            continue
        cmd += ["--" + name.replace("_", "-"), str(value)]
    subprocess.run(cmd, check=True)


def gpu_count():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([device for device in visible.split(",") if device.strip()])
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True)
    return len(output.splitlines())


_EVAL_SPECS = (


    ("locomo", "eval/locomo.py", "eval_locomo_concurrency",
     r"^\s*F1\s*:\s*([0-9.]+)\s*$", "f1_percent"),
    ("longbench", "eval/longbench.py", "eval_longbench_concurrency",
     r"^\s*F1\s*:\s*([0-9.]+)\s*$", "f1_percent"),
    ("longcite", "eval/longcite.py", "eval_longcite_concurrency",
     r"^\s*F1\s*:\s*([0-9.]+)\s*$", "f1_percent"),
    ("longmemeval", "eval/longmemeval.py", "eval_longmemeval_concurrency",
     r"^\s*F1\s*:\s*([0-9.]+)\s*$", "f1_percent"),
    ("longrlvr", "eval/longrlvr.py", "eval_longrlvr_concurrency",
     r"^\s*F1\s*:\s*([0-9.]+)\s*$", "f1_percent"),
    ("bc200", "eval/bc200.py", "eval_bc200_concurrency",
     r"^\s*accuracy\s*:\s*([0-9.]+)", "accuracy"),
)


def evaluate(args, step):
    if not args.eval_every or step % args.eval_every:
        return {}

    requested = {name.strip() for name in args.eval_benchmarks.split(",") if name.strip()}
    available = {spec[0] for spec in _EVAL_SPECS}
    if requested == {"all"}:
        specs = _EVAL_SPECS
    else:
        unknown = requested - available
        if unknown:
            raise ValueError(f" Unknown benchmark : {sorted(unknown)};  available  {sorted(available)}  or  all")
        specs = tuple(spec for spec in _EVAL_SPECS if spec[0] in requested)
    if not specs:
        raise ValueError("--eval-benchmarks  must not be empty ")

    root = Path(__file__).resolve().parents[1]
    eval_root = Path(args.eval_output or Path(args.output).parent / "eval")
    eval_root.mkdir(parents=True, exist_ok=True)
    adapter, weights = student_checkpoint(args)
    env = os.environ.copy()


    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root), env.get("PYTHONPATH")) if part)


    env.update(UNFOLD_SELECTION_MODE="model", BM25_UNION_TOP_K="0")


    env.setdefault("SUMMARY_DISK_CACHE", str(eval_root / "summary-cache.sqlite3"))

    print(f"\n=== oo eval after student step {step}: {len(specs)} benchmarks ===",
          flush=True)
    total_started = time.perf_counter()
    metrics = {}


    with launch_server(adapter, weights, port=None,
                       max_reside=args.eval_max_reside,
                       max_token=args.eval_max_token) as server:
        for name, script, concurrency_arg, score_pattern, metric_name in specs:
            trace = eval_root / f"{name}-oo-step-{step:06d}.jsonl"
            script_path = root / script
            cmd = [sys.executable]
            cmd += [str(script_path)]
            cmd += [
                "--model", args.model,
                "--out", str(trace),
                "--concurrency", str(getattr(args, concurrency_arg)),
                "--url", server.url + "/generate",
            ]
            if name == "bc200":
                cmd += ["--data", args.eval_bc200_data]
            else:
                cmd += ["--keep-last-k", str(args.keep_k)]

            print(f"\n--- {name} oo eval → {trace} ---", flush=True)
            started = time.perf_counter()
            score = None
            process = subprocess.Popen(
                cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,


                start_new_session=True)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                match = re.match(score_pattern, line)
                if match:
                    score = float(match.group(1))
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, cmd)

            elapsed = time.perf_counter() - started
            metrics[f"perf/eval_{name}"] = elapsed
            if score is not None:
                metrics[f"eval/{name}_oo_{metric_name}"] = score
            print(f"--- {name} oo eval step {step} finished in {elapsed:.1f}s ---", flush=True)

    metrics["perf/eval_all"] = time.perf_counter() - total_started
    print(f"=== all oo evals step {step} finished in {metrics['perf/eval_all']:.1f}s ===\n",
          flush=True)
    return metrics


def train(args):
    if not args.full_finetune:
        raise ValueError("End-to-end OPD requires full model checkpoints for the rollout server")
    validate_tokenizers(args.model, args.teacher_model)
    args.nproc = args.nproc or gpu_count()
    if args.start_step <= 0:
        args.start_step = next_step(args.lora_path)
    if args.opd_reverse_coef and (
            args.temperature != 1.0 or args.top_p != 1.0 or args.top_k != 0):
        raise ValueError("reverse KL PG  requires  temperature=1, top_p=1, top_k=0")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    run = wandb.init(
        project="opd",
        name=f"{Path(args.lora_path).name}-{datetime.now():%Y%m%d-%H%M%S}")

    for step in range(args.start_step, args.start_step + args.steps):
        trace = output / f"step-{step:06d}.jsonl"


        t0 = time.perf_counter()
        adapter, weights = student_checkpoint(args)
        with launch_server(
            adapter,
            weights,
            args.port,
            args.max_reside,
            args.max_token
        ) as server:
            loop.run_until_complete(rollout(
                model=args.model,
                data=args.data,
                output=trace,
                url=server.url + "/generate",
                num_prompts=args.num_prompts,
                samples_per_prompt=args.samples_per_prompt,
                concurrency=args.concurrency,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                sampling_seed=args.sampling_seed,
                keep_last_k=args.keep_k,
                max_answer_tokens=args.max_answer_tokens,
                offset=(step - 1) * args.num_prompts))
        t_roll = time.perf_counter() - t0


        t1 = time.perf_counter()
        teacher = argparse.Namespace(
            model_path=args.teacher_model,
            input_path=trace,
            output=trace,
            top_k=args.opd_top_k,
            chunk_size=args.chunk_size)
        distributed("opd.teacher", args.nproc, teacher)
        t_teacher = time.perf_counter() - t1


        t2 = time.perf_counter()
        student = argparse.Namespace(
            model_path=args.model,
            input_path=trace,
            lora_path=args.lora_path,
            step=step,
            lr=args.lr,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            chunk_size=args.chunk_size,
            max_grad_norm=args.max_grad_norm,
            full_finetune=args.full_finetune,
            keep_checkpoints=args.keep_checkpoints,
            load_optimizer_state=int(args.load_optimizer_state or step > args.start_step),
            **{f"opd_{f.name}": getattr(args, f"opd_{f.name}") for f in fields(OPD)})
        distributed("opd.student", args.nproc, student)
        t_student = time.perf_counter() - t2


        metrics = json.loads((Path(args.lora_path) / ".metrics.json").read_text())
        train_total = time.perf_counter() - t0
        eval_metrics = evaluate(args, step)
        wandb.log({**metrics, "perf/rollout": t_roll, "perf/teacher": t_teacher,
                   "perf/student": t_student,
                   "perf/total": train_total, **eval_metrics}, step=step)
    run.finish()
    loop.close()


def main():


    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="trace/opd/rollout")
    p.add_argument("--lora-path", default="trace/opd/ckpt")
    p.add_argument("--start-step", type=int, default=0, help="0 =  resume from the latest  checkpoint  resume training ")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--nproc", type=int, default=0, help="0 =  all visible  GPU")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--max-reside", type=int, default=32)
    p.add_argument("--max-token", type=int, default=1048576)
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--samples-per-prompt", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--sampling-seed", type=int, default=20260822,
                   help="rollout  Reproducible training prompt  shuffle  seed ")
    p.add_argument("--opd-top-k", type=int, default=10, help=" Number of tokens stored per annotated position  top-k")
    p.add_argument("--keep-k", type=int, default=3, help=" Window size ； fold all paragraphs outside the window ")
    p.add_argument("--max-answer-tokens", type=int, default=128,
                   help="longrlvr  answers are full sentences ， do not use  locomo  that  32")
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--load-optimizer-state", type=int, choices=(0, 1), default=1,
                   help="1 =  Restore  AdamW  momentum ；0 =  first step of this invocation  step  reset ")
    p.add_argument("--full-finetune", type=int, choices=(0, 1), default=1,
                   help="1 =  Full parameter finetuning （--lr  should be  ~1e-6）")
    p.add_argument("--keep-checkpoints", type=int, default=2,
                   help=" full finetuning only ： Number to retain on disk  step-*  directory ")
    p.add_argument("--eval-every", type=int, default=0,
                   help=" every  N  items  student step  Run selected  oo  evaluation （ excluding  RULER）；0 =  disabled ")
    p.add_argument("--eval-benchmarks", default="locomo",
                   help=" Comma-separated  oo  benchmark names （ default  locomo； use  all  run all ）")
    p.add_argument("--eval-output", default=None,
                   help=" evaluation  trace  directory ； default  <output  parent directory of >/eval")
    p.add_argument("--eval-max-reside", type=int, default=32)
    p.add_argument("--eval-max-token", type=int, default=1048576)
    p.add_argument("--eval-locomo-concurrency", type=int, default=32)
    p.add_argument("--eval-longbench-concurrency", type=int, default=32)
    p.add_argument("--eval-longcite-concurrency", type=int, default=16)
    p.add_argument("--eval-longmemeval-concurrency", type=int, default=8)
    p.add_argument("--eval-longrlvr-concurrency", type=int, default=4)
    p.add_argument("--eval-bc200-concurrency", type=int, default=32)
    p.add_argument("--eval-bc200-data", default=os.environ.get("GDRIVE_LOCAL", "") +
                   "/data/search_eval/eval_bc200_reformulate.jsonl")
    train(OPD.add_arguments(p).parse_args())


if __name__ == "__main__":
    main()
