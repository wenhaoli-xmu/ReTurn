from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class OPD:
    forward_coef: float = 3.0
    reverse_coef: float = 1.0
    entropy_coef: float = 0.01
    tis_cap: float = 2.0

    @classmethod
    def add_arguments(cls, parser):

        for f in fields(cls):
            parser.add_argument(f"--opd-{f.name.replace('_', '-')}",
                                type=type(f.default), default=f.default)
        return parser

    @classmethod
    def from_args(cls, args):
        return cls(**{f.name: getattr(args, f"opd_{f.name}") for f in fields(cls)})


METRIC_KEYS = (
    "forward_kl", "reverse_kl", "teacher_mass", "student_mass", "entropy",
    "teacher_topk_hit_rate", "train_infer_logp_diff", "tis_clip_frac")


def opd_loss(student_logits, sampled, rollout_logp, teacher_topk_ids,
             teacher_topk_logp, teacher_logp,
             forward_coef=3.0, reverse_coef=1.0, entropy_coef=0.01, tis_cap=2.0):
    logp = F.log_softmax(student_logits.float(), dim=-1)
    sampled_logp = logp.gather(-1, sampled[:, None]).squeeze(-1)


    student_topk_logp = logp.gather(-1, teacher_topk_ids)
    teacher_topk_p = teacher_topk_logp.float().exp()
    forward_kl = (teacher_topk_p *
                  (teacher_topk_logp - student_topk_logp)).sum(-1)
    if teacher_topk_ids.size(-1) < student_logits.size(-1):
        teacher_rest = (1.0 - teacher_topk_p.sum(-1)).clamp_min(0.0)
        rest_logits = student_logits.float().scatter(-1, teacher_topk_ids, -torch.inf)
        student_rest_logp = rest_logits.logsumexp(-1) - student_logits.float().logsumexp(-1)
        forward_kl = forward_kl + teacher_rest * (
            teacher_rest.clamp_min(torch.finfo(torch.float32).tiny).log() - student_rest_logp)


    advantage = (sampled_logp - teacher_logp.float()).detach()
    logp_diff = sampled_logp.detach() - rollout_logp
    tis = logp_diff.exp().clamp(max=tis_cap)
    reverse_kl_pg = tis * (sampled_logp - sampled_logp.detach()).exp() * advantage

    entropy = -(logp.exp() * logp).sum(-1)
    token_loss = (forward_coef * forward_kl + reverse_coef * reverse_kl_pg
                  - entropy_coef * entropy)
    metrics = {
        "forward_kl": forward_kl.mean(),
        "reverse_kl": advantage.mean(),
        "teacher_mass": teacher_topk_p.sum(-1).mean(),
        "student_mass": student_topk_logp.exp().sum(-1).mean(),
        "entropy": entropy.mean(),
        "teacher_topk_hit_rate": teacher_topk_ids.eq(
            sampled[:, None]).any(-1).float().mean(),


        "train_infer_logp_diff": logp_diff.abs().mean(),
        "tis_clip_frac": (tis >= tis_cap).float().mean(),
    }
    return token_loss.mean(), metrics


def chunked_opd_loss(hidden, head, sampled, rollout_logp, teacher_topk_ids,
                     teacher_topk_logp, teacher_logp, chunk_size=4096, **kwargs):
    n = hidden.size(0)
    total = hidden.new_zeros((), dtype=torch.float32)
    metrics = dict.fromkeys(METRIC_KEYS, 0.0)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        def run(x, start=start, end=end):
            loss, values = opd_loss(
                head(x), sampled[start:end], rollout_logp[start:end],
                teacher_topk_ids[start:end], teacher_topk_logp[start:end],
                teacher_logp[start:end], **kwargs)
            return loss, *(values[key] for key in METRIC_KEYS)

        loss, *values = checkpoint(run, hidden[start:end], use_reentrant=False)
        total = total + loss * (end - start) / n
        for key, value in zip(METRIC_KEYS, values):
            metrics[key] += float(value.detach()) * (end - start)
    return total, metrics
