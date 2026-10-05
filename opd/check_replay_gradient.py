import argparse
import math
import os

os.environ.setdefault("FLA_DISABLE_TENSOR_CACHE", "1")

import torch
from transformers import AutoModelForCausalLM

from opd.data import Unit
from opd.forward import Runtime
from opd.lora import inject_lora
from opd.replay import State, replay
from opd.utils import training_linear_fallback


def _units():


    token = lambda seed, n: [101 + (seed * 97 + i * 41) % 9000 for i in range(n)]
    return [
        Unit(token(0, 7), 0, []),
        Unit(token(1, 6), 7, [0]),
        Unit(token(2, 5), 13, [0]),
        Unit(token(3, 8), 18, [0, 2]),
    ]


def _ctx(states, indices, n_attn):
    if not indices:
        return None
    return [
        (
            torch.cat([states[index].kv[layer][0] for index in indices], dim=0),
            torch.cat([states[index].kv[layer][1] for index in indices], dim=0),
        )
        for layer in range(n_attn)
    ]


def _loss(hidden, index):
    if index not in (1, 3):
        return None

    value = hidden[-3:].float()
    return (value.tanh().square() + 0.03 * value.sin()).mean()


def _direct(run, units):
    n_attn = sum(
        layer.layer_type != "linear_attention" for layer in run.body.layers)
    states = []
    loss = None
    for index, unit in enumerate(units):
        ctx = _ctx(states, unit.ctx, n_attn)
        gdn = states[index - 1].gdn if index else None
        hidden, kv, gdn = run.forward(unit.ids, ctx, gdn)
        states.append(State(kv, gdn))
        local = _loss(hidden, index)
        if local is not None:
            loss = local if loss is None else loss + local
    assert loss is not None
    loss.backward()
    return loss.detach()


def _grads(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _compare(reference, replayed):
    assert reference.keys() == replayed.keys()
    dot = ref_sq = got_sq = diff_sq = 0.0
    max_abs = 0.0
    worst = []
    for name in reference:
        left, right = reference[name], replayed[name]
        assert (left is None) == (right is None), f"{name}:  one side  grad=None"
        if left is None:
            continue
        delta = left - right
        left64, right64, delta64 = left.double(), right.double(), delta.double()
        dot += float((left64 * right64).sum())
        ref_sq += float(left64.square().sum())
        got_sq += float(right64.square().sum())
        diff_sq += float(delta64.square().sum())
        local = float(delta.abs().max())
        max_abs = max(max_abs, local)
        worst.append((local, name, float(left.norm()), float(right.norm())))
    rel_l2 = math.sqrt(diff_sq) / max(math.sqrt(ref_sq), 1e-30)
    cosine = dot / max(math.sqrt(ref_sq * got_sq), 1e-30)
    return rel_l2, cosine, max_abs, sorted(worst, reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--relative-l2", type=float, default=0.02)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--loss-atol", type=float, default=0.002)
    parser.add_argument("--loss-rtol", type=float, default=0.01)
    args = parser.parse_args()

    training_linear_fallback()
    torch.manual_seed(1234)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16)
    inject_lora(model, rank=2, alpha=4)
    model.to(device).train()


    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(5678)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name.endswith(".b"):
                parameter.normal_(mean=0.0, std=0.03, generator=generator)

    units = _units()
    run = Runtime.of(model, device, max_pos=sum(len(unit.ids) for unit in units) + 8)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    model.zero_grad(set_to_none=True)
    direct_loss = _direct(run, units)
    direct_grad = _grads(model)

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    replay_loss = replay(
        run,
        units,
        lambda hidden, index: _loss(hidden, index),
    )
    replay_grad = _grads(model)

    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
    rel_l2, cosine, max_abs, worst = _compare(direct_grad, replay_grad)
    loss_diff = abs(float(direct_loss) - float(replay_loss))

    print(
        f"loss direct={float(direct_loss):.9g} replay={float(replay_loss):.9g} "
        f"|Δ|={loss_diff:.3e}"
    )
    print(
        f"all LoRA grads: relative_L2={rel_l2:.3e} "
        f"cosine={cosine:.9f} max|Δ|={max_abs:.3e}"
    )
    for local, name, ref_norm, got_norm in worst[:8]:
        print(
            f"  {name}: max|Δ|={local:.3e} "
            f"norm(direct/replay)={ref_norm:.3e}/{got_norm:.3e}"
        )

    torch.testing.assert_close(direct_loss, replay_loss,
                               atol=args.loss_atol, rtol=args.loss_rtol)
    assert rel_l2 <= args.relative_l2, (rel_l2, args.relative_l2)
    assert cosine >= args.min_cosine, (cosine, args.min_cosine)
    print("OPD parallel replay gradient: OK")


if __name__ == "__main__":
    main()
