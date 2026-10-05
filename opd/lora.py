import json
from pathlib import Path

import torch
import torch.nn as nn

TARGETS = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        weight = base.weight


        kw = {"dtype": torch.float32, "device": weight.device}
        self.a = nn.Parameter(torch.zeros(rank, weight.shape[1], **kw))
        self.b = nn.Parameter(torch.zeros(weight.shape[0], rank, **kw))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)

    def forward(self, x):
        base = self.base(x)
        delta = (x.to(self.a.dtype) @ self.a.T @ self.b.T) * self.scale
        return base + delta.to(base.dtype)


def _walk(model, kind):
    for module in model.modules():
        for attr, child in list(module.named_children()):
            if attr in TARGETS and isinstance(child, kind):
                yield module, attr, child


def inject_lora(model, rank=8, alpha=16):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parent, attr, child in list(_walk(model, nn.Linear)):
        setattr(parent, attr, LoRALinear(child, rank, alpha))
    return model


def save_lora(model, path, rank, alpha):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if k.endswith((".a", ".b"))}
    torch.save(state, path / "adapter_model.pt")
    (path / "adapter_config.json").write_text(
        json.dumps({"rank": rank, "alpha": alpha, "targets": list(TARGETS)}))


def load_lora(model, path, trainable=False):
    path = Path(path)
    config = json.loads((path / "adapter_config.json").read_text())
    inject_lora(model, config["rank"], config["alpha"])
    state = torch.load(path / "adapter_model.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    if not trainable:
        model.requires_grad_(False)
    return model


@torch.no_grad()
def merge_lora(model):


    for parent, attr, child in list(_walk(model, LoRALinear)):
        weight = child.base.weight
        delta = (child.b.float() @ child.a.float()) * child.scale
        weight.copy_((weight.float() + delta).to(weight.dtype))
        setattr(parent, attr, child.base)
    return model
