import json
import os
import httpx
from dataclasses import dataclass, asdict, replace
from typing import Any


@dataclass
class Judge:
    score: float
    golden: list
    predicted: str | None
    prompt: str | None = None
    verdict: str | None = None
    error: str | None = None


@dataclass
class Request:
    input_ids: list[int]
    url: str
    tokenizer: Any
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    def with_sampling(self, temperature, top_p, top_k):
        return replace(self, **{
            name: default if getattr(self, name) is None else getattr(self, name)
            for name, default in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k))})


@dataclass(eq=False)
class Para:
    start: int
    kind: str


    pid: int | None = None
    uid: str | None = None
    summary: str | None = None
    folded: bool = False
    task: object = None


    text: str | None = None


    selected_ids: list[int | str] | None = None
    selected_pids: list[int] | None = None


    swap: list | None = None


@dataclass
class Traj:

    tokens: list[int]
    text: str
    status: str
    judge: Judge | None = None
    prompt_length: int = 0
    loss_mask: list[int] | None = None
    rollout_logprobs: list[float] | None = None
    paras: list[Para] | None = None


def dump(traj: Traj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(traj), ensure_ascii=False) + "\n")
