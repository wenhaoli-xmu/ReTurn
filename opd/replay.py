from dataclasses import dataclass

import torch


@dataclass
class State:
    kv: object
    gdn: object


def replay(run, units, loss_fn):
    hidden = run.forward_parallel(units)
    total = None
    for index, value in enumerate(hidden):
        loss = loss_fn(value, index)
        if loss is not None:
            total = loss if total is None else total + loss
    if total is not None:
        total.backward()
        return total.detach()
    return None
