from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ForwardContext:
    is_prefill: bool = False
    n: int = 0
    T: int = 0
    num_splits: int = 1
    conv: Any = None
    sids: tuple = ()


DECODE_CONTEXT = ForwardContext(is_prefill=False)
