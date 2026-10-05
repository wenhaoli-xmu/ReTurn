from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch

_NULL = nullcontext()


class _Prof:
    def __init__(self):
        self.enabled = False
        self.device = None
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    def enable(self, device=None):
        self.enabled = True
        self.device = device

    def disable(self):
        self.enabled = False

    def reset(self):
        self.t.clear()
        self.n.clear()

    def __call__(self, name):
        if not self.enabled:
            return _NULL
        return self._timed(name)

    @contextmanager
    def _timed(self, name):
        if self.device is not None:
            torch.cuda.synchronize(self.device)
        t0 = _now()
        try:
            yield
        finally:
            if self.device is not None:
                torch.cuda.synchronize(self.device)
            self.t[name] += _now() - t0
            self.n[name] += 1

    def summary(self, sort=True):
        if not self.t:
            return "(profiler  no data )"
        rows = list(self.t.items())
        if sort:
            rows.sort(key=lambda kv: kv[1], reverse=True)
        total = max(self.t.values())
        w = max(len(k) for k in self.t)
        head = f"{'section'.ljust(w)}  {'total(ms)':>10}  {'calls':>7}  {'avg(ms)':>9}  {'%max':>6}"
        lines = [head, "-" * len(head)]
        for name, sec in rows:
            calls = self.n[name]
            lines.append(
                f"{name.ljust(w)}  {sec*1e3:>10.2f}  {calls:>7}  "
                f"{sec/calls*1e3:>9.3f}  {sec/total*100:>5.1f}%")
        return "\n".join(lines)


def _now():
    import time
    return time.perf_counter()


prof = _Prof()
