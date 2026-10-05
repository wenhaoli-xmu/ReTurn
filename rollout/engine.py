import asyncio
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field


class KVCapacityError(MemoryError):
    pass


@dataclass
class Request:
    id: str
    prompt_ids: list
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop_ids: list = field(default_factory=list)
    split_ids: list = field(default_factory=list)
    subs: list = field(default_factory=list)
    swap: list = field(default_factory=list)
    ephem_at: int | None = None
    swap_at: int | None = None
    output: list = field(default_factory=list)
    output_logprobs: list = field(default_factory=list)
    done: asyncio.Future = field(init=False)
    _ctx: list = field(default=None, init=False)
    _kv_start: int | None = field(default=None, init=False)
    _ephem_kvlen: int | None = field(default=None, init=False)
    _next_logprob: float = field(default=0.0, init=False)
    _decode_reserve: int = field(default=0, init=False)

    def __post_init__(self):
        assert not self.swap or self.swap_at is not None, \
            "swap  requires an explicit boundary when nonempty:  swap_at"
        self.done = asyncio.get_running_loop().create_future()

    def push(self, tok):
        self.output.append(tok)
        self.output_logprobs.append(self._next_logprob)

    @property
    def last_token(self):
        return self.output[-1]

    @property
    def finished(self):
        if len(self.output) >= self.max_new_tokens:
            return True
        for seq in self.stop_ids:
            n = len(seq)
            if n and len(self.output) >= n and self.output[-n:] == seq:
                return True
        return False

    def finish(self):

        if not self.done.done():
            self.done.set_result(self)


class Engine:


    def __init__(self, engine_id, model):
        self.id = engine_id
        self.model = model
        self.kv = model.get_kv_cache()
        self.lin = model.get_lin_caches()
        self.max_reside = self.kv.max_reside
        self._task = None

        self.prefilling = {}
        self.decoding = {}
        self.reside = {}
        self.pending = deque()

        self._wake = asyncio.Event()
        self._release_req = set()

        self._t_start = time.perf_counter()
        self._acc_decode = 0.0
        self._acc_prefill = 0.0
        self._dc_tok = 0
        self._pf_tok = 0
        self._decode_streak = 0
        self._streak_tot = 0
        self._streak_cnt = 0

    @property
    def resident(self):
        return len(self.prefilling) + len(self.decoding) + len(self.reside)

    @property
    def load(self):
        return self.resident + len(self.pending)

    @property
    def run_time(self):
        return time.perf_counter() - self._t_start

    @property
    def decode_tps(self):
        return self._dc_tok / self._acc_decode if self._acc_decode > 0 else None

    @property
    def prefill_tps(self):
        return self._pf_tok / self._acc_prefill if self._acc_prefill > 0 else None

    @property
    def decode_duty(self):
        rt = self.run_time
        return self._acc_decode / rt if rt > 0 else 0.0

    @property
    def prefill_duty(self):
        rt = self.run_time
        return self._acc_prefill / rt if rt > 0 else 0.0

    @property
    def decode_streak(self):
        return self._streak_tot / self._streak_cnt if self._streak_cnt else None

    def start(self):
        self._task = asyncio.create_task(self.run())
        self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            os._exit(1)

    async def stop(self):
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def _add(self, rid):
        self.kv.alloc(rid)
        for c in self.lin:
            c.alloc(rid)

    def _free(self, rid):
        self.kv.free(rid)
        for c in self.lin:
            c.free(rid)

    def _decode_pages(self, r):

        pages = self.kv.pages.get(r.id, [])
        ntok = self.kv.ntok.get(r.id, [])
        spare = (self.kv.page_size - ntok[-1]
                 if pages and not self.kv.closed[r.id] else 0)
        remaining = 1 if not r.finished else 0
        return (max(0, remaining - spare) + self.kv.page_size - 1) // self.kv.page_size

    def _request_pages(self, r):

        P = self.kv.page_size
        start = self.reside.get(r.id, 0)
        ids = r.prompt_ids[start:]
        splits = set(r.split_ids)
        ephem = r.ephem_at - start if r.ephem_at is not None else None
        swap = r.swap_at - start if r.swap and r.swap_at is not None else None
        pages = self.kv.pages.get(r.id, [])
        ntok = self.kv.ntok.get(r.id, [])
        spare = P - ntok[-1] if pages and not self.kv.closed[r.id] else 0

        total = 0
        for i, token in enumerate(ids):
            if i == ephem or i == swap or (token in splits and (ephem is None or i < ephem)):
                spare = 0
            if spare == 0:
                total += 1
                spare = P
            spare -= 1
        if r.max_new_tokens > 1 and spare == 0:
            total += 1
        alternatives = self.kv.alt.get(r.id, {})
        for pid in r.swap:
            variant = alternatives.get(pid)
            if variant and variant[0] and variant[0][0] < 0:
                total += len(variant[0])

        total += sum((len(tokens) + P - 1) // P for _, tokens in r.subs)
        return total

    def _admit(self):

        avail = self.max_reside - self.resident
        reserved = sum(self._decode_pages(r) for r in self.decoding.values())
        decode_reserve = reserved
        active = set(self.decoding) | set(self.prefilling)
        pending = list(self.pending)
        self.pending.clear()
        ordered = ([r for r in pending if r.id in self.reside]
                   + [r for r in pending if r.id not in self.reside])
        admitted, wait = [], set()
        protected = set()
        offloader = getattr(self.kv, 'offloader', None)
        for r in ordered:
            if r.done.done():
                continue
            if r.id in active or (r.id not in self.reside and avail <= 0):
                wait.add(id(r))
                continue
            needed = self._request_pages(r)
            if needed > self.kv.max_pages:
                r.done.set_exception(KVCapacityError('Request prefill exceeds the GPU KV page capacity'))
                continue
            wanted = reserved + needed
            incoming = {(r.id, pid) for pid in r.swap}
            capacity = len(self.kv.free_pages)
            if r.swap_at is not None or r.ephem_at is not None:
                if offloader is not None:
                    capacity += offloader.reclaimable_pages(
                        (r.id,), start_cycle=r.ephem_at is not None,
                        protected=protected | incoming, swaps=[(r.id, pid) for pid in r.swap])
                at = r.swap_at if r.swap_at is not None else r.ephem_at
                before = Request(r.id, r.prompt_ids[:at], max_new_tokens=0,
                                 split_ids=r.split_ids, subs=r.subs)
                if reserved + self._request_pages(before) > len(self.kv.free_pages):
                    wait.add(id(r))
                    continue
            if wanted > capacity:
                wait.add(id(r))
                continue
            if r.id not in self.reside:
                avail -= 1
            active.add(r.id)
            r._decode_reserve = decode_reserve
            reserved += needed
            protected.update(incoming)
            admitted.append(r)
        self.pending.extend(r for r in pending if id(r) in wait)
        return admitted

    def _prepare(self, new):
        for r in new:
            if r._ctx is None:
                r._ctx = r.prompt_ids
            S = self.reside.get(r.id, 0)
            for at_name in ("ephem_at", "swap_at"):
                at = getattr(r, at_name)
                if at is None:
                    continue
                assert S <= at < len(r._ctx), (
                    f"{at_name}={at}  is outside the uncached range  [{S}, {len(r._ctx)})")
                setattr(r, at_name, at - S)
            if not self._resume(r):
                self._add(r.id)
            r._kv_start = self.kv.seqlen(r.id)
        return new

    def _resume(self, r):

        S = self.reside.get(r.id)
        if S is None:
            return False
        del self.reside[r.id]
        assert 0 < S < len(r.prompt_ids)
        r.prompt_ids = r.prompt_ids[S:]
        return True

    def _do_release(self):


        active = set(self.decoding) | set(self.prefilling) | {r.id for r in self.pending}
        for rid in self._release_req - active:
            if rid in self.reside:
                self._free(rid)
                del self.reside[rid]
            self._release_req.remove(rid)

    def _finish(self, r):

        is_ephemeral = r.ephem_at is not None
        drop = self.kv.pop_paragraph(r.id) if is_ephemeral else 0
        if is_ephemeral:
            assert r._ephem_kvlen is not None, "ephemeral  request is missing  KV  boundary snapshot "
            assert self.kv.seqlen(r.id) == r._ephem_kvlen, (
                f"ephemeral KV  rollback length mismatch : {self.kv.seqlen(r.id)} != {r._ephem_kvlen}")
        for c in self.lin:
            c.persist(r.id)
            if is_ephemeral:
                c.rollback(r.id)


        self.reside[r.id] = len(r._ctx) + len(r.output) - 1 - drop
        if r.ephem_at is not None:
            prompt_start = len(r._ctx) - len(r.prompt_ids)
            ephem_cursor = prompt_start + r.ephem_at
            assert self.reside[r.id] == ephem_cursor, (
                f"ephem_at  rollback cursor mismatch : {self.reside[r.id]} != {ephem_cursor}")
        r.finish()

    def stats(self):
        offloader = getattr(self.kv, 'offloader', None)
        offload = offloader.stats() if offloader is not None else None
        return {
            "run_time": self.run_time,
            "page_used": offload['gpu_used_pages'] if offload else self.kv.max_pages - len(self.kv.free_pages),
            "page_total": self.kv.max_pages,
            "kv_offload": offload}

    def _fail(self, r, error):
        self.decoding.pop(r.id, None)
        self.prefilling.pop(r.id, None)
        self.reside.pop(r.id, None)
        if r.id in self.kv.pages:
            self._free(r.id)
        if not r.done.done():
            r.done.set_exception(error)

    def _decode_ready(self):
        reqs, wait = [], []
        available = len(self.kv.free_pages)
        for r in self.decoding.values():
            needed = self._decode_pages(r)
            if needed <= available:
                reqs.append(r)
                available -= needed
            else:
                wait.append(r)
        if not reqs and wait:
            self._fail(wait[0], KVCapacityError(
                'Active KV exhausted the GPU pool before the next probe boundary'))
        return reqs

    async def submit(self, req):
        self.pending.append(req)
        self._wake.set()
        return await req.done

    def release(self, rid):
        self._release_req.add(rid)
        self._wake.set()

    async def run(self):
        while True:
            self._wake.clear()
            self._do_release()
            worked = False

            new = self._prepare(self._admit())

            if new:


                if self._decode_streak > 0:
                    self._streak_tot += self._decode_streak
                    self._streak_cnt += 1
                    self._decode_streak = 0

                self.prefilling = {r.id: r for r in new}
                ntok = sum(len(r.prompt_ids) for r in new)
                t0 = time.perf_counter()
                try:
                    toks = await asyncio.to_thread(self.model.prefill, new)
                except KVCapacityError as error:
                    for r in new:
                        self._fail(r, error)
                    toks = []
                self._acc_prefill += time.perf_counter() - t0
                self._pf_tok += ntok

                for r, t in zip(new, toks):
                    r.push(t)
                    if r.finished:
                        self._finish(r)
                    else:
                        self.decoding[r.id] = r
                self.prefilling = {}
                worked = True

            if self.decoding:

                reqs = self._decode_ready()
                if not reqs:
                    await asyncio.sleep(0)
                    continue
                t0 = time.perf_counter()
                toks = await asyncio.to_thread(self.model.decode, reqs)
                self._acc_decode += time.perf_counter() - t0
                self._dc_tok += len(reqs)

                for r, t in zip(reqs, toks):
                    r.push(t)
                for r in [r for r in reqs if r.finished]:
                    self._finish(self.decoding.pop(r.id))

                self._decode_streak += 1
                worked = True

            if not worked:
                await self._wake.wait()
            else:
                await asyncio.sleep(0)
