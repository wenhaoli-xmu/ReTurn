from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from rollout.engine import KVCapacityError


@triton.jit
def _move_pages(PTRS, IDS, PACKED, WIDTH: tl.constexpr, LAYERS: tl.constexpr,
                RESTORE: tl.constexpr, BLOCK: tl.constexpr):
    block, page, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offset = block * BLOCK + tl.arange(0, BLOCK)
    physical = tl.load(IDS + page).to(tl.int64)
    pool = tl.load(PTRS + layer).to(tl.pointer_type(tl.bfloat16))
    packed = PACKED + (page.to(tl.int64) * LAYERS + layer) * WIDTH + offset
    pool = pool + physical * WIDTH + offset
    if RESTORE:
        tl.store(pool, tl.load(packed, offset < WIDTH, 0), offset < WIDTH)
    else:
        tl.store(packed, tl.load(pool, offset < WIDTH, 0), offset < WIDTH)


@dataclass
class Mirror:
    buffer: torch.Tensor
    count: int
    gpu_pages: tuple | None


class KVOffload:
    def __init__(self, cache, max_cpu_bytes=16 << 30,
                 transfer_bytes=64 << 20):
        if max_cpu_bytes <= 0 or transfer_bytes <= 0:
            raise ValueError('CPU and transfer budgets must be positive')
        if torch.device(cache.device).type != 'cuda' or not cache.k_pool:
            raise ValueError('KV offload requires a nonempty CUDA KV pool')
        self.kv, self.device = cache, cache.device
        pools = [p for pair in zip(cache.k_pool, cache.v_pool) for p in pair]
        self.layers = len(pools)
        self.width = cache.page_size * cache.num_kv_heads * cache.head_dim
        self.page_bytes = self.layers * self.width * 2
        self.max_cpu_bytes = int(max_cpu_bytes)
        self.chunk_pages = max(1, min(cache.max_pages, transfer_bytes // self.page_bytes))
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
            self.ptrs = torch.tensor([p.data_ptr() for p in pools], device=self.device)
            self.ids = torch.empty(self.chunk_pages, dtype=torch.int64, device=self.device)
            self.staging = torch.empty((self.chunk_pages, self.layers, self.width),
                                       dtype=torch.bfloat16, device=self.device)
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.cycles = {}
        self.inactive = OrderedDict()
        self.mirrors = OrderedDict()
        self.buffers = defaultdict(list)
        self.host_allocated = 0
        self.retired = deque()
        self.inflight = deque()
        self.counters = dict(d2h_bytes=0, h2d_bytes=0, offload_batches=0,
                             reload_batches=0, mirror_reuse_pages=0,
                             offloaded_pages=0, reloaded_pages=0)

    def _reap(self, wait=False):
        compute = torch.cuda.current_stream(self.device)
        while self.retired:
            event, pages = self.retired[0]
            if not wait and not event.query():
                break
            if wait:

                compute.wait_event(event)
            self.kv.free_pages.extend(pages)
            self.retired.popleft()
        while self.inflight and self.inflight[0][0].query():
            self.inflight.popleft()

    def _release_buffer(self, buffer):


        self.buffers[buffer.shape[0]].append(buffer)

    def _drop_mirror(self, key):
        entry = self.mirrors.pop(key)
        self._release_buffer(entry.buffer)

    def _allocate(self, pages):
        capacity = 1 << (pages - 1).bit_length()
        needed = capacity * self.page_bytes
        if needed > self.max_cpu_bytes:
            raise MemoryError('Paragraph exceeds the configured pinned-host budget')
        self._reap()
        if self.buffers[capacity]:
            return self.buffers[capacity].pop()

        for key, entry in list(self.mirrors.items()):
            if self.host_allocated + needed <= self.max_cpu_bytes:
                break
            if entry.gpu_pages is not None:
                self._drop_mirror(key)
                if self.buffers[capacity]:
                    return self.buffers[capacity].pop()
                self._discard_free()
        if self.host_allocated + needed > self.max_cpu_bytes:
            self._discard_free()
        if self.host_allocated + needed > self.max_cpu_bytes:
            raise MemoryError('Pinned-host budget is occupied by CPU-only KV variants')
        buffer = torch.empty((capacity, self.layers, self.width), dtype=torch.bfloat16,
                             device='cpu', pin_memory=True)
        self.host_allocated += needed
        return buffer

    def _discard_free(self):
        for buffers in self.buffers.values():
            for buffer in buffers:
                self.host_allocated -= buffer.numel() * buffer.element_size()
            buffers.clear()

    def touch(self, sid, pid):
        key = (sid, pid)
        self.inactive.pop(key, None)
        pages = self.kv.alt[sid][pid][0]
        if pages and pages[0] >= 0:
            self.inactive[key] = None

    def _chunks(self, records):
        segments, physical, used = [], [], 0
        for entry, pages in records:
            offset = 0
            while offset < len(pages):
                take = min(len(pages)-offset, self.chunk_pages-used)
                segments.append((entry, offset, take, used))
                physical.extend(pages[offset:offset+take])
                used += take
                offset += take
                if used == self.chunk_pages:
                    yield segments, physical
                    segments, physical, used = [], [], 0
        if used:
            yield segments, physical

    def _transfer(self, records, restore):
        refs = [entry.buffer for entry, _ in records]
        compute = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(compute)
            for segments, physical in self._chunks(records):

                indices = torch.tensor(physical, dtype=torch.int64, pin_memory=True)
                refs.append(indices)
                self.ids[:len(physical)].copy_(indices, non_blocking=True)
                if restore:
                    for entry, start, count, target in segments:
                        self.staging[target:target+count].copy_(entry.buffer[start:start+count],
                                                              non_blocking=True)
                _move_pages[(triton.cdiv(self.width, 1024), len(physical), self.layers)](
                    self.ptrs, self.ids, self.staging, self.width, self.layers, restore, 1024)
                if not restore:
                    for entry, start, count, source in segments:
                        entry.buffer[start:start+count].copy_(self.staging[source:source+count],
                                                            non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        self.inflight.append((done, refs))
        return done

    def _evict(self, wanted, protected=(), allow_transfer=True):
        records, pages_to_retire, copied = [], [], 0
        freed = 0
        for key in list(self.inactive):
            if freed >= wanted:
                break
            if key in protected:
                continue
            sid, pid = key
            pages, ntok = self.kv.alt[sid][pid]
            if not pages or pages[0] < 0:
                self.inactive.pop(key, None)
                continue
            entry = self.mirrors.get(key)
            if entry is not None and entry.gpu_pages == tuple(pages):

                entry.gpu_pages = None
                self.kv.free_pages.extend(pages)
                self.counters['mirror_reuse_pages'] += len(pages)
            else:
                if not allow_transfer:
                    continue
                if entry is not None:
                    self._drop_mirror(key)
                try:
                    buffer = self._allocate(len(pages))
                except MemoryError:
                    continue
                entry = Mirror(buffer, len(pages), None)
                self.mirrors[key] = entry
                records.append((entry, list(pages)))
                pages_to_retire.extend(pages)
                copied += len(pages)
            self.kv.alt[sid][pid] = ([-1] * len(pages), ntok)
            self.inactive.pop(key, None)
            freed += len(pages)
        if records:
            done = self._transfer(records, restore=False)
            self.retired.append((done, pages_to_retire))
            self.counters['offload_batches'] += 1
            self.counters['d2h_bytes'] += copied * self.page_bytes
        self.counters['offloaded_pages'] += freed
        return freed

    def reclaimable_pages(self, sids=(), start_cycle=False, protected=(), swaps=()):
        allow_transfer = start_cycle or all(
            not self.cycles.get(sid, {}).get('offload', False) for sid in sids)
        total = 0
        for key in self.inactive:
            if key in protected:
                continue
            pages = self.kv.alt[key[0]][key[1]][0]
            entry = self.mirrors.get(key)
            if pages and pages[0] >= 0 and (
                    allow_transfer or (entry is not None and entry.gpu_pages == tuple(pages))):
                total += len(pages)
        for sid, pid in swaps:
            incoming = self.kv.alt.get(sid, {}).get(pid)
            if incoming is None or (incoming[0] and incoming[0][0] < 0):
                continue
            index = self.kv.para_id[sid].index(pid)
            a, b = self.kv._bounds(sid, index)
            pages = self.kv.pages[sid][a:b]
            entry = self.mirrors.get((sid, pid))
            if allow_transfer or (entry is not None and entry.gpu_pages == tuple(pages)):
                total += len(pages)
        return total

    def _boundary_capacity(self, pages, sids):
        self._reap(wait=True)
        if pages <= len(self.kv.free_pages):
            return
        allow_transfer = all(not self.cycles[sid]['offload'] for sid in sids)
        before = self.counters['offload_batches']
        self._evict(pages - len(self.kv.free_pages), allow_transfer=allow_transfer)
        if self.counters['offload_batches'] > before:
            for sid in sids:
                self.cycles[sid]['offload'] = True
        self._reap(wait=True)
        if pages > len(self.kv.free_pages):
            raise KVCapacityError('Probe boundary cannot obtain sufficient KV capacity within one offload batch')

    def swap_many(self, pairs, *, required_pages=0, boundary_sids=(), start_cycle=()):
        pairs = list(pairs)
        sids = set(boundary_sids)
        start_cycle = set(start_cycle)
        if ((pairs or required_pages or start_cycle) and not sids
                or not start_cycle <= sids
                or any(sid not in sids for sid, _ in pairs)):
            raise ValueError('KV transfers require an explicit probe boundary')
        for sid in start_cycle:
            self.cycles[sid] = dict(offload=False, reload=False)
        for sid in sids:
            self.cycles.setdefault(sid, dict(offload=False, reload=False))
        if len(set(pairs)) != len(pairs):
            raise ValueError('Duplicate paragraph in one swap batch')
        for sid, pid in pairs:
            self.kv._validate_swap(sid, pid)
        restore, ready = [], []
        for key in pairs:
            pages = self.kv.alt[key[0]][key[1]][0]
            (restore if pages and pages[0] < 0 else ready).append(key)

        for sid, pid in ready:
            self.kv._swap_resident(sid, pid)
            self.touch(sid, pid)
        total = sum(self.mirrors[key].count for key in restore)
        if restore and any(self.cycles[sid]['reload'] for sid, _ in restore):
            raise KVCapacityError('A probe cycle permits only one reload batch')
        self._boundary_capacity(total + required_pages, sids)
        if restore:
            records = []
            for key in restore:
                entry = self.mirrors[key]
                pages = [self.kv.free_pages.pop() for _ in range(entry.count)]
                records.append((entry, pages))
                entry.gpu_pages = tuple(pages)
                self.mirrors.move_to_end(key)
                sid, pid = key
                self.kv.alt[sid][pid] = (pages, self.kv.alt[sid][pid][1])
            done = self._transfer(records, restore=True)
            torch.cuda.current_stream(self.device).wait_event(done)
            self.counters['reload_batches'] += 1
            for sid, _ in restore:
                self.cycles[sid]['reload'] = True
            self.counters['h2d_bytes'] += total * self.page_bytes
            self.counters['reloaded_pages'] += total
            for sid, pid in restore:
                self.kv._swap_resident(sid, pid)
                self.touch(sid, pid)

        self._reap()

    def free(self, sid):
        self.cycles.pop(sid, None)
        for key in [key for key in self.inactive if key[0] == sid]:
            self.inactive.pop(key)
        for key in [key for key in self.mirrors if key[0] == sid]:
            self._drop_mirror(key)
        self._reap(wait=True)

    def synchronize(self):
        self.stream.synchronize()
        self._reap()

    def stats(self):


        mirrors, retired = list(self.mirrors.values()), list(self.retired)
        cpu_pages = sum(e.count for e in mirrors if e.gpu_pages is None)
        ready = sum(len(p) for e, p in retired if e.query())
        return dict(self.counters, cpu_pages=cpu_pages, host_pool_bytes=self.host_allocated,
                    pending_free_pages=sum(len(p) for _, p in retired),
                    gpu_used_pages=self.kv.max_pages-len(self.kv.free_pages)-ready,
                    staging_gpu_bytes=self.staging.numel()*self.staging.element_size())
