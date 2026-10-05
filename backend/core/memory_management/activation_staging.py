"""Bounded pageable activation staging and speculative H2D lookahead."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import weakref

import torch


class ActivationStaging:
    def __init__(self, arena, device, d2h_stream, h2d_stream, stats):
        self.size = min(arena.numel() // 8, 256 * 1024**2)
        self.d2h = arena.narrow(0, arena.numel() - 2*self.size, self.size)
        self.h2d = arena.narrow(0, arena.numel() - self.size, self.size)
        self.device, self.d2h_stream, self.h2d_stream = device, d2h_stream, h2d_stream
        self.gpu_budget = 2*self.size
        self.stats = stats
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="activation-stage")
        self.pending = deque()
        self.pending_bytes = 0
        self.saved = []
        self.consumed = set()
        self.prefetched = None
        self.h2d_pending = None

    def register(self, saved):
        saved.index = len(self.saved)
        self.saved.append(weakref.ref(saved))

    def pack(self, tensor, saved):
        nbytes = tensor.numel()*tensor.element_size()
        if nbytes > self.gpu_budget:
            return False
        while self.pending and (self.pending[0][0].done() or self.pending_bytes+nbytes > self.gpu_budget):
            future, size = self.pending.popleft()
            future.result()
            self.pending_bytes -= size
        saved.cpu = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
        producer = torch.cuda.current_stream(self.device).record_event()
        saved.future = self.worker.submit(self._pack, tensor, saved.cpu, producer, saved.version)
        self.pending.append((saved.future, nbytes))
        self.pending_bytes += nbytes
        self.stats["staged_bytes"] = self.stats.get("staged_bytes", 0) + nbytes
        self.stats["staged_gpu_peak_bytes"] = max(self.stats.get("staged_gpu_peak_bytes", 0), self.pending_bytes)
        return True

    def _pack(self, tensor, cpu, producer, version):
        if tensor._version != version:
            raise RuntimeError("An asynchronously saved activation was modified in place")
        source = tensor.detach().reshape(-1)
        target = cpu.reshape(-1)
        count = self.size//tensor.element_size()
        with torch.cuda.stream(self.d2h_stream):
            self.d2h_stream.wait_event(producer)
            for start in range(0, source.numel(), count):
                length = min(count, source.numel()-start)
                stage = self.d2h[:length*tensor.element_size()].view(tensor.dtype)
                stage.copy_(source[start:start+length], non_blocking=True)
                self.d2h_stream.synchronize()
                target[start:start+length].copy_(stage)
        if tensor._version != version:
            raise RuntimeError("An asynchronously saved activation was modified in place")

    def _restore(self, saved):
        if saved.future is not None:
            saved.future.result()
        with torch.cuda.stream(self.h2d_stream):
            if saved.d2h_ready is not None:
                self.h2d_stream.wait_event(saved.d2h_ready)
            restored = torch.empty(saved.cpu.shape, dtype=saved.cpu.dtype, device=self.device)
            if saved.cpu.is_pinned():
                restored.copy_(saved.cpu, non_blocking=True)
            elif saved.cpu.numel()*saved.cpu.element_size() > self.gpu_budget:
                restored.copy_(saved.cpu, non_blocking=False)
            else:
                source, target = saved.cpu.reshape(-1), restored.reshape(-1)
                count = self.size//source.element_size()
                for start in range(0, source.numel(), count):
                    length = min(count, source.numel()-start)
                    stage = self.h2d[:length*source.element_size()].view(source.dtype)
                    if self.h2d_pending is not None:
                        self.h2d_pending.synchronize()
                    stage.copy_(source[start:start+length])
                    target[start:start+length].copy_(stage, non_blocking=True)
                    # Wait on reuse, so the last chunk need not stall the caller's CPU.
                    self.h2d_pending = self.h2d_stream.record_event()
            return restored, self.h2d_stream.record_event()

    def _take_prefetch(self, saved):
        prefetched = self.prefetched
        self.prefetched = None
        if prefetched is not None:
            expected, future = prefetched
            restored, ready = future.result()
            if expected() is saved:
                self.stats["prefetch_hits"] = self.stats.get("prefetch_hits", 0) + 1
                return restored, ready
            self.stats["prefetch_misses"] = self.stats.get("prefetch_misses", 0) + 1
        return None

    def unpack(self, saved):
        cached = self._take_prefetch(saved)
        restored, ready = cached if cached is not None else self.worker.submit(self._restore, saved).result()
        current = torch.cuda.current_stream(self.device)
        current.wait_event(ready)
        restored.record_stream(current)
        self.consumed.add(saved.index)
        attempts = self.stats.get("prefetch_hits", 0) + self.stats.get("prefetch_misses", 0)
        if attempts >= 8 and self.stats.get("prefetch_hits", 0)*4 < attempts:
            return restored
        for index in range(saved.index-1, -1, -1):
            candidate = self.saved[index]()
            if candidate is None or index in self.consumed:
                continue
            size = candidate.cpu.numel()*candidate.cpu.element_size()
            free, _ = torch.cuda.mem_get_info(self.device)
            if size <= self.gpu_budget and free > size + 2*1024**3:
                self.prefetched = (weakref.ref(candidate), self.worker.submit(self._restore, candidate))
                self.stats["prefetched_bytes"] = self.stats.get("prefetched_bytes", 0) + size
            break
        return restored

    def close(self):
        try:
            if self.prefetched is not None:
                self.prefetched[1].result()
            for future, _ in self.pending:
                future.result()
        finally:
            self.worker.shutdown(wait=True)
            self.prefetched = None
            self.pending.clear()
            self.saved.clear()
