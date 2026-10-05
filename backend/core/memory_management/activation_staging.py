"""Bounded pageable activation staging and speculative H2D lookahead."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import weakref
import time

import torch


class ActivationStaging:
    def __init__(self, arena, device, d2h_stream, h2d_stream, stats):
        self.size = min(arena.numel() // 8, 256 * 1024**2)
        self.d2h = arena.narrow(0, arena.numel() - 2*self.size, self.size)
        self.h2d = arena.narrow(0, arena.numel() - self.size, self.size)
        self.device, self.d2h_stream, self.h2d_stream = device, d2h_stream, h2d_stream
        self.gpu_budget = 2*self.size
        self.stats = stats
        self.d2h_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="activation-d2h")
        self.h2d_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="activation-h2d")
        self.pending = deque()
        self.pending_bytes = 0
        self.saved = []
        self.consumed = set()
        self.prefetched = None
        self.prefetch_age = 0
        self.h2d_pending = None

    def register(self, saved):
        saved.index = len(self.saved)
        self.saved.append(weakref.ref(saved))

    def pack(self, tensor, saved):
        nbytes = tensor.numel()*tensor.element_size()
        reservation = max(nbytes, tensor.untyped_storage().nbytes())
        if not tensor.is_contiguous():
            reservation += nbytes
        if reservation > self.gpu_budget:
            reason = "layout" if not tensor.is_contiguous() else "oversize"
            self.stats["fallback_"+reason+"_bytes"] = self.stats.get("fallback_"+reason+"_bytes", 0) + nbytes
            return False
        while self.pending and (self.pending[0][0].done() or self.pending_bytes+reservation > self.gpu_budget):
            future, size = self.pending.popleft()
            started = time.perf_counter()
            future.result()
            self.stats["pack_wait_seconds"] = self.stats.get("pack_wait_seconds", 0) + time.perf_counter()-started
            self.pending_bytes -= size
        saved.cpu = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
        producer = torch.cuda.current_stream(self.device).record_event()
        saved.future = self.d2h_worker.submit(self._pack, tensor, saved.cpu, producer, saved.version)
        self.pending.append((saved.future, reservation))
        self.pending_bytes += reservation
        if not tensor.is_contiguous():
            self.stats["staged_layout_bytes"] = self.stats.get("staged_layout_bytes", 0) + nbytes
        self.stats["staged_bytes"] = self.stats.get("staged_bytes", 0) + nbytes
        self.stats["staged_gpu_peak_bytes"] = max(self.stats.get("staged_gpu_peak_bytes", 0), self.pending_bytes)
        return True

    def _pack(self, tensor, cpu, producer, version):
        if tensor._version != version:
            raise RuntimeError("An asynchronously saved activation was modified in place")
        target = cpu.reshape(-1)
        count = self.size//tensor.element_size()
        with torch.cuda.stream(self.d2h_stream):
            self.d2h_stream.wait_event(producer)
            # A strided source's contiguous scratch must follow its producer event.
            source = tensor.detach().reshape(-1)
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
            self.stats["h2d_bytes"] = self.stats.get("h2d_bytes", 0) + saved.cpu.numel()*saved.cpu.element_size()
            return restored, self.h2d_stream.record_event()

    def _drop_prefetch(self):
        expected, future = self.prefetched
        self.prefetched = None
        self.prefetch_age = 0
        if not future.cancel():
            restored, _ = future.result()
            self.stats["prefetch_wasted_bytes"] = self.stats.get("prefetch_wasted_bytes", 0) + restored.numel()*restored.element_size()
        self.stats["prefetch_misses"] = self.stats.get("prefetch_misses", 0) + 1

    def _take_prefetch(self, saved):
        prefetched = self.prefetched
        if prefetched is not None:
            expected, future = prefetched
            if expected() is saved:
                self.prefetched = None
                self.prefetch_age = 0
                self.stats["prefetch_hits"] = self.stats.get("prefetch_hits", 0) + 1
                return future.result()
            self.prefetch_age += 1
            if expected() is None or self.prefetch_age >= 4:
                self._drop_prefetch()
            else:
                self.stats["prefetch_deferred"] = self.stats.get("prefetch_deferred", 0) + 1
        return None

    def unpack(self, saved):
        started = time.perf_counter()
        cached = self._take_prefetch(saved)
        restored, ready = cached if cached is not None else self.h2d_worker.submit(self._restore, saved).result()
        self.stats["restore_wait_seconds"] = self.stats.get("restore_wait_seconds", 0) + time.perf_counter()-started
        current = torch.cuda.current_stream(self.device)
        current.wait_event(ready)
        restored.record_stream(current)
        self.consumed.add(saved.index)
        attempts = self.stats.get("prefetch_hits", 0) + self.stats.get("prefetch_misses", 0)
        if self.prefetched is not None or (attempts >= 8 and self.stats.get("prefetch_hits", 0)*2 < attempts):
            return restored
        for index in range(saved.index-1, -1, -1):
            candidate = self.saved[index]()
            if candidate is None or index in self.consumed:
                continue
            size = candidate.cpu.numel()*candidate.cpu.element_size()
            free, _ = torch.cuda.mem_get_info(self.device)
            if size <= self.gpu_budget and free > size + 2*1024**3:
                self.prefetched = (weakref.ref(candidate), self.h2d_worker.submit(self._restore, candidate))
                self.prefetch_age = 0
                self.stats["prefetched_bytes"] = self.stats.get("prefetched_bytes", 0) + size
            break
        return restored

    def close(self):
        try:
            if self.prefetched is not None:
                self._drop_prefetch()
            for future, _ in self.pending:
                future.result()
        finally:
            # Drain D2H first: H2D tasks can be awaiting a pack future.
            try:
                self.d2h_worker.shutdown(wait=True)
            finally:
                self.h2d_worker.shutdown(wait=True)
            self.prefetched = None
            self.pending.clear()
            self.saved.clear()
