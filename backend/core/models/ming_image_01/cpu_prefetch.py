"""Exclusive frozen native conditioner with bounded CPU batch lookahead."""

import queue
import threading

import torch

from .cache import parameter_versions
from .dataset import load_shared_items, prepare_record_conditioning, shared_conditioning_context
from core.training.encoder_reuse import reuse_ming_conditioning


class _Cancelled(Exception):
    pass


class MingCpuPrefetch:
    def __init__(self, trainer, batches, *, epoch, batch_offset, start, depth, identity_fn, preparation_lock=None):
        if type(depth) is not int or depth < 1:
            raise ValueError("Ming CPU prefetch depth must be a positive integer")
        self.trainer, self.batches, self.epoch, self.offset, self.start = trainer, batches, epoch, batch_offset, start
        self.identity_fn = identity_fn
        self.conditioner = trainer.ming_image_01_components.conditioner
        if not self.conditioner.cacheable:
            raise ValueError("Ming CPU prefetch requires a frozen conditioner")
        self.conditioner.to(device="cpu").eval()
        self.versions = parameter_versions(self.conditioner)
        self.precision = tuple(value.dtype for value in self.conditioner.parameters())
        self.preparation_lock = preparation_lock if preparation_lock is not None else threading.RLock()
        self.queue = queue.Queue(maxsize=depth)
        self.stop_event, self.done = threading.Event(), threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="ming-cpu-prefetch", daemon=True)
        self.thread.start()

    def _check(self):
        if self.stop_event.is_set(): raise _Cancelled()
        if (not self.conditioner.cacheable or any(module.training for module in self.conditioner.modules())
                or parameter_versions(self.conditioner) != self.versions
                or tuple(value.dtype for value in self.conditioner.parameters()) != self.precision
                or any(value.device.type != "cpu" for value in self.conditioner.parameters())
                or any(value.device.type != "cpu" for value in self.conditioner.buffers())):
            raise RuntimeError("Ming CPU prefetch producer ownership changed")

    def _publish(self, value):
        while not self.stop_event.is_set():
            try:
                self.queue.put(value, timeout=0.05)
                return
            except queue.Full:
                continue

    def _run(self):
        handles = []
        def guard(module, args):
            if threading.current_thread() is not self.thread:
                raise RuntimeError("Ming prefetched conditioner is exclusively worker-owned")
            if self.stop_event.is_set(): raise _Cancelled()
        try:
            native = self.conditioner.native
            modules = [self.conditioner, *native.model.model.layers,
                       *native.connector.model.layers, *native.vision.blocks]
            for module in modules: handles.append(module.register_forward_pre_hook(guard))
            for index in range(self.start, len(self.batches)):
                self._check()
                self.trainer._check_stop_requested()
                pairs = self.batches[index]
                identity = self.identity_fn(pairs)
                with self.preparation_lock:
                    items = load_shared_items(self.trainer, pairs, epoch=self.epoch, batch_position=self.offset+index)
                    source = prepare_record_conditioning(self.trainer.ming_image_01_components, items,
                        resolution=self.trainer.ming_prompt_resolution,
                        cache_context=shared_conditioning_context(self.trainer, pairs))
                self._check()
                condition = reuse_ming_conditioning(self.trainer, source)
                self._check()
                if condition.queries.device.type != "cpu" or condition.direct.device.type != "cpu":
                    raise RuntimeError("Ming CPU prefetch produced non-CPU conditioning")
                self._publish((index, identity, condition))
                del source, items, condition
        except _Cancelled:
            pass
        except BaseException as error:
            self.error = error
        finally:
            for handle in handles: handle.remove()
            self.done.set()

    def take(self, index):
        while not self.stop_event.is_set():
            self.trainer._check_stop_requested()
            self._check()
            try:
                position, identity, condition = self.queue.get(timeout=0.05)
                if position < index: continue
                if position != index: raise RuntimeError("Ming CPU prefetch batch cursor mismatch")
                return identity, condition
            except queue.Empty:
                if self.done.is_set():
                    if self.error is not None: raise self.error
                    raise RuntimeError("Ming CPU prefetch ended before requested batch")
        raise RuntimeError("Ming CPU prefetch has stopped")

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        while True:
            try: self.queue.get_nowait()
            except queue.Empty: break
