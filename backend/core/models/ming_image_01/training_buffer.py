"""Bounded native CPU buffers over the resolved epoch batch suffix."""

import threading
from contextlib import contextmanager

from .cache import identity_key
from .dataset import (load_shared_items, prepare_shared_items_latents, prepare_record_conditioning,
                      shared_conditioning_context, supervision_identity, verify_record_content)
from .training_residency import grouped_encoder_phase


def batch_identity(pairs):
    return identity_key({"version": "ming_batch_buffer_v1", "items": [[getattr(dataset, "unique_id", None),
        supervision_identity(item["_ming_record"]), item.get("caption", ""),
        item.get("bucket_width") or item.get("width"), item.get("bucket_height") or item.get("height"),
        getattr(item.get("_crop_spec"), "crop_box", None)] for item, dataset in pairs]})


class MingBatchBuffer:
    def __init__(self, trainer, batches, *, epoch, batch_offset, latent_interval, text_interval, prefetch_depth=None):
        if any(type(v) is not int or v < 1 for v in (latent_interval, text_interval)):
            raise ValueError("Ming swap intervals must be positive integers")
        if any(type(v) is not int or v < 0 for v in (epoch, batch_offset)):
            raise ValueError("Ming buffer coordinates must be nonnegative integers")
        self.trainer, self.batches, self.epoch, self.offset = trainer, batches, epoch, batch_offset
        self.policy = trainer.ming_training_residency
        self.latent_interval, self.text_interval = latent_interval, text_interval
        self.latents, self.conditions = {}, {}
        self.closed = False
        self.next_cursor = 0
        self.preparation_lock = threading.RLock()
        self.prefetcher = None
        if self.policy.modes["text_encoding_mode"] == "cpu_prefetch":
            from api.param_defaults import TRAINING_DEFAULTS
            self.prefetch_depth = (TRAINING_DEFAULTS["text_encoding_prefetch_depth"]
                                   if prefetch_depth is None else prefetch_depth)
            self._start_prefetch(0)

    def _start_prefetch(self, start):
        from .cpu_prefetch import MingCpuPrefetch
        if self.prefetcher is not None: self.prefetcher.stop()
        self.prefetcher = MingCpuPrefetch(self.trainer, self.batches, epoch=self.epoch,
            batch_offset=self.offset, start=start, depth=self.prefetch_depth, identity_fn=batch_identity, preparation_lock=self.preparation_lock)

    def _items(self, index, pairs):
        return load_shared_items(self.trainer, pairs, epoch=self.epoch, batch_position=self.offset+index)

    def _fill_latents(self, start):
        self.latents.clear()
        with grouped_encoder_phase(self.policy, "vae") as grouped:
            for index in range(start, min(start+self.latent_interval, len(self.batches))):
                self.trainer._check_stop_requested()
                pairs = self.batches[index]
                identity = batch_identity(pairs)
                prepared = prepare_shared_items_latents(self.trainer, pairs, self._items(index, pairs), residency=grouped)
                prepared.residency = self.policy
                prepared.clean = prepared.clean.detach().cpu()
                if prepared.reference_latents is not None:
                    prepared.reference_latents = [None if value is None else value.detach().cpu()
                                                  for value in prepared.reference_latents]
                self.latents[index] = (identity, prepared)

    def _fill_conditions(self, start):
        self.conditions.clear()
        with grouped_encoder_phase(self.policy, "conditioner") as grouped:
            for index in range(start, min(start+self.text_interval, len(self.batches))):
                self.trainer._check_stop_requested()
                pairs = self.batches[index]
                source = prepare_record_conditioning(self.trainer.ming_image_01_components,
                    self._items(index, pairs), resolution=self.trainer.ming_prompt_resolution,
                    cache_context=shared_conditioning_context(self.trainer, pairs), residency=grouped)
                condition = source.for_step().detached()
                condition.queries, condition.direct = condition.queries.cpu(), condition.direct.cpu()
                self.conditions[index] = (batch_identity(pairs), condition)

    def prepare(self, index, pairs):
        if self.closed or type(index) is not int or not 0 <= index < len(self.batches):
            raise ValueError("Invalid Ming batch-buffer cursor")
        identity = batch_identity(pairs)
        for item, _dataset in pairs:
            verify_record_content(item["_ming_record"])
        with self.preparation_lock:
            if self.policy.modes["latent_encoding_mode"] == "swap_onthefly":
                if index not in self.latents or self.latents[index][0] != identity:
                    self._fill_latents(index)
                stored_identity, prepared = self.latents.pop(index)
                if stored_identity != identity:
                    prepared = prepare_shared_items_latents(self.trainer, pairs, self._items(index, pairs))
            else:
                prepared = prepare_shared_items_latents(self.trainer, pairs, self._items(index, pairs))
        frozen = self.trainer.ming_image_01_components.conditioner.cacheable
        if frozen and self.policy.modes["text_encoding_mode"] == "swap_onthefly":
            if index not in self.conditions or self.conditions[index][0] != identity:
                self._fill_conditions(index)
            if index in self.conditions and self.conditions[index][0] == identity:
                prepared.frozen_conditioning = self.conditions.pop(index)[1]
        elif self.policy.modes["text_encoding_mode"] == "cpu_prefetch":
            if not frozen:
                raise ValueError("Ming CPU prefetch producer became trainable")
            stored_identity, condition = self.prefetcher.take(index)
            if stored_identity != identity:
                self._start_prefetch(index)
                stored_identity, condition = self.prefetcher.take(index)
                if stored_identity != identity:
                    raise RuntimeError("Ming CPU prefetch batch changed during preparation")
            prepared.frozen_conditioning = condition
        elif not frozen:
            self.conditions.clear()
        self.next_cursor = index + 1
        return prepared

    @contextmanager
    def suspend_prefetch(self):
        worker = self.prefetcher
        if worker is not None:
            worker.stop()
            self.prefetcher = None
            if worker.error is not None:
                raise worker.error
        try:
            yield
        finally:
            if worker is not None and not self.closed and self.next_cursor < len(self.batches):
                self._start_prefetch(self.next_cursor)

    def close(self):
        if self.prefetcher is not None:
            self.prefetcher.stop()
            self.prefetcher = None
        self.latents.clear()
        self.conditions.clear()
        self.closed = True


def close_training_buffer(trainer):
    buffer = getattr(trainer, "ming_batch_buffer", None)
    if buffer is not None:
        buffer.close()
        trainer.ming_batch_buffer = None
