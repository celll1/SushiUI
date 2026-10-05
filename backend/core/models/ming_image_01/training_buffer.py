"""Bounded native CPU buffers over the resolved epoch batch suffix."""

import threading
from contextlib import closing, contextmanager, nullcontext

import torch

from .cache import identity_key
from .dataset import (load_shared_items, prepare_shared_items_latents, prepare_record_conditioning,
                      shared_conditioning_context, supervision_identity, verify_record_files)
from .training_residency import grouped_encoder_phase
from .pixel_prefetch import prefetch_pixels


def batch_identity(pairs):
    return identity_key({"version": "ming_batch_buffer_v1", "items": [[getattr(dataset, "unique_id", None),
        supervision_identity(item["_ming_record"]), item.get("caption", ""),
        item.get("bucket_width") or item.get("width"), item.get("bucket_height") or item.get("height"),
        getattr(item.get("_crop_spec"), "crop_box", None)] for item, dataset in pairs]})


class MingBatchBuffer:
    def __init__(self, trainer, batches, *, epoch, batch_offset, latent_interval, text_interval, prefetch_depth=None,
                 progress_callback=None):
        if any(type(v) is not int or v < 1 for v in (latent_interval, text_interval)):
            raise ValueError("Ming swap intervals must be positive integers")
        if any(type(v) is not int or v < 0 for v in (epoch, batch_offset)):
            raise ValueError("Ming buffer coordinates must be nonnegative integers")
        self.trainer, self.batches, self.epoch, self.offset = trainer, batches, epoch, batch_offset
        self.policy = trainer.ming_training_residency
        self.latent_interval, self.text_interval = latent_interval, text_interval
        self.progress_callback = progress_callback
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

    def _progress(self, phase, step, total):
        if self.progress_callback is not None:
            self.progress_callback(phase=phase, step=step, total=total)

    def _fill_latents(self, start):
        self.latents.clear()
        end = min(start+self.latent_interval, len(self.batches))
        total = sum(len(batch) for batch in self.batches[start:end])
        completed = 0
        self._progress("latent_cache", 0, total)
        previous_shape = None
        pixels = prefetch_pixels(self.batches, start, end, self._items, self.trainer._check_stop_requested)
        with grouped_encoder_phase(self.policy, "vae") as grouped, closing(pixels):
            for index, items in pixels:
                self.trainer._check_stop_requested()
                pairs = self.batches[index]
                shape = tuple((item.get("bucket_width") or item.get("width"),
                               item.get("bucket_height") or item.get("height")) for item, _ in pairs)
                if previous_shape is not None and shape != previous_shape and torch.device(self.trainer.device).type == "cuda":
                    # A swap window spans buckets that the outer training loop has not visited yet.
                    torch.cuda.empty_cache()
                previous_shape = shape
                identity = batch_identity(pairs)
                prepared = prepare_shared_items_latents(self.trainer, pairs, items, residency=grouped)
                del items
                prepared.residency = self.policy
                prepared.clean = prepared.clean.detach().cpu()
                if prepared.reference_latents is not None:
                    prepared.reference_latents = [None if value is None else value.detach().cpu()
                                                  for value in prepared.reference_latents]
                self.latents[index] = (identity, prepared)
                completed += len(pairs)
                if (index-start+1) % 10 == 0 or index+1 == end:
                    self._progress("latent_cache", completed, total)

    def _fill_conditions(self, start):
        self.conditions.clear()
        end = min(start+self.text_interval, len(self.batches))
        total = sum(len(batch) for batch in self.batches[start:end])
        completed = 0
        self._progress("text_encoder_cache", 0, total)
        with grouped_encoder_phase(self.policy, "conditioner") as grouped:
            for index in range(start, min(start+self.text_interval, len(self.batches))):
                self.trainer._check_stop_requested()
                pairs = self.batches[index]
                context = shared_conditioning_context(self.trainer, pairs)
                if context is None and all(item["_ming_record"].profile == "generation_edit"
                        and not item["_ming_record"].references for item, _ in pairs):
                    from .prompting import prepare_inputs
                    from .training_data import MingConditioningBatch
                    for item, _ in pairs:
                        verify_record_files(item["_ming_record"])
                    item = pairs[0][0]
                    prepared = prepare_inputs(self.trainer.ming_image_01_components.processor,
                        [item.get("caption", "") for item, _ in pairs], references=[[] for _ in pairs],
                        width=item.get("bucket_width") or item["width"],
                        height=item.get("bucket_height") or item["height"],
                        resolution=self.trainer.ming_prompt_resolution)
                    source = MingConditioningBatch(prepared, self.trainer.ming_image_01_components.conditioner,
                                                   residency=grouped)
                else:
                    source = prepare_record_conditioning(self.trainer.ming_image_01_components,
                        self._items(index, pairs), resolution=self.trainer.ming_prompt_resolution,
                        cache_context=context, residency=grouped)
                condition = source.for_step().detached()
                condition.queries, condition.direct = condition.queries.cpu(), condition.direct.cpu()
                self.conditions[index] = (batch_identity(pairs), condition)
                completed += len(pairs)
                if (index-start+1) % 10 == 0 or index+1 == end:
                    self._progress("text_encoder_cache", completed, total)

    def prepare(self, index, pairs):
        if self.closed or type(index) is not int or not 0 <= index < len(self.batches):
            raise ValueError("Invalid Ming batch-buffer cursor")
        identity = batch_identity(pairs)
        for item, _dataset in pairs:
            verify_record_files(item["_ming_record"])
        frozen = self.trainer.ming_image_01_components.conditioner.cacheable
        latent_due = (self.policy.modes["latent_encoding_mode"] == "swap_onthefly"
                      and (index not in self.latents or self.latents[index][0] != identity))
        condition_due = (frozen and self.policy.modes["text_encoding_mode"] == "swap_onthefly"
                         and (index not in self.conditions or self.conditions[index][0] != identity))
        refill = getattr(self.policy, "continuous_refill", None)
        with self.preparation_lock, refill() if latent_due and condition_due and refill else nullcontext():
            if latent_due:
                self._fill_latents(index)
            if condition_due:
                self._fill_conditions(index)
        with self.preparation_lock:
            if self.policy.modes["latent_encoding_mode"] == "swap_onthefly":
                if index not in self.latents or self.latents[index][0] != identity:
                    self._fill_latents(index)
                stored_identity, prepared = self.latents.pop(index)
                if stored_identity != identity:
                    prepared = prepare_shared_items_latents(self.trainer, pairs, self._items(index, pairs))
            else:
                prepared = prepare_shared_items_latents(self.trainer, pairs, self._items(index, pairs))
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
