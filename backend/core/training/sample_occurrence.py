"""Stable sample exposures with lazily resolved, thread-local captions."""

from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
import inspect
import json
import random


CAPTION_RNG_VERSION = 1
_caption_context = ContextVar("caption_occurrence", default=None)


def stable_seed(*parts):
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()
    return int.from_bytes(sha256(encoded).digest()[:8], "big")


def caption_random(operation=None):
    context = _caption_context.get()
    if context is None:
        return random
    seed, rng = context
    return random.Random(stable_seed(seed, operation)) if operation else rng


@contextmanager
def caption_context(seed):
    token = _caption_context.set((seed, random.Random(seed)))
    try:
        yield
    finally:
        _caption_context.reset(token)


def resolve_caption(item, dataset, exposure):
    from .caption_processor import (
        apply_caption_dropout, process_caption, process_caption_with_tag_data,
    )
    config = item.get("_caption_processing_config", getattr(dataset, "caption_config", {}))
    seed = stable_seed(CAPTION_RNG_VERSION, exposure["seed"], exposure["epoch"],
                       exposure["source"], exposure["id"])
    raw = item.get("raw_caption", item.get("caption", "")) or ""
    with caption_context(seed):
        if not item.get("is_tags_format", True):
            return apply_caption_dropout(raw, config.get("caption_dropout_rate", 0.0))
        tags = item.get("tag_data")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (TypeError, ValueError):
                tags = None
        if tags:
            token_rate = config.get("token_dropout_rate")
            if token_rate:
                keep = config.get("keep_tokens", 0)
                token_rng = caption_random("token_dropout")
                tags = [tag for index, tag in enumerate(tags)
                        if index < keep or token_rng.random() >= token_rate]
            return process_caption_with_tag_data(tags, exposure["epoch"], item["image_path"], config)
        allowed = inspect.signature(process_caption).parameters
        params = {key: value for key, value in config.items()
                  if key in allowed and key not in ("caption", "epoch_num", "item_path")}
        return process_caption(raw, epoch_num=exposure["epoch"], item_path=item["image_path"], **params)


class OccurrenceItem(dict):
    """A shallow item view; encoder windows own the resolved caption strings."""

    def __init__(self, item, dataset, exposure):
        super().__init__(item)
        self["_sample_occurrence"] = exposure
        self._dataset = dataset

    def __getitem__(self, key):
        if key == "caption":
            # Read the stored legacy caption without recursively resolving it.
            source = dict(self)
            return resolve_caption(source, self._dataset, dict.__getitem__(self, "_sample_occurrence"))
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key == "caption" or key in self else default


def make_occurrence(pair, *, epoch, seed, segment, index, kind):
    item, dataset = pair
    source = [str(dataset.unique_id), str(item["image_path"])]
    exposure = {"source": source, "epoch": epoch, "seed": seed, "segment": segment,
                "index": index, "kind": kind, "id": f"{kind}:{index}"}
    return OccurrenceItem(item, dataset, exposure), dataset


def text_buffer_key(item):
    exposure = item.get("_sample_occurrence")
    if exposure is None:
        return item["image_path"]
    return (*exposure["source"], exposure["id"])


def occurrence_batches(batches, device):
    """Isolate exposure draws from encoder prefill and resume preparation."""
    import numpy as np
    import torch
    target = torch.device(device)
    devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == "cuda" else []
    for batch in batches:
        exposures = [item.get("_sample_occurrence") for item, _ in batch]
        if not any(exposures):
            yield batch
            continue
        seed = stable_seed("training-exposure-v1", exposures)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                yield batch
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
