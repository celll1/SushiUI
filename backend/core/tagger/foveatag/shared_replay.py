"""Indexed views and explicit permutations for native shared-trainer resume."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import DataLoader, Subset

from ..tagger_dataset import tagger_collate_fn


@dataclass
class NativeReplayCursor:
    base_batches: int = 0
    valid_base_batches: int = 0
    last_batch_idx: int = -1
    pending_injection: bool = False

    def complete(self, batch_idx, *, is_injection, readable, injection_interval):
        if is_injection:
            self.pending_injection = False
        else:
            self.base_batches += 1
            self.valid_base_batches += int(readable)
            self.pending_injection = bool(readable and self.valid_base_batches % injection_interval == 0)
        self.last_batch_idx = batch_idx

    def to_dict(self):
        return asdict(self)


def epoch_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(f"fovea-sampler-v1:{seed}:{epoch}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def native_dataset_fingerprint(dataset):
    indices = list(range(len(dataset)))
    source = dataset
    while isinstance(source, Subset):
        indices = [int(source.indices[index]) for index in indices]
        source = source.dataset
    digest = hashlib.sha256()
    for index in indices:
        if hasattr(source, "source_path_for_index"):
            path = source.source_path_for_index(index)
        elif hasattr(source, "_samples"):
            path = source._samples[index][0]
        elif hasattr(source, "items"):
            item = source.items[index]
            path = item.get("image_path", "") if isinstance(item, dict) else getattr(item, "image_path", "")
        else:
            raise ValueError("native dataset lacks source identity metadata")
        digest.update(f"{index}:{path}\n".encode("utf-8"))
    return {"total_item_count": len(indices), "image_paths_hash": digest.hexdigest()}


def native_epoch_loader(loader, *, seed: int, epoch: int, base_batches: int):
    if base_batches < 0:
        raise ValueError("base batch cursor must be non-negative")
    source = loader.dataset
    while hasattr(source, "dataset"):
        source = source.dataset
    processor = getattr(source, "processor", None)
    if processor is not None and getattr(processor, "is_native_fovea_processor", False):
        processor.set_epoch(epoch)
    batch_size = loader.batch_size or 1
    total = (len(loader.dataset) + batch_size - 1) // batch_size
    if base_batches > total:
        raise ValueError("base batch cursor exceeds the epoch")
    permutation = torch.randperm(len(loader.dataset), generator=torch.Generator().manual_seed(epoch_seed(seed, epoch))).tolist()
    remaining = Subset(loader.dataset, permutation[base_batches * batch_size:])
    return DataLoader(
        remaining, batch_size=batch_size, shuffle=False, num_workers=loader.num_workers,
        collate_fn=tagger_collate_fn, pin_memory=False,
        generator=torch.Generator().manual_seed(epoch_seed(seed, epoch)),
    )
