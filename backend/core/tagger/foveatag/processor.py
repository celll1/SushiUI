"""Native multi-view inputs for the shared tagger data lifecycle."""

from __future__ import annotations

import hashlib
from dataclasses import fields
from typing import Sequence

import torch
from PIL import Image

from .batch import FoveaViewTensorBatch, collate_view_batches
from .views import ViewConfig, build_views


class FoveaProcessor:
    def __init__(self, config: ViewConfig, *, seed: int, training: bool,
                 max_source_pixels: int | None = None) -> None:
        if max_source_pixels is not None and max_source_pixels <= 0:
            raise ValueError("native source pixel cap must be positive")
        self.max_source_pixels = max_source_pixels
        self.config = config
        self.seed = int(seed)
        self.training = bool(training)
        self.epoch = 0
        self.is_native_fovea_processor = True

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def image_seed(self, image, sample_index=None):
        if self.training and sample_index is not None:
            identity = f"{self.seed}:{self.epoch}:{sample_index}:{image.width}:{image.height}"
            digest = hashlib.sha256(identity.encode("utf-8")).digest()
            return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
        if self.training:
            return int(torch.randint(0, 2**63 - 1, ()).item())
        digest = hashlib.sha256(image.resize((8, 8)).convert("RGB").tobytes()).digest()
        return (int.from_bytes(digest[:8], "little") ^ self.seed) & ((1 << 63) - 1)

    def __call__(self, *, images: Sequence[Image.Image], return_tensors: str,
                 sample_index: int | None = None) -> dict:
        if return_tensors != "pt":
            raise ValueError("FoveaProcessor only supports PyTorch tensors")
        views = []
        for image in images:
            if self.max_source_pixels is not None and image.width * image.height > self.max_source_pixels:
                raise ValueError("native source image exceeds the pixel cap")
            seed = self.image_seed(image, sample_index)
            views.append(build_views(image, self.config, seed=seed, training=self.training))
        return {"fovea_views": collate_view_batches(views)}


def concatenate_view_tensors(batches: Sequence[FoveaViewTensorBatch]) -> FoveaViewTensorBatch:
    if not batches:
        raise ValueError("at least one native view batch is required")
    expected = batches[0]
    for batch in batches:
        for field in fields(FoveaViewTensorBatch):
            value = getattr(batch, field.name)
            if value.shape[1:] != getattr(expected, field.name).shape[1:]:
                raise ValueError("native view tensor shapes differ within one batch")
    return FoveaViewTensorBatch(**{
        field.name: torch.cat([getattr(batch, field.name) for batch in batches], dim=0)
        for field in fields(FoveaViewTensorBatch)
    })
