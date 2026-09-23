"""Image loading, views and batching for style training (design §6, §7)."""

from __future__ import annotations

import contextlib
from typing import Iterator, NamedTuple, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.tagger.tagger_dataset import _load_image

from .style_ingest import ItemPaths
from .style_sampler import StyleBatchSampler
from .style_views import apply_block_masks, eval_view_entropy, train_view_entropy


class StyleSampleKey(NamedTuple):
    row: int                          # StyleIndex row
    view_entropy: tuple[int, ...]     # SeedSequence entropy of the masked view
    include_hsic: bool = False


class StyleSample(NamedTuple):
    pixel_values: torch.Tensor        # [N, patch_dim]
    attention_mask: torch.Tensor      # [N] global view
    masked_attention_mask: torch.Tensor  # [N] masked view, same pixels
    spatial_shapes: torch.Tensor      # [2]
    row: int
    hsic_pixel_values: Optional[torch.Tensor] = None
    hsic_attention_mask: Optional[torch.Tensor] = None
    hsic_spatial_shapes: Optional[torch.Tensor] = None


class StyleBatchTensors(NamedTuple):
    pixel_values: torch.Tensor           # [B, N, patch_dim]
    attention_mask: torch.Tensor         # [B, N]
    masked_attention_mask: torch.Tensor  # [B, N]
    spatial_shapes: torch.Tensor         # [B, 2]
    rows: torch.Tensor                   # [B] int64; unreadable images are absent
    hsic_pixel_values: Optional[torch.Tensor] = None
    hsic_attention_mask: Optional[torch.Tensor] = None
    hsic_spatial_shapes: Optional[torch.Tensor] = None


class StyleImageDataset(Dataset):
    """Map-style dataset keyed by `StyleSampleKey`, so a sample depends only on its key and exact resume
    reproduces masked views whatever the worker split."""

    def __init__(self, paths: ItemPaths, mask_ratios: np.ndarray, processor, max_num_patches: int,
                 hsic_processor=None, hsic_max_num_patches: int = 256, hsic_is_naflex: bool = True) -> None:
        if len(paths) != mask_ratios.shape[0]:
            raise ValueError(f"{len(paths)} paths but {mask_ratios.shape[0]} mask ratios")
        # Only what __getitem__ reads, since the dataset is pickled into every spawned worker.
        self._offsets = paths.offsets
        self._blob = paths.blob
        self.mask_ratios = np.asarray(mask_ratios, dtype=np.float64)
        self.processor = processor
        self.max_num_patches = int(max_num_patches)
        self.hsic_processor = hsic_processor
        self.hsic_max_num_patches = int(hsic_max_num_patches)
        self.hsic_is_naflex = bool(hsic_is_naflex)

    def __len__(self) -> int:
        return int(self._offsets.size - 1)

    def path(self, row: int) -> str:
        return self._blob[self._offsets[row]:self._offsets[row + 1]].tobytes().decode("utf-8")

    def __getitem__(self, key: StyleSampleKey) -> Optional[StyleSample]:
        path = self.path(key.row)
        try:
            image = _load_image(path)
            inputs = self.processor(images=[image], max_num_patches=self.max_num_patches, return_tensors="pt")
        except Exception as e:  # noqa: BLE001 - one unreadable file must not end the run
            print(f"[StyleDataset] Skipping unreadable image {path}: {e}")
            return None
        if "pixel_attention_mask" not in inputs or "spatial_shapes" not in inputs:
            raise ValueError("the style encoder needs a NaFlex image processor (pixel_attention_mask, spatial_shapes)")
        mask = inputs["pixel_attention_mask"]
        shapes = inputs["spatial_shapes"]
        rng = np.random.default_rng(np.random.SeedSequence(list(key.view_entropy)))
        masked = apply_block_masks(mask, shapes, self.mask_ratios[key.row:key.row + 1], rng)
        if key.include_hsic:
            if self.hsic_processor is None:
                raise RuntimeError("HSIC processor was not attached to the style dataset")
            kwargs = {"images": [image], "return_tensors": "pt"}
            if self.hsic_is_naflex:
                kwargs["max_num_patches"] = self.hsic_max_num_patches
            hsic = self.hsic_processor(**kwargs)
            hpv = hsic["pixel_values"][0]
            ham = hsic["pixel_attention_mask"][0] if self.hsic_is_naflex else None
            hss = hsic["spatial_shapes"][0] if self.hsic_is_naflex else None
        else:
            hpv = ham = hss = None
        return StyleSample(inputs["pixel_values"][0], mask[0], masked[0], shapes[0], int(key.row),
                           hpv, ham, hss)


def style_collate(samples: Sequence[Optional[StyleSample]]) -> Optional[StyleBatchTensors]:
    valid = [s for s in samples if s is not None]
    if not valid:
        return None
    n = max(s.pixel_values.shape[0] for s in valid)
    b = len(valid)
    pixel_values = torch.zeros(b, n, valid[0].pixel_values.shape[1], dtype=valid[0].pixel_values.dtype)
    mask = torch.zeros(b, n, dtype=valid[0].attention_mask.dtype)
    masked = torch.zeros_like(mask)
    for i, s in enumerate(valid):
        length = s.pixel_values.shape[0]
        pixel_values[i, :length] = s.pixel_values
        mask[i, :length] = s.attention_mask
        masked[i, :length] = s.masked_attention_mask
    hpv = ham = hss = None
    if valid[0].hsic_pixel_values is not None:
        if valid[0].hsic_attention_mask is None:
            hpv = torch.stack([s.hsic_pixel_values for s in valid])
        else:
            h_n = max(s.hsic_pixel_values.shape[0] for s in valid)
            hpv = valid[0].hsic_pixel_values.new_zeros((b, h_n, valid[0].hsic_pixel_values.shape[-1]))
            ham = valid[0].hsic_attention_mask.new_zeros((b, h_n))
            for i, sample in enumerate(valid):
                length = sample.hsic_pixel_values.shape[0]
                hpv[i, :length] = sample.hsic_pixel_values
                ham[i, :length] = sample.hsic_attention_mask
            hss = torch.stack([s.hsic_spatial_shapes for s in valid])
    return StyleBatchTensors(pixel_values, mask, masked, torch.stack([s.spatial_shapes for s in valid]),
                             torch.tensor([s.row for s in valid], dtype=torch.int64), hpv, ham, hss)


class TrainBatchKeys:
    """Batch sampler for steps [start_step, stop_step) of one epoch: step s is `sampler.batch(epoch, s)`,
    flattened artist-major, with position-indexed view entropy. Iterated in the main process, so
    `set_range` between iterations lets one persistent-worker DataLoader serve every segment."""

    def __init__(self, sampler: StyleBatchSampler, view_seed: int, epoch: int, start_step: int,
                 stop_step: int, include_hsic: bool = False) -> None:
        self.sampler = sampler
        self.view_seed = int(view_seed)
        self.include_hsic = include_hsic
        self.set_range(epoch, start_step, stop_step)

    def set_range(self, epoch: int, start_step: int, stop_step: int) -> None:
        if not 0 <= start_step <= stop_step <= self.sampler.epoch_length:
            raise ValueError(f"steps [{start_step}, {stop_step}) outside epoch of length "
                             f"{self.sampler.epoch_length}")
        self.epoch = int(epoch)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)

    def __len__(self) -> int:
        return self.stop_step - self.start_step

    def __iter__(self) -> Iterator[list[StyleSampleKey]]:
        for step in range(self.start_step, self.stop_step):
            rows = self.sampler.batch(self.epoch, step).rows.reshape(-1)
            yield [StyleSampleKey(int(row), train_view_entropy(self.view_seed, self.epoch, step, position),
                                  self.include_hsic)
                   for position, row in enumerate(rows)]


class EvalBatchKeys:
    """Fixed-order batches over `rows`; the masked view of an item is a function of its item key only."""

    def __init__(self, rows: np.ndarray, item_keys: np.ndarray, view_seed: int, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.rows = np.asarray(rows, dtype=np.int64)
        self.item_keys = np.asarray(item_keys, dtype=np.int64)
        self.view_seed = int(view_seed)
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return -(-self.rows.size // self.batch_size)

    def __iter__(self) -> Iterator[list[StyleSampleKey]]:
        for start in range(0, self.rows.size, self.batch_size):
            rows = self.rows[start:start + self.batch_size]
            yield [StyleSampleKey(int(r), eval_view_entropy(self.view_seed, int(self.item_keys[r]))) for r in rows]


class SwitchableBatchKeys:
    """Batch sampler of one persistent DataLoader that serves both training segments and evaluations:
    each `iter(loader)` draws from the source set at that moment, so no second worker set is spawned."""

    def __init__(self, source) -> None:
        self.source = source

    @contextlib.contextmanager
    def serving(self, source):
        previous, self.source = self.source, source
        try:
            yield
        finally:
            self.source = previous

    def __len__(self) -> int:
        return len(self.source)

    def __iter__(self):
        return iter(self.source)


def make_loader(dataset: StyleImageDataset, batch_keys, num_workers: int, persistent: bool = False) -> DataLoader:
    # pin_memory=False: the pin thread would not inherit the trainer thread's set_device (see tagger).
    return DataLoader(dataset, batch_sampler=batch_keys, collate_fn=style_collate, num_workers=int(num_workers),
                      pin_memory=False, persistent_workers=persistent and num_workers > 0)
