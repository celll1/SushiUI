"""Raw-pixel torch Dataset for VAE training.

Deliberately NOT the latent-cache path: VAE training is defined by a live
encode->decode forward on raw pixels (a pre-encoded cache is HARD-REFUSED in
``vae_config``). Bucketing is unnecessary too — a fixed square crop makes every
batch the same shape by construction.

The geometry/normalisation convention is the one from
``base_trainer.encode_image``'s preamble (base_trainer.py:4407-4508): shortest
side scaled up if needed, crop to the target, then ``(rgb/255 - 0.5) * 2`` into
a ``[3, H, W]`` float tensor in [-1, 1]. Those few lines are reproduced here
rather than imported, because ``encode_image`` continues into a
``torch.no_grad()`` VAE forward (base_trainer.py:4545) — precisely the thing a
VAE trainer must not inherit.

Captions are not read: a plain (non-text-conditioned) VAE has no use for them.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# Training datasets routinely contain slightly-truncated JPEGs; the rest of the
# repo's loaders tolerate them rather than crashing a multi-hour run.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image_tensor(
    path: str,
    resolution: int,
    *,
    random_crop: bool = True,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Load one image as a ``[3, resolution, resolution]`` float32 tensor in [-1, 1].

    Aspect ratio is preserved: the image is scaled so the SHORT side reaches
    ``resolution`` (up- or down-scaling as needed), then cropped — randomly when
    ``random_crop`` is set (training), centred otherwise (validation, so the
    held-out metric is deterministic across steps).
    """
    rng = rng or random
    with Image.open(path) as im:
        image = im.convert("RGB")

        w, h = image.size
        scale = resolution / min(w, h)
        if scale != 1.0:
            new_w = max(resolution, int(round(w * scale)))
            new_h = max(resolution, int(round(h * scale)))
            image = image.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h

        max_left, max_top = w - resolution, h - resolution
        if random_crop:
            left = rng.randint(0, max_left) if max_left > 0 else 0
            top = rng.randint(0, max_top) if max_top > 0 else 0
        else:
            left, top = max_left // 2, max_top // 2
        image = image.crop((left, top, left + resolution, top + resolution))

        arr = np.array(image).astype(np.float32) / 255.0
        arr = (arr - 0.5) * 2.0

    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class VaeRawImageDataset(Dataset):
    """Random square crops at ``resolution`` from the shared dataset items.

    ``items`` are the dicts produced by ``train_runner.get_dataset_items_fast``
    (shape at train_runner.py:549-556); only ``image_path`` is used.

    An unreadable/corrupt image is skipped by walking forward to the next index
    rather than aborting the run — a single bad file in a 100k-item dataset must
    not kill a multi-hour fine-tune. Repeated failures are reported once each.
    """

    def __init__(
        self,
        items: List[Dict],
        resolution: int,
        *,
        random_crop: bool = True,
        seed: int = 0,
    ):
        self.paths = [it["image_path"] for it in items if it.get("image_path")]
        if not self.paths:
            raise ValueError("VaeRawImageDataset: no items with an image_path")
        self.resolution = int(resolution)
        self.random_crop = bool(random_crop)
        self.seed = int(seed)
        self._reported_failures = set()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        n = len(self.paths)
        # Per-item RNG so a worker-parallel loader is reproducible for a seed.
        rng = random.Random((self.seed * 1000003) ^ index)
        for attempt in range(min(n, 32)):
            i = (index + attempt) % n
            path = self.paths[i]
            try:
                return load_image_tensor(
                    path, self.resolution,
                    random_crop=self.random_crop, rng=rng,
                )
            except Exception as e:
                if path not in self._reported_failures:
                    self._reported_failures.add(path)
                    print(f"[VaeDataset] Skipping unreadable image {path}: "
                          f"{type(e).__name__}: {e}")
        raise RuntimeError(
            f"VaeRawImageDataset: 32 consecutive images failed to load starting "
            f"at index {index}; the dataset paths are probably invalid."
        )


def make_validation_batch(
    items: List[Dict],
    resolution: int,
    count: int,
) -> torch.Tensor:
    """Deterministic held-out batch: the LAST ``count`` items, centre-cropped.

    Taking them from the tail (and excluding them from the training set, see
    ``VaeTrainer._split_items``) keeps the validation signal honest without
    needing a separate dataset registration.
    """
    picked = items[-count:] if count < len(items) else items
    tensors = []
    for it in picked:
        path = it.get("image_path")
        if not path:
            continue
        try:
            tensors.append(load_image_tensor(path, resolution, random_crop=False))
        except Exception as e:
            print(f"[VaeDataset] Validation image skipped {path}: "
                  f"{type(e).__name__}: {e}")
    if not tensors:
        raise ValueError("make_validation_batch: no loadable validation images")
    return torch.stack(tensors, dim=0)
