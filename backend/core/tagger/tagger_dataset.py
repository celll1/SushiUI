"""
Dataset for SigLIP2 tagger training.

Loads images from sushiUI DatasetItems + DatasetCaption(is_tags_format=True),
produces (pixel_values, pixel_attention_mask, spatial_shapes, label, loss_mask).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoProcessor

from .tag_vocabulary import (
    QUALITY_TAG_GROUPS,
    RATING_TAGS,
    TagVocabulary,
    normalize_tag,
)


class TaggerDataset(Dataset):
    """PyTorch Dataset for tagger training.

    Each item returns:
        pixel_values         : Tensor [num_patches, 768]     (NaFlex patch-flattened)
        pixel_attention_mask : Tensor [num_patches]           (int32)
        spatial_shapes       : Tensor [2]                     (int64, height x width patches)
        label                : Tensor [num_tags]              (float32, multi-hot)
        loss_mask            : Tensor [num_tags]              (float32, 0=ignore 1=use)
    """

    def __init__(
        self,
        dataset_ids: List[int],
        vocabulary: TagVocabulary,
        datasets_db,
        processor: AutoProcessor,
        caption_types: Optional[List[str]] = None,
        alias_resolver=None,
    ) -> None:
        """
        Parameters
        ----------
        dataset_ids    : list of Dataset.id to include
        vocabulary     : TagVocabulary built from the same datasets
        datasets_db    : SQLAlchemy session for datasets.db
        processor      : SigLIP2 AutoProcessor (handles NaFlex preprocessing)
        caption_types  : restrict to these caption_type values (None = all tags-format)
        alias_resolver : optional TagAliasResolver; when provided, deprecated
                         tags are resolved to canonical form during label construction
        """
        self.vocabulary = vocabulary
        self.processor = processor
        self.num_tags = vocabulary.num_tags
        self._alias_resolver = alias_resolver

        # Detect NaFlex vs standard by probing the processor output
        _probe = processor(images=[Image.new("RGB", (64, 64))], return_tensors="pt")
        self.is_naflex = "pixel_attention_mask" in _probe and "spatial_shapes" in _probe
        print(f"[TaggerDataset] Processor mode: {'NaFlex' if self.is_naflex else 'standard (fixed resolution)'}")

        self._samples: List[Tuple[str, List[str]]] = []  # (image_path, [tag, ...])
        self._build_samples(dataset_ids, datasets_db, caption_types)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_samples(
        self,
        dataset_ids: List[int],
        datasets_db,
        caption_types: Optional[List[str]],
    ) -> None:
        from database.models import DatasetItem, DatasetCaption

        for dataset_id in dataset_ids:
            print(f"[TaggerDataset] Loading dataset_id={dataset_id}...")

            # Bulk-load all items for this dataset in one query
            items = (
                datasets_db.query(DatasetItem)
                .filter(DatasetItem.dataset_id == dataset_id)
                .all()
            )
            print(f"[TaggerDataset]   {len(items)} items loaded")

            # Collect item ids that have a valid image path.
            # Record a small sample of skipped paths (capped to keep the log readable).
            item_path_map: Dict[int, str] = {}
            _MAX_SHOW = 5
            skipped_examples: List[str] = []
            skipped_no_path = 0
            for item in tqdm(items, desc=f"  Checking files (dataset {dataset_id})", unit="item", leave=False):
                if item.image_path and os.path.isfile(item.image_path):
                    item_path_map[item.id] = item.image_path
                else:
                    if not item.image_path:
                        skipped_no_path += 1
                        if len(skipped_examples) < _MAX_SHOW:
                            skipped_examples.append(f"<item_id={item.id} (no image_path)>")
                    elif len(skipped_examples) < _MAX_SHOW:
                        skipped_examples.append(item.image_path)
            valid_item_ids = list(item_path_map.keys())
            skipped = len(items) - len(valid_item_ids)
            if skipped:
                print(f"[TaggerDataset]   {skipped} items skipped (missing image files)")
                for ex in skipped_examples:
                    print(f"[TaggerDataset]     - {ex}")
                if skipped > _MAX_SHOW:
                    print(f"[TaggerDataset]     ... and {skipped - _MAX_SHOW} more")

            if not valid_item_ids:
                continue

            # Bulk-load captions in chunks to stay within SQLite's 999-variable limit
            CHUNK = 500
            from collections import defaultdict
            captions_by_item: Dict[int, list] = defaultdict(list)
            total_captions = 0
            n_chunks = (len(valid_item_ids) + CHUNK - 1) // CHUNK
            for i in tqdm(range(0, len(valid_item_ids), CHUNK), total=n_chunks,
                          desc=f"  Loading captions (dataset {dataset_id})", unit="chunk", leave=False):
                chunk_ids = valid_item_ids[i:i + CHUNK]
                q = (
                    datasets_db.query(DatasetCaption)
                    .filter(
                        DatasetCaption.item_id.in_(chunk_ids),
                        DatasetCaption.is_tags_format == True,  # noqa: E712
                    )
                )
                if caption_types:
                    q = q.filter(DatasetCaption.caption_type.in_(caption_types))
                for cap in q.all():
                    captions_by_item[cap.item_id].append(cap)
                    total_captions += 1
            print(f"[TaggerDataset]   {total_captions} tag captions loaded")

            # Build samples
            for item_id in tqdm(valid_item_ids, desc=f"  Building samples (dataset {dataset_id})", unit="item", leave=False):
                item_captions = captions_by_item.get(item_id)
                if not item_captions:
                    continue
                tags: List[str] = []
                for caption in item_captions:
                    tags.extend(self._extract_tags(caption))
                if tags:
                    self._samples.append((item_path_map[item_id], tags))

            print(f"[TaggerDataset]   {len(self._samples)} samples so far")

    def _extract_tags(self, caption) -> List[str]:
        raw_tags: List[str] = []
        if caption.tag_data:
            try:
                raw = json.loads(caption.tag_data) if isinstance(caption.tag_data, str) else caption.tag_data
                if isinstance(raw, list):
                    raw_tags = [r["tag"] for r in raw if isinstance(r, dict) and "tag" in r]
            except (json.JSONDecodeError, TypeError):
                pass
        if not raw_tags and caption.content:
            raw_tags = [t.strip() for t in caption.content.split(",") if t.strip()]
        if self._alias_resolver:
            return [self._alias_resolver.resolve(t) for t in raw_tags]
        return [normalize_tag(t) for t in raw_tags]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        image_path, tags = self._samples[idx]
        try:
            image = _load_image(image_path)
            inputs = self.processor(images=[image], return_tensors="pt")
        except Exception as e:
            print(f"[TaggerDataset] Skipping corrupt image {image_path}: {e}")
            return None  # filtered out by tagger_collate_fn

        if self.is_naflex:
            pixel_values         = inputs["pixel_values"].squeeze(0)          # [num_patches, patch_dim]
            pixel_attention_mask = inputs["pixel_attention_mask"].squeeze(0)  # [num_patches]
            spatial_shapes       = inputs["spatial_shapes"].squeeze(0)        # [2]
        else:
            pixel_values         = inputs["pixel_values"].squeeze(0)          # [3, H, W]
            pixel_attention_mask = torch.zeros(0, dtype=torch.int32)          # sentinel → [B, 0] after collate
            spatial_shapes       = torch.zeros(0, dtype=torch.int64)          # sentinel

        label, loss_mask = self._build_label_and_mask(tags)
        return pixel_values, pixel_attention_mask, spatial_shapes, label, loss_mask

    # ------------------------------------------------------------------
    # Label / mask construction
    # ------------------------------------------------------------------

    def _build_label_and_mask(
        self, tags: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        voc = self.vocabulary
        label     = torch.zeros(self.num_tags, dtype=torch.float32)
        loss_mask = torch.ones(self.num_tags,  dtype=torch.float32)

        # Set positive labels
        tag_set = set(tags)
        for tag in tag_set:
            if tag in voc.tag_to_idx:
                label[voc.tag_to_idx[tag]] = 1.0

        # Rating tags: mask out if none present in this sample
        has_rating = any(normalize_tag(r) in tag_set for r in RATING_TAGS)
        if not has_rating:
            for idx in voc.rating_indices:
                loss_mask[idx] = 0.0

        # Quality tags: mask ALL quality groups only when NO quality tag is present.
        # When any quality tag IS present, leave all quality groups unmasked so that
        # cross-group negative labels are trained (e.g. "best quality" sample must also
        # teach the model that "bad quality"=0).  Quality groups are mutually exclusive
        # by definition, so this negative signal is always correct.
        all_quality_norms = {
            normalize_tag(t)
            for gtags in QUALITY_TAG_GROUPS.values()
            for t in gtags
        }
        has_any_quality = any(t in tag_set for t in all_quality_norms)
        if not has_any_quality:
            for group_indices in voc.quality_indices.values():
                for idx in group_indices:
                    loss_mask[idx] = 0.0

        return label, loss_mask


# ------------------------------------------------------------------
# Collate function — supports both NaFlex (variable patches) and standard
# ------------------------------------------------------------------

def tagger_collate_fn(batch):
    """Collate batch items.

    - Filters out None entries (corrupt/unreadable images).
    - Handles NaFlex (variable num_patches per image) by padding to max.
    - Handles standard fixed-resolution models (pam/ss are sentinel zero-tensors).
    Returns None if the entire batch is corrupt.
    """
    valid = [item for item in batch if item is not None]
    if not valid:
        return None

    pixel_values_list, masks_list, shapes_list, labels_list, loss_masks_list = zip(*valid)

    # Stack pixel_values; for NaFlex with varying patch counts, pad to max.
    try:
        pixel_values         = torch.stack(pixel_values_list)
        pixel_attention_mask = torch.stack(masks_list)
        spatial_shapes       = torch.stack(shapes_list)
    except RuntimeError:
        # NaFlex: different num_patches across images — pad to the maximum.
        max_patches = max(p.shape[0] for p in pixel_values_list)
        patch_dim   = pixel_values_list[0].shape[1]
        B = len(pixel_values_list)
        pixel_values         = torch.zeros(B, max_patches, patch_dim)
        pixel_attention_mask = torch.zeros(B, max_patches, dtype=torch.int32)
        for i, (pv, pm) in enumerate(zip(pixel_values_list, masks_list)):
            n = pv.shape[0]
            pixel_values[i, :n]         = pv
            pixel_attention_mask[i, :n] = pm
        spatial_shapes = torch.stack(shapes_list)

    return (
        pixel_values,
        pixel_attention_mask,
        spatial_shapes,
        torch.stack(labels_list),
        torch.stack(loss_masks_list),
    )


# ------------------------------------------------------------------
# Image loading helper
# ------------------------------------------------------------------

def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    img.load()  # force full decode here so truncation raises OSError before reaching the processor
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img
