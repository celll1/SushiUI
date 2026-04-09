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
    ) -> None:
        """
        Parameters
        ----------
        dataset_ids   : list of Dataset.id to include
        vocabulary    : TagVocabulary built from the same datasets
        datasets_db   : SQLAlchemy session for datasets.db
        processor     : SigLIP2 AutoProcessor (handles NaFlex preprocessing)
        caption_types : restrict to these caption_type values (None = all tags-format)
        """
        self.vocabulary = vocabulary
        self.processor = processor
        self.num_tags = vocabulary.num_tags

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
        from backend.database.models import DatasetItem, DatasetCaption

        for dataset_id in dataset_ids:
            items = (
                datasets_db.query(DatasetItem)
                .filter(DatasetItem.dataset_id == dataset_id)
                .all()
            )
            for item in items:
                if not item.image_path or not os.path.isfile(item.image_path):
                    continue

                tags: List[str] = []
                for caption in item.captions:
                    if not caption.is_tags_format:
                        continue
                    if caption_types and caption.caption_type not in caption_types:
                        continue
                    tags.extend(self._extract_tags(caption))

                if tags:
                    self._samples.append((item.image_path, tags))

    @staticmethod
    def _extract_tags(caption) -> List[str]:
        if caption.tag_data:
            try:
                raw = json.loads(caption.tag_data) if isinstance(caption.tag_data, str) else caption.tag_data
                if isinstance(raw, list):
                    return [normalize_tag(r["tag"]) for r in raw if isinstance(r, dict) and "tag" in r]
            except (json.JSONDecodeError, TypeError):
                pass
        if caption.content:
            return [normalize_tag(t) for t in caption.content.split(",") if t.strip()]
        return []

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        image_path, tags = self._samples[idx]

        # Load and preprocess image with SigLIP2 NaFlex processor
        image = _load_image(image_path)
        inputs = self.processor(images=[image], return_tensors="pt")

        pixel_values         = inputs["pixel_values"].squeeze(0)          # [num_patches, 768]
        pixel_attention_mask = inputs["pixel_attention_mask"].squeeze(0)  # [num_patches]
        spatial_shapes       = inputs["spatial_shapes"].squeeze(0)        # [2]

        # Build multi-hot label and loss_mask
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

        # Quality tags: mask within each group unless at least one is present
        for group_name, group_indices in voc.quality_indices.items():
            group_tags = [normalize_tag(t) for t in QUALITY_TAG_GROUPS[group_name]]
            has_group = any(t in tag_set for t in group_tags)
            if not has_group:
                for idx in group_indices:
                    loss_mask[idx] = 0.0
            else:
                # Mask out quality tags from OTHER groups in the same sample
                # (mutual exclusivity across groups)
                for other_name, other_indices in voc.quality_indices.items():
                    if other_name == group_name:
                        continue
                    other_tags = [normalize_tag(t) for t in QUALITY_TAG_GROUPS[other_name]]
                    has_other = any(t in tag_set for t in other_tags)
                    if not has_other:
                        for idx in other_indices:
                            loss_mask[idx] = 0.0

        return label, loss_mask


# ------------------------------------------------------------------
# Collate function (for variable-length NaFlex tensors)
# ------------------------------------------------------------------

def tagger_collate_fn(batch):
    """Collate batch items with potentially different num_patches.

    SigLIP2 NaFlex always outputs 256 patches per image by default,
    so in practice shapes are identical. But we handle the general case.
    """
    pixel_values_list, masks_list, shapes_list, labels_list, loss_masks_list = zip(*batch)

    # Stack if all same shape (usual case with default processor settings)
    try:
        pixel_values         = torch.stack(pixel_values_list)
        pixel_attention_mask = torch.stack(masks_list)
        spatial_shapes       = torch.stack(shapes_list)
    except RuntimeError:
        # Pad to max patches if shapes differ
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

    labels     = torch.stack(labels_list)
    loss_masks = torch.stack(loss_masks_list)

    return pixel_values, pixel_attention_mask, spatial_shapes, labels, loss_masks


# ------------------------------------------------------------------
# Image loading helper
# ------------------------------------------------------------------

def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img
