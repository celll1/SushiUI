"""Label and loss-mask construction shared by tagger data sources."""

from __future__ import annotations

from typing import Any, List, Tuple

import torch

from .tag_vocabulary import QUALITY_TAG_GROUPS, RATING_TAGS, TagVocabulary, normalize_tag


_RATING_TAG_SET = frozenset(normalize_tag(tag) for tag in RATING_TAGS)
_QUALITY_TAG_SETS = {
    group: frozenset(normalize_tag(tag) for tag in tags)
    for group, tags in QUALITY_TAG_GROUPS.items()
}


def build_label_and_mask(
    tags: List[str],
    vocabulary: TagVocabulary,
    quality_masking_mode: str = "intra_group",
    alias_resolver: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build multi-hot labels and the rating/quality loss mask."""
    if alias_resolver is not None:
        resolved_tags = [alias_resolver.resolve(tag) for tag in tags]
    else:
        resolved_tags = [normalize_tag(tag) for tag in tags]

    label = torch.zeros(vocabulary.num_tags, dtype=torch.float32)
    loss_mask = torch.ones(vocabulary.num_tags, dtype=torch.float32)
    tag_set = set(resolved_tags)

    for tag in tag_set:
        idx = vocabulary.tag_to_idx.get(tag)
        if idx is not None:
            label[idx] = 1.0

    if tag_set.isdisjoint(_RATING_TAG_SET):
        for idx in vocabulary.rating_indices:
            loss_mask[idx] = 0.0

    present_groups = {
        group
        for group, group_tags in _QUALITY_TAG_SETS.items()
        if not tag_set.isdisjoint(group_tags)
    }
    if not present_groups:
        for group_indices in vocabulary.quality_indices.values():
            for idx in group_indices:
                loss_mask[idx] = 0.0
    elif quality_masking_mode == "intra_group":
        for group in present_groups:
            for idx in vocabulary.quality_indices[group]:
                if label[idx] == 0.0:
                    loss_mask[idx] = 0.0

    return label, loss_mask
