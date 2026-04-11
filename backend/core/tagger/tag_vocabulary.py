"""
Tag vocabulary management for SigLIP2 tagger training.

Collects tags from sushiUI Dataset (DatasetCaption with is_tags_format=True),
builds tag->index mapping with category information.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Category sort order for vocabulary organization
CATEGORY_ORDER: List[str] = [
    "General", "Character", "Copyright", "Artist", "Meta", "Rating", "Quality", "Model"
]

# Quality tag groups (same as tagutl/lora.py)
QUALITY_TAG_GROUPS: Dict[str, List[str]] = {
    "high_quality_group": ["best quality", "high quality", "normal quality", "medium quality"],
    "low_quality_group":  ["low quality", "bad quality", "worst quality"],
}

# Rating tags
RATING_TAGS: List[str] = ["rating:general", "rating:sensitive", "rating:questionable", "rating:explicit"]


def normalize_tag(tag: str) -> str:
    """Normalize tag for consistent matching.

    Steps applied in order:
    1. Strip leading/trailing whitespace
    2. Replace underscores with spaces  (danbooru convention)
    3. Lowercase
    4. Remove backslash escaping        e.g. \\( → (  so that
       "fate \\(series\\)" and "fate (series)" collapse to the same key
    """
    tag = tag.strip().replace("_", " ").lower()
    # Unescape backslash sequences repeatedly until stable
    # (handles multiply-escaped tags like \\\\( → \\( → ()
    while True:
        unescaped = re.sub(r"\\(.)", r"\1", tag)
        if unescaped == tag:
            break
        tag = unescaped
    return tag


class TagVocabulary:
    """
    Maps tag strings to integer indices.

    Attributes
    ----------
    tag_to_idx   : Dict[str, int]   normalized tag -> index
    idx_to_tag   : Dict[int, str]   index -> normalized tag
    tag_to_category : Dict[str, str]  normalized tag -> category name
    rating_indices  : List[int]  indices of rating tags
    quality_indices : Dict[str, List[int]]  group_name -> indices
    """

    def __init__(self) -> None:
        self.tag_to_idx: Dict[str, int] = {}
        self.idx_to_tag: Dict[int, str] = {}
        self.tag_to_category: Dict[str, str] = {}
        self.rating_indices: List[int] = []
        self.quality_indices: Dict[str, List[int]] = {k: [] for k in QUALITY_TAG_GROUPS}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build_from_dataset_ids(
        cls,
        dataset_ids: List[int],
        datasets_db,
        min_count: int = 1,
        excluded_categories: Optional[List[str]] = None,
        ban_tags: Optional[List[str]] = None,
    ) -> "TagVocabulary":
        """Build vocabulary by scanning DatasetCaption rows for given dataset IDs.

        Parameters
        ----------
        dataset_ids         : list of Dataset.id values
        datasets_db         : SQLAlchemy session for datasets.db
        min_count           : minimum occurrence count to include a tag
        excluded_categories : categories to exclude entirely (e.g. ["Artist"])
        ban_tags            : tag patterns to exclude; supports fnmatch wildcards
                              (e.g. ["some tag", "prefix_*", "bad*"])
        """
        from database.models import DatasetItem, DatasetCaption

        tag_counts: Dict[str, int] = defaultdict(int)
        tag_categories: Dict[str, str] = {}

        # Single JOIN query — avoids N+1 (one query per item for lazy caption load)
        captions = (
            datasets_db.query(DatasetCaption)
            .join(DatasetItem, DatasetCaption.item_id == DatasetItem.id)
            .filter(
                DatasetItem.dataset_id.in_(dataset_ids),
                DatasetCaption.is_tags_format == True,
            )
            .all()
        )
        for caption in captions:
            tags_with_cats = _parse_caption_tags(caption)
            for tag, category in tags_with_cats:
                norm = normalize_tag(tag)
                tag_counts[norm] += 1
                if norm not in tag_categories:
                    tag_categories[norm] = category

        # Resolve "__lookup__" sentinels via taglist_cache (batch, O(1) per tag)
        lookup_tags = [t for t, c in tag_categories.items() if c == "__lookup__"]
        if lookup_tags:
            try:
                from utils.taglist_cache import taglist_cache
                resolved = taglist_cache.get_categories_batch(lookup_tags)
                for norm_tag in lookup_tags:
                    tag_categories[norm_tag] = resolved.get(norm_tag, "General")
            except Exception:
                # taglist_cache not initialized or unavailable — fall back to General
                for norm_tag in lookup_tags:
                    tag_categories[norm_tag] = "General"

        # Filter by min_count
        filtered: Dict[str, int] = {t: c for t, c in tag_counts.items() if c >= min_count}

        # Filter by excluded_categories
        if excluded_categories:
            excl: Set[str] = {c.strip() for c in excluded_categories}
            filtered = {t: c for t, c in filtered.items()
                        if tag_categories.get(t, "General") not in excl}

        # Filter by ban_tags (fnmatch wildcards supported)
        if ban_tags:
            ban_patterns = [p.strip() for p in ban_tags if p.strip()]
            filtered = {t: c for t, c in filtered.items()
                        if not any(fnmatch.fnmatch(t, pat) for pat in ban_patterns)}

        # Sort: category order first, then alphabetically within each category
        def _sort_key(tag: str) -> tuple:
            cat = tag_categories.get(tag, "General")
            cat_rank = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
            return (cat_rank, tag)

        selected = sorted(filtered.keys(), key=_sort_key)

        vocab = cls()
        for idx, tag in enumerate(selected):
            vocab.tag_to_idx[tag] = idx
            vocab.idx_to_tag[idx] = tag
            vocab.tag_to_category[tag] = tag_categories.get(tag, "General")

        vocab._build_special_indices()
        return vocab

    @classmethod
    def from_dict(cls, data: dict) -> "TagVocabulary":
        """Restore from a plain dict (e.g. loaded from JSON)."""
        vocab = cls()
        vocab.tag_to_idx = {k: int(v) for k, v in data["tag_to_idx"].items()}
        vocab.idx_to_tag = {int(k): v for k, v in data["idx_to_tag"].items()}
        vocab.tag_to_category = data.get("tag_to_category", {})
        vocab._build_special_indices()
        return vocab

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        # Build categories section: category -> sorted list of tags
        categories: Dict[str, List[str]] = {}
        for tag, cat in self.tag_to_category.items():
            categories.setdefault(cat, []).append(tag)
        # Sort within each category alphabetically; sort category keys by CATEGORY_ORDER
        sorted_categories: Dict[str, List[str]] = {}
        cat_keys = sorted(
            categories.keys(),
            key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else len(CATEGORY_ORDER),
        )
        for cat in cat_keys:
            sorted_categories[cat] = sorted(categories[cat])

        return {
            "tag_to_idx": self.tag_to_idx,
            "idx_to_tag": {str(k): v for k, v in self.idx_to_tag.items()},
            "tag_to_category": self.tag_to_category,
            "num_tags": len(self.tag_to_idx),
            "categories": sorted_categories,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_tags(self) -> int:
        return len(self.tag_to_idx)

    def category_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for cat in self.tag_to_category.values():
            counts[cat] += 1
        return dict(counts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_special_indices(self) -> None:
        """Populate rating_indices and quality_indices from current tag_to_idx."""
        self.rating_indices = []
        self.quality_indices = {k: [] for k in QUALITY_TAG_GROUPS}

        for tag, idx in self.tag_to_idx.items():
            norm = normalize_tag(tag)
            if norm in [normalize_tag(r) for r in RATING_TAGS]:
                self.rating_indices.append(idx)
            for group_name, group_tags in QUALITY_TAG_GROUPS.items():
                if norm in [normalize_tag(t) for t in group_tags]:
                    self.quality_indices[group_name].append(idx)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_caption_tags(caption) -> List[Tuple[str, str]]:
    """Extract (tag, category) pairs from a DatasetCaption row.

    Prefers tag_data JSON (has category info); falls back to content string
    with category="__lookup__" sentinel so the caller can batch-resolve
    categories via taglist_cache.
    """
    if caption.tag_data:
        try:
            raw = json.loads(caption.tag_data) if isinstance(caption.tag_data, str) else caption.tag_data
            if isinstance(raw, list):
                result = []
                for item in raw:
                    if isinstance(item, dict) and "tag" in item:
                        result.append((item["tag"], item.get("category", "General")))
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: parse comma-separated content.
    # Use "__lookup__" sentinel so build_from_dataset_ids can resolve via taglist_cache.
    if caption.content:
        tags = [t.strip() for t in caption.content.split(",") if t.strip()]
        return [(t, "__lookup__") for t in tags]

    return []
