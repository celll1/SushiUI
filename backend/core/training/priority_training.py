"""Priority Training - Focused training on specific tags/concepts.

Allows prioritizing certain data items (matching specific tags or captions)
at the beginning of each epoch with optional repetition (multiplier),
reducing forgetting in large-scale training.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from core.training.tag_group_utils import normalize_tag_for_matching


@dataclass
class PriorityEntry:
    """A single entry in the priority training list."""
    tags: Optional[List[str]] = None          # Tag match (AND condition)
    caption_contains: Optional[str] = None    # Caption substring match

    def __post_init__(self):
        if self.tags and self.caption_contains:
            raise ValueError("PriorityEntry cannot have both 'tags' and 'caption_contains'")
        if not self.tags and not self.caption_contains:
            raise ValueError("PriorityEntry must have either 'tags' or 'caption_contains'")


@dataclass
class PriorityTrainingConfig:
    """Configuration for priority training."""
    entries: List[PriorityEntry] = field(default_factory=list)
    multiplier: int = 1
    timing: str = "epoch_start"  # "epoch_start" for now

    @staticmethod
    def _parse_entry(entry_data) -> Optional["PriorityEntry"]:
        """Parse a single entry from various formats.

        Formats:
        - str: "tag_name" → single tag match
        - str: "tag1, tag2" → AND condition (comma-separated)
        - str: "caption:text" → caption substring match
        - dict: {"tags": [...]} → tag AND match
        - dict: {"caption_contains": "..."} → caption match
        """
        if isinstance(entry_data, str):
            entry_data = entry_data.strip()
            if not entry_data:
                return None
            if entry_data.startswith("caption:"):
                text = entry_data[len("caption:"):].strip()
                return PriorityEntry(caption_contains=text) if text else None
            elif "," in entry_data:
                tags = [t.strip() for t in entry_data.split(",") if t.strip()]
                return PriorityEntry(tags=tags) if tags else None
            else:
                return PriorityEntry(tags=[entry_data])
        elif isinstance(entry_data, dict):
            if "tags" in entry_data:
                return PriorityEntry(tags=entry_data["tags"])
            elif "caption_contains" in entry_data:
                return PriorityEntry(caption_contains=entry_data["caption_contains"])
        print(f"[PriorityTraining] Skipping invalid entry: {entry_data}")
        return None

    @staticmethod
    def from_dict(data: Dict) -> "PriorityTrainingConfig":
        """Create config from inline dict (embedded in training YAML)."""
        if not data or not isinstance(data, dict):
            return PriorityTrainingConfig()

        entries = []
        for entry_data in data.get("entries", []):
            entry = PriorityTrainingConfig._parse_entry(entry_data)
            if entry:
                entries.append(entry)

        config = PriorityTrainingConfig(
            entries=entries,
            multiplier=data.get("multiplier", 1),
            timing=data.get("timing", "epoch_start"),
        )
        print(f"[PriorityTraining] Loaded {len(config.entries)} entries, "
              f"multiplier={config.multiplier}, timing={config.timing}")
        return config

    @staticmethod
    def load(yaml_path: str) -> "PriorityTrainingConfig":
        """Load priority training config from YAML file (legacy support)."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Priority training config not found: {yaml_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return PriorityTrainingConfig.from_dict(data)


def match_item_to_entry(item: Dict, entry: PriorityEntry) -> bool:
    """Check if an item matches a priority entry.

    For tag entries: uses normalized tag matching against tag_data (fast path)
    or caption text (fallback).
    For caption_contains entries: case-insensitive substring match.
    """
    if entry.tags:
        search_tags = [normalize_tag_for_matching(t) for t in entry.tags]

        # Fast path: use pre-categorized tag_data JSON
        tag_data_str = item.get("tag_data")
        if tag_data_str and item.get("is_tags_format", False):
            try:
                tag_data = json.loads(tag_data_str) if isinstance(tag_data_str, str) else tag_data_str
                item_tags = {normalize_tag_for_matching(td["tag"]) for td in tag_data}
                return all(st in item_tags for st in search_tags)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Fallback: search in raw caption text
        raw_caption = item.get("raw_caption", item.get("caption", ""))
        caption_normalized = normalize_tag_for_matching(raw_caption)
        return all(st in caption_normalized for st in search_tags)

    elif entry.caption_contains:
        caption = item.get("raw_caption", item.get("caption", ""))
        return entry.caption_contains.lower() in caption.lower()

    return False


def classify_items(
    all_items: List[Tuple[Dict, Any]],
    config: PriorityTrainingConfig,
) -> Tuple[List[Tuple[Dict, Any, int]], List[Tuple[Dict, Any]]]:
    """Classify items into priority and normal groups.

    Args:
        all_items: List of (item_dict, dataset) tuples
        config: Priority training configuration

    Returns:
        (priority_items, normal_items) where:
        - priority_items: [(item, dataset, entry_index), ...] sorted by entry_index
        - normal_items: [(item, dataset), ...]
    """
    priority_items = []
    normal_items = []

    for item, dataset in all_items:
        matched_idx = None
        for idx, entry in enumerate(config.entries):
            if match_item_to_entry(item, entry):
                matched_idx = idx
                break  # Use first matching entry (highest priority)
        if matched_idx is not None:
            priority_items.append((item, dataset, matched_idx))
        else:
            normal_items.append((item, dataset))

    # Sort by entry index so nearby list entries are batched together
    priority_items.sort(key=lambda x: x[2])

    print(f"[PriorityTraining] Classification: {len(priority_items)} priority, "
          f"{len(normal_items)} normal items")
    for idx, entry in enumerate(config.entries):
        count = sum(1 for _, _, ei in priority_items if ei == idx)
        label = f"tags={entry.tags}" if entry.tags else f"caption_contains='{entry.caption_contains}'"
        print(f"  Entry {idx}: {label} -> {count} items")

    return priority_items, normal_items


def build_priority_batches(
    priority_items: List[Tuple[Dict, Any, int]],
    batch_size: int,
    bucket_manager: Any = None,
) -> List[List[Tuple[Dict, Any]]]:
    """Build batches from priority items, respecting bucket constraints.

    Items are already sorted by entry_index. Within the same entry_index,
    items are grouped by resolution bucket (if bucketing is enabled).

    Args:
        priority_items: [(item, dataset, entry_index), ...] sorted by entry_index
        batch_size: Batch size
        bucket_manager: Optional BucketManager for resolution grouping

    Returns:
        List of batches, each batch is [(item, dataset), ...]
    """
    if not priority_items:
        return []

    batches = []

    if bucket_manager:
        # Group by (entry_index, bucket_resolution) for optimal batching
        groups: Dict[Tuple[int, Tuple[int, int]], List[Tuple[Dict, Any]]] = {}
        for item, dataset, entry_idx in priority_items:
            bucket_key = (entry_idx, (item.get("width", 1024), item.get("height", 1024)))
            if bucket_key not in groups:
                groups[bucket_key] = []
            groups[bucket_key].append((item, dataset))

        # Build batches from groups, ordered by entry_index
        for group_key in sorted(groups.keys()):
            group_items = groups[group_key]
            for i in range(0, len(group_items), batch_size):
                batch = group_items[i:i + batch_size]
                batches.append(batch)
    else:
        # No bucketing: just split by batch_size, items already sorted by entry_index
        items_no_idx = [(item, dataset) for item, dataset, _ in priority_items]
        for i in range(0, len(items_no_idx), batch_size):
            batches.append(items_no_idx[i:i + batch_size])

    print(f"[PriorityTraining] Built {len(batches)} priority batches")
    return batches
