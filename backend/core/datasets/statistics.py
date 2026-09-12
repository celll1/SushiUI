"""Derived dataset statistics."""

from __future__ import annotations

import json
from collections.abc import Callable

from sqlalchemy import func
from sqlalchemy.orm import Session

from database.models import DatasetCaption, DatasetItem
from utils.taglist_cache import taglist_cache


def caption_item_counts(db: Session, dataset_id: int) -> tuple[int, int]:
    """Count distinct items with tag and training-caption content."""
    item_ids = db.query(DatasetItem.id).filter(DatasetItem.dataset_id == dataset_id)
    with_tags = db.query(DatasetCaption.item_id).filter(
        DatasetCaption.item_id.in_(item_ids),
        DatasetCaption.is_tags_format == True,
    ).distinct().count()
    with_captions = db.query(DatasetCaption.item_id).filter(
        DatasetCaption.item_id.in_(item_ids),
        DatasetCaption.is_tags_format == False,
        DatasetCaption.field_category == "training",
        DatasetCaption.content.isnot(None),
        func.trim(DatasetCaption.content) != "",
    ).distinct().count()
    return with_tags, with_captions


def compute_tag_statistics(
    dataset_id: int,
    db: Session,
    *,
    root_dir: str,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict[str, int | str]]:
    """Count tags while retaining the first known category for each tag."""
    taglist_cache.initialize(root_dir, enable_gelbooru=True)
    rows = db.query(
        DatasetCaption.content,
        DatasetCaption.tag_data,
    ).join(
        DatasetItem, DatasetCaption.item_id == DatasetItem.id
    ).filter(
        DatasetItem.dataset_id == dataset_id,
        DatasetCaption.caption_type == "tags",
    ).order_by(DatasetCaption.id).yield_per(1000)

    tag_counts: dict[str, int] = {}
    categories: dict[str, str] = {}
    processed = 0

    def add(tag: str, category: str = "Unknown") -> None:
        tag = tag.strip()
        if not tag:
            return
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        previous = categories.get(tag)
        if previous is None or (previous == "Unknown" and category != "Unknown"):
            categories[tag] = category

    for content, tag_data_json in rows:
        parsed_tags = None
        if tag_data_json:
            try:
                parsed = json.loads(tag_data_json)
                if not isinstance(parsed, list):
                    raise ValueError("tag_data must be a list")
                parsed_tags = [
                    (item.get("tag", "").strip(), item.get("category", "Unknown"))
                    for item in parsed
                ]
            except (AttributeError, TypeError, ValueError):
                parsed_tags = None
        if parsed_tags is not None:
            for tag, category in parsed_tags:
                add(tag, category)
        elif content:
            for tag in content.split(","):
                add(tag)

        processed += 1
        if progress is not None and processed % 10_000 == 0:
            progress(processed, len(tag_counts))

    unresolved = [tag for tag, category in categories.items() if category == "Unknown"]
    if unresolved:
        resolved = taglist_cache.get_categories_batch(unresolved)
        for tag in unresolved:
            category = resolved.get(tag, "Unknown")
            if category != "Unknown":
                categories[tag] = category

    return {
        tag: {"count": count, "category": categories.get(tag, "Unknown")}
        for tag, count in tag_counts.items()
    }
