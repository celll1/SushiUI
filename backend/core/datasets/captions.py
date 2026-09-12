"""Consistent dataset caption mutation."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from database.models import Dataset, DatasetCaption, DatasetItem
from .sidecars import (
    SidecarWriteResult,
    capture_sidecars,
    restore_sidecars,
    write_indexed_caption,
)


class CaptionSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class CaptionUpdateResult:
    caption: DatasetCaption
    sidecar: SidecarWriteResult | None


def select_caption(
    db: Session,
    item_id: int,
    caption_type: str,
    *,
    caption_id: int | None,
    source_field: str | None,
) -> DatasetCaption | None:
    query = db.query(DatasetCaption).filter(DatasetCaption.item_id == item_id)
    if caption_id is not None:
        caption = query.filter(DatasetCaption.id == caption_id).first()
        if caption is None:
            raise CaptionSelectionError("Caption does not belong to this dataset item")
        if caption.caption_type != caption_type:
            raise CaptionSelectionError("Caption type does not match caption_id")
        return caption

    query = query.filter(DatasetCaption.caption_type == caption_type)
    if source_field is not None:
        query = query.filter(DatasetCaption.source_field == source_field)
    matches = query.limit(2).all()
    if len(matches) > 1:
        raise CaptionSelectionError(
            "Multiple captions match; provide caption_id or source_field"
        )
    return matches[0] if matches else None


def _tag_set(content: str | None) -> set[str]:
    return {tag.strip() for tag in (content or "").split(",") if tag.strip()}


def _update_tag_statistics(
    dataset: Dataset,
    old_content: str | None,
    new_content: str,
    tag_data: list[dict[str, str]] | None,
) -> None:
    statistics: dict[str, Any] = copy.deepcopy(dataset.tag_statistics or {})
    old_tags = _tag_set(old_content)
    new_tags = _tag_set(new_content)
    categories = {
        entry.get("tag", ""): entry.get("category", "Unknown")
        for entry in (tag_data or [])
    }

    for tag in old_tags - new_tags:
        if tag in statistics:
            statistics[tag]["count"] -= 1
            if statistics[tag]["count"] <= 0:
                del statistics[tag]
    for tag in new_tags - old_tags:
        if tag in statistics:
            statistics[tag]["count"] += 1
        else:
            statistics[tag] = {
                "count": 1,
                "category": categories.get(tag, "Unknown"),
            }
    dataset.tag_statistics = statistics


def update_caption(
    db: Session,
    *,
    item_id: int,
    dataset_id: int | None,
    caption_type: str,
    content: str,
    tag_data: list[dict[str, str]] | None = None,
    caption_id: int | None = None,
    source_field: str | None = None,
    persist_sidecar: bool = False,
    source: str | None = None,
) -> CaptionUpdateResult:
    item_query = db.query(DatasetItem).filter(DatasetItem.id == item_id)
    if dataset_id is not None:
        item_query = item_query.filter(DatasetItem.dataset_id == dataset_id)
    item = item_query.first()
    if item is None:
        raise LookupError("Dataset item not found")

    dataset = db.query(Dataset).filter(Dataset.id == item.dataset_id).first()
    item_had_tag_caption = False
    if caption_type == "tags":
        item_had_tag_caption = db.query(DatasetCaption.id).filter(
            DatasetCaption.item_id == item.id,
            DatasetCaption.is_tags_format == True,
        ).first() is not None
    caption = select_caption(
        db,
        item.id,
        caption_type,
        caption_id=caption_id,
        source_field=source_field,
    )
    old_content = caption.content if caption else None
    snapshot = capture_sidecars(item.image_path) if persist_sidecar else None
    sidecar = None

    try:
        if persist_sidecar:
            sidecar = write_indexed_caption(
                item.image_path,
                content,
                caption_type=caption_type,
                source_field=(caption.source_field if caption else source_field),
            )

        if caption is None:
            caption = DatasetCaption(
                item_id=item.id,
                caption_type=caption_type,
                content=content,
                source=source or "manual",
                source_field=source_field,
            )
            db.add(caption)
        else:
            caption.content = content
            caption.updated_at = datetime.utcnow()
            if source is not None:
                caption.source = source

        if caption_type == "tags":
            caption.field_category = "training"
            caption.is_tags_format = True
            if dataset is not None:
                if not item_had_tag_caption:
                    dataset.total_tags = (dataset.total_tags or 0) + 1
                _update_tag_statistics(dataset, old_content, content, tag_data)
            caption.tag_data = (
                json.dumps(tag_data, ensure_ascii=False)
                if tag_data is not None
                else None
            )
        elif tag_data is not None:
            caption.tag_data = json.dumps(tag_data, ensure_ascii=False)

        db.commit()
        db.refresh(caption)
        return CaptionUpdateResult(caption=caption, sidecar=sidecar)
    except Exception:
        db.rollback()
        if snapshot is not None:
            restore_sidecars(snapshot)
        raise
