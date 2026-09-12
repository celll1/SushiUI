"""Shared dataset query semantics."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import func
from sqlalchemy.orm import load_only


def parse_exact_tags(content: str) -> set[str]:
    return {part.strip().casefold() for part in (content or "").split(",") if part.strip()}


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def exact_tag_item_ids(
    db,
    dataset_id: int,
    requested_tags: Iterable[str],
    *,
    search: str | None = None,
) -> list[int]:
    """Return ordered item IDs whose tags caption contains every exact token."""
    from database.models import DatasetCaption, DatasetItem

    wanted = {tag.strip().casefold() for tag in requested_tags if tag.strip()}
    if not wanted:
        return []

    query = (
        db.query(DatasetItem.id, DatasetCaption.content)
        .join(DatasetCaption, DatasetCaption.item_id == DatasetItem.id)
        .filter(
            DatasetItem.dataset_id == dataset_id,
            DatasetCaption.caption_type == "tags",
        )
    )
    if search:
        query = query.filter(DatasetItem.base_name.like(f"%{search}%"))
    for tag in wanted:
        if tag.isascii():
            query = query.filter(
                func.lower(DatasetCaption.content).like(
                    f"%{_escape_like(tag)}%", escape="\\"
                )
            )

    matched: set[int] = set()
    for item_id, content in query.order_by(DatasetItem.id).yield_per(2000):
        if wanted <= parse_exact_tags(content):
            matched.add(item_id)
    return sorted(matched)


def dataset_item_page(
    db,
    dataset_id: int,
    *,
    page: int,
    page_size: int,
    search: str | None,
    tags: str | None,
    grid_projection: bool = False,
):
    from database.models import DatasetItem

    query = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id)
    if search:
        query = query.filter(DatasetItem.base_name.like(f"%{search}%"))

    exact_ids = None
    if tags:
        requested = [tag.strip() for tag in tags.split(",") if tag.strip()]
        if requested:
            exact_ids = exact_tag_item_ids(
                db, dataset_id, requested, search=search
            )

    total = len(exact_ids) if exact_ids is not None else query.count()
    offset = (page - 1) * page_size
    if exact_ids is not None:
        page_ids = exact_ids[offset:offset + page_size]
        query = db.query(DatasetItem).filter(DatasetItem.id.in_(page_ids))
    else:
        query = query.order_by(DatasetItem.id).offset(offset).limit(page_size)
    if grid_projection:
        query = query.options(load_only(
            DatasetItem.id,
            DatasetItem.dataset_id,
            DatasetItem.item_type,
            DatasetItem.base_name,
            DatasetItem.image_path,
            DatasetItem.width,
            DatasetItem.height,
            DatasetItem.file_size,
        ))
    if exact_ids is not None:
        items = query.order_by(DatasetItem.id).all() if page_ids else []
    else:
        items = query.all()
    return items, total
