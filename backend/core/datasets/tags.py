"""Shared tag-list metadata helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable


def build_tag_data(
    tags: Iterable[str],
    *,
    existing_json: str | None = None,
    resolve_categories: Callable[[list[str]], dict[str, str]],
) -> list[dict[str, str]]:
    """Preserve indexed categories and resolve metadata only for new tags."""
    tag_list = list(tags)
    existing: dict[str, str] = {}
    if existing_json:
        try:
            existing = {
                item["tag"]: item.get("category", "Unknown")
                for item in json.loads(existing_json)
                if isinstance(item, dict) and item.get("tag")
            }
        except (TypeError, ValueError):
            existing = {}
    missing = [tag for tag in tag_list if tag not in existing]
    resolved = resolve_categories(missing) if missing else {}
    return [
        {"tag": tag, "category": existing.get(tag, resolved.get(tag, "Unknown"))}
        for tag in tag_list
    ]
