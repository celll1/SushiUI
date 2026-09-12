"""Caption tag extraction shared by tagger ingestion paths."""

from __future__ import annotations

import json
from typing import Any, List

from .tag_vocabulary import normalize_tag


def extract_caption_tags(tag_data: Any, content: str | None) -> List[str]:
    """Prefer structured tag data and fall back to comma-separated content."""
    raw_tags: List[str] = []
    if tag_data:
        try:
            raw = json.loads(tag_data) if isinstance(tag_data, str) else tag_data
            if isinstance(raw, list):
                raw_tags = [
                    row["tag"]
                    for row in raw
                    if isinstance(row, dict) and "tag" in row
                ]
        except (json.JSONDecodeError, TypeError):
            pass
    if not raw_tags and content:
        raw_tags = [tag.strip() for tag in content.split(",") if tag.strip()]
    return raw_tags


def resolve_caption_tags(
    tag_data: Any,
    content: str | None,
    comma_resolver: Any = None,
    alias_resolver: Any = None,
) -> List[str]:
    """Return normalized caption tags with configured canonicalization applied."""
    norm_tokens = [
        tag for tag in (normalize_tag(raw) for raw in extract_caption_tags(tag_data, content)) if tag
    ]
    if comma_resolver is not None:
        norm_tokens = comma_resolver.resolve(norm_tokens)
    if alias_resolver:
        return [
            tag
            if comma_resolver is not None and comma_resolver.category_of(tag) is not None
            else alias_resolver.resolve(tag)
            for tag in norm_tokens
        ]
    return norm_tokens
