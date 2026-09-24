"""Deterministic Artist-tag selection for a future style Danbooru source."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any


def select_artist_tags(
    pages: Iterable[Iterable[Mapping[str, Any]]], *, min_post_count: int,
    order: str, seed: int, limit: int,
) -> tuple[str, ...]:
    """Select eligible names from a completed tag scan; no network or files are touched."""
    if isinstance(min_post_count, bool) or not isinstance(min_post_count, int) or min_post_count < 1:
        raise ValueError("min_post_count must be a positive integer")
    if order not in ("count_desc", "random"):
        raise ValueError("order must be count_desc or random")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    counts: dict[str, int] = {}
    for page in pages:
        for tag in page:
            name, count = tag.get("name"), tag.get("post_count")
            if (tag.get("category") != 1 or tag.get("is_deprecated") is True
                    or not isinstance(name, str) or not name.strip()
                    or isinstance(count, bool) or not isinstance(count, int) or count < min_post_count):
                continue
            counts[name] = max(count, counts.get(name, 0))

    names = sorted(counts, key=lambda name: (-counts[name], name))
    if order == "random":
        random.Random(seed).shuffle(names)
    return tuple(names[:limit])
