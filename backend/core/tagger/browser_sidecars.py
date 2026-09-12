"""Sidecar serialization shared by tagger browser operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable, List

from core.datasets.sidecars import read_text_sidecar, write_text_tags


def prediction_tag_names(result: Mapping[str, Any]) -> List[str]:
    """Return the sidecar tag order represented by an inference response."""
    names: List[str] = []
    seen = set()

    def append(item: Any) -> None:
        if not isinstance(item, Mapping):
            return
        tag = item.get("tag")
        if isinstance(tag, str) and tag and tag not in seen:
            seen.add(tag)
            names.append(tag)

    for item in result.get("tags") or ():
        append(item)
    append(result.get("quality_top"))
    append(result.get("rating_top"))
    return names


def read_image_sidecar(image_path: str) -> tuple[List[str], str]:
    """Read an image sidecar; a missing sidecar is a valid empty value."""
    return read_text_sidecar(image_path)


def write_image_sidecar(image_path: str, tags: Iterable[str]) -> str:
    """Atomically replace the image's comma-separated UTF-8 sidecar."""
    return write_text_tags(image_path, tags).path
