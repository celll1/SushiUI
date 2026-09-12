"""Sidecar serialization shared by tagger browser operations."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from typing import Any, Iterable, List


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


def write_image_sidecar(image_path: str, tags: Iterable[str]) -> str:
    """Atomically replace the image's comma-separated UTF-8 sidecar."""
    sidecar_path = os.path.splitext(image_path)[0] + ".txt"
    directory = os.path.dirname(sidecar_path) or "."
    fd, temporary_path = tempfile.mkstemp(
        dir=directory,
        prefix=os.path.basename(sidecar_path) + ".",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(", ".join(tags))
        os.replace(temporary_path, sidecar_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise
    return sidecar_path
