"""Canonical dataset sidecar persistence."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SidecarFormatError(ValueError):
    """The existing sidecar cannot be updated without guessing its shape."""


@dataclass(frozen=True)
class SidecarWriteResult:
    path: str
    format: str
    field: str | None = None


@dataclass(frozen=True)
class SidecarSnapshot:
    files: tuple[tuple[Path, bytes | None], ...]


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def capture_sidecars(image_path: str) -> SidecarSnapshot:
    media_path = Path(image_path)
    files = tuple(
        (path, path.read_bytes() if path.exists() else None)
        for path in (media_path.with_suffix(".txt"), media_path.with_suffix(".json"))
    )
    return SidecarSnapshot(files)


def restore_sidecars(snapshot: SidecarSnapshot) -> None:
    for path, payload in snapshot.files:
        if payload is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        else:
            _atomic_write_bytes(path, payload)


def _load_json_object(path: Path) -> MutableMapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SidecarFormatError(f"Cannot read JSON sidecar {path}: {exc}") from exc
    if not isinstance(value, MutableMapping):
        raise SidecarFormatError(f"JSON sidecar must contain an object: {path}")
    return value


def _find_field(root: Mapping[str, Any], field_path: str) -> bool:
    current: Any = root
    for part in field_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _set_field(root: MutableMapping[str, Any], field_path: str, value: str) -> None:
    parts = field_path.split(".")
    current = root
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def read_text_sidecar(image_path: str) -> tuple[list[str], str]:
    """Read the conventional TXT tag sidecar for a media path."""
    sidecar_path = Path(image_path).with_suffix(".txt")
    if not sidecar_path.is_file():
        return [], ""
    content = sidecar_path.read_text(encoding="utf-8").strip()
    return [tag.strip() for tag in content.split(",") if tag.strip()], content


def write_text_tags(image_path: str, tags: Iterable[str]) -> SidecarWriteResult:
    """Atomically replace the conventional TXT tag sidecar."""
    sidecar_path = Path(image_path).with_suffix(".txt")
    _atomic_write(sidecar_path, ", ".join(tags))
    return SidecarWriteResult(str(sidecar_path), "txt")


def write_indexed_caption(
    image_path: str,
    content: str,
    *,
    caption_type: str = "tags",
    source_field: str | None = None,
) -> SidecarWriteResult:
    """Persist an indexed caption without changing the sidecar's established shape.

    An existing JSON field wins when it matches the indexed source field or a
    known legacy tag field. Otherwise an existing TXT sidecar is retained. A
    JSON file with no matching field receives the canonical ``tags`` field;
    when neither sidecar exists, the conventional TXT file is created.
    """
    media_path = Path(image_path)
    txt_path = media_path.with_suffix(".txt")
    json_path = media_path.with_suffix(".json")
    json_data: MutableMapping[str, Any] | None = None

    if json_path.exists():
        json_data = _load_json_object(json_path)
        candidates: list[str] = []
        if source_field:
            candidates.append(source_field)
        if caption_type == "tags":
            candidates.extend(("tags", "caption"))
        for field in dict.fromkeys(candidates):
            if field and _find_field(json_data, field):
                _set_field(json_data, field, content)
                _atomic_write(
                    json_path,
                    json.dumps(json_data, ensure_ascii=False, indent=2) + "\n",
                )
                return SidecarWriteResult(str(json_path), "json", field)

    if txt_path.exists():
        _atomic_write(txt_path, content)
        return SidecarWriteResult(str(txt_path), "txt")

    if json_data is not None:
        field = source_field if source_field and "." not in source_field else caption_type
        field = field or "tags"
        _set_field(json_data, field, content)
        _atomic_write(
            json_path,
            json.dumps(json_data, ensure_ascii=False, indent=2) + "\n",
        )
        return SidecarWriteResult(str(json_path), "json", field)

    _atomic_write(txt_path, content)
    return SidecarWriteResult(str(txt_path), "txt")
