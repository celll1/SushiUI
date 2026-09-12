"""Bounded dataset media preview cache."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageOps

from database.models import DatasetItem
from utils.image_utils import dataset_thumbnail_key


PREVIEW_SIZES = (128, 256, 512)
MAX_CACHE_FILES = 4096
MAX_CACHE_BYTES = 512 * 1024 * 1024
_cache_lock = threading.RLock()


class PreviewUnavailableError(FileNotFoundError):
    pass


def _existing_media_preview(item: DatasetItem, thumbnails_dir: str) -> Path:
    root = Path(thumbnails_dir)
    key = dataset_thumbnail_key(item.image_path)
    candidates = (
        root / f"{key}.webp",
        root / f"{key}.png",
        root / f"{item.base_name}.webp",
        root / f"{item.base_name}.png",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise PreviewUnavailableError(f"No preview is available for item {item.id}")


def _source_path(item: DatasetItem, thumbnails_dir: str) -> Path:
    if item.item_type in ("video", "audio"):
        return _existing_media_preview(item, thumbnails_dir)
    path = Path(item.image_path)
    if not path.is_file():
        raise PreviewUnavailableError(f"Dataset media is missing for item {item.id}")
    return path


def _fingerprint(item: DatasetItem, source: Path, size: int) -> str:
    media = Path(item.image_path)
    media_stat = media.stat()
    source_stat = source.stat()
    identity = "\0".join((
        str(item.dataset_id),
        str(item.id),
        os.path.normcase(os.path.abspath(item.image_path)),
        str(media_stat.st_mtime_ns),
        str(media_stat.st_size),
        os.path.normcase(str(source.resolve())),
        str(source_stat.st_mtime_ns),
        str(source_stat.st_size),
        str(size),
    ))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def prune_preview_cache(
    cache_dir: Path,
    *,
    max_files: int = MAX_CACHE_FILES,
    max_bytes: int = MAX_CACHE_BYTES,
    keep: Path | None = None,
) -> None:
    entries = []
    total_bytes = 0
    for path in cache_dir.glob("*.webp"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        entries.append((stat.st_mtime_ns, path, stat.st_size))
        total_bytes += stat.st_size
    entries.sort()
    while len(entries) > max_files or total_bytes > max_bytes:
        removable = next(
            (index for index, (_, path, _) in enumerate(entries) if path != keep),
            None,
        )
        if removable is None:
            break
        _, path, file_size = entries.pop(removable)
        try:
            path.unlink()
            total_bytes -= file_size
        except FileNotFoundError:
            pass


def get_or_create_preview(
    item: DatasetItem,
    size: int,
    *,
    thumbnails_dir: str,
    cache_root: str,
) -> tuple[Path, str]:
    if size not in PREVIEW_SIZES:
        raise ValueError(f"Unsupported preview size: {size}")
    source = _source_path(item, thumbnails_dir)
    fingerprint = _fingerprint(item, source, size)
    cache_dir = Path(cache_root) / "dataset_previews" / "v1"
    output = cache_dir / f"{fingerprint}.webp"

    with _cache_lock:
        if output.is_file():
            os.utime(output, None)
            return output, fingerprint
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            dir=str(cache_dir), prefix=fingerprint + ".", suffix=".tmp"
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened)
                image.thumbnail((size, size), Image.Resampling.LANCZOS)
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGBA" if "transparency" in image.info else "RGB")
                image.save(temporary, format="WEBP", quality=82, method=4)
            os.replace(temporary, output)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        prune_preview_cache(cache_dir, keep=output)
    return output, fingerprint
