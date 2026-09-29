"""Persistent, coordinate-aligned tiles for bounded FoveaTag source reads."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

from PIL import Image

from .views import ViewSpec


TILED_PYRAMID_FORMAT_VERSION = 1
_BICUBIC = Image.Resampling.BICUBIC


@dataclass(frozen=True)
class PyramidLevel:
    index: int
    width: int
    height: int
    columns: int
    rows: int


class TiledPyramidSource:
    """Read only the pyramid tiles needed by one overview or local view."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        payload = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if payload.get("format_version") != TILED_PYRAMID_FORMAT_VERSION:
            raise ValueError("unsupported FoveaTag tiled-pyramid format")
        if payload.get("kind") != "foveatag_tiled_pyramid":
            raise ValueError("directory is not a FoveaTag tiled pyramid")
        self.width = int(payload["source"]["width"])
        self.height = int(payload["source"]["height"])
        self.tile_size = int(payload["tile_size"])
        if min(self.width, self.height, self.tile_size) <= 0:
            raise ValueError("tiled-pyramid dimensions must be positive")
        self.source_sha256 = str(payload["source"]["sha256"])
        self.source_path = Path(str(payload["source"].get("path", ""))).resolve()
        self.source_bytes = payload["source"].get("bytes")
        self.source_mtime_ns = payload["source"].get("mtime_ns")
        self.levels = tuple(PyramidLevel(**row) for row in payload["levels"])
        self._validate_levels()

    def validate_source(self, image_path: str | Path) -> None:
        path = Path(image_path).resolve()
        if path != self.source_path:
            raise ValueError("tiled-pyramid cache belongs to a different source path")
        stat = path.stat()
        if self.source_bytes is None or self.source_mtime_ns is None:
            raise ValueError("tiled-pyramid cache lacks cheap source identity")
        if stat.st_size != int(self.source_bytes) or stat.st_mtime_ns != int(
            self.source_mtime_ns
        ):
            raise ValueError("tiled-pyramid source changed after cache creation")

    def render_view(
        self,
        spec: ViewSpec,
        pad_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Image.Image:
        canvas = Image.new("RGB", (spec.canvas_width, spec.canvas_height), pad_color)
        box = spec.source_box
        source_left = max(0.0, box.x)
        source_top = max(0.0, box.y)
        source_right = min(float(self.width), box.x + box.width)
        source_bottom = min(float(self.height), box.y + box.height)
        if source_right <= source_left or source_bottom <= source_top:
            return canvas

        level = self._select_level(spec.pixels_per_canvas_pixel)
        scale_x = level.width / self.width
        scale_y = level.height / self.height
        level_box = (
            source_left * scale_x,
            source_top * scale_y,
            source_right * scale_x,
            source_bottom * scale_y,
        )
        integer_box = (
            min(level.width - 1, max(0, round(level_box[0]))),
            min(level.height - 1, max(0, round(level_box[1]))),
            min(level.width, round(level_box[2])),
            min(level.height, round(level_box[3])),
        )
        integer_box = (
            integer_box[0],
            integer_box[1],
            max(integer_box[0] + 1, integer_box[2]),
            max(integer_box[1] + 1, integer_box[3]),
        )
        region = self._assemble(level, integer_box)
        canvas_left, canvas_top = spec.source_to_canvas(source_left, source_top)
        canvas_right, canvas_bottom = spec.source_to_canvas(source_right, source_bottom)
        target_width = max(1, round(canvas_right - canvas_left))
        target_height = max(1, round(canvas_bottom - canvas_top))
        rendered = region.resize((target_width, target_height), _BICUBIC)
        canvas.paste(rendered, (round(canvas_left), round(canvas_top)))
        return canvas

    def _select_level(self, source_pixels_per_canvas_pixel: float) -> PyramidLevel:
        allowed = max(1.0, float(source_pixels_per_canvas_pixel))
        candidates = [
            level
            for level in self.levels
            if self._level_downsample(level) <= allowed * (1.0 + 1e-6)
        ]
        return candidates[-1] if candidates else self.levels[0]

    def _level_downsample(self, level: PyramidLevel) -> float:
        return math.sqrt((self.width * self.height) / (level.width * level.height))

    def _assemble(
        self,
        level: PyramidLevel,
        box: tuple[int, int, int, int],
    ) -> Image.Image:
        left, top, right, bottom = box
        region = Image.new("RGB", (right - left, bottom - top))
        first_column, last_column = left // self.tile_size, (right - 1) // self.tile_size
        first_row, last_row = top // self.tile_size, (bottom - 1) // self.tile_size
        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                tile = self._read_tile(level.index, column, row)
                tile_left = column * self.tile_size
                tile_top = row * self.tile_size
                overlap = (
                    max(left, tile_left),
                    max(top, tile_top),
                    min(right, tile_left + tile.width),
                    min(bottom, tile_top + tile.height),
                )
                crop = tile.crop((
                    overlap[0] - tile_left,
                    overlap[1] - tile_top,
                    overlap[2] - tile_left,
                    overlap[3] - tile_top,
                ))
                region.paste(crop, (overlap[0] - left, overlap[1] - top))
        return region

    @lru_cache(maxsize=32)
    def _read_tile(self, level: int, column: int, row: int) -> Image.Image:
        path = self.root / f"level_{level}" / f"{column}_{row}.png"
        with Image.open(path) as source:
            tile = source.convert("RGB")
            tile.load()
        return tile

    def _validate_levels(self) -> None:
        if not self.levels or self.levels[0].width != self.width or self.levels[0].height != self.height:
            raise ValueError("pyramid level zero must match the source dimensions")
        previous = None
        for expected_index, level in enumerate(self.levels):
            if level.index != expected_index or min(level.width, level.height) <= 0:
                raise ValueError("tiled-pyramid levels must be contiguous and positive")
            expected_columns = math.ceil(level.width / self.tile_size)
            expected_rows = math.ceil(level.height / self.tile_size)
            if (level.columns, level.rows) != (expected_columns, expected_rows):
                raise ValueError("tiled-pyramid tile grid is inconsistent")
            if previous is not None and (
                level.width != math.ceil(previous.width / 2)
                or level.height != math.ceil(previous.height / 2)
            ):
                raise ValueError("tiled-pyramid levels must use aligned 2x reduction")
            previous = level


def build_tiled_pyramid(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    tile_size: int = 512,
    coarsest_max_side: int = 512,
    max_source_pixels: Optional[int] = None,
) -> Path:
    """Build the persistent cache once; inference subsequently reads bounded tiles."""
    if tile_size <= 0 or coarsest_max_side <= 0:
        raise ValueError("tile_size and coarsest_max_side must be positive")
    if max_source_pixels is not None and max_source_pixels <= 0:
        raise ValueError("max_source_pixels must be positive when supplied")
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"tiled-pyramid output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    source_stat = image_path.stat()
    source_sha256 = _file_sha256(image_path)
    with Image.open(image_path) as opened:
        if max_source_pixels is not None and opened.width * opened.height > max_source_pixels:
            raise ValueError("source exceeds the configured pyramid-build pixel cap")
        source_width, source_height = opened.size
        current = opened.convert("RGB")
        current.load()

    with TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temporary:
        temporary_root = Path(temporary)
        levels = []
        level_index = 0
        while True:
            level_dir = temporary_root / f"level_{level_index}"
            level_dir.mkdir()
            columns = math.ceil(current.width / tile_size)
            rows = math.ceil(current.height / tile_size)
            for row in range(rows):
                for column in range(columns):
                    left, top = column * tile_size, row * tile_size
                    tile = current.crop((
                        left,
                        top,
                        min(current.width, left + tile_size),
                        min(current.height, top + tile_size),
                    ))
                    tile.save(level_dir / f"{column}_{row}.png", format="PNG")
            levels.append({
                "index": level_index,
                "width": current.width,
                "height": current.height,
                "columns": columns,
                "rows": rows,
            })
            if max(current.size) <= coarsest_max_side:
                break
            current = current.resize(
                (math.ceil(current.width / 2), math.ceil(current.height / 2)),
                _BICUBIC,
            )
            level_index += 1
        final_stat = image_path.stat()
        if (
            final_stat.st_size != source_stat.st_size
            or final_stat.st_mtime_ns != source_stat.st_mtime_ns
        ):
            raise RuntimeError("source changed while its tiled pyramid was being built")
        manifest = {
            "format_version": TILED_PYRAMID_FORMAT_VERSION,
            "kind": "foveatag_tiled_pyramid",
            "source": {
                "path": str(image_path.resolve()),
                "width": source_width,
                "height": source_height,
                "bytes": source_stat.st_size,
                "mtime_ns": source_stat.st_mtime_ns,
                "sha256": source_sha256,
            },
            "tile_size": tile_size,
            "encoding": "lossless_png_rgb",
            "levels": levels,
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_root, output_dir)
    return output_dir


def tiled_pyramid_cache_path(
    cache_root: str | Path, image_path: str | Path
) -> Path:
    """Map a source path to a stable cache location without decoding it."""
    normalized = os.path.normcase(str(Path(image_path).resolve()))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return Path(cache_root) / digest[:2] / digest


def build_cached_tiled_pyramid(
    image_path: str | Path,
    cache_root: str | Path,
    *,
    tile_size: int = 512,
    coarsest_max_side: int = 512,
    max_source_pixels: Optional[int] = None,
) -> Path:
    return build_tiled_pyramid(
        image_path,
        tiled_pyramid_cache_path(cache_root, image_path),
        tile_size=tile_size,
        coarsest_max_side=coarsest_max_side,
        max_source_pixels=max_source_pixels,
    )


def resolve_cached_tiled_pyramid(
    image_path: str | Path, cache_root: str | Path
) -> Optional[TiledPyramidSource]:
    root = tiled_pyramid_cache_path(cache_root, image_path)
    if not (root / "manifest.json").is_file():
        return None
    source = TiledPyramidSource(root)
    source.validate_source(image_path)
    return source


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
