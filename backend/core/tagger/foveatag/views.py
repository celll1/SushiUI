"""Coordinate-preserving overview and local-view construction for FoveaTag."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


@dataclass(frozen=True)
class SourceBox:
    """A possibly padded rectangle in source-image pixel coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0


@dataclass(frozen=True)
class ViewSpec:
    """Geometry needed to reproduce one model view and map evidence back."""

    role: str
    source_box: SourceBox
    canvas_width: int
    canvas_height: int
    content_box: Tuple[float, float, float, float]
    pixels_per_canvas_pixel: float

    def source_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        left, top, content_width, content_height = self.content_box
        canvas_x = left + (x - self.source_box.x) * content_width / self.source_box.width
        canvas_y = top + (y - self.source_box.y) * content_height / self.source_box.height
        return canvas_x, canvas_y

    def canvas_to_source(self, x: float, y: float) -> Tuple[float, float]:
        left, top, content_width, content_height = self.content_box
        source_x = self.source_box.x + (x - left) * self.source_box.width / content_width
        source_y = self.source_box.y + (y - top) * self.source_box.height / content_height
        return source_x, source_y

    def normalized_geometry(self, image_width: int, image_height: int) -> Tuple[float, ...]:
        """Continuous geometry features; no discrete resolution bucket is used."""
        center_x, center_y = self.source_box.center
        content_left, content_top, content_width, content_height = self.content_box
        field_fraction = math.sqrt(
            (self.source_box.width * self.source_box.height) / float(image_width * image_height)
        )
        return (
            center_x / image_width,
            center_y / image_height,
            self.source_box.width / image_width,
            self.source_box.height / image_height,
            math.log2(max(field_fraction, 1e-8)),
            math.log2(max(self.pixels_per_canvas_pixel, 1e-8)),
            content_left / self.canvas_width,
            content_top / self.canvas_height,
            content_width / self.canvas_width,
            content_height / self.canvas_height,
        )


@dataclass(frozen=True)
class ViewBatch:
    overview: Image.Image
    overview_spec: ViewSpec
    local_views: Tuple[Image.Image, ...]
    local_specs: Tuple[ViewSpec, ...]
    local_canvas_size: int


@dataclass(frozen=True)
class ViewConfig:
    overview_size: int = 448
    local_size: int = 256
    local_views: int = 4
    field_levels: Tuple[float, ...] = (2.0, 1.0, 0.5)
    scale_jitter: Tuple[float, float] = (0.8, 1.2)
    pad_color: Tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        if self.overview_size <= 0 or self.local_size <= 0:
            raise ValueError("view canvas sizes must be positive")
        if self.local_views < 0:
            raise ValueError("local_views must be non-negative")
        if len(self.field_levels) < 2 or any(level <= 0 for level in self.field_levels):
            raise ValueError("field_levels must contain at least two positive levels")
        low, high = self.scale_jitter
        if not (0 < low <= high):
            raise ValueError("scale_jitter must be a positive ordered range")


def overview_spec(image_width: int, image_height: int, canvas_size: int) -> ViewSpec:
    """Fit the full image into a square canvas without enlarging it."""
    if image_width <= 0 or image_height <= 0 or canvas_size <= 0:
        raise ValueError("image and canvas dimensions must be positive")
    scale = min(1.0, canvas_size / float(max(image_width, image_height)))
    content_width = max(1, round(image_width * scale))
    content_height = max(1, round(image_height * scale))
    left = (canvas_size - content_width) / 2.0
    top = (canvas_size - content_height) / 2.0
    return ViewSpec(
        role="overview",
        source_box=SourceBox(0.0, 0.0, float(image_width), float(image_height)),
        canvas_width=canvas_size,
        canvas_height=canvas_size,
        content_box=(left, top, float(content_width), float(content_height)),
        pixels_per_canvas_pixel=1.0 / scale,
    )


def identity_overview_spec(image_width: int, image_height: int) -> ViewSpec:
    """Describe an unchanged source image for a strict baseline control."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    return ViewSpec(
        role="baseline-overview",
        source_box=SourceBox(0.0, 0.0, float(image_width), float(image_height)),
        canvas_width=image_width,
        canvas_height=image_height,
        content_box=(0.0, 0.0, float(image_width), float(image_height)),
        pixels_per_canvas_pixel=1.0,
    )


def preserve_source_overview(image: Image.Image, views: ViewBatch) -> ViewBatch:
    """Keep local views but restore the baseline's unmodified overview input."""
    return ViewBatch(
        overview=image.copy(),
        overview_spec=identity_overview_spec(image.width, image.height),
        local_views=views.local_views,
        local_specs=views.local_specs,
        local_canvas_size=views.local_canvas_size,
    )


def local_spec(
    *,
    center_x: float,
    center_y: float,
    canvas_size: int,
    pixels_per_canvas_pixel: float,
    role: str = "local",
) -> ViewSpec:
    if canvas_size <= 0 or pixels_per_canvas_pixel <= 0:
        raise ValueError("canvas_size and pixels_per_canvas_pixel must be positive")
    source_side = canvas_size * pixels_per_canvas_pixel
    return ViewSpec(
        role=role,
        source_box=SourceBox(
            x=center_x - source_side / 2.0,
            y=center_y - source_side / 2.0,
            width=source_side,
            height=source_side,
        ),
        canvas_width=canvas_size,
        canvas_height=canvas_size,
        content_box=(0.0, 0.0, float(canvas_size), float(canvas_size)),
        pixels_per_canvas_pixel=pixels_per_canvas_pixel,
    )


def render_view(image: Image.Image, spec: ViewSpec, pad_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Render a view while preserving the exact affine transform in ``spec``."""
    image = image.convert("RGB")
    canvas = Image.new("RGB", (spec.canvas_width, spec.canvas_height), pad_color)
    box = spec.source_box
    source_left = max(0.0, box.x)
    source_top = max(0.0, box.y)
    source_right = min(float(image.width), box.x + box.width)
    source_bottom = min(float(image.height), box.y + box.height)
    if source_right <= source_left or source_bottom <= source_top:
        return canvas

    crop = image.crop((source_left, source_top, source_right, source_bottom))
    canvas_left, canvas_top = spec.source_to_canvas(source_left, source_top)
    canvas_right, canvas_bottom = spec.source_to_canvas(source_right, source_bottom)
    target_width = max(1, round(canvas_right - canvas_left))
    target_height = max(1, round(canvas_bottom - canvas_top))
    crop = crop.resize((target_width, target_height), _BICUBIC)
    canvas.paste(crop, (round(canvas_left), round(canvas_top)))
    return canvas


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert RGB PIL input to contiguous float32 CHW in [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _stratified_centers(
    image_width: int,
    image_height: int,
    count: int,
    rng: random.Random,
) -> List[Tuple[float, float]]:
    if count <= 0:
        return []
    columns = math.ceil(math.sqrt(count * image_width / max(image_height, 1)))
    columns = max(1, columns)
    rows = math.ceil(count / columns)
    cells = [(column, row) for row in range(rows) for column in range(columns)]
    rng.shuffle(cells)
    centers: List[Tuple[float, float]] = []
    for column, row in cells[:count]:
        x0 = column / columns
        x1 = (column + 1) / columns
        y0 = row / rows
        y1 = (row + 1) / rows
        centers.append((rng.uniform(x0, x1) * image_width, rng.uniform(y0, y1) * image_height))
    return centers


def sample_local_specs(
    image_width: int,
    image_height: int,
    config: ViewConfig,
    *,
    seed: int,
    training: bool,
) -> Tuple[ViewSpec, ...]:
    """Sample fixed-budget views with spatial coverage and multiple fields of view."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    rng = random.Random(seed)
    centers = _stratified_centers(image_width, image_height, config.local_views, rng)
    levels = list(config.field_levels)
    if training:
        rng.shuffle(levels)
    specs: List[ViewSpec] = []
    for index, (center_x, center_y) in enumerate(centers):
        base_level = levels[index % len(levels)]
        if training:
            low, high = config.scale_jitter
            jitter = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:
            jitter = 1.0
        density = max(0.5, base_level * jitter)
        specs.append(local_spec(
            center_x=center_x,
            center_y=center_y,
            canvas_size=config.local_size,
            pixels_per_canvas_pixel=density,
            role="local",
        ))
    return tuple(specs)


def build_views(
    image: Image.Image,
    config: ViewConfig,
    *,
    seed: int,
    training: bool,
    specs: Optional[Sequence[ViewSpec]] = None,
) -> ViewBatch:
    """Build an overview and local views; callers may replay explicit specs."""
    ov_spec = overview_spec(image.width, image.height, config.overview_size)
    local_specs = tuple(specs) if specs is not None else sample_local_specs(
        image.width, image.height, config, seed=seed, training=training,
    )
    overview = render_view(image, ov_spec, config.pad_color)
    locals_rendered = tuple(render_view(image, spec, config.pad_color) for spec in local_specs)
    return ViewBatch(
        overview=overview,
        overview_spec=ov_spec,
        local_views=locals_rendered,
        local_specs=local_specs,
        local_canvas_size=config.local_size,
    )


def stack_view_tensors(batch: ViewBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    overview = pil_to_tensor(batch.overview).unsqueeze(0)
    if batch.local_views:
        local = torch.stack([pil_to_tensor(image) for image in batch.local_views])
    else:
        local = torch.empty((0, 3, batch.local_canvas_size, batch.local_canvas_size), dtype=torch.float32)
    return overview, local
