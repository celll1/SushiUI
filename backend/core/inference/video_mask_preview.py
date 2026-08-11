"""Pure-Python downscaling and sprite assembly for a video mask timeline preview.

This module adds NOTHING to the generation-path rasterization itself -- it
reuses `video_mask_timeline.rasterize_mask_timeline` VERBATIM, unmodified, so
a preview and a real generation call rasterize an identical manifest into
identical soft masks. What this module adds is preview-only bookkeeping on
top of that output: bounding an arbitrary, possibly-sparse requested-frame
list to a single rasterization call, downscaling every returned frame the
same way, and packing them into one sprite strip PNG so a scrubbing UI needs
one HTTP round trip instead of one per frame.

No model, server, or GPU dependency, matching the module it wraps.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from core.inference.video_mask_timeline import (
    MaskRasterizationError,
    MaskTimelineManifest,
    VideoMaskTimelineError,
    rasterize_mask_timeline,
)

MAX_MASK_PREVIEW_FRAMES = 64
# Bounds the SPAN `rasterize_mask_timeline` is asked to cover -- every
# manifest keyframe's frame number UNION every requested frame -- rather than
# bounding how many frames are actually returned (`MAX_MASK_PREVIEW_FRAMES`
# already does that). `rasterize_mask_timeline` requires every manifest
# keyframe to lie inside the contiguous range it rasterizes, so a manifest
# with only two keyframes a huge distance apart would otherwise force a huge
# contiguous rasterization no matter how few frames were actually requested
# near them. No shipped video architecture's own clip-length ceiling exceeds
# 362 frames (MiniMax-H3's documented maximum); this is a generous multiple
# of that, not a per-architecture value, since this preview endpoint loads no
# model and therefore has no architecture context to size the cap from.
MAX_MASK_PREVIEW_SPAN_FRAMES = 4096
MIN_PREVIEW_MAX_SIZE = 16
MAX_PREVIEW_MAX_SIZE = 1024


class MaskPreviewError(VideoMaskTimelineError):
    """Raised for a preview-only input error (the frame list or max_size)."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_frames(frames: Any) -> list[int]:
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
        raise MaskPreviewError("frames must be an array of integers")
    if not frames:
        raise MaskPreviewError("frames must be a non-empty array")
    if len(frames) > MAX_MASK_PREVIEW_FRAMES:
        raise MaskPreviewError(
            f"frames may contain at most {MAX_MASK_PREVIEW_FRAMES} entries"
        )
    seen: set[int] = set()
    validated: list[int] = []
    for value in frames:
        if not _is_int(value):
            raise MaskPreviewError("frames entries must be integers")
        if value < 0:
            raise MaskPreviewError("frames entries must be non-negative")
        if value in seen:
            raise MaskPreviewError(f"duplicate frame number: {value}")
        seen.add(value)
        validated.append(value)
    return sorted(validated)


def _validate_max_size(max_size: Any) -> int:
    if not _is_int(max_size):
        raise MaskPreviewError("max_size must be an integer")
    if max_size < MIN_PREVIEW_MAX_SIZE or max_size > MAX_PREVIEW_MAX_SIZE:
        raise MaskPreviewError(
            f"max_size must be between {MIN_PREVIEW_MAX_SIZE} and {MAX_PREVIEW_MAX_SIZE}"
        )
    return max_size


def _downscaled_size(canvas_width: int, canvas_height: int, max_size: int) -> tuple[int, int]:
    longest = max(canvas_width, canvas_height)
    scale = min(1.0, max_size / float(longest))
    width = max(1, round(canvas_width * scale))
    height = max(1, round(canvas_height * scale))
    return width, height


def build_mask_preview_strip(
    timeline: MaskTimelineManifest,
    mask_by_id: Mapping[str, Any],
    frames: Any,
    max_size: Any,
    *,
    sdf_fallback_warnings: list[str] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Rasterize the requested frames and pack them into one sprite strip PNG.

    Returns ``(png_bytes, metadata)``. ``metadata["frames"]`` lists the
    ascending, de-duplicated requested frame numbers in the same order they
    appear left-to-right in the strip, each carrying its own ``x_offset``;
    every frame shares one ``frame_width``/``frame_height`` (every frame in a
    timeline shares one canvas). ``metadata`` does not include ``warnings``
    or the PNG itself -- the caller decides how to surface those (this
    module's own contract is the raster + sprite geometry, not the response
    envelope, the same separation `video_mask_timeline` keeps between
    rasterizing and serving).
    """

    if not isinstance(timeline, MaskTimelineManifest):
        raise MaskPreviewError("timeline must be a MaskTimelineManifest")
    validated_frames = _validate_frames(frames)
    validated_max_size = _validate_max_size(max_size)

    keyframe_frames = [keyframe.frame for keyframe in timeline.keyframes]
    span_start = min(keyframe_frames + validated_frames)
    span_end = max(keyframe_frames + validated_frames) + 1
    if span_end - span_start > MAX_MASK_PREVIEW_SPAN_FRAMES:
        raise MaskPreviewError(
            "the manifest's keyframes and requested frames together span "
            f"{span_end - span_start} frames, more than the "
            f"{MAX_MASK_PREVIEW_SPAN_FRAMES}-frame preview limit"
        )

    # Unmodified call into the generation-path rasterizer: same manifest,
    # same masks, same function. This is the one call in this module that
    # can raise `MaskRasterizationError` for a reason unrelated to the
    # preview-only checks above (e.g. an sdf/affine mode error on a
    # hand-built manifest bypassing the parser); it is not caught here so the
    # caller sees the same error type `rasterize_mask_timeline` itself raises.
    rasterized = rasterize_mask_timeline(
        timeline, mask_by_id, span_start, span_end,
        sdf_fallback_warnings=sdf_fallback_warnings,
    )

    frame_width, frame_height = _downscaled_size(
        timeline.canvas.width, timeline.canvas.height, validated_max_size
    )
    needs_resize = (frame_width, frame_height) != (timeline.canvas.width, timeline.canvas.height)
    strip = Image.new("L", (frame_width * len(validated_frames), frame_height), color=0)
    frame_meta: list[dict[str, int]] = []
    for index, frame_number in enumerate(validated_frames):
        mask = rasterized[frame_number - span_start]
        pixel_values = np.clip(np.rint(mask * 255.0), 0.0, 255.0).astype(np.uint8)
        tile = Image.fromarray(pixel_values, mode="L")
        if needs_resize:
            tile = tile.resize((frame_width, frame_height), resample=Image.Resampling.LANCZOS)
        x_offset = index * frame_width
        strip.paste(tile, (x_offset, 0))
        frame_meta.append({"frame": frame_number, "x_offset": x_offset})

    buffer = BytesIO()
    strip.save(buffer, format="PNG")
    metadata = {
        "canvas_width": timeline.canvas.width,
        "canvas_height": timeline.canvas.height,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frames": frame_meta,
    }
    return buffer.getvalue(), metadata


__all__ = [
    "MAX_MASK_PREVIEW_FRAMES",
    "MAX_MASK_PREVIEW_SPAN_FRAMES",
    "MIN_PREVIEW_MAX_SIZE",
    "MAX_PREVIEW_MAX_SIZE",
    "MaskPreviewError",
    "build_mask_preview_strip",
]
