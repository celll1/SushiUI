"""Pure-Python utilities for video spatial-mask timelines.

The module deliberately has no model, server, or GPU dependency.  A timeline
uses output-canvas pixel coordinates and rasterizes to soft masks whose white
values mean "generate this pixel".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
import json
import math
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


MAX_MASK_PIXELS = 16_777_216
MAX_MASK_KEYFRAMES = 128
MAX_MASK_ASSETS = 64
MAX_TOTAL_MASK_PIXELS = 67_108_864
SDF_ZERO_THRESHOLD = 1.0 / 255.0
_INTERPOLATION_MODES = frozenset({"hold", "affine", "sdf"})
_ROOT_FIELDS = frozenset(
    {"version", "coordinate_space", "polarity", "canvas", "keyframes", "composite_feather_px"}
)
_CANVAS_FIELDS = frozenset({"width", "height"})
_KEYFRAME_FIELDS = frozenset(
    {"id", "frame", "mask_id", "interpolation_to_next", "transform"}
)
_TRANSFORM_FIELDS = frozenset({"x", "y", "scale_x", "scale_y", "rotation"})


class VideoMaskTimelineError(ValueError):
    """Base error for invalid manifests, masks, and rasterization inputs."""


class ManifestValidationError(VideoMaskTimelineError):
    """Raised when a timeline manifest does not satisfy its schema."""


class MaskDecodeError(VideoMaskTimelineError):
    """Raised when a keyframe mask cannot be decoded or does not fit the canvas."""


class MaskRasterizationError(VideoMaskTimelineError):
    """Raised when timeline masks cannot be rasterized or composited."""


@dataclass(frozen=True)
class MaskCanvas:
    """The output canvas dimensions used by every timeline mask."""

    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


@dataclass(frozen=True)
class MaskTransform:
    """A transform in output-canvas pixels and image-coordinate degrees.

    ``x`` and ``y`` translate the mask right and down.  Positive rotation is
    clockwise on an image array because its y axis points down.
    """

    x: float = 0.0
    y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0


@dataclass(frozen=True)
class MaskKeyframe:
    """One mask asset and interpolation mode starting at a frame."""

    frame: int
    mask_id: str
    interpolation_to_next: str
    transform: MaskTransform = MaskTransform()
    id: str | None = None


@dataclass(frozen=True)
class MaskTimelineManifest:
    """Validated version-one spatial-mask timeline manifest."""

    version: int
    coordinate_space: str
    polarity: str
    canvas: MaskCanvas
    keyframes: tuple[MaskKeyframe, ...]
    composite_feather_px: float = 0.0


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{path} must be an object")
    return value


def _require_string(value: Any, path: str, *, non_empty: bool = False) -> str:
    if not isinstance(value, str) or (non_empty and not value):
        suffix = " and non-empty" if non_empty else ""
        raise ManifestValidationError(f"{path} must be a string{suffix}")
    return value


def _require_int(value: Any, path: str) -> int:
    if not _is_int(value):
        raise ManifestValidationError(f"{path} must be an integer")
    return value


def _parse_transform(value: Any, path: str) -> MaskTransform:
    if value is None:
        raise ManifestValidationError(f"{path} must be an object when provided")
    transform = _require_mapping(value, path)
    unknown = set(transform) - _TRANSFORM_FIELDS
    if unknown:
        raise ManifestValidationError(
            f"{path} contains unknown fields: {sorted(map(str, unknown))}"
        )

    values: dict[str, float] = {}
    defaults = {"x": 0.0, "y": 0.0, "scale_x": 1.0, "scale_y": 1.0, "rotation": 0.0}
    for field_name, default in defaults.items():
        raw_value = transform.get(field_name, default)
        if not _is_number(raw_value):
            raise ManifestValidationError(f"{path}.{field_name} must be a number")
        numeric_value = float(raw_value)
        if not math.isfinite(numeric_value):
            raise ManifestValidationError(f"{path}.{field_name} must be finite")
        values[field_name] = numeric_value

    if values["scale_x"] <= 0 or values["scale_y"] <= 0:
        raise ManifestValidationError(f"{path}.scale_x and scale_y must be positive")
    return MaskTransform(**values)


def parse_mask_timeline_manifest(
    manifest: str | bytes | bytearray | Mapping[str, Any],
) -> MaskTimelineManifest:
    """Parse and validate a version-one spatial-mask manifest.

    The keyframe frame numbers are decoded-video indices.  They must be
    strictly ascending and are not implicitly clamped or sorted.
    """

    if isinstance(manifest, Mapping):
        data: Any = manifest
    elif isinstance(manifest, (str, bytes, bytearray)):
        try:
            data = json.loads(manifest)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestValidationError("manifest must contain valid JSON") from exc
    else:
        raise ManifestValidationError("manifest must be a JSON string or object")

    root = _require_mapping(data, "manifest")
    unknown_root = set(root) - _ROOT_FIELDS
    if unknown_root:
        raise ManifestValidationError(
            f"manifest contains unknown fields: {sorted(map(str, unknown_root))}"
        )
    version = _require_int(root.get("version"), "version")
    if version != 1:
        raise ManifestValidationError("version must be 1")
    coordinate_space = _require_string(root.get("coordinate_space"), "coordinate_space")
    if coordinate_space != "output_canvas":
        raise ManifestValidationError("coordinate_space must be 'output_canvas'")
    polarity = _require_string(root.get("polarity"), "polarity")
    if polarity != "white_generate":
        raise ManifestValidationError("polarity must be 'white_generate'")

    canvas_data = _require_mapping(root.get("canvas"), "canvas")
    unknown_canvas = set(canvas_data) - _CANVAS_FIELDS
    if unknown_canvas:
        raise ManifestValidationError(
            f"canvas contains unknown fields: {sorted(map(str, unknown_canvas))}"
        )
    canvas_width = _require_int(canvas_data.get("width"), "canvas.width")
    canvas_height = _require_int(canvas_data.get("height"), "canvas.height")
    if canvas_width <= 0 or canvas_height <= 0:
        raise ManifestValidationError("canvas.width and canvas.height must be positive")
    if canvas_width * canvas_height > MAX_MASK_PIXELS:
        raise ManifestValidationError("canvas is too large")
    canvas = MaskCanvas(width=canvas_width, height=canvas_height)

    raw_feather = root.get("composite_feather_px", 0.0)
    if not _is_number(raw_feather) or not math.isfinite(float(raw_feather)):
        raise ManifestValidationError("composite_feather_px must be a finite number")
    composite_feather_px = float(raw_feather)
    if composite_feather_px < 0.0 or composite_feather_px > 128.0:
        raise ManifestValidationError("composite_feather_px must be between 0 and 128")

    raw_keyframes = root.get("keyframes")
    if not isinstance(raw_keyframes, list) or not raw_keyframes:
        raise ManifestValidationError("keyframes must be a non-empty array")
    if len(raw_keyframes) > MAX_MASK_KEYFRAMES:
        raise ManifestValidationError(f"keyframes may contain at most {MAX_MASK_KEYFRAMES} entries")

    keyframes: list[MaskKeyframe] = []
    previous_frame = -1
    seen_keyframe_ids: set[str] = set()
    for index, raw_keyframe in enumerate(raw_keyframes):
        path = f"keyframes[{index}]"
        keyframe = _require_mapping(raw_keyframe, path)
        unknown_keyframe = set(keyframe) - _KEYFRAME_FIELDS
        if unknown_keyframe:
            raise ManifestValidationError(
                f"{path} contains unknown fields: {sorted(map(str, unknown_keyframe))}"
            )
        keyframe_id: str | None = None
        if "id" in keyframe:
            keyframe_id = _require_string(keyframe["id"], f"{path}.id", non_empty=True)
            if keyframe_id in seen_keyframe_ids:
                raise ManifestValidationError(f"duplicate keyframe id: {keyframe_id}")
            seen_keyframe_ids.add(keyframe_id)
        frame = _require_int(keyframe.get("frame"), f"{path}.frame")
        if frame < 0:
            raise ManifestValidationError(f"{path}.frame must be non-negative")
        if frame <= previous_frame:
            raise ManifestValidationError("keyframes must have strictly ascending frames")
        previous_frame = frame

        mask_id = _require_string(keyframe.get("mask_id"), f"{path}.mask_id", non_empty=True)
        interpolation = _require_string(
            keyframe.get("interpolation_to_next"),
            f"{path}.interpolation_to_next",
        )
        if interpolation not in _INTERPOLATION_MODES:
            allowed = ", ".join(sorted(_INTERPOLATION_MODES))
            raise ManifestValidationError(
                f"{path}.interpolation_to_next must be one of: {allowed}"
            )

        transform = (
            MaskTransform()
            if "transform" not in keyframe
            else _parse_transform(keyframe["transform"], f"{path}.transform")
        )
        keyframes.append(
            MaskKeyframe(
                frame=frame,
                mask_id=mask_id,
                interpolation_to_next=interpolation,
                transform=transform,
                id=keyframe_id,
            )
        )

    for left_keyframe, right_keyframe in zip(keyframes, keyframes[1:]):
        if (
            left_keyframe.interpolation_to_next == "affine"
            and left_keyframe.mask_id != right_keyframe.mask_id
        ):
            raise ManifestValidationError(
                "affine interpolation requires the same mask_id on both keyframes"
            )

    return MaskTimelineManifest(
        version=version,
        coordinate_space=coordinate_space,
        polarity=polarity,
        canvas=canvas,
        keyframes=tuple(keyframes),
        composite_feather_px=composite_feather_px,
    )


def _decode_png(png_bytes: bytes, canvas: MaskCanvas, mask_id: str, max_pixels: int) -> np.ndarray:
    try:
        with Image.open(BytesIO(png_bytes)) as image:
            if image.format != "PNG":
                raise MaskDecodeError(f"mask {mask_id!r} must be a PNG")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise MaskDecodeError(f"mask {mask_id!r} is too large")
            if (width, height) != (canvas.width, canvas.height):
                raise MaskDecodeError(
                    f"mask {mask_id!r} has size {(width, height)}, "
                    f"expected {(canvas.width, canvas.height)}"
                )
            image.load()
            if image.mode == "L":
                values = np.asarray(image, dtype=np.float32) / 255.0
            elif image.mode == "LA":
                values = np.asarray(image.getchannel(0), dtype=np.float32) / 255.0
            elif image.mode in {"RGB", "RGBA"}:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
                values = (
                    0.299 * rgb[..., 0]
                    + 0.587 * rgb[..., 1]
                    + 0.114 * rgb[..., 2]
                ) / 255.0
            else:
                raise MaskDecodeError(
                    f"mask {mask_id!r} must use an 8-bit L, LA, RGB, or RGBA PNG"
                )
    except MaskDecodeError:
        raise
    except Exception as exc:
        raise MaskDecodeError(f"mask {mask_id!r} is not a readable PNG") from exc

    values = np.asarray(values, dtype=np.float32)
    if values.shape != canvas.shape or not np.isfinite(values).all():
        raise MaskDecodeError(f"mask {mask_id!r} decoded to an invalid array")
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


def decode_mask_pngs(
    png_by_mask_id: Mapping[str, bytes],
    timeline: MaskTimelineManifest,
    *,
    max_pixels: int = MAX_MASK_PIXELS,
) -> dict[str, np.ndarray]:
    """Decode one PNG per manifest mask ID into float32 ``[0, 1]`` arrays."""

    if not isinstance(timeline, MaskTimelineManifest):
        raise MaskDecodeError("timeline must be a MaskTimelineManifest")
    if not isinstance(png_by_mask_id, Mapping):
        raise MaskDecodeError("png_by_mask_id must be a mapping")
    if not _is_int(max_pixels) or max_pixels <= 0:
        raise MaskDecodeError("max_pixels must be a positive integer")
    if timeline.canvas.width * timeline.canvas.height > max_pixels:
        raise MaskDecodeError("canvas is too large for max_pixels")

    expected_ids = {keyframe.mask_id for keyframe in timeline.keyframes}
    actual_ids = set(png_by_mask_id)
    missing = expected_ids - actual_ids
    unknown = actual_ids - expected_ids
    if missing:
        raise MaskDecodeError(f"missing mask IDs: {sorted(missing)}")
    if unknown:
        raise MaskDecodeError(f"unknown mask IDs: {sorted(map(str, unknown))}")
    if len(expected_ids) > MAX_MASK_ASSETS:
        raise MaskDecodeError(f"at most {MAX_MASK_ASSETS} unique mask assets are allowed")
    total_pixels = timeline.canvas.width * timeline.canvas.height * len(expected_ids)
    if total_pixels > MAX_TOTAL_MASK_PIXELS:
        raise MaskDecodeError("decoded mask assets exceed the total pixel budget")

    decoded: dict[str, np.ndarray] = {}
    for mask_id in expected_ids:
        png_bytes = png_by_mask_id[mask_id]
        if not isinstance(png_bytes, (bytes, bytearray, memoryview)):
            raise MaskDecodeError(f"mask {mask_id!r} must be bytes")
        decoded[mask_id] = _decode_png(bytes(png_bytes), timeline.canvas, mask_id, max_pixels)
    return decoded


def _validate_mask_array(
    mask: Any,
    canvas: MaskCanvas,
    *,
    error_type: type[VideoMaskTimelineError],
    label: str,
) -> np.ndarray:
    try:
        array = np.asarray(mask, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise error_type(f"{label} must be a numeric array") from exc
    if array.shape != canvas.shape:
        raise error_type(f"{label} has shape {array.shape}, expected {canvas.shape}")
    if not np.isfinite(array).all():
        raise error_type(f"{label} contains non-finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise error_type(f"{label} values must be in [0, 1]")
    return array.astype(np.float32, copy=False)


def _validated_mask_mapping(
    mask_by_id: Mapping[str, Any],
    timeline: MaskTimelineManifest,
) -> dict[str, np.ndarray]:
    if not isinstance(mask_by_id, Mapping):
        raise MaskRasterizationError("mask_by_id must be a mapping")
    expected_ids = {keyframe.mask_id for keyframe in timeline.keyframes}
    actual_ids = set(mask_by_id)
    missing = expected_ids - actual_ids
    unknown = actual_ids - expected_ids
    if missing:
        raise MaskRasterizationError(f"missing mask IDs: {sorted(missing)}")
    if unknown:
        raise MaskRasterizationError(f"unknown mask IDs: {sorted(map(str, unknown))}")
    return {
        mask_id: _validate_mask_array(
            mask_by_id[mask_id],
            timeline.canvas,
            error_type=MaskRasterizationError,
            label=f"mask {mask_id!r}",
        )
        for mask_id in expected_ids
    }


def _validate_output_shape(output_shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(output_shape, tuple)
        or len(output_shape) != 2
        or not all(_is_int(value) for value in output_shape)
        or any(value <= 0 for value in output_shape)
    ):
        raise MaskRasterizationError("output_shape must be a pair of positive integers")
    return output_shape


def apply_mask_transform(
    mask: np.ndarray,
    transform: MaskTransform,
    *,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply a canvas-pixel transform and return a float32 soft mask."""

    if not isinstance(transform, MaskTransform):
        raise MaskRasterizationError("transform must be a MaskTransform")
    source = np.asarray(mask, dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise MaskRasterizationError("mask must be a finite two-dimensional array")
    if np.any((source < 0.0) | (source > 1.0)):
        raise MaskRasterizationError("mask values must be in [0, 1]")
    target_shape = source.shape if output_shape is None else _validate_output_shape(output_shape)

    angle = math.radians(transform.rotation)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    forward_xy = np.array(
        [
            [transform.scale_x * cosine, -transform.scale_y * sine],
            [transform.scale_x * sine, transform.scale_y * cosine],
        ],
        dtype=np.float64,
    )
    if (
        source.shape == target_shape
        and np.array_equal(forward_xy, np.eye(2))
        and transform.x == 0.0
        and transform.y == 0.0
    ):
        return source.copy()

    try:
        inverse_xy = np.linalg.inv(forward_xy)
    except np.linalg.LinAlgError as exc:
        raise MaskRasterizationError("transform is not invertible") from exc
    swap_xy_rc = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    matrix_rc = swap_xy_rc @ inverse_xy @ swap_xy_rc
    offset_rc = -swap_xy_rc @ inverse_xy @ np.array(
        [transform.x, transform.y], dtype=np.float64
    )
    transformed = ndimage.affine_transform(
        source,
        matrix=matrix_rc,
        offset=offset_rc,
        output_shape=target_shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.asarray(transformed, dtype=np.float32)


def _interpolate_transform(left: MaskTransform, right: MaskTransform, amount: float) -> MaskTransform:
    return MaskTransform(
        x=left.x + (right.x - left.x) * amount,
        y=left.y + (right.y - left.y) * amount,
        scale_x=left.scale_x + (right.scale_x - left.scale_x) * amount,
        scale_y=left.scale_y + (right.scale_y - left.scale_y) * amount,
        rotation=left.rotation + (right.rotation - left.rotation) * amount,
    )


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = mask >= 0.5
    distance_inside = ndimage.distance_transform_edt(inside)
    distance_outside = ndimage.distance_transform_edt(~inside)
    return (distance_inside - distance_outside).astype(np.float32)


def _sdf_interpolate(left: np.ndarray, right: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return left.copy()
    if amount >= 1.0:
        return right.copy()
    signed_distance = (1.0 - amount) * _signed_distance(left) + amount * _signed_distance(right)
    # A hard sign threshold can erase a translated region when the two
    # intermediate level sets do not overlap.  A unit-width logistic keeps the
    # interpolated field soft while retaining the signed-distance boundary.
    signed_distance = np.clip(signed_distance, -60.0, 60.0)
    result = (1.0 / (1.0 + np.exp(-signed_distance))).astype(np.float32)
    result[result < SDF_ZERO_THRESHOLD] = 0.0
    result[result > 1.0 - SDF_ZERO_THRESHOLD] = 1.0
    return result


def rasterize_mask_timeline(
    timeline: MaskTimelineManifest,
    mask_by_id: Mapping[str, Any],
    start_frame: int,
    end_frame: int,
) -> list[np.ndarray]:
    """Rasterize ``[start_frame, end_frame)`` into one mask per frame.

    Every manifest keyframe must lie inside the requested range.  Frames before
    the first keyframe and after the last one use that nearest keyframe's
    transformed mask, which provides the requested terminal hold behavior.
    """

    if not isinstance(timeline, MaskTimelineManifest):
        raise MaskRasterizationError("timeline must be a MaskTimelineManifest")
    if not _is_int(start_frame) or not _is_int(end_frame):
        raise MaskRasterizationError("frame range must contain integer frame numbers")
    if start_frame < 0 or end_frame <= start_frame:
        raise MaskRasterizationError("frame range must be non-empty and end after start")
    if any(
        keyframe.frame < start_frame or keyframe.frame >= end_frame
        for keyframe in timeline.keyframes
    ):
        raise MaskRasterizationError("all keyframes must lie inside the requested frame range")

    masks = _validated_mask_mapping(mask_by_id, timeline)
    keyframes = timeline.keyframes
    transformed_keyframes = [
        apply_mask_transform(
            masks[keyframe.mask_id],
            keyframe.transform,
            output_shape=timeline.canvas.shape,
        )
        for keyframe in keyframes
    ]
    frame_numbers = [keyframe.frame for keyframe in keyframes]
    rasterized: list[np.ndarray] = []

    for frame in range(start_frame, end_frame):
        right_index = 0
        while right_index < len(frame_numbers) and frame_numbers[right_index] <= frame:
            right_index += 1
        left_index = right_index - 1

        if left_index < 0:
            rasterized.append(transformed_keyframes[0].copy())
            continue
        if right_index >= len(keyframes):
            rasterized.append(transformed_keyframes[-1].copy())
            continue

        left_keyframe = keyframes[left_index]
        right_keyframe = keyframes[right_index]
        left_mask = transformed_keyframes[left_index]
        right_mask = transformed_keyframes[right_index]
        amount = (frame - left_keyframe.frame) / (right_keyframe.frame - left_keyframe.frame)
        mode = left_keyframe.interpolation_to_next

        if mode == "hold":
            rasterized.append(left_mask.copy())
        elif mode == "affine":
            if left_keyframe.mask_id == right_keyframe.mask_id:
                interpolated = _interpolate_transform(
                    left_keyframe.transform,
                    right_keyframe.transform,
                    amount,
                )
                rasterized.append(
                    apply_mask_transform(
                        masks[left_keyframe.mask_id],
                        interpolated,
                        output_shape=timeline.canvas.shape,
                    )
                )
            else:
                rasterized.append(
                    ((1.0 - amount) * left_mask + amount * right_mask).astype(np.float32)
                )
        elif mode == "sdf":
            rasterized.append(_sdf_interpolate(left_mask, right_mask, amount))
        else:
            raise MaskRasterizationError(f"unsupported interpolation mode: {mode!r}")

    return rasterized


def max_pool_mask_to_latent(
    pixel_masks: Any,
    patch_h: int,
    patch_w: int,
    *,
    generate_threshold: float | None = 0.5,
) -> np.ndarray:
    """Max-pool ``[T, H, W]`` generate masks onto a patch grid.

    The result is ordered as ``[T, latent_h, latent_w]`` (H-major, then
    W-major).  Partial edge patches are included using zero padding.  Since
    white means generate, max pooling expands a patch whenever any source
    pixel requests it.  ``generate_threshold`` can turn soft masks into an
    explicit generate/preserve decision for latent pinning.
    """

    if not _is_int(patch_h) or not _is_int(patch_w) or patch_h <= 0 or patch_w <= 0:
        raise MaskRasterizationError("patch_h and patch_w must be positive integers")
    if generate_threshold is not None and (
        not _is_number(generate_threshold)
        or not math.isfinite(float(generate_threshold))
        or not 0.0 <= float(generate_threshold) <= 1.0
    ):
        raise MaskRasterizationError("generate_threshold must be a finite number in [0, 1]")
    try:
        masks = np.asarray(pixel_masks, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise MaskRasterizationError("pixel_masks must be a numeric array") from exc
    if masks.ndim != 3:
        raise MaskRasterizationError("pixel_masks must have shape [T, H, W]")
    if not np.isfinite(masks).all():
        raise MaskRasterizationError("pixel_masks contains non-finite values")
    if np.any((masks < 0.0) | (masks > 1.0)):
        raise MaskRasterizationError("pixel_masks values must be in [0, 1]")

    frames, height, width = masks.shape
    latent_height = (height + patch_h - 1) // patch_h
    latent_width = (width + patch_w - 1) // patch_w
    pooled = np.zeros((frames, latent_height, latent_width), dtype=np.float32)
    for latent_y in range(latent_height):
        y0 = latent_y * patch_h
        y1 = min(y0 + patch_h, height)
        for latent_x in range(latent_width):
            x0 = latent_x * patch_w
            x1 = min(x0 + patch_w, width)
            pooled[:, latent_y, latent_x] = np.max(masks[:, y0:y1, x0:x1], axis=(1, 2))
    if generate_threshold is not None:
        return (pooled >= float(generate_threshold)).astype(np.float32)
    return pooled


def composite_masked_frames(
    source_frames: Any,
    generated_frames: Any,
    soft_masks: Any,
) -> np.ndarray:
    """Composite source and generated RGB frames with a soft generate mask."""

    source = np.asarray(source_frames)
    generated = np.asarray(generated_frames)
    masks = np.asarray(soft_masks, dtype=np.float32)
    if source.ndim != 4 or source.shape[-1] != 3:
        raise MaskRasterizationError("source_frames must have shape [T, H, W, 3]")
    if generated.shape != source.shape:
        raise MaskRasterizationError("generated_frames must match source_frames shape")
    if masks.shape != source.shape[:3]:
        raise MaskRasterizationError("soft_masks must have shape [T, H, W]")
    if source.dtype != np.uint8 or generated.dtype != np.uint8:
        raise MaskRasterizationError("frame arrays must use uint8 RGB values")
    if not np.isfinite(source).all() or not np.isfinite(generated).all():
        raise MaskRasterizationError("frame arrays must contain finite values")
    if not np.isfinite(masks).all() or np.any((masks < 0.0) | (masks > 1.0)):
        raise MaskRasterizationError("soft_masks values must be finite and in [0, 1]")

    amount = masks[..., None]
    if np.all(masks == 0.0):
        return source.copy()
    if np.all(masks == 1.0):
        return generated.copy()
    blended = (1.0 - amount) * source.astype(np.float32) + amount * generated.astype(np.float32)
    return np.clip(np.rint(blended), 0.0, 255.0).astype(np.uint8)


__all__ = [
    "MAX_MASK_PIXELS",
    "ManifestValidationError",
    "MaskCanvas",
    "MaskDecodeError",
    "MaskKeyframe",
    "MaskRasterizationError",
    "MaskTimelineManifest",
    "MaskTransform",
    "VideoMaskTimelineError",
    "apply_mask_transform",
    "composite_masked_frames",
    "decode_mask_pngs",
    "max_pool_mask_to_latent",
    "parse_mask_timeline_manifest",
    "rasterize_mask_timeline",
]
