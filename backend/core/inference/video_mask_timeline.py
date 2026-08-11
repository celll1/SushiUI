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
# The frontend timeline editor (frontend/src/utils/videoMaskTimeline.ts) caps a
# keyframe transform's scale at 100x and never lets it reach 0; this backend
# is the only place that actually enforces it, since a direct API caller
# bypasses the editor entirely. MIN guards against a positive-but-denormal
# value (e.g. 1e-320) blowing up `apply_mask_transform`'s matrix inverse into
# `inf`/`nan`, which used to surface as an opaque 400 far from its cause.
MAX_MASK_TRANSFORM_SCALE = 100.0
MIN_MASK_TRANSFORM_SCALE = 0.01
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

    if (
        values["scale_x"] < MIN_MASK_TRANSFORM_SCALE or values["scale_x"] > MAX_MASK_TRANSFORM_SCALE
        or values["scale_y"] < MIN_MASK_TRANSFORM_SCALE or values["scale_y"] > MAX_MASK_TRANSFORM_SCALE
    ):
        raise ManifestValidationError(
            f"{path}.scale_x and scale_y must be between {MIN_MASK_TRANSFORM_SCALE} and "
            f"{MAX_MASK_TRANSFORM_SCALE}"
        )
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
    source_center_xy = np.array(
        [(source.shape[1] - 1) / 2.0, (source.shape[0] - 1) / 2.0],
        dtype=np.float64,
    )
    target_center_xy = np.array(
        [(target_shape[1] - 1) / 2.0, (target_shape[0] - 1) / 2.0],
        dtype=np.float64,
    )
    offset_xy = source_center_xy - inverse_xy @ (
        target_center_xy + np.array([transform.x, transform.y], dtype=np.float64)
    )
    offset_rc = swap_xy_rc @ offset_xy
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


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """The (row, col) centroid of a mask's ``>= 0.5`` region, or None if empty."""
    ys, xs = np.nonzero(mask >= 0.5)
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def _shift_field(field: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    if shift[0] == 0.0 and shift[1] == 0.0:
        return field
    return ndimage.shift(field, shift, order=1, mode="nearest").astype(np.float32)


def _sdf_fields(mask: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
    """The signed-distance field and centroid a keyframe pair's SDF blend reuses.

    Computed once per (left, right) keyframe pair by ``rasterize_mask_timeline``
    and reused for every intermediate frame between them, rather than recomputed
    per frame: the distance transform is the expensive part of an SDF blend and
    does not depend on ``amount``.
    """
    return _signed_distance(mask), _mask_centroid(mask)


def _sdf_blend(
    left: np.ndarray,
    right: np.ndarray,
    signed_distance_left: np.ndarray,
    signed_distance_right: np.ndarray,
    centroid_left: tuple[float, float] | None,
    centroid_right: tuple[float, float] | None,
    amount: float,
) -> tuple[np.ndarray, bool]:
    """Centroid-aligned signed-distance morph between two masks.

    Blending two shapes' signed-distance fields directly (the naive approach)
    only has a positive (mask >= 0.5) region at a blend fraction where the two
    shapes' intermediate level sets actually overlap. Two shapes on opposite
    sides of the canvas -- or even two disjoint shapes a modest distance apart
    -- can blend to a field that is negative EVERYWHERE, i.e. an interpolated
    frame with no generate pixels at all, even though both endpoints are
    non-empty. A signed logistic keeps the field soft but does not change
    where it is negative, so it does not fix this.

    Aligning each field's centroid to the amount-weighted target centroid
    before blending keeps the two shapes registered on top of each other for
    the blend, which is what makes a morph between disjoint shapes produce a
    non-empty in-between frame. There is no "shift back": the target IS the
    common position both fields are shifted toward, not either source's own
    position.

    Returns ``(result, fell_back)``. ``fell_back`` is True when the blend
    still produced no generate pixels despite both endpoints being non-empty
    (this can still happen for shapes whose alignment does not make them
    overlap, e.g. very different sizes or elongated shapes at right angles);
    the caller then holds the nearer endpoint's mask instead of returning an
    empty frame.
    """
    if amount <= 0.0:
        return left.copy(), False
    if amount >= 1.0:
        return right.copy(), False

    sd_left = signed_distance_left
    sd_right = signed_distance_right
    if centroid_left is not None and centroid_right is not None:
        target = (
            centroid_left[0] + (centroid_right[0] - centroid_left[0]) * amount,
            centroid_left[1] + (centroid_right[1] - centroid_left[1]) * amount,
        )
        sd_left = _shift_field(
            sd_left, (target[0] - centroid_left[0], target[1] - centroid_left[1])
        )
        sd_right = _shift_field(
            sd_right, (target[0] - centroid_right[0], target[1] - centroid_right[1])
        )

    signed_distance = (1.0 - amount) * sd_left + amount * sd_right
    signed_distance = np.clip(signed_distance, -60.0, 60.0)
    result = (1.0 / (1.0 + np.exp(-signed_distance))).astype(np.float32)

    left_nonempty = centroid_left is not None
    right_nonempty = centroid_right is not None
    if left_nonempty and right_nonempty and not bool((result >= 0.5).any()):
        return (left.copy() if amount < 0.5 else right.copy()), True
    return result, False


def rasterize_mask_timeline(
    timeline: MaskTimelineManifest,
    mask_by_id: Mapping[str, Any],
    start_frame: int,
    end_frame: int,
    *,
    sdf_fallback_warnings: list[str] | None = None,
) -> list[np.ndarray]:
    """Rasterize ``[start_frame, end_frame)`` into one mask per frame.

    Every manifest keyframe must lie inside the requested range.  Frames before
    the first keyframe and after the last one use that nearest keyframe's
    transformed mask, which provides the requested terminal hold behavior.

    ``sdf_fallback_warnings``, if supplied, has at most one human-readable
    message appended PER KEYFRAME PAIR summarizing every frame in that pair's
    span where the ``sdf`` blend fell back to holding an endpoint's mask
    instead of returning an empty frame (see ``_sdf_blend``), rather than one
    message per individual frame (Medium-2 final-audit fix: an ``sdf``
    segment spanning a 100-frame clip could otherwise append up to 99
    near-identical messages into a response's ``warnings[]`` and the log).
    Left ``None`` this function does no I/O of its own, matching the module's
    no-server-dependency contract.
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
    # Keyed by (left_index, right_index): the signed-distance field and
    # centroid of each mask do not depend on `amount`, so they are computed
    # once per keyframe pair and reused for every intermediate frame between
    # them rather than recomputed per frame.
    sdf_field_cache: dict[tuple[int, int], tuple] = {}
    # Medium-2 (final audit): collect fallback frames per keyframe PAIR
    # instead of formatting a message per frame, so a long `sdf` segment
    # produces one summary instead of up to one message per frame.
    sdf_fallback_frames_by_pair: dict[tuple[int, int], list[int]] = {}

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
            # L-3: `parse_mask_timeline_manifest` already rejects an `affine`
            # keyframe pair whose mask_id differs (same error message below),
            # so this branch is unreachable through the parser -- but this
            # function also accepts a hand-built `MaskTimelineManifest` that
            # bypasses the parser (a test, or a future caller), and "affine"
            # names TRANSFORM interpolation of one mask, not a cross-fade
            # between two different mask assets (that would be a different,
            # not-yet-named mode), so a mismatched pair is refused here rather
            # than silently blended.
            if left_keyframe.mask_id != right_keyframe.mask_id:
                raise MaskRasterizationError(
                    "affine interpolation requires the same mask_id on both keyframes"
                )
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
        elif mode == "sdf":
            cache_key = (left_index, right_index)
            cached = sdf_field_cache.get(cache_key)
            if cached is None:
                cached = (_sdf_fields(left_mask), _sdf_fields(right_mask))
                sdf_field_cache[cache_key] = cached
            (sd_left, centroid_left), (sd_right, centroid_right) = cached
            blended, fell_back = _sdf_blend(
                left_mask, right_mask, sd_left, sd_right, centroid_left, centroid_right, amount,
            )
            if fell_back and sdf_fallback_warnings is not None:
                sdf_fallback_frames_by_pair.setdefault(
                    (left_keyframe.frame, right_keyframe.frame), []
                ).append(frame)
            rasterized.append(blended)
        else:
            raise MaskRasterizationError(f"unsupported interpolation mode: {mode!r}")

    if sdf_fallback_warnings is not None:
        for (left_frame, right_frame), frames in sdf_fallback_frames_by_pair.items():
            sdf_fallback_warnings.append(
                f"frames {frames[0]}-{frames[-1]} ({len(frames)} of "
                f"{right_frame - left_frame} frames) between keyframes at frames "
                f"{left_frame} and {right_frame}: sdf morph produced no generate pixels after "
                f"centroid alignment on those frames; held the nearer keyframe's mask instead."
            )

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


def feather_mask_edges(pixel_masks: Any, feather_px: float) -> np.ndarray:
    """Soften spatial mask edges without mixing neighboring video frames."""

    try:
        masks = np.asarray(pixel_masks, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise MaskRasterizationError("pixel_masks must be a numeric array") from exc
    if masks.ndim != 3:
        raise MaskRasterizationError("pixel_masks must have shape [T, H, W]")
    if not _is_number(feather_px) or not math.isfinite(float(feather_px)):
        raise MaskRasterizationError("feather_px must be a finite number")
    feather_px = float(feather_px)
    if feather_px < 0.0 or feather_px > 128.0:
        raise MaskRasterizationError("feather_px must be between 0 and 128")
    if not np.isfinite(masks).all() or np.any((masks < 0.0) | (masks > 1.0)):
        raise MaskRasterizationError("pixel_masks values must be finite and in [0, 1]")
    if feather_px == 0.0:
        return masks.copy()
    sigma = max(feather_px / 2.0, 0.5)
    softened = ndimage.gaussian_filter(
        masks,
        sigma=(0.0, sigma, sigma),
        mode="nearest",
        truncate=3.0,
    )
    return np.clip(softened, 0.0, 1.0).astype(np.float32, copy=False)


def validate_spatial_mask_plan_cheap(
    timeline: MaskTimelineManifest,
    mask_by_id: Mapping[str, Any],
    *,
    spatial_scale: int,
    patch_h: int,
    patch_w: int,
) -> None:
    """A cheap, keyframe-only pre-check of a SUBSET of the invariants
    ``build_spatial_mask_plan`` enforces, for a caller that wants to fail fast
    BEFORE reserving a GPU generation slot without paying for a full-clip
    rasterization.

    ``build_spatial_mask_plan`` rasterizes and max-pools EVERY frame in the
    regenerate range (``O(clip_frames * canvas_pixels)``); at the largest
    nominal MiniMax-H3 clip and canvas that call alone measured at 6.6-19.8s
    of CPU time (longer with ``sdf``, which recomputes distance transforms).
    Calling it twice per request -- once from the route to validate, once from
    the backend to actually use the result -- pays that cost twice for a
    result the route immediately discards (H-4). This function instead pools
    each keyframe's OWN transformed mask once (``O(unique keyframes)``, at
    most 128, typically far fewer, and with no distance transform), which
    catches the "generates at least one token" half of the invariant for
    every keyframe individually. ``timeline.composite_feather_px``, if
    nonzero, is applied to each keyframe's own transformed mask before
    pooling (Medium-2): feathering only softens edges independently per
    frame, so this is exactly the operation the full rasterization performs,
    just once per keyframe instead of once per output frame.

    ONLY the "generates a token" side is checked here, not "preserves a
    token": the preserve-side invariant in ``build_spatial_mask_plan`` is
    defined over the max-pool of the ENTIRE regenerate range, not any single
    keyframe (see ``generated_count >= total_count`` there) -- a keyframe
    whose own mask pools to fully white is a normal, legal way to say "fully
    regenerate this instant", as long as some OTHER frame in the range still
    preserves a token. Rejecting it per-keyframe would refuse manifests
    ``build_spatial_mask_plan`` accepts (e.g. a fully-white keyframe morphing
    into a partial-white keyframe later in the same range).

    NOT EXHAUSTIVE beyond that: it has no notion of interpolation between
    keyframes, so it cannot see a case where an ``affine`` cross-fade between
    two DIFFERENT mask assets thins an intermediate frame below the pooling
    threshold while every keyframe's OWN mask still pools fine on its own (two
    masks whose union covers a token but whose weighted average at some blend
    fraction does not). After the ``sdf`` centroid-alignment fix (H-1), an
    ``sdf`` segment cannot go empty when both its endpoints do not, since the
    fallback holds an endpoint's mask rather than returning an empty frame, so
    ``sdf`` segments ARE exhaustively covered by the "generates a token" check
    here. The full, per-frame-exact check (both the generate- and
    preserve-side invariants, evaluated over the whole range) still runs once
    in the backend (``pipeline_backends/minimax_h3.py``) before any GPU
    compute happens, so a manifest that slips past this cheap check is still
    refused before a single denoise step -- just after the generation slot is
    already reserved, rather than before.
    """
    masks = _validated_mask_mapping(mask_by_id, timeline)
    if not _is_int(spatial_scale) or spatial_scale <= 0:
        raise MaskRasterizationError("spatial_scale must be a positive integer")
    if not _is_int(patch_h) or not _is_int(patch_w) or patch_h <= 0 or patch_w <= 0:
        raise MaskRasterizationError("patch_h and patch_w must be positive integers")
    if timeline.canvas.height % spatial_scale or timeline.canvas.width % spatial_scale:
        raise MaskRasterizationError("mask canvas is not aligned to the latent token grid")
    latent_height = timeline.canvas.height // spatial_scale
    latent_width = timeline.canvas.width // spatial_scale
    if latent_height % patch_h or latent_width % patch_w:
        raise MaskRasterizationError("mask canvas is not aligned to the latent token grid")

    token_pixels = f"{spatial_scale * patch_h}x{spatial_scale * patch_w}"
    for keyframe in timeline.keyframes:
        transformed = apply_mask_transform(
            masks[keyframe.mask_id], keyframe.transform, output_shape=timeline.canvas.shape,
        )
        # Mirror `build_spatial_mask_plan`'s feathering (Medium-2): feathering
        # only softens edges, it never creates or destroys white area outside
        # an existing edge, so applying it to this single transformed frame
        # is exactly the per-frame operation the full rasterization performs
        # (`feather_mask_edges` uses sigma=0 on the time axis, i.e. every
        # frame is softened independently). Skipping this let a mask whose
        # generate region survives at full opacity but thins below the 0.5
        # pooling threshold once feathered pass the cheap check while the
        # full check correctly rejected it.
        if timeline.composite_feather_px:
            transformed = feather_mask_edges(transformed[None], timeline.composite_feather_px)[0]
        pooled = max_pool_mask_to_latent(
            transformed[None],
            spatial_scale * patch_h,
            spatial_scale * patch_w,
            generate_threshold=0.5,
        )
        if not pooled.any():
            raise MaskRasterizationError(
                f"keyframe at frame {keyframe.frame} generates no video token after max-pooling "
                f"its mask onto the {token_pixels}px latent token grid"
            )


def build_spatial_mask_plan(
    timeline: MaskTimelineManifest,
    mask_by_id: Mapping[str, Any],
    *,
    clip_frames: int,
    start_frame: int,
    end_frame: int,
    latent_frame_spans: Any,
    spatial_scale: int,
    patch_h: int,
    patch_w: int,
    sdf_fallback_warnings: list[str] | None = None,
    warnings: list[str] | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build full pixel masks and frame-major pinned token rows on the CPU.

    ``sdf_fallback_warnings``: see ``rasterize_mask_timeline``.

    ``warnings``, if supplied, also collects a single aggregate message
    reporting how many latent tokens had a max source-pixel value that was
    non-zero (some white was painted) but fell below the 0.5 max-pool
    generate threshold -- i.e. tokens that were partially requested but pin
    to source anyway because the effective mask granularity is one latent
    token (``spatial_scale * patch_h`` by ``spatial_scale * patch_w`` output
    pixels).
    """

    if not isinstance(timeline, MaskTimelineManifest):
        raise MaskRasterizationError("timeline must be a MaskTimelineManifest")
    if not all(_is_int(value) for value in (clip_frames, start_frame, end_frame)):
        raise MaskRasterizationError("clip and mask ranges must contain integers")
    if clip_frames <= 0 or not 0 <= start_frame < end_frame <= clip_frames:
        raise MaskRasterizationError("mask range must be inside the clip")
    if not _is_int(spatial_scale) or spatial_scale <= 0:
        raise MaskRasterizationError("spatial_scale must be a positive integer")
    if not _is_int(patch_h) or not _is_int(patch_w) or patch_h <= 0 or patch_w <= 0:
        raise MaskRasterizationError("patch_h and patch_w must be positive integers")

    spans = tuple((int(lo), int(hi)) for lo, hi in latent_frame_spans)
    if (
        not spans
        or spans[0][0] != 0
        or spans[-1][1] != clip_frames
        or any(lo < 0 or lo >= hi or hi > clip_frames for lo, hi in spans)
    ):
        raise MaskRasterizationError("latent frame spans must cover the clip exactly")

    sdf_target = sdf_fallback_warnings if sdf_fallback_warnings is not None else warnings
    range_masks = rasterize_mask_timeline(
        timeline, mask_by_id, start_frame, end_frame,
        sdf_fallback_warnings=sdf_target,
    )
    if warnings is not None and sdf_target is not None and sdf_target is not warnings:
        warnings.extend(sdf_target)
    range_array = np.asarray(range_masks, dtype=np.float32)
    expected_range_shape = (end_frame - start_frame, timeline.canvas.height, timeline.canvas.width)
    if range_array.shape != expected_range_shape:
        raise MaskRasterizationError(
            f"rasterized mask has shape {range_array.shape}, expected {expected_range_shape}"
        )
    if not np.isfinite(range_array).all() or np.any((range_array < 0.0) | (range_array > 1.0)):
        raise MaskRasterizationError("rasterized mask values must be finite and in [0, 1]")
    if timeline.composite_feather_px:
        range_array = feather_mask_edges(range_array, timeline.composite_feather_px)

    full_masks = np.zeros((clip_frames, timeline.canvas.height, timeline.canvas.width), dtype=np.float32)
    full_masks[start_frame:end_frame] = range_array
    latent_height = timeline.canvas.height // spatial_scale
    latent_width = timeline.canvas.width // spatial_scale
    if (
        timeline.canvas.height % spatial_scale
        or timeline.canvas.width % spatial_scale
        or latent_height % patch_h
        or latent_width % patch_w
    ):
        raise MaskRasterizationError("mask canvas is not aligned to the latent token grid")
    pooled_raw = max_pool_mask_to_latent(
        full_masks,
        spatial_scale * patch_h,
        spatial_scale * patch_w,
        generate_threshold=None,
    )
    expected_shape = (clip_frames, latent_height // patch_h, latent_width // patch_w)
    if pooled_raw.shape != expected_shape:
        raise MaskRasterizationError(
            f"pooled mask has shape {pooled_raw.shape}, expected {expected_shape}"
        )
    token_pixels = f"{spatial_scale * patch_h}x{spatial_scale * patch_w}"
    latent_generate_raw = np.stack([pooled_raw[lo:hi].max(axis=0) for lo, hi in spans], axis=0)
    latent_generate = (latent_generate_raw >= 0.5).astype(np.float32)
    generated_count = int(np.count_nonzero(latent_generate >= 0.5))
    total_count = int(latent_generate.size)
    if generated_count <= 0:
        raise MaskRasterizationError(
            "spatial mask must generate at least one video token: after max-pooling onto the "
            f"{token_pixels}px latent token grid, no token reached the 0.5 generate threshold. "
            "A generate region narrower than this grid, or a transform that scales a region's "
            "peak value below 0.5, pools to nothing even though some pixels were marked white."
        )
    if generated_count >= total_count:
        raise MaskRasterizationError(
            "spatial mask must preserve at least one video token: after max-pooling onto the "
            f"{token_pixels}px latent token grid, every token reached the 0.5 generate threshold."
        )
    pinned_rows = tuple(int(index) for index in np.flatnonzero(latent_generate.reshape(-1) < 0.5))
    if warnings is not None:
        partial_loss_count = int(
            np.count_nonzero((latent_generate_raw > 0.0) & (latent_generate_raw < 0.5))
        )
        if partial_loss_count:
            warnings.append(
                f"{partial_loss_count} of {total_count} latent token(s) had some non-zero "
                f"generate-mask coverage but stayed below the 0.5 max-pool threshold on the "
                f"{token_pixels}px latent token grid, so they were pinned to source instead of "
                "generated."
            )
    return full_masks, pinned_rows


def composite_masked_frames(
    source_frames: Any,
    generated_frames: Any,
    soft_masks: Any,
) -> np.ndarray:
    """Composite source and generated RGB frames with a soft generate mask.

    M-3: pixel-exact only where a mask value IS EXACTLY 0.0 (the returned
    pixel is then bit-identical to ``source_frames``, asserted by
    ``video_mask_timeline_test.py``). A latent token whose pooled mask value
    is below the 0.5 generate threshold is PINNED (its content is never
    denoised -- see ``build_spatial_mask_plan``'s ``pinned_rows``), but a
    feathered mask can still carry a non-zero, sub-0.5 value over that same
    pixel span, and this function blends by that continuous value, not by the
    token-level pin decision: the two operate at different granularities on
    purpose. The pin decides what the MODEL recomputes (one decision per
    latent token); this composite decides what the OUTPUT PIXELS show, and
    follows the manifest's own white=generate polarity continuously, which is
    the point of feathering -- a feathered edge that also became a hard
    pixel-level cut at the token boundary would produce the same double-edge
    artifact feathering exists to avoid. So a pixel inside a feather band can
    show partial (or even majority) generated content even though its own
    token was pinned; what it shows there is a coherent model output at that
    location's OWN pinned-content-adjacent context, not garbage -- it is
    exact "the model was never asked to change that token"'s effect on
    conditioning, not exact "that pixel is unmixed with generated content".
    """

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

    # Composited FRAME BY FRAME rather than as one float32 blend over the whole
    # clip: at the largest nominal MiniMax-H3 clip (362x768x1344) the whole-clip
    # float32 blend allocates 4-5 temporaries at ~4.5 GB each (~16 GB peak,
    # measured) on top of whatever the caller still has resident, which is
    # enough to OOM a host that just finished staging a 21 GB DiT. The
    # all-0/all-1 shortcut below used to be checked over the WHOLE clip, which
    # only ever fires for a uniform mask; checked per frame it also fires for
    # any frame entirely inside or entirely outside the regenerated range --
    # the common case for a spatial mask that only touches a fraction of the
    # clip's frames.
    output = np.empty_like(source)
    for frame_index in range(source.shape[0]):
        frame_mask = masks[frame_index]
        if not frame_mask.any():
            output[frame_index] = source[frame_index]
            continue
        if np.all(frame_mask == 1.0):
            output[frame_index] = generated[frame_index]
            continue
        amount = frame_mask[..., None]
        blended = (
            (1.0 - amount) * source[frame_index].astype(np.float32)
            + amount * generated[frame_index].astype(np.float32)
        )
        output[frame_index] = np.clip(np.rint(blended), 0.0, 255.0).astype(np.uint8)
    return output


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
    "build_spatial_mask_plan",
    "composite_masked_frames",
    "decode_mask_pngs",
    "feather_mask_edges",
    "max_pool_mask_to_latent",
    "parse_mask_timeline_manifest",
    "rasterize_mask_timeline",
    "validate_spatial_mask_plan_cheap",
]
