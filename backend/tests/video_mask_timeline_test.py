"""GPU-free tests for the video spatial-mask timeline utilities."""

from __future__ import annotations

from io import BytesIO
import os
import sys

import numpy as np
from PIL import Image
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND_ROOT = os.path.join(_REPO_ROOT, "backend")
for _path in (_REPO_ROOT, _BACKEND_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.inference.video_mask_timeline import (  # noqa: E402
    ManifestValidationError,
    MaskDecodeError,
    MaskRasterizationError,
    composite_masked_frames,
    decode_mask_pngs,
    max_pool_mask_to_latent,
    parse_mask_timeline_manifest,
    rasterize_mask_timeline,
)


def _manifest(keyframes, *, width=5, height=3):
    return {
        "version": 1,
        "coordinate_space": "output_canvas",
        "polarity": "white_generate",
        "canvas": {"width": width, "height": height},
        "keyframes": keyframes,
    }


def _png(values, mode="L"):
    image = Image.fromarray(np.asarray(values, dtype=np.uint8), mode=mode)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_manifest_validation_builds_readable_dataclasses():
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {
                    "frame": 2,
                    "mask_id": "subject",
                    "interpolation_to_next": "affine",
                    "transform": {"x": 1, "scale_x": 1.25},
                },
                {"frame": 5, "mask_id": "subject", "interpolation_to_next": "hold"},
            ]
        )
    )
    assert timeline.canvas.shape == (3, 5)
    assert timeline.keyframes[0].transform.x == 1.0
    assert timeline.keyframes[0].transform.scale_x == 1.25
    assert timeline.keyframes[0].transform.scale_y == 1.0


@pytest.mark.parametrize(
    "mutator",
    [
        lambda manifest: manifest.update(version=2),
        lambda manifest: manifest.update(coordinate_space="source_video"),
        lambda manifest: manifest.update(polarity="black_generate"),
        lambda manifest: manifest.update(
            keyframes=[
                {"frame": 3, "mask_id": "a", "interpolation_to_next": "hold"},
                {"frame": 3, "mask_id": "b", "interpolation_to_next": "hold"},
            ]
        ),
        lambda manifest: manifest.update(
            keyframes=[
                {
                    "frame": 1,
                    "mask_id": "a",
                    "interpolation_to_next": "hold",
                    "transform": {"rotation": float("nan")},
                }
            ]
        ),
    ],
)
def test_manifest_validation_rejects_invalid_values(mutator):
    manifest = _manifest([{"frame": 1, "mask_id": "a", "interpolation_to_next": "hold"}])
    mutator(manifest)
    with pytest.raises(ManifestValidationError):
        parse_mask_timeline_manifest(manifest)


def test_manifest_validation_rejects_unsorted_and_invalid_interpolation():
    with pytest.raises(ManifestValidationError):
        parse_mask_timeline_manifest(
            _manifest(
                [
                    {"frame": 4, "mask_id": "a", "interpolation_to_next": "hold"},
                    {"frame": 2, "mask_id": "b", "interpolation_to_next": "hold"},
                ]
            )
        )
    with pytest.raises(ManifestValidationError):
        parse_mask_timeline_manifest(
            _manifest([{"frame": 1, "mask_id": "a", "interpolation_to_next": "linear"}])
        )


def test_manifest_rejects_unknown_fields_and_cross_asset_affine():
    manifest = _manifest(
        [
            {"frame": 0, "mask_id": "a", "interpolation_to_next": "affine"},
            {"frame": 1, "mask_id": "b", "interpolation_to_next": "hold"},
        ]
    )
    with pytest.raises(ManifestValidationError, match="same mask_id"):
        parse_mask_timeline_manifest(manifest)
    manifest["unexpected"] = True
    with pytest.raises(ManifestValidationError, match="unknown fields"):
        parse_mask_timeline_manifest(manifest)


def test_decode_pngs_supports_luminance_and_ignores_alpha():
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "gray", "interpolation_to_next": "hold"},
                {"frame": 1, "mask_id": "rgb", "interpolation_to_next": "hold"},
            ],
            width=2,
            height=1,
        )
    )
    rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    alpha = np.array([[[0], [255]]], dtype=np.uint8)
    rgba = np.dstack((rgb, alpha))
    decoded = decode_mask_pngs(
        {"gray": _png([[0, 255]]), "rgb": _png(rgba, mode="RGBA")},
        timeline,
    )
    assert np.array_equal(decoded["gray"], np.array([[0.0, 1.0]], dtype=np.float32))
    assert np.allclose(decoded["rgb"], [[0.299, 0.587]], atol=1e-3)


def test_decode_pngs_rejects_unknown_ids_size_and_large_images():
    timeline = parse_mask_timeline_manifest(
        _manifest([{"frame": 0, "mask_id": "a", "interpolation_to_next": "hold"}], width=2, height=2)
    )
    with pytest.raises(MaskDecodeError, match="unknown"):
        decode_mask_pngs({"a": _png(np.zeros((2, 2))), "extra": _png(np.zeros((2, 2)))}, timeline)
    with pytest.raises(MaskDecodeError, match="size"):
        decode_mask_pngs({"a": _png(np.zeros((1, 2)))}, timeline)
    with pytest.raises(MaskDecodeError, match="too large"):
        decode_mask_pngs({"a": _png(np.zeros((2, 2)))}, timeline, max_pixels=3)


def test_hold_rasterization_fills_before_first_and_after_last_keyframe():
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 2, "mask_id": "a", "interpolation_to_next": "hold"},
                {"frame": 4, "mask_id": "b", "interpolation_to_next": "hold"},
            ]
        )
    )
    masks = {"a": np.ones((3, 5), dtype=np.float32), "b": np.zeros((3, 5), dtype=np.float32)}
    frames = rasterize_mask_timeline(timeline, masks, 0, 6)
    assert len(frames) == 6
    assert all(np.array_equal(frame, masks["a"]) for frame in frames[:4])
    assert all(np.array_equal(frame, masks["b"]) for frame in frames[4:])


def test_rasterization_rejects_keyframes_outside_requested_range():
    timeline = parse_mask_timeline_manifest(
        _manifest([{"frame": 3, "mask_id": "a", "interpolation_to_next": "hold"}])
    )
    with pytest.raises(MaskRasterizationError, match="inside"):
        rasterize_mask_timeline(timeline, {"a": np.zeros((3, 5))}, 0, 3)


def test_affine_rasterization_interpolates_a_mask_transform():
    base = np.zeros((3, 5), dtype=np.float32)
    base[1, 0] = 1.0
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {
                    "frame": 0,
                    "mask_id": "dot",
                    "interpolation_to_next": "affine",
                    "transform": {"x": 0},
                },
                {
                    "frame": 2,
                    "mask_id": "dot",
                    "interpolation_to_next": "hold",
                    "transform": {"x": 2},
                },
            ]
        )
    )
    frames = rasterize_mask_timeline(timeline, {"dot": base}, 0, 3)
    assert np.array_equal(frames[0], base)
    assert frames[1][1, 1] == pytest.approx(1.0)
    assert frames[2][1, 2] == pytest.approx(1.0)


def test_sdf_rasterization_preserves_endpoints_and_interpolates_boundaries():
    left = np.zeros((3, 5), dtype=np.float32)
    right = np.zeros((3, 5), dtype=np.float32)
    left[:, :2] = 1.0
    right[:, 3:] = 1.0
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "left", "interpolation_to_next": "sdf"},
                {"frame": 2, "mask_id": "right", "interpolation_to_next": "hold"},
            ]
        )
    )
    frames = rasterize_mask_timeline(timeline, {"left": left, "right": right}, 0, 3)
    assert np.array_equal(frames[0], left)
    assert np.array_equal(frames[2], right)
    assert np.all((frames[1] >= 0.0) & (frames[1] <= 1.0))
    assert np.all((frames[1][:, 1:4] > 0.0) & (frames[1][:, 1:4] < 1.0))


def test_max_pool_mask_to_latent_uses_generate_max_semantics_and_ceil_edges():
    masks = np.zeros((2, 5, 6), dtype=np.float32)
    masks[0, 1, 4] = 0.4
    masks[0, 4, 5] = 0.7
    masks[1, 0, 0] = 0.9
    pooled = max_pool_mask_to_latent(masks, patch_h=2, patch_w=4)
    assert pooled.shape == (2, 3, 2)
    np.testing.assert_allclose(pooled[0], [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(pooled[1], [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])


def test_max_pool_can_apply_explicit_generate_threshold_for_latent_pinning():
    masks = np.array([[[0.01, 0.6], [0.4, 0.0]]], dtype=np.float32)
    pooled = max_pool_mask_to_latent(masks, patch_h=1, patch_w=1)
    np.testing.assert_array_equal(pooled, [[[0.0, 1.0], [0.0, 0.0]]])


def test_composite_masked_frames_preserves_source_and_generated_pixels_exactly():
    source = np.array([[[[10, 20, 30], [40, 50, 60], [70, 80, 90]]]], dtype=np.uint8)
    generated = np.array([[[[110, 120, 130], [140, 150, 160], [170, 180, 190]]]], dtype=np.uint8)
    masks = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    result = composite_masked_frames(source, generated, masks)
    assert np.array_equal(result[0, 0, 0], source[0, 0, 0])
    assert np.array_equal(result[0, 0, 2], generated[0, 0, 2])
    assert np.array_equal(result[0, 0, 1], np.array([90, 100, 110], dtype=np.uint8))


def test_composite_rejects_non_uint8_frames():
    source = np.zeros((1, 1, 1, 3), dtype=np.float32)
    with pytest.raises(MaskRasterizationError, match="uint8"):
        composite_masked_frames(source, source, np.zeros((1, 1, 1), dtype=np.float32))
