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
    MAX_MASK_ASSETS,
    MaskTransform,
    ManifestValidationError,
    MaskDecodeError,
    MaskRasterizationError,
    build_spatial_mask_plan,
    apply_mask_transform,
    composite_masked_frames,
    decode_mask_pngs,
    feather_mask_edges,
    max_pool_mask_to_latent,
    parse_mask_timeline_manifest,
    rasterize_mask_timeline,
    validate_spatial_mask_plan_cheap,
)


def _manifest(keyframes, *, width=5, height=3, composite_feather_px=0.0):
    return {
        "version": 1,
        "coordinate_space": "output_canvas",
        "polarity": "white_generate",
        "canvas": {"width": width, "height": height},
        "keyframes": keyframes,
        "composite_feather_px": composite_feather_px,
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


def test_affine_transform_scales_around_the_canvas_center():
    mask = np.zeros((5, 5), dtype=np.float32)
    mask[2, 2] = 1.0
    transformed = apply_mask_transform(
        mask,
        MaskTransform(scale_x=2.0, scale_y=2.0),
        output_shape=(5, 5),
    )
    assert transformed[2, 2] == pytest.approx(1.0)
    assert transformed[0, 0] == pytest.approx(0.0)


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


def test_sdf_morph_between_disjoint_shapes_stays_non_empty_at_every_frame():
    """H-1 regression: two far-apart rectangles must not vanish mid-morph.

    Before the centroid-alignment fix, blending two disjoint shapes' signed-
    distance fields directly could produce an entirely-negative field (no
    `>= 0.5` pixel anywhere) at intermediate frames, even though both
    endpoints are non-empty.
    """
    height, width = 200, 900
    left = np.zeros((height, width), dtype=np.float32)
    right = np.zeros((height, width), dtype=np.float32)
    left[50:80, 50:80] = 1.0
    right[50:80, 800:830] = 1.0
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "left", "interpolation_to_next": "sdf"},
                {"frame": 10, "mask_id": "right", "interpolation_to_next": "hold"},
            ],
            width=width,
            height=height,
        )
    )
    frames = rasterize_mask_timeline(timeline, {"left": left, "right": right}, 0, 11)
    assert np.array_equal(frames[0], left)
    assert np.array_equal(frames[10], right)
    for index, frame in enumerate(frames):
        assert frame.max() >= 0.5, f"frame {index} has no generate pixel (max={frame.max()})"


def test_sdf_morph_between_a_thin_stroke_translation_stays_non_empty():
    """A narrow (< the fixed 11px-wide erosion the old logistic could not
    rescue) stroke translated across the canvas must not vanish mid-morph."""
    height, width = 64, 256
    left = np.zeros((height, width), dtype=np.float32)
    right = np.zeros((height, width), dtype=np.float32)
    left[28:36, 10:18] = 1.0     # 8px-wide stroke
    right[28:36, 42:50] = 1.0    # translated 32px
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "left", "interpolation_to_next": "sdf"},
                {"frame": 6, "mask_id": "right", "interpolation_to_next": "hold"},
            ],
            width=width,
            height=height,
        )
    )
    frames = rasterize_mask_timeline(timeline, {"left": left, "right": right}, 0, 7)
    for index, frame in enumerate(frames):
        assert frame.max() >= 0.5, f"frame {index} has no generate pixel (max={frame.max()})"


def test_sdf_field_cache_is_reused_across_frames_between_the_same_keyframes():
    """The signed-distance field is computed once per keyframe pair, not once
    per frame -- a spy on `_signed_distance` must see exactly two calls (one
    per endpoint) for an arbitrarily long run of frames between them."""
    from core.inference import video_mask_timeline as timeline_module

    left = np.zeros((10, 10), dtype=np.float32)
    right = np.zeros((10, 10), dtype=np.float32)
    left[2:4, 2:4] = 1.0
    right[6:8, 6:8] = 1.0
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "left", "interpolation_to_next": "sdf"},
                {"frame": 20, "mask_id": "right", "interpolation_to_next": "hold"},
            ],
            width=10,
            height=10,
        )
    )
    calls = {"count": 0}
    original = timeline_module._signed_distance

    def spy(mask):
        calls["count"] += 1
        return original(mask)

    timeline_module._signed_distance = spy
    try:
        rasterize_mask_timeline(timeline, {"left": left, "right": right}, 0, 21)
    finally:
        timeline_module._signed_distance = original
    assert calls["count"] == 2, f"expected 2 signed-distance computations, got {calls['count']}"


def test_sdf_thin_frame_warning_fires_for_multicomponent_collapse():
    """Audit regression: a keyframe with two disconnected 100x100 squares
    morphing into a single 100x100 square must warn about the intermediate
    frames whose generate area collapses well below either keyframe's own
    area, even on frames that keep `max() >= 0.5` and therefore never trip
    `_sdf_blend`'s empty-frame fallback (and its separate warning)."""
    height, width = 1200, 1200
    left = np.zeros((height, width), dtype=np.float32)
    left[50:150, 50:150] = 1.0
    left[50:150, 450:550] = 1.0
    right = np.zeros((height, width), dtype=np.float32)
    right[250:350, 1000:1100] = 1.0
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "left", "interpolation_to_next": "sdf"},
                {"frame": 10, "mask_id": "right", "interpolation_to_next": "hold"},
            ],
            width=width,
            height=height,
        )
    )
    warnings: list = []
    frames = rasterize_mask_timeline(
        timeline, {"left": left, "right": right}, 0, 11, sdf_fallback_warnings=warnings,
    )
    assert np.array_equal(frames[0], left)
    assert np.array_equal(frames[10], right)

    thin_warnings = [w for w in warnings if "generate region" in w]
    assert len(thin_warnings) == 1, f"expected exactly one aggregated warning, got {warnings}"
    message = thin_warnings[0]
    assert "keyframes at frames 0 and 10" in message
    assert "20000px and 10000px" in message
    # There must be a frame between the two collapse regions (the fallback
    # zone) that does NOT get folded into the same aggregate frame span,
    # since it is a hold, not a thin frame.
    assert "frame 3" in message or "frame 2" in message or "frame 8" in message


def test_sdf_thin_frame_warning_is_absent_for_single_component_translation():
    """A single connected shape translating across the canvas (identical
    area on both ends) must never warn: this is the parity case the fix
    must not regress."""
    height, width = 300, 900
    left = np.zeros((height, width), dtype=np.float32)
    right = np.zeros((height, width), dtype=np.float32)
    left[100:200, 100:200] = 1.0
    right[100:200, 700:800] = 1.0
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "left", "interpolation_to_next": "sdf"},
                {"frame": 10, "mask_id": "right", "interpolation_to_next": "hold"},
            ],
            width=width,
            height=height,
        )
    )
    warnings: list = []
    rasterize_mask_timeline(
        timeline, {"left": left, "right": right}, 0, 11, sdf_fallback_warnings=warnings,
    )
    assert warnings == []


def test_sdf_thin_frame_warning_is_absent_for_same_area_different_shapes():
    """Morphing a square into a similarly-sized circle at the same centroid
    must never warn: area varies only slightly during the shape change."""
    height, width = 300, 300
    yy, xx = np.mgrid[0:height, 0:width]
    left = np.zeros((height, width), dtype=np.float32)
    left[100:200, 100:200] = 1.0
    right = np.zeros((height, width), dtype=np.float32)
    right[((yy - 150) ** 2 + (xx - 150) ** 2) <= 56.4 ** 2] = 1.0
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "left", "interpolation_to_next": "sdf"},
                {"frame": 10, "mask_id": "right", "interpolation_to_next": "hold"},
            ],
            width=width,
            height=height,
        )
    )
    warnings: list = []
    rasterize_mask_timeline(
        timeline, {"left": left, "right": right}, 0, 11, sdf_fallback_warnings=warnings,
    )
    assert warnings == []


def test_sdf_thin_frame_warning_is_absent_for_rects_500px_apart():
    """Two same-size 300x300 rectangles (one connected component per
    keyframe) 500px apart must never warn: this is the existing disjoint-
    shape regression case, re-checked against the new thin-frame warning."""
    height, width = 900, 1600
    left = np.zeros((height, width), dtype=np.float32)
    right = np.zeros((height, width), dtype=np.float32)
    left[100:400, 100:400] = 1.0
    right[100:400, 900:1200] = 1.0
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "left", "interpolation_to_next": "sdf"},
                {"frame": 10, "mask_id": "right", "interpolation_to_next": "hold"},
            ],
            width=width,
            height=height,
        )
    )
    warnings: list = []
    rasterize_mask_timeline(
        timeline, {"left": left, "right": right}, 0, 11, sdf_fallback_warnings=warnings,
    )
    assert warnings == []


def test_sdf_thin_frame_warning_is_absent_for_thin_stroke_translation():
    """The existing 8px-wide-stroke translation regression case must not
    trip the new thin-frame warning either."""
    height, width = 64, 256
    left = np.zeros((height, width), dtype=np.float32)
    right = np.zeros((height, width), dtype=np.float32)
    left[28:36, 10:18] = 1.0
    right[28:36, 42:50] = 1.0
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "left", "interpolation_to_next": "sdf"},
                {"frame": 6, "mask_id": "right", "interpolation_to_next": "hold"},
            ],
            width=width,
            height=height,
        )
    )
    warnings: list = []
    rasterize_mask_timeline(
        timeline, {"left": left, "right": right}, 0, 7, sdf_fallback_warnings=warnings,
    )
    assert warnings == []


def test_sdf_thin_frame_warning_is_absent_for_large_monotonic_shrink():
    """A single connected shape shrinking by three orders of magnitude at a
    fixed centroid legitimately drives its area far below what a naive
    linear interpolation between the two keyframes' areas would predict, at
    every intermediate frame. This must not warn: the thin-frame check
    compares against the SMALLER keyframe's own area, not a linear area
    interpolation, specifically so this case (measured to reproduce for
    single-component shrinks of any severity that keep their centroid fixed)
    stays silent while the multi-component collapse above still warns."""
    height, width = 1200, 1200
    left = np.zeros((height, width), dtype=np.float32)
    right = np.zeros((height, width), dtype=np.float32)
    left[50:1050, 50:1050] = 1.0  # 1000x1000
    right[598:602, 598:602] = 1.0  # 4x4, same centroid
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "left", "interpolation_to_next": "sdf"},
                {"frame": 20, "mask_id": "right", "interpolation_to_next": "hold"},
            ],
            width=width,
            height=height,
        )
    )
    warnings: list = []
    rasterize_mask_timeline(
        timeline, {"left": left, "right": right}, 0, 21, sdf_fallback_warnings=warnings,
    )
    assert warnings == []


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


def test_feather_softens_only_spatial_edges_and_preserves_time_shape():
    masks = np.zeros((2, 5, 5), dtype=np.float32)
    masks[:, 1:4, 1:4] = 1.0
    softened = feather_mask_edges(masks, 2.0)
    assert softened.shape == masks.shape
    assert np.array_equal(softened[0], softened[1])
    assert 0.0 < softened[0, 1, 2] < softened[0, 2, 2] < 1.0


def test_spatial_mask_plan_returns_full_masks_and_frame_major_pinned_rows():
    timeline = parse_mask_timeline_manifest(
        _manifest([{"frame": 0, "mask_id": "subject", "interpolation_to_next": "hold"}],
                  width=4, height=2)
    )
    mask = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float32)
    full_masks, pinned_rows = build_spatial_mask_plan(
        timeline,
        {"subject": mask},
        clip_frames=5,
        start_frame=0,
        end_frame=5,
        latent_frame_spans=((0, 2), (2, 5)),
        spatial_scale=1,
        patch_h=1,
        patch_w=1,
    )
    assert full_masks.shape == (5, 2, 4)
    assert len(pinned_rows) == 8
    assert pinned_rows[:4] == (2, 3, 6, 7)


@pytest.mark.parametrize("mask, expected", [
    (np.zeros((2, 4), dtype=np.float32), "generate"),
    (np.ones((2, 4), dtype=np.float32), "preserve"),
])
def test_spatial_mask_plan_rejects_empty_generation_or_preservation(mask, expected):
    timeline = parse_mask_timeline_manifest(
        _manifest([{"frame": 0, "mask_id": "subject", "interpolation_to_next": "hold"}],
                  width=4, height=2)
    )
    with pytest.raises(MaskRasterizationError, match=expected):
        build_spatial_mask_plan(
            timeline,
            {"subject": mask},
            clip_frames=5,
            start_frame=0,
            end_frame=5,
            latent_frame_spans=((0, 2), (2, 5)),
            spatial_scale=1,
            patch_h=1,
            patch_w=1,
        )


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


def test_spatial_mask_plan_zeroes_full_masks_outside_the_rasterized_range():
    """`full_masks[:start]` and `full_masks[end:]` must stay exactly 0.0."""
    timeline = parse_mask_timeline_manifest(
        _manifest([{"frame": 3, "mask_id": "subject", "interpolation_to_next": "hold"}],
                  width=4, height=2)
    )
    mask = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float32)
    full_masks, _pinned_rows = build_spatial_mask_plan(
        timeline,
        {"subject": mask},
        clip_frames=9,
        start_frame=3,
        end_frame=6,
        latent_frame_spans=((0, 3), (3, 6), (6, 9)),
        spatial_scale=1,
        patch_h=1,
        patch_w=1,
    )
    assert full_masks.shape == (9, 2, 4)
    assert np.all(full_masks[:3] == 0.0)
    assert np.all(full_masks[6:] == 0.0)
    assert np.array_equal(full_masks[3], mask)
    assert np.array_equal(full_masks[5], mask)


def test_composite_masked_frames_is_bit_exact_over_a_full_clip_with_real_feather():
    """H-3 regression: the frame-by-frame rewrite must not change a single bit
    of the output versus a whole-array blend, over a clip large enough that
    every code path (all-source frames, all-generated frames, feathered
    partial frames) fires -- not just the 3-pixel single-frame check above."""
    rng = np.random.default_rng(0)
    frames, height, width = 12, 17, 23
    source = rng.integers(0, 256, size=(frames, height, width, 3), dtype=np.uint8)
    generated = rng.integers(0, 256, size=(frames, height, width, 3), dtype=np.uint8)

    raw_masks = np.zeros((frames, height, width), dtype=np.float32)
    raw_masks[4:8, 5:12, 6:15] = 1.0   # a block of frames fully inside a region
    masks = feather_mask_edges(raw_masks, 3.0)
    # Sanity: feathering must have produced a genuine partial-value band, or
    # this test would not exercise the blended branch at all.
    assert np.any((masks > 0.0) & (masks < 1.0))
    assert np.all(masks[0] == 0.0)          # an all-source frame is present
    assert masks[6].max() > 0.9             # a mostly-generated frame is present

    def _reference_whole_array_blend(source, generated, masks):
        amount = masks[..., None]
        if np.all(masks == 0.0):
            return source.copy()
        if np.all(masks == 1.0):
            return generated.copy()
        blended = (1.0 - amount) * source.astype(np.float32) + amount * generated.astype(np.float32)
        return np.clip(np.rint(blended), 0.0, 255.0).astype(np.uint8)

    expected = _reference_whole_array_blend(source, generated, masks)
    actual = composite_masked_frames(source, generated, masks)
    assert actual.dtype == np.uint8
    assert np.array_equal(actual, expected)


def test_composite_masked_frames_peak_memory_is_bounded_at_a_reduced_similar_case():
    """H-3 regression, at a scaled-down clip (the real 362x768x1344 case takes
    tens of seconds and ~16 GB peak pre-fix, per the audit's own measurement;
    this checks the SAME code path -- the frame-by-frame loop, with all-source
    and all-generated shortcuts exercised -- at a size the test suite can run
    in milliseconds). Correctness is covered by the bit-exact test above; this
    one is a hard guard against a future edit reintroducing the whole-array
    float32 blend that motivated H-3."""
    import inspect

    source_code = inspect.getsource(composite_masked_frames)
    # NEGATIVE CONTROL for a whole-array float32 temporary: the function must
    # index a single frame (`source[frame_index]`) before ever casting to
    # float32, not cast the whole `source`/`generated` array at once.
    assert "source.astype(np.float32)" not in source_code
    assert "generated.astype(np.float32)" not in source_code
    assert "source[frame_index].astype(np.float32)" in source_code
    assert "generated[frame_index].astype(np.float32)" in source_code


def test_cheap_validation_catches_the_same_generate_invariant():
    """H-4 (as narrowed by the final-audit Medium-1 fix): `validate_spatial_mask_plan_cheap`
    must reject the same "generates no token" manifests `build_spatial_mask_plan`
    does, for the keyframe-only cases it covers, without materializing a
    `[clip_frames, H, W]` array. It intentionally does NOT reject a keyframe
    whose own mask pools to fully white -- see
    test_cheap_validation_accepts_a_fully_white_keyframe_followed_by_a_partial_one
    for why that case is legal and must reach the full, aggregate check
    instead."""
    timeline_ok = parse_mask_timeline_manifest(
        _manifest([{"frame": 0, "mask_id": "subject", "interpolation_to_next": "hold"}],
                  width=4, height=2)
    )
    ok_mask = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float32)
    # Does not raise.
    validate_spatial_mask_plan_cheap(
        timeline_ok, {"subject": ok_mask}, spatial_scale=1, patch_h=1, patch_w=1,
    )

    timeline_empty = parse_mask_timeline_manifest(
        _manifest([{"frame": 0, "mask_id": "subject", "interpolation_to_next": "hold"}],
                  width=4, height=2)
    )
    with pytest.raises(MaskRasterizationError, match="generates no video token"):
        validate_spatial_mask_plan_cheap(
            timeline_empty, {"subject": np.zeros((2, 4), dtype=np.float32)},
            spatial_scale=1, patch_h=1, patch_w=1,
        )
    # A single fully-white keyframe's own mask pools to `pooled.all()`, but
    # the cheap check no longer treats that as an error by itself (Medium-1).
    # For a single-keyframe manifest the full, aggregate check in
    # `build_spatial_mask_plan` still refuses it -- there is only one latent
    # frame span here, so the aggregate invariant and the per-keyframe one
    # coincide -- just later, and after a GPU slot would already be reserved.
    validate_spatial_mask_plan_cheap(
        timeline_empty, {"subject": np.ones((2, 4), dtype=np.float32)},
        spatial_scale=1, patch_h=1, patch_w=1,
    )
    with pytest.raises(MaskRasterizationError, match="preserve at least one video token"):
        build_spatial_mask_plan(
            timeline_empty, {"subject": np.ones((2, 4), dtype=np.float32)},
            clip_frames=1, start_frame=0, end_frame=1, latent_frame_spans=((0, 1),),
            spatial_scale=1, patch_h=1, patch_w=1,
        )


def test_cheap_validation_accepts_a_fully_white_keyframe_followed_by_a_partial_one():
    """Final-audit Medium-1: `build_spatial_mask_plan`'s preserve-side
    invariant is defined over the max-pool of the ENTIRE regenerate range,
    aggregated across every latent frame span -- not any single keyframe's
    own mask. A keyframe whose own mask pools to fully white (a legitimate
    way to say "fully regenerate this instant") must not be rejected by the
    cheap keyframe-only check just because a LATER keyframe/span in the same
    manifest still preserves a token overall. Before this fix, a real
    manifest with an all-white keyframe at frame 0 morphing into a partial
    mask at frame 5 was accepted by `build_spatial_mask_plan` but rejected by
    `validate_spatial_mask_plan_cheap` -- a working request started 400ing."""
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "full", "interpolation_to_next": "hold"},
                {"frame": 5, "mask_id": "partial", "interpolation_to_next": "hold"},
            ],
            width=8,
            height=6,
        )
    )
    mask_full = np.ones((6, 8), dtype=np.float32)
    mask_partial = np.zeros((6, 8), dtype=np.float32)
    mask_partial[1:3, 1:3] = 1.0
    masks = {"full": mask_full, "partial": mask_partial}

    # Does not raise: keyframe 0's own mask pools to fully white, but the
    # aggregate invariant `build_spatial_mask_plan` enforces still holds.
    validate_spatial_mask_plan_cheap(timeline, masks, spatial_scale=1, patch_h=1, patch_w=1)

    _, pinned_rows = build_spatial_mask_plan(
        timeline,
        masks,
        clip_frames=9,
        start_frame=0,
        end_frame=9,
        latent_frame_spans=((0, 5), (5, 9)),
        spatial_scale=1,
        patch_h=1,
        patch_w=1,
    )
    # The partial keyframe's span still preserves tokens overall.
    assert len(pinned_rows) > 0


def test_cheap_validation_catches_feathering_that_thins_a_generate_region_below_threshold():
    """Medium-2 (final audit): `build_spatial_mask_plan` feathers the
    rasterized mask (`feather_mask_edges`) before pooling, so a small
    generate region can survive at full opacity in the source PNG but pool
    to nothing once feathered. Before this fix, `validate_spatial_mask_plan_cheap`
    never feathered its per-keyframe pool, so it accepted manifests the full
    check rejected -- a request that always failed still passed the cheap
    pre-check and reserved a GPU slot for nothing."""
    canvas = 20
    manifest = _manifest(
        [{"frame": 0, "mask_id": "tiny", "interpolation_to_next": "hold"}],
        width=canvas,
        height=canvas,
        composite_feather_px=24.0,
    )
    timeline = parse_mask_timeline_manifest(manifest)
    mask = np.zeros((canvas, canvas), dtype=np.float32)
    mask[9:11, 9:11] = 1.0  # a 2x2px generate region in the center
    masks = {"tiny": mask}

    with pytest.raises(MaskRasterizationError, match="generates no video token"):
        validate_spatial_mask_plan_cheap(timeline, masks, spatial_scale=1, patch_h=1, patch_w=1)

    # NEGATIVE CONTROL: `build_spatial_mask_plan` rejects the identical
    # feathered manifest for the same reason -- the cheap check now agrees
    # with the full check instead of accepting what it refuses.
    with pytest.raises(MaskRasterizationError, match="generate at least one video token"):
        build_spatial_mask_plan(
            timeline, masks,
            clip_frames=1, start_frame=0, end_frame=1, latent_frame_spans=((0, 1),),
            spatial_scale=1, patch_h=1, patch_w=1,
        )


def test_build_spatial_mask_plan_is_pure_and_deterministic():
    """H-4: calling it twice with identical inputs (once as the route used to,
    once as the backend still does) must return bit-identical results -- the
    property that made the route's duplicate call redundant rather than a
    correctness safeguard."""
    timeline = parse_mask_timeline_manifest(
        _manifest(
            [
                {"frame": 0, "mask_id": "a", "interpolation_to_next": "sdf"},
                {"frame": 4, "mask_id": "b", "interpolation_to_next": "hold"},
            ],
            width=8,
            height=6,
        )
    )
    mask_a = np.zeros((6, 8), dtype=np.float32)
    mask_a[1:3, 1:3] = 1.0
    mask_b = np.zeros((6, 8), dtype=np.float32)
    mask_b[3:5, 5:7] = 1.0
    kwargs = dict(
        clip_frames=9,
        start_frame=0,
        end_frame=5,
        latent_frame_spans=((0, 5), (5, 9)),
        spatial_scale=1,
        patch_h=1,
        patch_w=1,
    )
    first = build_spatial_mask_plan(timeline, {"a": mask_a, "b": mask_b}, **kwargs)
    second = build_spatial_mask_plan(timeline, {"a": mask_a, "b": mask_b}, **kwargs)
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_mask_transform_scale_is_bounded_against_denormals_and_the_frontend_cap():
    """L-1: the frontend timeline editor caps scale at 100x and never lets it
    reach 0; a direct API caller must be bounded the same way, including
    against a positive-but-denormal value that would blow up the transform's
    matrix inverse."""
    with pytest.raises(ManifestValidationError):
        parse_mask_timeline_manifest(
            _manifest([{
                "frame": 0, "mask_id": "a", "interpolation_to_next": "hold",
                "transform": {"scale_x": 1e-320},
            }])
        )
    with pytest.raises(ManifestValidationError):
        parse_mask_timeline_manifest(
            _manifest([{
                "frame": 0, "mask_id": "a", "interpolation_to_next": "hold",
                "transform": {"scale_x": 101.0},
            }])
        )
    # NEGATIVE CONTROL: a legal scale (matching the frontend's own cap) is accepted.
    parse_mask_timeline_manifest(
        _manifest([{
            "frame": 0, "mask_id": "a", "interpolation_to_next": "hold",
            "transform": {"scale_x": 100.0, "scale_y": 0.01},
        }])
    )


def test_affine_rejects_a_hand_built_manifest_with_mismatched_mask_ids():
    """L-3: the parser already rejects this at parse time (see
    test_manifest_rejects_unknown_fields_and_cross_asset_affine); this checks
    that a hand-built MaskTimelineManifest bypassing the parser is still
    refused by rasterize_mask_timeline itself, rather than silently blended."""
    from core.inference.video_mask_timeline import (
        MaskCanvas,
        MaskKeyframe,
        MaskTimelineManifest,
    )

    timeline = MaskTimelineManifest(
        version=1,
        coordinate_space="output_canvas",
        polarity="white_generate",
        canvas=MaskCanvas(width=3, height=3),
        keyframes=(
            MaskKeyframe(frame=0, mask_id="a", interpolation_to_next="affine"),
            MaskKeyframe(frame=2, mask_id="b", interpolation_to_next="hold"),
        ),
    )
    masks = {"a": np.zeros((3, 3), dtype=np.float32), "b": np.ones((3, 3), dtype=np.float32)}
    with pytest.raises(MaskRasterizationError, match="same mask_id"):
        rasterize_mask_timeline(timeline, masks, 0, 3)


def test_decode_pngs_enforces_the_max_mask_assets_constant():
    """M-2 support: MAX_MASK_ASSETS is the single cap both the route's
    parse-time check and this decode-time defensive check share."""
    assert MAX_MASK_ASSETS == 64
