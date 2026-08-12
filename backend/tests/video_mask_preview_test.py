"""GPU-free tests for the video mask preview: pure rasterization/sprite
packing (`core/inference/video_mask_preview.py`) and the
`POST /video-mask/preview` route.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_mask_preview_test.py -v

WHY THIS FILE EXISTS
--------------------
The preview route exists so a UI can show what a spatial mask timeline
rasterizes to WITHOUT reimplementing `sdf`'s distance-transform morph in the
browser. The tests below pin the two things that make that claim true rather
than aspirational:

* the preview's rasterization must be the SAME numbers `rasterize_mask_timeline`
  itself produces for the same manifest/masks/frame -- not a re-derivation
  that happens to look similar;
* the preview's own added surface (frame list, span cap, max_size, sprite
  packing) enforces bounds independent of and no looser than the generation
  route's own spatial-mask validation, and the route maps every invalid input
  to a 400 (`ValidationError`), never an unguarded 500.
"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
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

from core.inference.video_mask_preview import (  # noqa: E402
    MASK_PREVIEW_RASTER_BUDGET_BYTES,
    MASK_PREVIEW_RASTER_PEAK_MULTIPLIER,
    MAX_MASK_PREVIEW_FRAMES,
    MAX_PREVIEW_MAX_SIZE,
    MIN_PREVIEW_MAX_SIZE,
    MaskPreviewError,
    build_mask_preview_strip,
)
from core.inference.video_mask_timeline import (  # noqa: E402
    MaskCanvas,
    MaskKeyframe,
    MaskRasterizationError,
    MaskTimelineManifest,
    MaskTransform,
    parse_mask_timeline_manifest,
    rasterize_mask_timeline,
)

WIDTH, HEIGHT = 16, 8


def _manifest_dict(keyframes, *, width=WIDTH, height=HEIGHT):
    return {
        "version": 1,
        "coordinate_space": "output_canvas",
        "polarity": "white_generate",
        "canvas": {"width": width, "height": height},
        "keyframes": keyframes,
    }


def _half_split_mask(width=WIDTH, height=HEIGHT):
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:, width // 2:] = 1.0
    return mask


def _png_bytes(mask: np.ndarray) -> bytes:
    image = Image.fromarray(np.clip(np.rint(mask * 255.0), 0, 255).astype(np.uint8), mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _sdf_timeline_and_masks():
    """A timeline whose sdf segment (frames 10..30) actually interpolates,
    so the strip's middle frame is neither endpoint's own mask."""
    timeline = parse_mask_timeline_manifest(_manifest_dict([
        {"frame": 10, "mask_id": "left", "interpolation_to_next": "sdf"},
        {"frame": 30, "mask_id": "right", "interpolation_to_next": "hold"},
    ]))
    left = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    left[:, :4] = 1.0
    right = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    right[:, -4:] = 1.0
    return timeline, {"left": left, "right": right}


# --------------------------------------------------------------------------
# The addressable unit: identical numbers to rasterize_mask_timeline
# --------------------------------------------------------------------------

def test_the_preview_rasterizes_the_exact_same_values_as_generation():
    """THE CORE CLAIM. Fails if the preview re-derives its own numbers
    instead of calling the shared rasterizer."""
    timeline, masks = _sdf_timeline_and_masks()

    warnings = []
    png_bytes, metadata = build_mask_preview_strip(
        timeline, masks, [10, 15, 20, 25, 30], max_size=1024,
        sdf_fallback_warnings=warnings,
    )
    # max_size=1024 > the canvas, so no downscale happened -- the sprite tile
    # values must be bit-identical to a direct rasterize_mask_timeline call.
    assert metadata["frame_width"] == WIDTH and metadata["frame_height"] == HEIGHT

    expected = rasterize_mask_timeline(timeline, masks, 10, 31, sdf_fallback_warnings=[])
    strip = np.asarray(Image.open(BytesIO(png_bytes)).convert("L"), dtype=np.uint8)
    for index, frame in enumerate([10, 15, 20, 25, 30]):
        tile = strip[:, index * WIDTH:(index + 1) * WIDTH]
        expected_tile = np.clip(np.rint(expected[frame - 10] * 255.0), 0, 255).astype(np.uint8)
        assert np.array_equal(tile, expected_tile), f"frame {frame} mismatch"


def test_frames_are_returned_ascending_and_deduplicated_regardless_of_request_order():
    timeline, masks = _sdf_timeline_and_masks()
    _png, metadata = build_mask_preview_strip(timeline, masks, [30, 10, 20], max_size=1024)
    assert [entry["frame"] for entry in metadata["frames"]] == [10, 20, 30]
    assert [entry["x_offset"] for entry in metadata["frames"]] == [0, WIDTH, 2 * WIDTH]


def test_a_frame_past_the_last_keyframe_holds_like_generation_does():
    timeline, masks = _sdf_timeline_and_masks()
    _png, metadata = build_mask_preview_strip(timeline, masks, [30, 60], max_size=1024)
    strip = np.asarray(Image.open(BytesIO(_png)).convert("L"), dtype=np.uint8)
    tile_30 = strip[:, 0:WIDTH]
    tile_60 = strip[:, WIDTH:2 * WIDTH]
    # frame 60 is past the last keyframe (30, hold): identical to frame 30.
    assert np.array_equal(tile_30, tile_60)


# --------------------------------------------------------------------------
# The downscale
# --------------------------------------------------------------------------

def test_max_size_downscales_and_preserves_aspect_ratio():
    timeline, masks = _sdf_timeline_and_masks()
    _png, metadata = build_mask_preview_strip(timeline, masks, [10], max_size=MIN_PREVIEW_MAX_SIZE)
    # WIDTH=16, HEIGHT=8 -> longest edge 16 scaled to MIN_PREVIEW_MAX_SIZE=16 -> unchanged.
    # Use a canvas twice as wide as MIN_PREVIEW_MAX_SIZE to force a real downscale.
    wide_timeline = parse_mask_timeline_manifest(_manifest_dict(
        [{"frame": 0, "mask_id": "m", "interpolation_to_next": "hold"}],
        width=MIN_PREVIEW_MAX_SIZE * 2, height=MIN_PREVIEW_MAX_SIZE,
    ))
    wide_masks = {"m": _half_split_mask(width=MIN_PREVIEW_MAX_SIZE * 2, height=MIN_PREVIEW_MAX_SIZE)}
    _png2, metadata2 = build_mask_preview_strip(
        wide_timeline, wide_masks, [0], max_size=MIN_PREVIEW_MAX_SIZE,
    )
    assert (metadata2["frame_width"], metadata2["frame_height"]) == (
        MIN_PREVIEW_MAX_SIZE, MIN_PREVIEW_MAX_SIZE // 2,
    )
    assert metadata["canvas_width"] == WIDTH and metadata["canvas_height"] == HEIGHT


def test_max_size_larger_than_canvas_does_not_upscale():
    timeline, masks = _sdf_timeline_and_masks()
    _png, metadata = build_mask_preview_strip(timeline, masks, [10], max_size=MAX_PREVIEW_MAX_SIZE)
    assert (metadata["frame_width"], metadata["frame_height"]) == (WIDTH, HEIGHT)


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------

def test_frames_must_be_non_empty_and_within_the_per_request_cap():
    timeline, masks = _sdf_timeline_and_masks()
    with pytest.raises(MaskPreviewError, match="non-empty"):
        build_mask_preview_strip(timeline, masks, [], max_size=256)
    with pytest.raises(MaskPreviewError, match=str(MAX_MASK_PREVIEW_FRAMES)):
        build_mask_preview_strip(
            timeline, masks, list(range(10, 10 + MAX_MASK_PREVIEW_FRAMES + 1)), max_size=256,
        )
    # NEGATIVE CONTROL: exactly the cap is accepted.
    _png, metadata = build_mask_preview_strip(
        timeline, masks, list(range(10, 10 + MAX_MASK_PREVIEW_FRAMES)), max_size=256,
    )
    assert len(metadata["frames"]) == MAX_MASK_PREVIEW_FRAMES


def test_duplicate_frame_numbers_are_refused():
    timeline, masks = _sdf_timeline_and_masks()
    with pytest.raises(MaskPreviewError, match="duplicate"):
        build_mask_preview_strip(timeline, masks, [10, 10], max_size=256)


def test_negative_frame_numbers_are_refused():
    timeline, masks = _sdf_timeline_and_masks()
    with pytest.raises(MaskPreviewError, match="non-negative"):
        build_mask_preview_strip(timeline, masks, [-1], max_size=256)


def test_max_size_must_be_within_bounds():
    timeline, masks = _sdf_timeline_and_masks()
    with pytest.raises(MaskPreviewError):
        build_mask_preview_strip(timeline, masks, [10], max_size=MIN_PREVIEW_MAX_SIZE - 1)
    with pytest.raises(MaskPreviewError):
        build_mask_preview_strip(timeline, masks, [10], max_size=MAX_PREVIEW_MAX_SIZE + 1)


def _frame_budget_for_canvas(width: int, height: int) -> int:
    """The exact span-frame cap `build_mask_preview_strip` computes for a
    canvas of this size, mirroring `video_mask_preview.py`'s own arithmetic
    so this test does not hardcode a canvas-size-independent constant that no
    longer exists post byte-budget cap."""
    canvas_px = width * height
    return int(
        MASK_PREVIEW_RASTER_BUDGET_BYTES
        // max(1, int(canvas_px * 4 * MASK_PREVIEW_RASTER_PEAK_MULTIPLIER))
    )


def test_a_span_far_past_the_keyframes_is_refused_independent_of_frame_count():
    """H-x: the DoS this guards -- two keyframes close together, one
    requested frame far away, must not force a multi-million-pixel
    contiguous rasterization just because only ONE frame was asked for.

    Byte-budget cap (post `a33347f9`, which removed MiniMax-H3's own
    clip-length ceiling): uses a large canvas so the byte budget, not an
    unrelated frame-count constant, is what is actually pinned here."""
    width, height = 1024, 1024
    frame_budget = _frame_budget_for_canvas(width, height)
    timeline = parse_mask_timeline_manifest(_manifest_dict([
        {"frame": 0, "mask_id": "m", "interpolation_to_next": "hold"},
    ], width=width, height=height))
    masks = {"m": _half_split_mask(width=width, height=height)}
    with pytest.raises(MaskPreviewError, match="GiB"):
        build_mask_preview_strip(timeline, masks, [frame_budget + 10], max_size=256)
    # NEGATIVE CONTROL: a span within the budget is accepted.
    _png, metadata = build_mask_preview_strip(
        timeline, masks, [frame_budget - 1], max_size=256,
    )
    assert metadata["frames"][0]["frame"] == frame_budget - 1


def test_the_wide_keyframe_span_itself_is_also_bounded_with_no_requested_frames_far_out():
    """The same cap catches a manifest whose OWN keyframes are far apart,
    even when every requested frame sits right next to one of them."""
    width, height = 1024, 1024
    frame_budget = _frame_budget_for_canvas(width, height)
    timeline = parse_mask_timeline_manifest(_manifest_dict([
        {"frame": 0, "mask_id": "m", "interpolation_to_next": "hold"},
        {"frame": frame_budget + 100, "mask_id": "m", "interpolation_to_next": "hold"},
    ], width=width, height=height))
    masks = {"m": _half_split_mask(width=width, height=height)}
    with pytest.raises(MaskPreviewError, match="GiB"):
        build_mask_preview_strip(timeline, masks, [0], max_size=256)


def test_an_underlying_rasterization_error_is_not_swallowed():
    """A hand-built manifest that bypasses the parser can still trip
    rasterize_mask_timeline's own invariants; that error type must propagate
    unmodified rather than being caught and re-labeled by this module."""
    timeline = MaskTimelineManifest(
        version=1,
        coordinate_space="output_canvas",
        polarity="white_generate",
        canvas=MaskCanvas(width=WIDTH, height=HEIGHT),
        keyframes=(
            MaskKeyframe(frame=0, mask_id="a", interpolation_to_next="affine",
                         transform=MaskTransform()),
            MaskKeyframe(frame=10, mask_id="b", interpolation_to_next="hold"),
        ),
    )
    masks = {"a": _half_split_mask(), "b": _half_split_mask()}
    with pytest.raises(MaskRasterizationError):
        build_mask_preview_strip(timeline, masks, [5], max_size=256)


# --------------------------------------------------------------------------
# The route: multipart request in, JSON preview out, 400 on bad input
# --------------------------------------------------------------------------

def _post_preview(app_module, manifest_dict, mask_png_by_id, frames, max_size=256):
    import httpx
    from fastapi import FastAPI

    from api.error_handlers import register_error_handlers

    app = FastAPI()
    register_error_handlers(app)
    app.post("/video-mask/preview")(app_module.preview_video_mask)

    boundary = "----VideoMaskPreviewBoundary"
    parts = [
        (f'--{boundary}\r\nContent-Disposition: form-data; name="spatial_mask_manifest"'
         f'\r\n\r\n{json.dumps(manifest_dict)}\r\n').encode()
    ]
    for value in frames:
        parts.append(
            (f'--{boundary}\r\nContent-Disposition: form-data; name="frames"\r\n\r\n'
             f'{value}\r\n').encode()
        )
    parts.append(
        (f'--{boundary}\r\nContent-Disposition: form-data; name="max_size"\r\n\r\n'
         f'{max_size}\r\n').encode()
    )
    for mask_id, png_bytes in mask_png_by_id.items():
        parts.append((
            f'--{boundary}\r\nContent-Disposition: form-data; name="spatial_mask_ids"'
            f'\r\n\r\n{mask_id}\r\n'
        ).encode())
        parts.append((
            f'--{boundary}\r\nContent-Disposition: form-data; name="spatial_mask_files"; '
            f'filename="{mask_id}.png"\r\nContent-Type: image/png\r\n\r\n'
        ).encode() + png_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/video-mask/preview", content=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )

    return asyncio.run(run())


def test_the_route_returns_a_json_sprite_matching_the_pure_function():
    import api.routes as routes

    timeline_dict = _manifest_dict([
        {"frame": 10, "mask_id": "left", "interpolation_to_next": "sdf"},
        {"frame": 30, "mask_id": "right", "interpolation_to_next": "hold"},
    ])
    left = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    left[:, :4] = 1.0
    right = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    right[:, -4:] = 1.0

    response = _post_preview(
        routes, timeline_dict,
        {"left": _png_bytes(left), "right": _png_bytes(right)},
        frames=[10, 20, 30],
        max_size=1024,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["canvas_width"] == WIDTH and body["canvas_height"] == HEIGHT
    assert [entry["frame"] for entry in body["frames"]] == [10, 20, 30]
    assert body["strip_png"].startswith("data:image/png;base64,")
    strip_bytes = base64.b64decode(body["strip_png"].split(",", 1)[1])
    strip = np.asarray(Image.open(BytesIO(strip_bytes)).convert("L"), dtype=np.uint8)
    assert strip.shape == (HEIGHT, WIDTH * 3)


def test_the_route_refuses_an_invalid_manifest_with_a_400_not_a_500():
    import api.routes as routes
    from api.error_handlers import ValidationError

    response = _post_preview(
        routes, {"version": 2}, {}, frames=[0],
    )
    assert response.status_code == 400, response.text
    assert "manifest" in response.json()["error"].lower() or "manifest" in str(response.json()).lower()


def test_the_route_refuses_a_mask_id_mismatch_with_a_400():
    import api.routes as routes

    timeline_dict = _manifest_dict([
        {"frame": 0, "mask_id": "subject", "interpolation_to_next": "hold"},
    ])
    response = _post_preview(
        routes, timeline_dict,
        {"WRONG_ID": _png_bytes(_half_split_mask())},
        frames=[0],
    )
    assert response.status_code == 400, response.text


def test_the_route_refuses_too_many_frames_with_a_400():
    import api.routes as routes

    timeline_dict = _manifest_dict([
        {"frame": 0, "mask_id": "subject", "interpolation_to_next": "hold"},
    ])
    response = _post_preview(
        routes, timeline_dict,
        {"subject": _png_bytes(_half_split_mask())},
        frames=list(range(MAX_MASK_PREVIEW_FRAMES + 1)),
    )
    assert response.status_code == 400, response.text


def test_the_route_does_not_import_or_call_any_pipeline_or_model_symbol():
    """Structural: the route body must not reach into pipeline_manager or any
    model-loading path -- it is a pure rasterization preview and must not
    require (or accidentally depend on) a loaded model."""
    import inspect

    import api.routes as routes

    source = inspect.getsource(routes.preview_video_mask)
    assert "pipeline_manager" not in source
    # Not a plain "minimax_h3" substring check: the docstring legitimately
    # names the sibling generation route and its test file by way of
    # explaining why this route's validation is a deliberate near-duplicate
    # rather than a shared helper. What must be absent is any USE of the
    # MiniMax-H3 mixin or model-specific components, not a mention of it.
    assert "MiniMaxH3Mixin" not in source
    assert "minimax_h3_components" not in source
    assert "current_model_info" not in source
