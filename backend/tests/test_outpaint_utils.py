"""
Tests for the image spatial outpaint orchestration helpers
(core/inference/outpaint_utils.py).

Run with:
    d:\\celll1\\webui_cl\\venv\\Scripts\\python.exe -m pytest backend/tests/test_outpaint_utils.py -v

Test coverage:
  1. paste_preserved_region -- the placed rect is BYTE-IDENTICAL to placed_img
     (the strict-preservation contract).
  2. build_outpaint_mask -- mask == 0 over the ENTIRE rect even with
     mask_blur > 0 (outward-only blur), and > 0 somewhere outside the rect.
  3. validate_and_snap_placement -- clamps in-bounds geometry and rejects
     degenerate cases (empty crop, too-small rect, rect fully covering canvas).
  4. build_outpaint_canvas -- replicate fill leaves the placed rect exactly
     equal to the resized (cropped) input, for every fill mode.
  5. reconcile_and_paste -- byte-exact preserved-rect paste even when the
     returned image size differs from the canvas the rect was computed
     against (arch re-rounding, floor-down AND round-up deltas), and a
     16-aligned canvas is left unchanged by the align pass.

No network access, no torch/pipeline dependencies -- outpaint_utils is pure
PIL/numpy/stdlib.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from PIL import Image

# ── path setup ───────────────────────────────────────────────────────────────
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from backend.core.inference.outpaint_utils import (
    build_outpaint_canvas,
    build_outpaint_mask,
    paste_preserved_region,
    reconcile_and_paste,
    validate_and_snap_placement,
)


def _make_gradient_image(w: int, h: int) -> Image.Image:
    """A deterministic non-uniform RGB image (gradient), so pixel-exactness
    checks are meaningful (a flat-color image would trivially pass)."""
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    ys = np.linspace(0, 255, h, dtype=np.uint8)
    r = np.tile(xs, (h, 1))
    g = np.tile(ys.reshape(-1, 1), (1, w))
    b = ((r.astype(int) + g.astype(int)) % 256).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    return Image.fromarray(arr, mode="RGB")


class TestValidateAndSnapPlacement(unittest.TestCase):
    def test_defaults_resolve_to_input_native_size_centered_clamped(self):
        resolved = validate_and_snap_placement(
            {"canvas_width": 1536, "canvas_height": 1536}, (512, 512)
        )
        self.assertEqual(resolved["canvas_width"], 1536)
        self.assertEqual(resolved["canvas_height"], 1536)
        self.assertEqual(resolved["place_width"], 512)
        self.assertEqual(resolved["place_height"], 512)
        self.assertEqual(resolved["place_x"], 0)
        self.assertEqual(resolved["place_y"], 0)

    def test_canvas_size_rounds_to_align_grid(self):
        resolved = validate_and_snap_placement(
            {"canvas_width": 1000, "canvas_height": 1003}, (256, 256), align=8
        )
        self.assertEqual(resolved["canvas_width"] % 8, 0)
        self.assertEqual(resolved["canvas_height"] % 8, 0)

    def test_already_16_aligned_canvas_unchanged_by_align_16(self):
        # PipelineManager.generate_outpaint calls with align=16 (7 of 9 image
        # archs re-round their own working resolution to a 16px grid). An
        # already 16-aligned canvas must be a no-op through that pass.
        resolved = validate_and_snap_placement(
            {"canvas_width": 1536, "canvas_height": 1536}, (256, 256), align=16
        )
        self.assertEqual(resolved["canvas_width"], 1536)
        self.assertEqual(resolved["canvas_height"], 1536)

        resolved2 = validate_and_snap_placement(
            {"canvas_width": 800, "canvas_height": 624}, (256, 256), align=16
        )
        self.assertEqual(resolved2["canvas_width"], 800)
        self.assertEqual(resolved2["canvas_height"], 624)

    def test_place_rect_clamped_into_canvas_bounds(self):
        resolved = validate_and_snap_placement(
            {
                "canvas_width": 512,
                "canvas_height": 512,
                "place_x": 10000,
                "place_y": 10000,
                "place_width": 256,
                "place_height": 256,
            },
            (256, 256),
        )
        self.assertLessEqual(resolved["place_x"] + resolved["place_width"], resolved["canvas_width"])
        self.assertLessEqual(resolved["place_y"] + resolved["place_height"], resolved["canvas_height"])
        self.assertGreaterEqual(resolved["place_x"], 0)
        self.assertGreaterEqual(resolved["place_y"], 0)

    def test_place_rect_larger_than_canvas_is_capped(self):
        # canvas_height is intentionally much larger than the capped
        # place_height, so this does not also trip the "fully covers the
        # canvas" rejection (only the width fully covers here).
        resolved = validate_and_snap_placement(
            {
                "canvas_width": 256,
                "canvas_height": 8192,
                "place_width": 4096,
                "place_height": 4096,
            },
            (4096, 4096),
        )
        self.assertLessEqual(resolved["place_width"], resolved["canvas_width"])
        self.assertLessEqual(resolved["place_height"], resolved["canvas_height"])

    def test_empty_crop_rejected(self):
        with self.assertRaises(ValueError):
            validate_and_snap_placement(
                {
                    "canvas_width": 512,
                    "canvas_height": 512,
                    "input_crop_x": 512,  # == input width -> nothing left
                    "input_crop_w": 100,
                },
                (512, 512),
            )

    def test_rect_too_small_rejected(self):
        with self.assertRaises(ValueError):
            validate_and_snap_placement(
                {
                    "canvas_width": 512,
                    "canvas_height": 512,
                    "place_width": 4,
                    "place_height": 4,
                },
                (512, 512),
                snap=0,  # isolate the too-small check from snapping
            )

    def test_rect_fully_covering_canvas_rejected(self):
        with self.assertRaises(ValueError):
            validate_and_snap_placement(
                {
                    "canvas_width": 256,
                    "canvas_height": 256,
                    "place_width": 256,
                    "place_height": 256,
                },
                (256, 256),
            )

    def test_partial_cover_one_axis_is_allowed(self):
        # place_width == canvas_width but place_height < canvas_height ->
        # still something to generate (top/bottom strips).
        resolved = validate_and_snap_placement(
            {
                "canvas_width": 256,
                "canvas_height": 512,
                "place_width": 256,
                "place_height": 256,
            },
            (256, 256),
        )
        self.assertEqual(resolved["place_width"], 256)
        self.assertEqual(resolved["place_height"], 256)


class TestBuildOutpaintCanvas(unittest.TestCase):
    def _run(self, fill_mode: str):
        # All placement values are multiples of 8 so the default snap-to-8
        # grid (see validate_and_snap_placement) is a no-op -- this test is
        # about fill-mode/crop correctness, not snapping.
        input_img = _make_gradient_image(304, 200)
        params = {
            "canvas_width": 800,
            "canvas_height": 608,  # 16-aligned (build_outpaint_canvas defaults to align=16)
            "place_x": 200,
            "place_y": 152,
            "place_width": 304,
            "place_height": 200,
            "outpaint_fill_mode": fill_mode,
        }
        canvas_img, placed_img, rect = build_outpaint_canvas(input_img, params)
        self.assertEqual(canvas_img.size, (800, 608))
        self.assertEqual(placed_img.size, (304, 200))
        x0, y0, x1, y1 = rect
        self.assertEqual((x1 - x0, y1 - y0), (304, 200))

        canvas_arr = np.array(canvas_img)
        placed_arr = np.array(placed_img)
        rect_slice = canvas_arr[y0:y1, x0:x1]
        np.testing.assert_array_equal(
            rect_slice, placed_arr,
            err_msg=f"placed rect not exactly preserved for fill_mode={fill_mode!r}",
        )
        return canvas_img, placed_img, rect

    def test_replicate_fill_preserves_placed_rect(self):
        self._run("replicate")

    def test_reflect_fill_preserves_placed_rect(self):
        self._run("reflect")

    def test_mean_fill_preserves_placed_rect(self):
        self._run("mean")

    def test_noise_fill_preserves_placed_rect(self):
        self._run("noise")

    def test_crop_before_placement(self):
        # Crop offset/size kept as multiples of 8 so the default snap-to-8
        # grid does not alter the auto-derived (0 = native) placed size.
        input_img = _make_gradient_image(400, 400)
        params = {
            "canvas_width": 600,
            "canvas_height": 600,
            "input_crop_x": 96,
            "input_crop_y": 96,
            "input_crop_w": 96,
            "input_crop_h": 96,
            "place_width": 0,  # native size of the CROPPED region (96x96)
            "place_height": 0,
        }
        canvas_img, placed_img, rect = build_outpaint_canvas(input_img, params)
        self.assertEqual(placed_img.size, (96, 96))
        expected_crop = input_img.crop((96, 96, 192, 192))
        np.testing.assert_array_equal(np.array(placed_img), np.array(expected_crop))

    def test_reflect_fill_handles_large_canvas_relative_to_small_place(self):
        # Regression guard: numpy's own reflect mode caps pad width at
        # dim - 1 per call; canvas >> placed size must not crash.
        input_img = _make_gradient_image(16, 16)
        params = {
            "canvas_width": 800,
            "canvas_height": 800,
            "place_x": 50,
            "place_y": 50,
            "place_width": 16,
            "place_height": 16,
            "outpaint_fill_mode": "reflect",
        }
        canvas_img, placed_img, rect = build_outpaint_canvas(input_img, params)
        self.assertEqual(canvas_img.size, (800, 800))
        x0, y0, x1, y1 = rect
        np.testing.assert_array_equal(
            np.array(canvas_img)[y0:y1, x0:x1], np.array(placed_img)
        )


class TestBuildOutpaintMask(unittest.TestCase):
    def test_mask_zero_over_entire_rect_outward_only_blur(self):
        rect = (100, 100, 300, 250)
        mask = build_outpaint_mask((512, 512), rect, mask_blur=12)
        arr = np.array(mask)
        x0, y0, x1, y1 = rect

        # Entire preserved rect must be exactly 0, regardless of blur.
        self.assertTrue(np.all(arr[y0:y1, x0:x1] == 0))

        # Somewhere clearly outside the rect (far from any blur bleed) must
        # be white (255 = generate).
        self.assertEqual(arr[10, 10], 255)

        # The transition band exists somewhere just outside the rect
        # (fractional values between 0 and 255).
        band = arr[max(0, y0 - 12):y0, x0:x1]
        self.assertTrue(np.any((band > 0) & (band < 255)), "expected a soft transition band just outside the rect")

    def test_mask_without_blur_is_hard_edged(self):
        rect = (50, 50, 150, 150)
        mask = build_outpaint_mask((256, 256), rect, mask_blur=0)
        arr = np.array(mask)
        x0, y0, x1, y1 = rect
        self.assertTrue(np.all(arr[y0:y1, x0:x1] == 0))
        self.assertEqual(arr[0, 0], 255)


class TestPastePreservedRegion(unittest.TestCase):
    def test_paste_is_byte_exact(self):
        canvas = _make_gradient_image(400, 400)
        # Simulate a "generated" canvas that differs everywhere from the
        # original placed content (so the test cannot pass by accident).
        generated = Image.eval(canvas, lambda p: 255 - p)
        placed_img = _make_gradient_image(120, 90)
        rect = (50, 60, 50 + 120, 60 + 90)

        result = paste_preserved_region(generated, placed_img, rect)

        x0, y0, x1, y1 = rect
        result_rect = np.array(result)[y0:y1, x0:x1]
        placed_arr = np.array(placed_img)
        np.testing.assert_array_equal(
            result_rect, placed_arr,
            err_msg="pasted rect must be byte-identical to placed_img",
        )

        # Outside the rect, the (unrelated) generated content must be
        # untouched by the paste.
        outside_result = np.array(result)[0:10, 0:10]
        outside_generated = np.array(generated)[0:10, 0:10]
        np.testing.assert_array_equal(outside_result, outside_generated)

    def test_paste_does_not_mutate_input_in_place(self):
        generated = _make_gradient_image(200, 200)
        generated_copy_before = generated.copy()
        placed_img = Image.new("RGB", (50, 50), (0, 0, 0))
        rect = (10, 10, 60, 60)

        paste_preserved_region(generated, placed_img, rect)

        np.testing.assert_array_equal(np.array(generated), np.array(generated_copy_before))


class TestReconcileAndPaste(unittest.TestCase):
    """Regression coverage for the HIGH-severity bug: an architecture that
    re-rounds its working resolution can return a decoded image whose size
    != the canvas `rect` was computed against. reconcile_and_paste must
    re-square the result to the canvas size before pasting, so the preserved
    rect still lands byte-exact regardless."""

    def _assert_rect_byte_exact(self, result, placed_img, rect, canvas_size):
        self.assertEqual(result.size, canvas_size)
        x0, y0, x1, y1 = rect
        result_rect = np.array(result)[y0:y1, x0:x1]
        np.testing.assert_array_equal(
            result_rect, np.array(placed_img),
            err_msg="preserved rect must be byte-identical to placed_img "
                    "even when the returned image size != the canvas",
        )

    def test_arch_floor_down_delta(self):
        # e.g. FLUX.2/Anima floor the canvas down to their own 16px grid --
        # simulate a decoded result 8px SMALLER in each dimension than the
        # canvas the rect was computed against.
        canvas_size = (800, 600)
        placed_img = _make_gradient_image(200, 150)
        rect = (300, 225, 500, 375)
        # "Generated" result at a DIFFERENT (smaller) size than canvas_size.
        result_img = _make_gradient_image(792, 592)

        result = reconcile_and_paste(result_img, placed_img, rect, canvas_size)
        self._assert_rect_byte_exact(result, placed_img, rect, canvas_size)

    def test_arch_round_up_delta(self):
        # e.g. Krea2 rounds the canvas UP -- simulate a decoded result 16px
        # LARGER in each dimension than the canvas the rect was computed
        # against.
        canvas_size = (800, 600)
        placed_img = _make_gradient_image(200, 150)
        rect = (300, 225, 500, 375)
        result_img = _make_gradient_image(816, 616)

        result = reconcile_and_paste(result_img, placed_img, rect, canvas_size)
        self._assert_rect_byte_exact(result, placed_img, rect, canvas_size)

    def test_no_size_delta_is_a_plain_paste(self):
        canvas_size = (800, 600)
        placed_img = _make_gradient_image(200, 150)
        rect = (300, 225, 500, 375)
        result_img = _make_gradient_image(800, 600)

        result = reconcile_and_paste(result_img, placed_img, rect, canvas_size)
        self._assert_rect_byte_exact(result, placed_img, rect, canvas_size)

        # Outside the rect, the untouched "generated" content must be
        # unchanged (no spurious resize happened).
        outside_result = np.array(result)[0:10, 0:10]
        outside_original = np.array(result_img)[0:10, 0:10]
        np.testing.assert_array_equal(outside_result, outside_original)

    def test_end_to_end_with_build_outpaint_canvas(self):
        # Realistic pipeline: build the real canvas/rect via
        # build_outpaint_canvas, then simulate the arch returning a
        # differently-sized "decoded" image (e.g. re-rounded down),
        # reproducing the exact HIGH-severity failure mode before the fix.
        input_img = _make_gradient_image(304, 200)
        params = {
            "canvas_width": 800,
            "canvas_height": 608,
            "place_x": 200,
            "place_y": 152,
            "place_width": 304,
            "place_height": 200,
        }
        canvas_img, placed_img, rect = build_outpaint_canvas(input_img, params, align=16)
        self.assertEqual(canvas_img.size, (800, 608))

        # Simulate an arch that floored the (already 16-aligned) canvas down
        # to a DIFFERENT working resolution anyway, to exercise the
        # reconcile path even when generate_outpaint did everything right.
        mismatched_result = _make_gradient_image(784, 592)

        result = reconcile_and_paste(mismatched_result, placed_img, rect, canvas_img.size)
        self._assert_rect_byte_exact(result, placed_img, rect, canvas_img.size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
