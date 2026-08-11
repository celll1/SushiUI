"""H-2: spatial mask + FBCache exclusivity, at the backend layer.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_spatial_mask_fbcache_test.py -v

WHY THIS FILE EXISTS
--------------------
`MiniMaxH3BlockLoopWrapper._custom_forward`'s FBCache guard indicator reshapes
the free (non-pinned) video rows into whole latent frames of `rows_per_frame`
rows each. A row-level spatial mask pin can pin part of a frame while leaving
the rest free, so the free-row count need not divide by `rows_per_frame` --
which raised a RuntimeError deep in the denoise loop, after the text encode,
source clip VAE encode and DiT staging had already run (measured: 620 leftover
rows out of 1008 at 362 frames / 1344x768).

The route (`POST /generate/inpaint/video`) now refuses the combination before
any of that work starts; this file defends the SAME invariant at the backend
layer (`MiniMaxH3Mixin._generate_vidinpaint_minimax_h3`), which a caller that
bypasses the route (an internal script, a future second caller) would
otherwise still hit.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.inference.video_mask_timeline import (  # noqa: E402
    MaskCanvas,
    MaskKeyframe,
    MaskTimelineManifest,
)
from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin  # noqa: E402

WIDTH, HEIGHT = 64, 32
CLIP_FRAMES = 22   # on the 17n+5 grid; below the production floor (124), so the
                   # smoke override env var below is required to accept it.


@pytest.fixture(autouse=True)
def _smoke_floor(monkeypatch):
    # Scoped to this file's tests only (monkeypatch restores it after each
    # test) -- a module-level `os.environ[...] = ...` would leak into every
    # OTHER test module that imports after this one in the same pytest
    # session, silently lowering their clip-length floor too.
    monkeypatch.setenv("SUSHI_TEMPORAL_SMOKE", "1")


def _runner():
    class Runner(MiniMaxH3Mixin):
        minimax_h3_components = {
            "variant": "fl2va",
            "audio_sample_rate": 32000,
            "vae_scale_factor_spatial": 16,
            "transformer_config": {"patch_size": (1, 2, 2)},
        }
        current_model_info = {"type": "minimax_h3", "variant": "fl2va"}

        def _generate_minimax_h3(self, params, **kwargs):
            raise AssertionError(
                "must not reach generation: the FBCache/spatial-mask combination "
                "must be refused before this call"
            )

    return Runner()


def _source_clip():
    return np.zeros((CLIP_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)


def _half_split_manifest_and_masks():
    """A mask whose right half generates and left half preserves, so it
    passes `build_spatial_mask_plan`'s own generate/preserve invariants at
    this canvas's 32x32px token grid (spatial_scale=16 * patch=2)."""
    timeline = MaskTimelineManifest(
        version=1,
        coordinate_space="output_canvas",
        polarity="white_generate",
        canvas=MaskCanvas(width=WIDTH, height=HEIGHT),
        keyframes=(MaskKeyframe(frame=0, mask_id="subject", interpolation_to_next="hold"),),
    )
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    mask[:, WIDTH // 2:] = 1.0
    return timeline, {"subject": mask}


def _params(**overrides):
    params = {
        "width": WIDTH, "height": HEIGHT, "frame_rate": 24.0,
        "regenerate_start_frame": 0, "regenerate_end_frame": CLIP_FRAMES,
        "inpaint_video_audio_mode": "regenerate",
        "fbcache_enable": False, "fbcache_threshold": 0.12,
    }
    params.update(overrides)
    return params


def test_spatial_mask_with_fbcache_enabled_is_refused_before_generation():
    from api.error_handlers import ValidationError

    runner = _runner()
    timeline, masks = _half_split_manifest_and_masks()
    params = _params(fbcache_enable=True)
    with pytest.raises(ValidationError) as error:
        runner._generate_vidinpaint_minimax_h3(
            params, _source_clip(), 24.0, None,
            spatial_mask_timeline=timeline, spatial_mask_arrays=masks,
        )
    assert "FBCache" in str(error.value) or "fbcache" in str(error.value).lower()


def test_spatial_mask_with_fbcache_threshold_zero_is_not_refused():
    """NEGATIVE CONTROL: `fbcache_enable=True` alone does not activate FBCache
    (`fbcache_active` also requires a positive threshold) -- this must reach
    generation rather than being refused, so the test above is really about
    the exclusivity guard and not a blanket rejection of the flag."""
    runner = _runner()
    timeline, masks = _half_split_manifest_and_masks()
    params = _params(fbcache_enable=True, fbcache_threshold=0.0)

    class Runner(MiniMaxH3Mixin):
        minimax_h3_components = runner.minimax_h3_components
        current_model_info = runner.current_model_info
        reached = False

        def _generate_minimax_h3(self, params, **kwargs):
            Runner.reached = True
            frames = np.zeros((int(params["num_frames"]), HEIGHT, WIDTH, 3), dtype=np.uint8)
            return frames, None, None, 1

    Runner()._generate_vidinpaint_minimax_h3(
        params, _source_clip(), 24.0, None,
        spatial_mask_timeline=timeline, spatial_mask_arrays=masks,
    )
    assert Runner.reached is True


def test_spatial_mask_alone_without_fbcache_is_not_refused():
    """NEGATIVE CONTROL: a spatial mask by itself (fbcache_enable=False, the
    default) must reach generation -- otherwise the test above would be
    exercising "spatial masks are always refused" rather than the
    combination."""
    timeline, masks = _half_split_manifest_and_masks()
    params = _params(fbcache_enable=False)

    class Runner(MiniMaxH3Mixin):
        minimax_h3_components = {
            "variant": "fl2va",
            "audio_sample_rate": 32000,
            "vae_scale_factor_spatial": 16,
            "transformer_config": {"patch_size": (1, 2, 2)},
        }
        current_model_info = {"type": "minimax_h3", "variant": "fl2va"}
        reached = False

        def _generate_minimax_h3(self, params, **kwargs):
            Runner.reached = True
            frames = np.zeros((int(params["num_frames"]), HEIGHT, WIDTH, 3), dtype=np.uint8)
            return frames, None, None, 1

    Runner()._generate_vidinpaint_minimax_h3(
        params, _source_clip(), 24.0, None,
        spatial_mask_timeline=timeline, spatial_mask_arrays=masks,
    )
    assert Runner.reached is True
