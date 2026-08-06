"""Golden tests for the temporal (video) training call chain.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/temporal_bucketing_test.py -v

WHY THIS FILE EXISTS
--------------------
`bucketing.py`'s temporal section, `video_loader.py` and `LatentCache`'s clip
functions serve LTX-2.3 training in production and had NO test coverage at all.
The MiniMax-H3 integration (Phase 6a) threads a `TemporalSpec` through every one
of them, adds timestamp-based 24 fps resampling, and extends the clip cache key.

Every expected value below was recorded from the code as it stood BEFORE that
refactor and is frozen: the refactor is only allowed to change how these
functions are CALLED, never what they answer for LTX-2.3. In particular
`test_cache_key_hashes_are_frozen` hardcodes md5 digests -- a changed digest
silently invalidates every cached latent a user already has, so it must fail
loudly instead.

Sections:
  1. clip-length validity        (bucketing + video_loader, incl. the odd
                                  int()-coercion behaviour of both)
  2. clip_span / pick_clip_length (empty + duplicate + all-invalid lists, the
                                  `[1]` fallback, all-short videos)
  3. sample_clip_window          (exact fit, overflow, stride, val centering)
  4. spatial bucket
  5. load_clip                   (source indices, stride, loop padding,
                                  normalisation range, resize)
  6. cache keys                  (frozen digests, window/fps discrimination)
  7. VideoBucketManager          (assignment, buckets, batch grouping)

Sections 8+ are the H3 additions and are marked as such.
"""

import hashlib
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.training import bucketing as B  # noqa: E402
from core.training import video_loader as VL  # noqa: E402
from core.training.latent_cache import LatentCache  # noqa: E402
from core.models.components.wiring import (  # noqa: E402
    LTX2_TEMPORAL,
    MINIMAX_H3_TEMPORAL,
)


# ===========================================================================
# helpers
# ===========================================================================

def _write_video(path: str, num_frames: int, fps: float, w: int = 64, h: int = 48):
    """Write a lossless FFV1 clip whose frame ``i`` is the constant colour
    ``i*8``, so a decoded frame identifies its SOURCE index exactly."""
    import cv2

    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), float(fps), (w, h))
    assert vw.isOpened(), "FFV1 writer unavailable"
    for i in range(num_frames):
        vw.write(np.full((h, w, 3), (i * 8) % 256, np.uint8))
    vw.release()
    return path


def _indices_of(clip: torch.Tensor):
    """Recover the source frame index of every frame of a `_write_video` clip.

    `load_clip` returns [-1, 1]; undo that to a 0-255 level, then /8.
    """
    out = []
    for t in clip:
        level = float(((t.mean().item() / 2.0) + 0.5) * 255.0)
        out.append(int(round(level / 8.0)))
    return out


# ===========================================================================
# 1. clip-length validity
# ===========================================================================

# (value, expected) -- pinned from the pre-refactor implementation. Note the
# two quirks that are behaviour, not accident: both validators int()-coerce, so
# "9" and 9.5 are ACCEPTED, while "x"/None are rejected via the except branch.
_VALID_CASES = [
    (1, True), (2, False), (8, False), (9, True), (16, False), (17, True),
    (25, True), (33, True), (49, True), (121, True), (0, False), (-7, False),
    ("9", True), ("x", False), (None, False), (9.0, True), (9.5, True),
]


@pytest.mark.parametrize("value,expected", _VALID_CASES)
def test_bucketing_is_valid_clip_length(value, expected):
    assert B.is_valid_clip_length(value) is expected


@pytest.mark.parametrize("value,expected", _VALID_CASES)
def test_video_loader_agrees_with_bucketing(value, expected):
    """The loader's validity check must agree with bucketing's, value for value.

    (Pre-refactor these were two copies of the same rule, `bucketing.
    is_valid_clip_length` and `video_loader.is_valid_ltx_clip_length`; the
    loader now imports the one function. The answers are the same either way,
    which is what this pins.)
    """
    from core.training.video_loader import is_valid_clip_length as loader_valid
    assert loader_valid(value) is expected


@pytest.mark.parametrize("length,expected", [(1, 1), (9, 2), (17, 3), (25, 4), (49, 7)])
def test_ltx_latent_frames(length, expected):
    assert VL.clip_latent_frames(length) == expected


def test_default_clip_lengths_unchanged():
    assert B.DEFAULT_CLIP_LENGTHS == [9, 17, 25, 33, 49]
    assert B.LTX_SPATIAL_DIVISIBILITY == 32


# ===========================================================================
# 2. clip_span / pick_clip_length
# ===========================================================================

@pytest.mark.parametrize("length,stride,expected", [
    (9, 1, 9), (9, 2, 17), (1, 5, 1), (49, 1, 49), (0, 0, 1), (17, 3, 49),
])
def test_clip_span(length, stride, expected):
    assert B.clip_span(length, stride) == expected


@pytest.mark.parametrize("num_frames,stride,allowed,expected", [
    # longest fitting length wins
    (100, 1, None, 49),
    (48, 1, None, 33),
    (100, 2, None, 49),          # span(49,2)=97 <= 100
    (96, 2, None, 33),           # span(49,2)=97 > 96 -> 33
    # ALL-SHORT video: nothing fits -> the SMALLEST allowed length, which
    # load_clip then loop-pads (bucketing.py:521 docstring).
    (8, 1, None, 9),
    (0, 1, None, 9),
    (1, 1, None, 9),
    # empty list -> falls back to DEFAULT_CLIP_LENGTHS (`allowed or DEFAULT`)
    (100, 1, [], 49),
    # all-invalid list -> the `[1]` fallback
    (100, 1, [3, 4], 1),
    (0, 1, [3, 4], 1),
    # duplicates are deduped, invalid entries are filtered out
    (100, 1, [9, 9, 17], 17),
    (100, 1, [49, 9, 17], 49),
    (100, 1, [9, 12, 17, 40], 17),
    (20, 1, [9, 17, 25], 17),
])
def test_pick_clip_length(num_frames, stride, allowed, expected):
    assert B.pick_clip_length(num_frames, stride, allowed) == expected


# ===========================================================================
# 3. sample_clip_window
# ===========================================================================

@pytest.mark.parametrize("num_frames,length,stride,expected", [
    (100, 9, 1, 45),     # centered: (100-9)//2
    (9, 9, 1, 0),        # EXACT fit -> max_start == 0 -> 0
    (8, 9, 1, 0),        # too short -> 0 (load_clip loop-pads)
    (0, 9, 1, 0),
    (100, 9, 2, 41),     # span 17 -> (100-17)//2
    (17, 9, 2, 0),       # exact fit with stride
    (100, 1, 1, 49),
])
def test_sample_clip_window_validation_is_centered(num_frames, length, stride, expected):
    # Phase 6a returns ClipWindow(start_frame, start_time); the start FRAME is
    # the pre-refactor return value and must be unchanged.
    assert VL.sample_clip_window(num_frames, length, stride, training=False).start_frame == expected


def test_sample_clip_window_training_stays_in_range():
    import random
    random.seed(1234)
    for _ in range(200):
        s = VL.sample_clip_window(100, 9, 2, training=True).start_frame
        assert 0 <= s <= 100 - 17


# ===========================================================================
# 4. spatial bucket
# ===========================================================================

@pytest.mark.parametrize("w,h,res,expected", [
    (1920, 1080, 768, (992, 576)),
    (512, 512, 512, (512, 512)),
    (640, 360, 512, (608, 352)),
    (1080, 1920, 1024, (768, 1344)),
])
def test_spatial_bucket(w, h, res, expected):
    b = B.get_video_spatial_bucket(w, h, resolution=res)
    assert (b.width, b.height) == expected
    assert b.width % 32 == 0 and b.height % 32 == 0


# ===========================================================================
# 5. load_clip
# ===========================================================================

@pytest.fixture(scope="module")
def clip24(tmp_path_factory):
    d = tmp_path_factory.mktemp("vid")
    return str(_write_video(str(d / "src24.avi"), 30, 24.0))


@pytest.fixture(scope="module")
def clip30(tmp_path_factory):
    d = tmp_path_factory.mktemp("vid30")
    return str(_write_video(str(d / "src30.avi"), 40, 30.0))


@pytest.fixture(scope="module")
def clip_short(tmp_path_factory):
    d = tmp_path_factory.mktemp("vidshort")
    return str(_write_video(str(d / "short.avi"), 5, 24.0))


def test_load_clip_shape_range_and_indices(clip24):
    clip = VL.load_clip(clip24, 9, start_frame=0, stride=1, target_w=64, target_h=48)
    assert tuple(clip.shape) == (9, 3, 48, 64)
    assert clip.dtype == torch.float32
    assert clip.min() >= -1.0 and clip.max() <= 1.0
    assert _indices_of(clip) == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_load_clip_start_and_stride(clip24):
    clip = VL.load_clip(clip24, 9, start_frame=5, stride=2, target_w=64, target_h=48)
    assert _indices_of(clip) == [5, 7, 9, 11, 13, 15, 17, 19, 21]


def test_load_clip_resizes(clip24):
    clip = VL.load_clip(clip24, 9, 0, 1, target_w=32, target_h=32)
    assert tuple(clip.shape) == (9, 3, 32, 32)


def test_load_clip_loop_pads_short_video(clip_short):
    """5-frame source, 9-frame request: the tail repeats the last good frame."""
    clip = VL.load_clip(clip_short, 9, 0, 1, target_w=64, target_h=48)
    assert tuple(clip.shape)[0] == 9
    idx = _indices_of(clip)
    assert idx[:5] == [0, 1, 2, 3, 4]
    assert idx[5:] == [4, 4, 4, 4]


def test_load_clip_rejects_invalid_length(clip24):
    with pytest.raises(ValueError):
        VL.load_clip(clip24, 22, 0, 1, target_w=64, target_h=48)


def test_load_clip_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError):
        VL.load_clip(str(tmp_path / "nope.avi"), 9, 0, 1)


# ===========================================================================
# 6. cache keys -- FROZEN DIGESTS
# ===========================================================================

# Recorded from the pre-refactor `compute_clip_hash`. These are user-visible
# cache filenames; a change here invalidates existing caches silently.
_FROZEN_CLIP_KEYS = [
    (("D:/v/a.webm", 768, 512, 0, 49, 1, None), "9fda49a2f0e9a0ae8a822d0622496164"),
    (("D:/v/a.webm", 768, 512, 0, 49, 1, 30.0), "53cf47df31ea8759def58bacf8bf31d4"),
    (("D:/v/a.webm", 768, 512, 17, 25, 2, 29.97), "fe57e29c63b49f935bbaa4088c9bc84a"),
    (("/mnt/v/b.mp4", 512, 512, 3, 9, 1, 24.0), "75e68f0cf7f8db673ba6891c95f41f1f"),
    (("D:/v/\u65e5\u672c.webm", 640, 384, 0, 17, 1, 25.0), "91d973a028d40189873654d220af0196"),
]


@pytest.mark.parametrize("args,digest", _FROZEN_CLIP_KEYS)
def test_cache_key_hashes_are_frozen(args, digest):
    assert LatentCache.compute_clip_hash(*args) == digest


def test_cache_key_discriminates_windows_and_fps():
    base = ("D:/v/a.webm", 768, 512, 0, 49, 1, 30.0)
    keys = {
        LatentCache.compute_clip_hash(*base),
        LatentCache.compute_clip_hash("D:/v/a.webm", 768, 512, 8, 49, 1, 30.0),
        LatentCache.compute_clip_hash("D:/v/a.webm", 768, 512, 0, 33, 1, 30.0),
        LatentCache.compute_clip_hash("D:/v/a.webm", 768, 512, 0, 49, 2, 30.0),
        LatentCache.compute_clip_hash("D:/v/a.webm", 640, 512, 0, 49, 1, 30.0),
        LatentCache.compute_clip_hash("D:/v/a.webm", 768, 512, 0, 49, 1, 24.0),
        LatentCache.compute_clip_hash("D:/v/a.webm", 768, 512, 0, 49, 1, None),
    }
    assert len(keys) == 7


def test_clip_key_never_collides_with_image_key():
    assert (LatentCache.compute_clip_hash("D:/v/a.webm", 768, 512, 0, 49, 1)
            != LatentCache.compute_image_hash("D:/v/a.webm", 768, 512))


# ===========================================================================
# 7. VideoBucketManager
# ===========================================================================

def test_vbm_defaults_and_filtering():
    vbm = B.VideoBucketManager(base_resolutions=[512])
    assert vbm.allowed_clip_lengths == [9, 17, 25, 33, 49]
    assert B.VideoBucketManager(base_resolutions=[512],
                                allowed_clip_lengths=[17, 9, 9, 12]).allowed_clip_lengths == [9, 17]
    # all-invalid -> the `or [1]` fallback
    assert B.VideoBucketManager(base_resolutions=[512],
                                allowed_clip_lengths=[3, 4]).allowed_clip_lengths == [1]


def test_vbm_assign_and_info_fields():
    vbm = B.VideoBucketManager(base_resolutions=[512])
    key, info = vbm.assign_video_to_bucket(
        "D:/v/a.webm", 1920, 1080, num_frames=100, caption="c", fps=30.0,
    )
    assert key[1] == 49
    assert info["clip_length"] == 49 and info["stride"] == 1
    assert info["item_type"] == "video" and info["fps"] == 30.0
    assert info["bucket_width"] % 32 == 0 and info["bucket_height"] % 32 == 0
    params = vbm.clip_cache_params(info, clip_start=7)
    assert params == {
        "video_path": "D:/v/a.webm",
        "width": info["bucket_width"], "height": info["bucket_height"],
        "clip_start": 7, "clip_length": 49, "stride": 1, "fps": 30.0,
    }


def test_vbm_short_video_gets_smallest_length():
    vbm = B.VideoBucketManager(base_resolutions=[512])
    key, info = vbm.assign_video_to_bucket("D:/v/s.webm", 640, 480, num_frames=4)
    assert info["clip_length"] == 9  # smallest allowed; load_clip loop-pads


def test_vbm_batches_are_uniform_in_bucket_and_length():
    vbm = B.VideoBucketManager(base_resolutions=[512])
    for i in range(5):
        vbm.assign_video_to_bucket(f"D:/v/long{i}.webm", 1024, 1024, num_frames=200)
    for i in range(3):
        vbm.assign_video_to_bucket(f"D:/v/short{i}.webm", 1024, 1024, num_frames=20)
    counts = vbm.get_bucket_counts()
    assert sum(counts.values()) == 8
    assert len(counts) == 2  # same spatial bucket, two clip lengths
    for batch in vbm.build_batch_indices(2):
        assert len({(b["bucket_width"], b["bucket_height"], b["clip_length"])
                    for b in batch}) == 1
        assert 1 <= len(batch) <= 2


def test_vbm_mixed_fps_share_a_bucket_but_not_a_cache_key():
    """fps is NOT part of the bucket key (batches stack on shape alone) but IS
    part of the clip cache key."""
    vbm = B.VideoBucketManager(base_resolutions=[512])
    k1, i1 = vbm.assign_video_to_bucket("D:/v/a.webm", 1024, 1024, 200, fps=24.0)
    k2, i2 = vbm.assign_video_to_bucket("D:/v/b.webm", 1024, 1024, 200, fps=29.97)
    assert k1 == k2
    assert (LatentCache.compute_clip_hash(**vbm.clip_cache_params(i1, 0))
            != LatentCache.compute_clip_hash(**vbm.clip_cache_params(i2, 0)))


# ===========================================================================
# 8. Phase 6a: the LTX-2.3 spec must reproduce the hardcoded rule exactly
# ===========================================================================

def test_explicit_ltx_spec_equals_the_hardcoded_rule():
    """Passing LTX-2.3's spec must be indistinguishable from passing nothing.

    This is the whole preservation argument in one test: `spec=None` is the old
    code path, `spec=LTX2_TEMPORAL` is the new declarative one, and if they ever
    disagree on any input the refactor has changed LTX-2.3's behaviour.
    """
    for cl in list(range(-3, 130)) + ["9", "x", None, 9.5]:
        assert (B.is_valid_clip_length(cl)
                is B.is_valid_clip_length(cl, LTX2_TEMPORAL))
    for cl in range(1, 60):
        for stride in (1, 2, 5):
            assert (B.clip_span(cl, stride)
                    == B.clip_span(cl, stride, LTX2_TEMPORAL, source_fps=30.0))
    for nf in (0, 1, 8, 20, 48, 100, 5000):
        for stride in (1, 2):
            assert (B.pick_clip_length(nf, stride)
                    == B.pick_clip_length(nf, stride, spec=LTX2_TEMPORAL,
                                          source_fps=30.0))
    for nf, cl, st in [(100, 9, 1), (9, 9, 1), (8, 9, 1), (100, 9, 2), (17, 9, 2)]:
        assert (VL.sample_clip_window(nf, cl, st, training=False).start_frame
                == VL.sample_clip_window(nf, cl, st, training=False,
                                         spec=LTX2_TEMPORAL, source_fps=30.0).start_frame)


def test_ltx_cache_key_unchanged_by_the_spec():
    """An LTX-2.3 clip key must be byte-identical with or without the spec."""
    vbm_none = B.VideoBucketManager(base_resolutions=[512])
    vbm_spec = B.VideoBucketManager(base_resolutions=[512], temporal_spec=LTX2_TEMPORAL)
    _, i1 = vbm_none.assign_video_to_bucket("D:/v/a.webm", 1024, 1024, 200, fps=30.0)
    _, i2 = vbm_spec.assign_video_to_bucket("D:/v/a.webm", 1024, 1024, 200, fps=30.0)
    p1 = vbm_none.clip_cache_params(i1, 7)
    p2 = vbm_spec.clip_cache_params(i2, 7)
    assert p1 == p2                      # no extra key fields appear
    assert set(p1) == {"video_path", "width", "height", "clip_start",
                       "clip_length", "stride", "fps"}
    assert "target_fps" not in i2        # and no target_fps on the item
    assert (LatentCache.compute_clip_hash(**p1)
            == LatentCache.compute_clip_hash(**p2))


def test_ltx_vbm_allowed_lengths_unchanged_by_the_spec():
    assert (B.VideoBucketManager(base_resolutions=[512]).allowed_clip_lengths
            == B.VideoBucketManager(base_resolutions=[512],
                                    temporal_spec=LTX2_TEMPORAL).allowed_clip_lengths)
    # the all-invalid `[1]` fallback survives too
    assert B.VideoBucketManager(base_resolutions=[512], allowed_clip_lengths=[3, 4],
                                temporal_spec=LTX2_TEMPORAL).allowed_clip_lengths == [1]


def test_ltx_index_policy_and_no_extra_key_fields():
    assert LTX2_TEMPORAL.resample_policy == "index"
    assert LTX2_TEMPORAL.clip_duration(49) is None
    # "index" is the implicit historical policy, so passing it explicitly must
    # NOT change a key (that is what keeps existing caches readable).
    assert (LatentCache.compute_clip_hash("D:/v/a.webm", 768, 512, 0, 49, 1, 30.0)
            == LatentCache.compute_clip_hash("D:/v/a.webm", 768, 512, 0, 49, 1, 30.0,
                                             resample_policy="index"))


# ===========================================================================
# 9. Phase 6a: MiniMax-H3 (17n+5, 24 fps, 22-frame decodable floor)
# ===========================================================================

H3 = MINIMAX_H3_TEMPORAL


@pytest.mark.parametrize("length,expected", [
    (5, False),    # ON the grid but NOT decodable (num_chunks == 0 at T_lat=2)
    (22, True), (39, True), (56, True), (124, True), (345, True),
    (21, False), (23, False), (1, False), (9, False), (0, False), (-17, False),
])
def test_h3_valid_clip_lengths(length, expected):
    assert B.is_valid_clip_length(length, H3) is expected
    from core.training.video_loader import is_valid_clip_length as loader_valid
    assert loader_valid(length, H3) is expected


@pytest.mark.parametrize("length,expected", [(22, 7), (39, 12), (56, 17), (124, 37)])
def test_h3_latent_frames(length, expected):
    assert VL.clip_latent_frames(length, H3) == expected


def test_h3_defaults_and_policy():
    assert H3.default_clip_lengths == (22, 39)
    assert H3.fps_fixed == 24.0
    assert H3.resample_policy == "timestamp_nearest"
    assert H3.min_decodable_frames == 22
    assert abs(H3.clip_duration(22) - 22 / 24.0) < 1e-9


def test_h3_clip_span_converts_duration_to_source_frames():
    # 22 frames at 24 fps = 21 gaps = 0.875 s; a 30 fps source covers that in
    # round(0.875*30)+1 = 27 frames, NOT 22.
    assert B.clip_span(22, 1, H3, source_fps=30.0) == 27
    assert B.clip_span(22, 1, H3, source_fps=24.0) == 22
    # Slow source: 0.875 s of a 12 fps video is round(10.5)+1 = 11 frames, and
    # the clip repeats frames to reach 22. clip_span agrees with the planner's
    # last index by construction (both are round(gaps/fps_fixed * source_fps)).
    assert B.clip_span(22, 1, H3, source_fps=12.0) == 11
    assert (VL.plan_source_indices(22, 0, 1, H3, start_time=0.0, source_fps=12.0)[-1]
            == B.clip_span(22, 1, H3, source_fps=12.0) - 1)
    # unknown source fps -> the index form (no silent mislabelling)
    assert B.clip_span(22, 1, H3) == 22


def test_h3_pick_clip_length():
    # 200 frames of 24 fps source: 39 fits (span 39), so the longer bucket wins
    assert B.pick_clip_length(200, 1, spec=H3, source_fps=24.0) == 39
    # a 30-frame source fits neither cleanly -> smallest allowed (22), loop-padded
    assert B.pick_clip_length(20, 1, spec=H3, source_fps=24.0) == 22
    # invalid configured lengths fall back to the arch's own shortest valid
    # length (22), NOT to LTX-2.3's `[1]` -- a 1-frame H3 clip is not loadable.
    assert B.pick_clip_length(200, 1, [9, 17], spec=H3, source_fps=24.0) == 22


def test_h3_bucket_manager_carries_target_fps():
    vbm = B.VideoBucketManager(base_resolutions=[512], temporal_spec=H3)
    assert vbm.allowed_clip_lengths == [22, 39]
    _, info = vbm.assign_video_to_bucket("D:/v/h3.mp4", 1280, 720, num_frames=300,
                                         fps=29.97)
    assert info["clip_length"] in (22, 39)
    assert info["fps"] == 29.97          # SOURCE rate
    assert info["target_fps"] == 24.0    # what the resampled clip plays at
    params = vbm.clip_cache_params(info, clip_start=90, start_time=3.003,
                                   tiling_policy="official:tile256_overlap64")
    assert params["resample_policy"] == "timestamp_nearest"
    assert params["target_fps"] == 24.0
    assert params["source_fps"] == 29.97
    assert params["start_time"] == 3.003
    assert params["tiling_policy"] == "official:tile256_overlap64"
    assert LatentCache.compute_clip_hash(**params)


def test_h3_key_discriminates_policy_tiling_and_start_time():
    base = dict(video_path="D:/v/h3.mp4", width=640, height=384, clip_start=0,
                clip_length=22, stride=1, fps=30.0)
    k_legacy = LatentCache.compute_clip_hash(**base)
    k_h3 = LatentCache.compute_clip_hash(
        **base, source_fps=30.0, target_fps=24.0,
        resample_policy="timestamp_nearest", start_time=0.0)
    k_tile_on = LatentCache.compute_clip_hash(
        **base, source_fps=30.0, target_fps=24.0,
        resample_policy="timestamp_nearest", start_time=0.0,
        tiling_policy="tile256_overlap64")
    k_tile_off = LatentCache.compute_clip_hash(
        **base, source_fps=30.0, target_fps=24.0,
        resample_policy="timestamp_nearest", start_time=0.0,
        tiling_policy="none")
    k_t4 = LatentCache.compute_clip_hash(
        **base, source_fps=30.0, target_fps=24.0,
        resample_policy="timestamp_nearest", start_time=4.0)
    k_audio = LatentCache.compute_clip_hash(
        **base, source_fps=30.0, target_fps=24.0,
        resample_policy="timestamp_nearest", start_time=0.0,
        audio_prep_version="h3-a1")
    keys = [k_legacy, k_h3, k_tile_on, k_tile_off, k_t4, k_audio]
    assert len(set(keys)) == len(keys), "every policy field must discriminate"


# ---------------------------------------------------------------------------
# 9b. timestamp resampling -- the sampled SOURCE indices
# ---------------------------------------------------------------------------

def test_h3_source_indices_24fps_are_consecutive():
    """A 24 fps source needs no resampling: 1:1, same as the index policy."""
    idx = VL.plan_source_indices(22, 0, 1, H3, start_time=0.0, source_fps=24.0)
    assert idx == list(range(22))


def test_h3_source_indices_30fps_are_genuinely_resampled():
    """30 -> 24 fps drops every 5th source frame.

    Expected list derived from the rule, not from the output: target frame i is
    at i/24 s, the nearest source frame is round(i*30/24) = round(i*1.25), and
    Python's round() is ties-to-even (i=2 -> 2.5 -> 2, i=6 -> 7.5 -> 8), which
    is deterministic and matches the Phase 0T prototype.
    """
    idx = VL.plan_source_indices(22, 0, 1, H3, start_time=0.0, source_fps=30.0)
    assert idx == [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20,
                   21, 22, 24, 25, 26]
    # and it is NOT a relabelling of the legacy indices
    assert idx != list(range(22))
    # the window really covers 22/24 s of source time
    assert idx[-1] == round((21 / 24.0) * 30.0)


def test_h3_source_indices_slow_source_repeats_frames():
    # round(i*0.5), ties-to-even: 0,0,1,2,2,2,3,4,4,4,...
    idx = VL.plan_source_indices(22, 0, 1, H3, start_time=0.0, source_fps=12.0)
    assert idx == [0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 8, 8, 8, 9, 10,
                   10, 10]
    assert len(idx) == 22


def test_h3_source_indices_respect_start_time_and_clamp():
    idx = VL.plan_source_indices(22, 0, 1, H3, start_time=4.0, source_fps=30.0,
                                 num_frames=200)
    assert idx[0] == 120  # 4.0 s * 30 fps
    clamped = VL.plan_source_indices(22, 0, 1, H3, start_time=0.0, source_fps=30.0,
                                     num_frames=5)
    assert max(clamped) == 4 and len(clamped) == 22


def test_index_policy_is_untouched_by_the_new_planner():
    assert VL.plan_source_indices(9, 5, 2) == [5, 7, 9, 11, 13, 15, 17, 19, 21]
    assert VL.plan_source_indices(9, 5, 2, LTX2_TEMPORAL, source_fps=30.0) == \
        [5, 7, 9, 11, 13, 15, 17, 19, 21]
    # no source fps -> fall back to index sampling rather than mislabel
    assert VL.plan_source_indices(22, 0, 1, H3) == list(range(22))


def test_h3_sample_window_returns_seconds():
    w = VL.sample_clip_window(300, 22, 1, training=False, spec=H3, source_fps=30.0)
    assert w.start_frame == (300 - 27) // 2
    assert abs(w.start_time - w.start_frame / 30.0) < 1e-9


# ---------------------------------------------------------------------------
# 9c. the blocker this phase exists to remove: load_clip rejecting 22 frames
# ---------------------------------------------------------------------------

def test_h3_lengths_load_through_the_shared_loader(clip30):
    """22 and 39 frames were REJECTED by `load_clip` before Phase 6a."""
    for length in (22, 39):
        clip = VL.load_clip(clip30, length, 0, 1, target_w=64, target_h=48,
                            spec=H3, start_time=0.0, source_fps=30.0)
        assert tuple(clip.shape) == (length, 3, 48, 64)
        assert clip.min() >= -1.0 and clip.max() <= 1.0


def test_h3_loaded_frames_are_the_resampled_source_frames(clip30):
    clip = VL.load_clip(clip30, 22, 0, 1, target_w=64, target_h=48,
                        spec=H3, start_time=0.0, source_fps=30.0)
    assert _indices_of(clip) == VL.plan_source_indices(
        22, 0, 1, H3, start_time=0.0, source_fps=30.0, num_frames=40)


def test_h3_still_rejects_an_off_grid_length(clip30):
    with pytest.raises(ValueError):
        VL.load_clip(clip30, 23, 0, 1, target_w=64, target_h=48, spec=H3,
                     source_fps=30.0)
    # and 5, which is ON the grid but below the decodable floor
    with pytest.raises(ValueError):
        VL.load_clip(clip30, 5, 0, 1, target_w=64, target_h=48, spec=H3,
                     source_fps=30.0)


def test_ltx_loader_still_rejects_an_h3_length(clip24):
    """Without a spec the loader is still strictly LTX-2.3."""
    with pytest.raises(ValueError):
        VL.load_clip(clip24, 22, 0, 1, target_w=64, target_h=48)
