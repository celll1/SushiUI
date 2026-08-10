"""What the video routes are allowed to DROP from an uploaded clip.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_decode_bounds_test.py -v

Both routes below bounded their ffmpeg decode by a number that has nothing to
do with the uploaded clip's own length, and ffmpeg's `-frames:v` obeys it
silently: the frames past the bound never reach the pipeline, and every
downstream check sees a perfectly consistent shorter clip. There is no crash
and no warning -- the user just gets back less of their own footage than they
uploaded.

* `/generate/outpaint/video` bounded the decode by `total_frames`. On a
  boundary-conditioned architecture the preserved clip is PASTED at a timeline
  edge, so its length is independent of `total_frames`; the per-architecture
  default `total_frames` is smaller than a long upload, so this was reachable
  without the user touching the length control at all.
* `/generate/inpaint/video` bounded it by the architecture's `max_frames`.
  That route's output length EQUALS its trimmed input length, so a clip longer
  than the cap cannot be served at any setting -- it has to be a 400 naming
  both numbers, not a head-crop that looks like a successful run.

Both tests read the live source of `backend/api/routes.py` rather than
exercising the routes: the defect is which literal is passed to the decode
call, and reproducing it for real needs a GPU generation.
"""

from __future__ import annotations

import os
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ROUTES_PATH = os.path.join(_BACKEND, "api", "routes.py")


def _routes_source() -> str:
    with open(_ROUTES_PATH, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1. Video outpaint's bridge clip is not silently truncated
# ---------------------------------------------------------------------------
class OutpaintVideoBridgeDecodeTest(unittest.TestCase):
    """The bridge upload is only accepted on a boundary-conditioned
    architecture (`"bridge" not in _placements` is refused earlier in the
    same route), where the preserved clip is PASTED at a timeline edge and
    is not bounded by `total_frames` -- exactly the reasoning already applied
    to the HEAD clip's own decode bound. Reverting the fix (going back to
    `max_frames=total_frames` for the bridge clip) must fail this test."""

    def test_bridge_decode_is_not_bounded_by_total_frames(self):
        source = _routes_source()
        # The pre-fix defect, verbatim: bounding the bridge decode by
        # total_frames, which on a boundary placement is the GENERATED
        # span's target length, not the preserved bridge clip's own length.
        self.assertNotIn(
            "bridge_frames, bridge_source_fps = load_video_frames(\n"
            "                bridge_data, max_frames=total_frames",
            source,
            "the bridge clip's decode is bounded by total_frames again -- "
            "on a boundary placement this silently drops the tail of any "
            "bridge clip longer than the request's total_frames",
        )

    def test_bridge_decode_uses_the_same_bound_as_the_head_clip(self):
        source = _routes_source()
        self.assertIn(
            "bridge_frames, bridge_source_fps = load_video_frames(\n"
            "                bridge_data, max_frames=_decode_max_frames",
            source,
            "the bridge clip must be decode-bounded by the SAME "
            "_decode_max_frames the head clip uses (None on a boundary "
            "placement), not a separately-computed bound",
        )


# ---------------------------------------------------------------------------
# 2. Video inpaint refuses (400) a clip too long to ever fit, rather than
#    silently head-cropping it
# ---------------------------------------------------------------------------
class InpaintVideoOverlongClipTest(unittest.TestCase):
    """`/generate/inpaint/video`'s output length equals the TRIMMED input
    length, so a clip whose trimmed length exceeds the loaded architecture's
    own longest producible clip cannot be served at any setting. Reverting
    the fix (removing the pre-decode ffprobe refusal) must fail this test:
    the old behaviour decoded exactly `_decode_max_frames` frames via
    ffmpeg's `-frames:v`, which is ALWAYS a valid (on-grid, in-range)
    trimmed length, so nothing downstream ever saw the truncation."""

    def test_route_probes_and_refuses_before_decoding(self):
        source = _routes_source()
        # inpaint_vid's decode section, isolated so the assertions below
        # can't accidentally match the unrelated outpaint route's similarly-
        # named locals.
        anchor = source.index("_arch_max_frames = int(")
        section = source[anchor:anchor + 3000]

        self.assertIn(
            "probe_upload_clip(video_data)", section,
            "the inpaint route must ffprobe the upload BEFORE the bounded "
            "ffmpeg decode, so an over-long clip can be named in a 400 "
            "instead of silently cropped by the decode bound",
        )
        self.assertIn(
            "the trimmed clip is longer than this model can produce", section,
            "no CustomValidationError refusing an over-long trimmed clip "
            "was found in the inpaint route's decode section",
        )
        # The refusal must be judged on the TRIMMED length (after the
        # user's own input_trim_start_frames/input_trim_end_frames), not the
        # raw upload length -- a clip that is only too long BEFORE the
        # user's own trim must still be accepted.
        self.assertIn("input_trim_start_frames", section)
        self.assertIn("input_trim_end_frames", section)
        self.assertIn("_probed_trimmed_len", section)
        self.assertIn("_probed_trimmed_len > _arch_max_frames", section)

    def test_refusal_names_both_numbers(self):
        """The 400 must name the model's own cap and the upload's actual
        (post-trim) length -- a generic "too long" message would leave the
        user unable to tell how much to trim."""
        source = _routes_source()
        anchor = source.index("_arch_max_frames = int(")
        section = source[anchor:anchor + 3000]
        self.assertIn("This model's longest clip is {_arch_max_frames}", section)
        self.assertIn("{_probed_trimmed_len}", section)


if __name__ == "__main__":
    unittest.main()
