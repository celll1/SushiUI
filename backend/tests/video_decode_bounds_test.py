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


# ---------------------------------------------------------------------------
# 3. Video outpaint's "free" placement (LTX-2.3) is not silently truncated
#    either -- G3: the "total_frames genuinely is the timeline" reasoning
#    only holds when the upload is SHORTER than total_frames, and nothing
#    enforced that.
# ---------------------------------------------------------------------------
class OutpaintVideoFreePlacementDecodeTest(unittest.TestCase):
    """`load_video_frames(video_data, max_frames=...)` in the outpaint/video
    route must not bound the "free" placement's decode by
    `input_trim_start_frames + total_frames` -- an upload longer than that
    sum would silently lose its tail before `_generate_vidoutpaint_ltx2`'s
    own `outpaint_video_tail_frames_dropped` warning ever gets a chance to
    see (and report) the true clip length. Reverting the fix (restoring the
    placement-conditional `total_frames`-based cap) must fail this test."""

    def _outpaint_decode_section(self, source: str) -> str:
        anchor = source.index("_decode_max_frames = None")
        # Back up to the start of the explanatory comment block so the
        # assertions below can see the surrounding reasoning too.
        start = source.rindex("# Decode the uploaded clip", 0, anchor)
        return source[start:anchor + 200]

    def test_free_placement_decode_is_not_bounded_by_total_frames(self):
        source = _routes_source()
        section = self._outpaint_decode_section(source)
        # The pre-fix defect, verbatim: a placement-conditional cap that
        # bounds the "free" arm by input_trim_start_frames + total_frames.
        self.assertNotIn(
            'None if "free" not in _placements',
            section,
            "the outpaint/video route reintroduced a placement-conditional "
            "decode bound -- on the \"free\" placement this silently drops "
            "the tail of any upload longer than input_trim_start_frames + "
            "total_frames, with no warning, because "
            "outpaint_video_tail_frames_dropped only ever sees the "
            "already-truncated decoded frame count",
        )
        self.assertNotIn(
            "else max(0, input_trim_start_frames) + total_frames", section,
        )

    def test_decode_max_frames_is_unconditionally_none(self):
        source = _routes_source()
        section = self._outpaint_decode_section(source)
        self.assertIn(
            "_decode_max_frames = None", section,
            "the outpaint/video route's decode bound must be unconditionally "
            "None (every placement decodes the whole trimmed clip; RAM is "
            "bounded separately by _refuse_if_decode_too_large, and an "
            "over-long clip is reported downstream instead of silently "
            "cropped)",
        )

    def test_ram_guard_still_applies_regardless_of_placement(self):
        """Removing the placement-conditional decode cap must not remove the
        pre-decode RAM/size guard -- an unbounded decode of an arbitrarily
        long/high-resolution upload can OOM the process."""
        source = _routes_source()
        anchor = source.index("def _refuse_if_decode_too_large(")
        section = source[anchor:anchor + 1500]
        self.assertIn("MAX_VIDEO_UPLOAD_DECODE_BYTES", section)
        # Called unconditionally for both the head and bridge clips.
        self.assertIn(
            '_refuse_if_decode_too_large(video_data, label="input clip")',
            source,
        )
        self.assertIn(
            '_refuse_if_decode_too_large(bridge_data, label="bridge clip")',
            source,
        )


# ---------------------------------------------------------------------------
# 4. Video inpaint's over-length refusal is judged on the DECODED frame
#    count too, not only on an estimate that can under-count -- G5.
# ---------------------------------------------------------------------------
class InpaintVideoPostDecodeOverlongClipTest(unittest.TestCase):
    """`dataset_scanner.probe_video_metadata`'s frame count is an ESTIMATE on
    a container/codec without `nb_frames` (VFR webm/matroska): it falls back
    to `round(fps * duration)`, which can UNDER-count the true length. An
    estimate that under-counts below `_arch_max_frames` used to sail straight
    past the pre-decode refusal, after which the (un-bumped) ffmpeg
    `-frames:v` bound silently cropped the clip to exactly the cap -- always
    a valid, in-range trimmed length, so nothing downstream ever caught it.
    Reverting either half of the fix (the `+ 1` decode headroom, or the
    post-decode refusal) must fail this test."""

    def _inpaint_decode_section(self, source: str) -> str:
        anchor = source.index("_arch_max_frames = int(")
        return source[anchor:anchor + 4000]

    def test_decode_bound_has_one_frame_of_headroom_past_the_cap(self):
        source = _routes_source()
        section = self._inpaint_decode_section(source)
        # The pre-fix defect, verbatim: no "+ 1", so a truly over-length clip
        # decodes down to EXACTLY _arch_max_frames -- indistinguishable from
        # a clip that really was that long.
        self.assertNotIn(
            "_decode_max_frames = max(0, input_trim_start_frames) + _arch_max_frames\n",
            section,
            "the inpaint/video decode bound lost its +1 headroom past "
            "_arch_max_frames -- without it, an over-length clip decodes to "
            "exactly the cap and is indistinguishable from a clip that "
            "really was that long, so the post-decode refusal can never fire",
        )
        self.assertIn(
            "_decode_max_frames = max(0, input_trim_start_frames) + _arch_max_frames + 1",
            section,
        )

    def test_post_decode_refusal_exists_and_is_judged_on_true_decoded_length(self):
        source = _routes_source()
        # `_arch_max_frames = int(` anchors the inpaint route's decode
        # section; the outpaint route's own (unrelated) "trimmed_len ="
        # assignment appears earlier in the file and must not be matched.
        inpaint_anchor = source.index("_arch_max_frames = int(")
        anchor = source.index("trimmed_len = video_frames.shape[0]", inpaint_anchor)
        section = source[anchor:anchor + 2600]
        # Must check the DECODED trimmed length against the arch cap, not
        # only rely on the pre-decode (possibly under-counting) estimate.
        self.assertIn("if trimmed_len > _arch_max_frames:", section)
        self.assertIn(
            "the trimmed clip is longer than this model can produce", section,
            "no post-decode CustomValidationError refusing an over-long "
            "trimmed clip was found after the ffmpeg decode -- the "
            "pre-decode ffprobe estimate is not a substitute, since it can "
            "under-count a VFR/matroska clip's true frame count",
        )
        # And this refusal must come BEFORE plan_video_inpaint_span is
        # called, not rely solely on that function's own max_frames check
        # (which is fed the same, now-corrected, trimmed_len anyway, but the
        # explicit route-level check gives an accurate, non-generic message).
        self.assertLess(
            section.index("if trimmed_len > _arch_max_frames:"),
            section.index("plan_video_inpaint_span(params, _vid_arch"),
        )


if __name__ == "__main__":
    unittest.main()
