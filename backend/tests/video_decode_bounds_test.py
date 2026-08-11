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
* `/generate/inpaint/video` used to bound it by the architecture's
  `max_frames`, since that route's output length EQUALS its trimmed input
  length. MiniMax-H3's `max_frames` is now `None` (362 is
  `trained_max_frames`, a DOCUMENTED trained-range top, not an enforced
  decoder limit -- see `core/models/components/wiring.py`), so "too long to
  produce" is no longer a length question there either: it decodes the whole
  trimmed clip unconditionally and is bounded only by a pre-decode RAM guard
  (`_refuse_if_decode_too_large`, the same one `/generate/outpaint/video`
  already uses), which still has to be a 400 naming both numbers, not a
  head-crop that looks like a successful run.

Tests read the live source of `backend/api/routes.py` rather than exercising
the routes: the defect is which literal is passed to the decode call, and
reproducing it for real needs a GPU generation.
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
# 2. Video inpaint refuses (400) a clip too long to DECODE (a RAM question),
#    rather than silently head-cropping it -- there is no longer an
#    arch-length cap to refuse against (MiniMax-H3's `max_frames` is None;
#    362 is `trained_max_frames`, DOCUMENTED not enforced, see wiring.py).
# ---------------------------------------------------------------------------
class InpaintVideoOverlongClipTest(unittest.TestCase):
    """`/generate/inpaint/video`'s output length equals the TRIMMED input
    length, and the arch no longer enforces a maximum length at all, so
    "too long" stopped being a length question and became a RESOURCE one: a
    raw decoded uint8 RGB clip costs `width * height * 3` bytes per frame, and
    an unbounded upload can OOM the process. Reverting the fix (removing the
    pre-decode `_refuse_if_decode_too_large` RAM guard, or reintroducing a
    bounded `max_frames=...` decode that silently crops) must fail this test:
    a bounded decode via ffmpeg's `-frames:v` always lands on a length nothing
    downstream would flag."""

    def test_route_probes_and_refuses_before_decoding(self):
        source = _routes_source()
        # inpaint_vid's decode section, isolated so the assertions below
        # can't accidentally match the unrelated outpaint route's similarly-
        # named locals/helper (both define a `_refuse_if_decode_too_large`).
        anchor = source.index("This used to be bounded by the arch's `max_frames`")
        section = source[anchor:anchor + 3000]

        self.assertIn(
            "def _refuse_if_decode_too_large(", section,
            "the inpaint route must ffprobe the upload and refuse an "
            "over-large decode BEFORE calling load_video_frames, the same "
            "RAM guard /generate/outpaint/video already applies",
        )
        self.assertIn("MAX_VIDEO_UPLOAD_DECODE_BYTES", section)
        self.assertIn(
            '_refuse_if_decode_too_large(video_data, label="input clip")',
            section,
        )
        # The decode itself must be unbounded: nothing left silently crops a
        # clip that passed the RAM guard.
        self.assertIn(
            "video_data, max_frames=None, trim_end_frames=input_trim_end_frames",
            section,
            "the inpaint route's decode must not be bounded by a computed "
            "frame cap anymore -- max_frames must be passed as None",
        )

    def test_no_arch_length_cap_remains_in_the_decode_section(self):
        """The pre-fix defect, verbatim: a computed `_arch_max_frames` /
        `_decode_max_frames` pair bounding the ffmpeg decode by the arch's
        (now nonexistent) production ceiling."""
        source = _routes_source()
        anchor = source.index("This used to be bounded by the arch's `max_frames`")
        section = source[anchor:anchor + 3000]
        self.assertNotIn("_arch_max_frames = int(", section)
        self.assertNotIn("_decode_max_frames = max(0, input_trim_start_frames)", section)


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
# 4. Video inpaint's decode is judged on the TRUE decoded frame count, with
#    no length-cap-vs-estimate gap left to under-count past -- G5's original
#    defect class (an ffprobe ESTIMATE that can under-count a VFR/matroska
#    clip sailing past a length refusal, then getting silently cropped to
#    exactly the cap by a bounded ffmpeg decode) is now structurally
#    impossible: there is no computed decode cap to crop to, or to sail past.
# ---------------------------------------------------------------------------
class InpaintVideoPostDecodeOverlongClipTest(unittest.TestCase):
    """Reverting the fix (reintroducing a computed `_arch_max_frames`-based
    decode bound in place of the unconditional `max_frames=None`) must fail
    this test: with a bounded decode restored, the true frame count
    (`video_frames.shape[0]`) can no longer differ from the (possibly
    under-counting) pre-decode ffprobe estimate in the way that hid an
    over-length clip before."""

    def _inpaint_decode_section(self, source: str) -> str:
        anchor = source.index("This used to be bounded by the arch's `max_frames`")
        return source[anchor:anchor + 4500]

    def test_decode_is_unconditionally_unbounded(self):
        source = _routes_source()
        section = self._inpaint_decode_section(source)
        self.assertIn(
            "video_data, max_frames=None, trim_end_frames=input_trim_end_frames",
            section,
        )
        self.assertNotIn("_decode_max_frames = max(0, input_trim_start_frames)", section)

    def test_trimmed_len_is_computed_from_the_true_decoded_shape_only(self):
        source = _routes_source()
        inpaint_anchor = source.index("This used to be bounded by the arch's `max_frames`")
        anchor = source.index("trimmed_len = video_frames.shape[0]", inpaint_anchor)
        section = source[anchor:anchor + 1500]
        # No post-decode length-cap refusal left: plan_video_inpaint_span is
        # what judges trimmed_len now (off-grid or below floor is still a
        # 400; on-grid past the DOCUMENTED trained range is accepted and
        # warned as untested, not refused for being long).
        self.assertNotIn("if trimmed_len > _arch_max_frames:", section)
        self.assertIn("plan_video_inpaint_span(", section)


if __name__ == "__main__":
    unittest.main()
