"""Frontend-only: user-settable video chain segment length
(`chain_segment_frames`), plus the `trained_max_frames` advisory field that
motivated it (MiniMax-H3's `max_frames` went `null` -- see
`core/models/components/wiring.py`'s `TemporalSpec`).

Companion to `video_block_swap_threading_test.py` (same source-anchored
style, reading the frontend TS/TSX sources as text): a future edit that drops
one of these sites should fail loudly rather than silently regressing to
"chaining is unreachable on an uncapped architecture" (the bug this task
fixed) or "chaining is auto-triggered/mutated behind the user's back" (the
CLAUDE.md opt-in requirement this task preserves).

Per-FIELD assertions throughout, never a whole-line/whole-object match: a
line-shaped assertion in this exact codebase broke earlier in this session
when a third disjunct was added to a condition it pinned verbatim.
"""

from __future__ import annotations

import os
import re
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _read(*parts: str) -> str:
    with open(os.path.join(_REPO, *parts), encoding="utf-8") as handle:
        return handle.read()


def _function_source(source: str, name: str) -> str:
    """Slice out one `export const NAME = (...` arrow-function definition,
    the same convention `video_block_swap_threading_test.py`'s
    `FrontendApiTest._function_source` uses."""
    start = source.index(f"export const {name} =")
    end = source.find("\nexport const ", start + 1)
    return source[start:end if end >= 0 else None]


# ---------------------------------------------------------------------------
# 1. api.ts: VideoConstraints declares the advisory `trained_max_frames`
#    field, kept separate from the hard `max_frames` field it is served
#    alongside for MiniMax-H3.
# ---------------------------------------------------------------------------
class VideoConstraintsFieldTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "utils", "api.ts")

    def test_video_constraints_declares_trained_max_frames(self):
        match = re.search(
            r"export interface VideoConstraints \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("trained_max_frames?: number | null;", match.group(1))

    def test_trained_max_frames_is_a_distinct_field_from_max_frames(self):
        """Guards against someone collapsing the two into one optional field
        -- they answer different questions (hard wall vs advisory range) and
        can both be present-but-different (MiniMax-H3: max_frames=null,
        trained_max_frames=362)."""
        match = re.search(
            r"export interface VideoConstraints \{(.*?)\n\}", self.source, re.DOTALL)
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertIn("max_frames: number | null;", body)
        self.assertIn("trained_max_frames?: number | null;", body)


# ---------------------------------------------------------------------------
# 2. api.ts: the four chain-arithmetic helpers all take an optional
#    `segmentFrames` parameter, and share one `chainSegmentCap` resolver
#    rather than each re-deriving the fallback chain independently.
# ---------------------------------------------------------------------------
class ChainHelperSignatureTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "utils", "api.ts")

    def test_chain_segment_cap_resolver_exists(self):
        self.assertIn("const chainSegmentCap = (", self.source)
        # Falls back to the architecture's own max_frames before Infinity --
        # NOT straight to Infinity -- so a still-hard-capped architecture
        # keeps chaining automatically with the control left at its default.
        self.assertIn("return c?.max_frames ?? Number.POSITIVE_INFINITY;", self.source)

    def test_next_video_chain_total_frames_takes_segment_frames(self):
        fn = _function_source(self.source, "nextVideoChainTotalFrames")
        self.assertIn("segmentFrames?: number | null", fn)
        self.assertIn("chainSegmentCap(c, segmentFrames)", fn)

    def test_plan_video_chain_takes_segment_frames(self):
        fn = _function_source(self.source, "planVideoChain")
        self.assertIn("segmentFrames?: number | null", fn)
        self.assertIn("chainSegmentCap(c, segmentFrames)", fn)

    def test_plan_video_chain_segments_takes_segment_frames(self):
        fn = _function_source(self.source, "planVideoChainSegments")
        self.assertIn("segmentFrames?: number | null", fn)
        self.assertIn("chainSegmentCap(c, segmentFrames)", fn)

    def test_effective_segment_frames_takes_segment_frames(self):
        fn = _function_source(self.source, "effectiveSegmentFrames")
        self.assertIn("segmentFrames?: number | null", fn)
        self.assertIn("chainSegmentCap(c, segmentFrames)", fn)

    def test_null_segment_frames_means_never_split_by_default(self):
        """The resolver's own guard: only a POSITIVE FINITE `segmentFrames`
        overrides the max_frames/Infinity fallback -- null/undefined/0/NaN
        all fall through, which is what makes null the "never split unless
        the architecture still has a hard wall" default."""
        fn = self.source[self.source.index("const chainSegmentCap = ("):]
        fn = fn[:fn.index("\n};") + 3]
        self.assertIn("segmentFrames != null", fn)
        self.assertIn("Number.isFinite(segmentFrames)", fn)
        self.assertIn("segmentFrames > 0", fn)


# ---------------------------------------------------------------------------
# 3. api.ts: videoFrameLabel always states the floor, even with no ceiling.
# ---------------------------------------------------------------------------
class VideoFrameLabelTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "utils", "api.ts")

    def test_label_states_trained_ceiling_when_max_frames_is_null(self):
        fn = _function_source(self.source, "videoFrameLabel")
        self.assertIn("c.trained_max_frames != null", fn)
        self.assertIn("trained to ${c.trained_max_frames}", fn)

    def test_label_states_bare_floor_when_neither_ceiling_is_known(self):
        fn = _function_source(self.source, "videoFrameLabel")
        self.assertIn("`, ${c.min_frames}+`", fn)


# ---------------------------------------------------------------------------
# 4. GenerationQueueContext: QueueItem carries `chainSegmentFrames`, frozen
#    at enqueue time (a chain already running must not be retargeted by a
#    later change to the panel's control).
# ---------------------------------------------------------------------------
class QueueItemFieldTest(unittest.TestCase):
    def test_queue_item_declares_chain_segment_frames(self):
        source = _read(
            "frontend", "src", "contexts", "GenerationQueueContext.tsx")
        match = re.search(r"export interface QueueItem \{(.*?)\n\}", source, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertIn("chainSegmentFrames?: number | null;", match.group(1))


# ---------------------------------------------------------------------------
# 5. videoChain.ts: continuation-building and chain-advancing both thread
#    the segment length through, and it survives onto each queue item.
# ---------------------------------------------------------------------------
class VideoChainUtilTest(unittest.TestCase):
    def setUp(self):
        self.source = _read("frontend", "src", "utils", "videoChain.ts")

    def test_build_continuation_items_accepts_segment_frames(self):
        fn_start = self.source.index("export function buildChainContinuationQueueItems(")
        fn_end = self.source.index("\nexport ", fn_start + 1)
        fn = self.source[fn_start:fn_end]
        self.assertIn("segmentFrames?: number | null;", fn)
        self.assertIn(
            "planVideoChainSegments(args.caps, args.arch, args.targetFrames, args.segmentFrames)",
            fn,
        )
        self.assertIn("chainSegmentFrames: args.segmentFrames ?? null,", fn)

    def test_advance_video_chain_reads_segment_frames_from_the_item(self):
        """NOT from any freshly-passed argument -- `item.chainSegmentFrames`
        is what was frozen onto the item at enqueue time, which is the whole
        point (a live panel control must not retarget a running chain)."""
        fn_start = self.source.index("export async function advanceVideoChain(")
        fn = self.source[fn_start:]
        self.assertIn(
            "args.caps, args.arch, args.resultFrames, target, item.chainSegmentFrames", fn)


# ---------------------------------------------------------------------------
# 6. Txt2ImgPanel / Img2ImgPanel: local `chainSegmentFrames` state (default
#    null), the checkbox+NumberInput control, the arch-switch snap effect,
#    and every chain-trigger call site passing it through.
# ---------------------------------------------------------------------------
class PanelChainSegmentStateTest(unittest.TestCase):
    def _read(self, name: str) -> str:
        return _read("frontend", "src", "components", "generation", name)

    def test_both_panels_declare_chain_segment_frames_state_defaulting_null(self):
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn(
                    "const [chainSegmentFrames, setChainSegmentFrames] = useState<number | null>(null);",
                    source,
                )

    def test_both_panels_snap_a_held_segment_length_on_arch_switch(self):
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn("setChainSegmentFrames((prev) => {", source)
                self.assertIn("if (prev == null) return prev;", source)

    def test_both_panels_render_the_segment_length_checkbox(self):
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn("checked={chainSegmentFrames != null}", source)
                self.assertIn("Chain segment length", source)

    def test_both_panels_pass_chain_segment_frames_to_the_generate_time_gate(self):
        """The gate that decides whether the chain-choice dialog opens."""
        source = self._read("Txt2ImgPanel.tsx")
        self.assertIn(
            "planVideoChain(archCapabilities, loadedArch, params.num_frames ?? 0, chainSegmentFrames)",
            source,
        )
        source = self._read("Img2ImgPanel.tsx")
        self.assertIn(
            "planVideoChain(archCapabilities, loadedArch, params.num_frames ?? 0, chainSegmentFrames)",
            source,
        )

    def test_both_panels_freeze_the_segment_length_onto_the_enqueued_chain(self):
        """The main (segment 1) queue item's own `chainSegmentFrames` field
        -- distinct from `buildChainContinuationQueueItems`'s `segmentFrames`
        arg (covered by `VideoChainUtilTest`), which only reaches segments
        2..N."""
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn("chainSegmentFrames: segmentFrames,", source)

    def test_videochainprompt_state_carries_segment_frames(self):
        for panel in ("Txt2ImgPanel.tsx", "Img2ImgPanel.tsx"):
            with self.subTest(panel=panel):
                source = self._read(panel)
                self.assertIn("segmentFrames: number | null;", source)


# ---------------------------------------------------------------------------
# 7. VideoChainConfirmDialog / VideoFrameCountSlider wording: "single
#    inference" is not universally true any more once the segment length can
#    be a user choice rather than a physical wall.
# ---------------------------------------------------------------------------
class DialogAndSliderWordingTest(unittest.TestCase):
    def test_dialog_header_says_segment_length_not_single_inference_limit(self):
        source = _read(
            "frontend", "src", "components", "common", "VideoChainConfirmDialog.tsx")
        self.assertIn("Length exceeds the current segment length", source)
        self.assertNotIn("Length exceeds the single-inference limit", source)

    def test_dialog_button_says_single_request_not_single_inference(self):
        source = _read(
            "frontend", "src", "components", "common", "VideoChainConfirmDialog.tsx")
        self.assertIn("Generate at {capFrames} frames (single request)", source)

    def test_slider_over_cap_threshold_falls_back_to_trained_max_frames(self):
        source = _read(
            "frontend", "src", "components", "common", "VideoFrameCountSlider.tsx")
        self.assertIn(
            "const overCapThreshold = c.max_frames ?? c.trained_max_frames;", source)

    def test_slider_states_the_untested_fact_not_invalid(self):
        source = _read(
            "frontend", "src", "components", "common", "VideoFrameCountSlider.tsx")
        self.assertIn("documented trained range", source)
        self.assertIn("longer is untested", source)

    def test_slider_track_derives_reach_from_trained_max_frames(self):
        source = _read(
            "frontend", "src", "components", "common", "VideoFrameCountSlider.tsx")
        self.assertIn("c.trained_max_frames != null", source)
        self.assertIn("TRAINED_RANGE_SLIDER_HEADROOM", source)


if __name__ == "__main__":
    unittest.main()
