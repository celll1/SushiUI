"""Unit tests for the video chain context module (design §14.1).

Numbered comments refer to the checklist in
scratchpad/video_chain_context_design.md §14.1.
"""

import random
import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.inference.video_chain_context import (  # noqa: E402
    DEFAULT_CHAIN_DRIFT_TOLERANCE_FRAMES,
    VIDEO_CHAIN_ANCHOR_FRAMES,
    ChainPlanRequest,
    ChainReference,
    PersistentContext,
    TimelineEvent,
    VideoChainPlanError,
    VideoGridSpec,
    accumulated_after,
    anchor_global_frame,
    build_segment_spans,
    canonical_json,
    chain_segment_cap,
    compute_plan_hash,
    derive_segment_seed,
    derive_token_bindings,
    evaluate_chain_drift,
    format_timestamp,
    new_output_frames,
    next_chain_total_frames,
    plan_h3_chain_from_prompt,
    plan_video_chain,
    plan_video_chain_manifest,
    plan_video_chain_segments,
    resolve_reference_bindings,
    resolve_root_seed,
    resolve_segment_seeds,
    rewrite_reference_tokens,
    segment_reference_ids,
    segment_token_map,
    shots_to_events,
    parse_h3_structured_prompt,
    validate_manifest,
    validate_reference_binding,
    validate_timeline,
)

# backend/core/models/components/wiring.py:412 (MINIMAX_H3_TEMPORAL) and
# LTX-2.3's own spec, reduced to what the chain planner reads.
H3_GRID = VideoGridSpec(
    frame_multiple=17, frame_offset=5, min_frames=124, min_decodable_frames=22,
    max_frames=None,
)
CAPPED_GRID = VideoGridSpec(
    frame_multiple=8, frame_offset=1, min_frames=9, min_decodable_frames=9,
    max_frames=121,
)


def reference_frontend_plan(spec, target_frames, segment_frames=None):
    """A SECOND, independent transcription of frontend/src/utils/api.ts:1556-1675.

    Written straight from the TypeScript control flow (`chainSegmentCap`,
    `nextVideoChainTotalFrames`, `planVideoChain`, `planVideoChainSegments`,
    `snapUpValidVideoFrameCount`) rather than by calling the module under test,
    so an equality assertion between the two really does pin the port to the
    shipping frontend behaviour instead of to itself.

    Returns `(cap, segments, final_frames, continuation_totals)` or None.
    """
    def snap_up(frames):
        lo = max(spec.min_frames, spec.min_decodable_frames)
        k_lo = -(-(lo - spec.frame_offset) // spec.frame_multiple)
        k = max(-(-(frames - spec.frame_offset) // spec.frame_multiple), k_lo)
        if spec.max_frames is not None:
            k = min(k, (spec.max_frames - spec.frame_offset) // spec.frame_multiple)
        return k * spec.frame_multiple + spec.frame_offset

    def cap_of():
        if segment_frames is not None and segment_frames > 0:
            return segment_frames
        return spec.max_frames  # None == Infinity in the TS

    def next_total(accumulated):
        cap = cap_of()
        if cap is None:
            return None
        remaining = target_frames - accumulated
        if remaining <= 0:
            return None
        span = snap_up(min(remaining, cap))
        if span <= 1:
            return None
        return accumulated + span - 1

    cap = cap_of()
    if cap is None or target_frames <= cap:
        return None
    accumulated = cap
    totals = []
    guard = 0
    while guard < 500 and accumulated < target_frames:
        nxt = next_total(accumulated)
        if nxt is None:
            break
        totals.append(nxt)
        accumulated = nxt
        guard += 1
    return cap, 1 + len(totals), accumulated, totals


class GridAndLengthPlanTest(unittest.TestCase):
    """§14.1.1 -- model grid snap, identical to the frontend planner."""

    def test_snap_up_rounds_up_onto_the_grid_and_respects_the_floor(self):
        self.assertEqual(H3_GRID.snap_up(1), 124)      # floored at min_frames
        self.assertEqual(H3_GRID.snap_up(124), 124)
        self.assertEqual(H3_GRID.snap_up(125), 141)    # rounds UP, never down
        self.assertEqual(H3_GRID.snap_up(362), 362)
        self.assertEqual(CAPPED_GRID.snap_up(500), 121)  # clamped at max_frames

    def test_chain_segment_cap_prefers_the_user_segment_length(self):
        self.assertEqual(chain_segment_cap(H3_GRID, 362), 362)
        self.assertIsNone(chain_segment_cap(H3_GRID, None))   # uncapped arch
        self.assertIsNone(chain_segment_cap(H3_GRID, 0))      # 0 is "unset"
        self.assertEqual(chain_segment_cap(CAPPED_GRID, None), 121)

    def test_uncapped_arch_without_a_segment_length_plans_nothing(self):
        self.assertIsNone(plan_video_chain(H3_GRID, 1000))
        self.assertIsNone(plan_video_chain_segments(H3_GRID, 1000))
        self.assertIsNone(next_chain_total_frames(H3_GRID, 362, 1000))

    def test_target_that_already_fits_plans_nothing(self):
        self.assertIsNone(plan_video_chain(H3_GRID, 300, 362))

    def test_matches_the_frontend_planner_on_the_h3_grid(self):
        for target in (400, 500, 723, 724, 1000, 1600, 3000):
            for segment in (124, 141, 362, 300):
                expected = reference_frontend_plan(H3_GRID, target, segment)
                plan = plan_video_chain(H3_GRID, target, segment)
                if expected is None:
                    self.assertIsNone(plan, (target, segment))
                    continue
                cap, segments, final, totals = expected
                self.assertEqual(plan.cap_frames, cap, (target, segment))
                self.assertEqual(plan.segments, segments, (target, segment))
                self.assertEqual(plan.final_frames, final, (target, segment))
                self.assertEqual(list(plan.continuation_totals), totals, (target, segment))
                self.assertEqual(
                    plan_video_chain_segments(H3_GRID, target, segment), totals
                )

    def test_matches_the_frontend_planner_on_a_capped_arch(self):
        for target in (130, 200, 400, 1000):
            for segment in (None, 41, 121):
                expected = reference_frontend_plan(CAPPED_GRID, target, segment)
                plan = plan_video_chain(CAPPED_GRID, target, segment)
                if expected is None:
                    self.assertIsNone(plan, (target, segment))
                    continue
                cap, segments, final, totals = expected
                self.assertEqual(
                    (plan.cap_frames, plan.segments, plan.final_frames,
                     list(plan.continuation_totals)),
                    (cap, segments, final, totals),
                    (target, segment),
                )

    def test_h3_1000_frame_chain_is_the_shipping_arithmetic(self):
        """Pins the concrete numbers so a refactor cannot quietly change them.

        NOTE: the design doc's §5 example (362/362/294 -> 1016) treats the
        remainder as NEW frames and adds the shared anchor back before snapping.
        The shipping frontend snaps the raw remainder instead, so the real plan
        is the one asserted here. Kept as a deliberate parity decision, not an
        accident: this module has to describe the requests the queue actually
        makes today.
        """
        plan = plan_video_chain(H3_GRID, 1000, 362)
        self.assertEqual(plan.cap_frames, 362)
        self.assertEqual(list(plan.continuation_totals), [723, 999, 1122])
        self.assertEqual(plan.segments, 4)
        self.assertEqual(plan.final_frames, 1122)


class FrameArithmeticTest(unittest.TestCase):
    """§14.1.1 / §14.1.2 -- shared-anchor arithmetic, both directions."""

    def test_anchor_and_accumulation_formulas(self):
        self.assertEqual(anchor_global_frame(362), 361)
        self.assertEqual(new_output_frames(362), 361)
        self.assertEqual(accumulated_after(362, 362), 723)

    def test_segment_spans_tile_the_output_without_gaps_or_overlap(self):
        spans = build_segment_spans(H3_GRID, 1000, 362)
        self.assertEqual([s.index for s in spans], [0, 1, 2, 3])
        self.assertIsNone(spans[0].anchor_global_frame)
        self.assertEqual(spans[0].owned_start_frame, 0)
        cursor = 0
        for span in spans:
            self.assertEqual(span.owned_start_frame, cursor)
            cursor = span.owned_end_frame
        self.assertEqual(cursor, 1122)
        for span in spans[1:]:
            self.assertEqual(span.anchor_global_frame, span.owned_start_frame - 1)
            self.assertEqual(
                span.owned_frames, span.generated_span_frames - VIDEO_CHAIN_ANCHOR_FRAMES
            )
            self.assertEqual(span.requested_total_frames, span.owned_end_frame)

    def test_global_and_local_frame_are_exact_inverses(self):
        spans = build_segment_spans(H3_GRID, 1000, 362)
        for span in spans:
            for k in (0, 1, span.generated_span_frames - 1):
                self.assertEqual(span.local_frame(span.global_frame(k)), k)
            # A continuation's local 0 IS the shared anchor, i.e. the last frame
            # the previous segment already produced -- not its first new frame.
            if span.anchor_global_frame is not None:
                self.assertEqual(span.global_frame(0), span.anchor_global_frame)
                self.assertEqual(span.global_frame(1), span.owned_start_frame)
            else:
                self.assertEqual(span.global_frame(0), 0)

    def test_out_of_range_local_index_is_refused(self):
        span = build_segment_spans(H3_GRID, 1000, 362)[1]
        with self.assertRaises(VideoChainPlanError):
            span.global_frame(span.generated_span_frames)
        with self.assertRaises(VideoChainPlanError):
            span.local_frame(span.anchor_global_frame - 1)

    def test_timestamp_formatting(self):
        self.assertEqual(format_timestamp(0, 24.0), "00:00.000")
        self.assertEqual(format_timestamp(24, 24.0), "00:01.000")
        self.assertEqual(format_timestamp(24 * 65 + 12, 24.0), "01:05.500")


# A 700-frame target with a 362-frame segment cap plans two segments:
# [0, 362) and [362, 706) (the second request asks for 345 frames and returns
# 706 accumulated). The timeline below tiles exactly that.
PLANNED_FINAL_FRAMES = 706


def _events():
    return [
        TimelineEvent(
            id="e0", kind="shot", start_frame=0, end_frame=200,
            description="The baker unlocks the door.",
            resulting_state="The door is unlocked.",
        ),
        TimelineEvent(
            id="e1", kind="shot", start_frame=200, end_frame=362,
            description='The baker says "Good morning, everyone."',
            resulting_state="The baker stands behind the counter.",
            verbatim=['"Good morning, everyone."'],
        ),
        TimelineEvent(
            id="e2", kind="shot", start_frame=362, end_frame=PLANNED_FINAL_FRAMES,
            description="The camera pushes in on the pastry case.",
            resulting_state="The camera holds on the pastry case.",
        ),
    ]


def _plan_request(**changes):
    values = dict(
        architecture="generic_video",
        variant="t2v",
        root_prompt="a bakery opens for the day",
        grid=H3_GRID,
        fps=24.0,
        target_frames=700,
        segment_frames=362,
        persistent_context=PersistentContext(
            subjects=["A tall baker in a blue apron."],
            visual_style=["Warm morning light, shot on 35 mm."],
        ),
        events=_events(),
        chain_id="chain-test",
        root_seed=1234,
    )
    values.update(changes)
    return ChainPlanRequest(**values)


class TimelineValidationTest(unittest.TestCase):
    """§14.1.8 -- out-of-range timestamps and shot gap / overlap are refused."""

    def test_valid_timeline_passes(self):
        validate_timeline(_events(), PLANNED_FINAL_FRAMES)

    def test_timestamp_past_the_plan_is_refused(self):
        events = _events()
        events[-1].end_frame = 900
        with self.assertRaises(VideoChainPlanError):
            validate_timeline(events, PLANNED_FINAL_FRAMES)

    def test_gap_between_shots_is_refused(self):
        events = _events()
        events[1].start_frame = 250
        with self.assertRaises(VideoChainPlanError):
            validate_timeline(events, PLANNED_FINAL_FRAMES)

    def test_overlapping_shots_are_refused(self):
        events = _events()
        events[1].start_frame = 150
        with self.assertRaises(VideoChainPlanError):
            validate_timeline(events, PLANNED_FINAL_FRAMES)

    def test_reversed_range_and_duplicate_id_are_refused(self):
        events = _events()
        events[0].end_frame = 0
        with self.assertRaises(VideoChainPlanError):
            validate_timeline(events, PLANNED_FINAL_FRAMES)
        events = _events()
        events[1].id = "e0"
        with self.assertRaises(VideoChainPlanError):
            validate_timeline(events, PLANNED_FINAL_FRAMES)

    def test_shot_crossing_a_segment_boundary_is_refused_by_default(self):
        # Segment 1 owns [0, 362); an event that starts inside it and ends past
        # it must not be split silently (design §17-4).
        events = [
            TimelineEvent(id="a", kind="shot", start_frame=0, end_frame=400,
                          description="one long take"),
            TimelineEvent(id="b", kind="shot", start_frame=400,
                          end_frame=PLANNED_FINAL_FRAMES,
                          description="the second take"),
        ]
        # `fixed` by name: with the boundaries resolved from the timeline the
        # planner would place one at 400 and there would be no crossing at all.
        with self.assertRaises(VideoChainPlanError):
            plan_video_chain_manifest(
                _plan_request(events=events, segment_length_mode="fixed")
            )
        # ... but the caller may opt in, and then it is a warning, not a split.
        manifest = plan_video_chain_manifest(
            _plan_request(events=events, segment_length_mode="fixed",
                          allow_boundary_split=True)
        )
        self.assertTrue(any("crosses the boundary" in w for w in manifest.warnings))
        self.assertEqual(manifest.segments[0].owned_event_ids, ["a"])


class ManifestCompilationTest(unittest.TestCase):
    def setUp(self):
        self.manifest = plan_video_chain_manifest(_plan_request())

    def test_two_segments_for_a_700_frame_target(self):
        self.assertEqual(len(self.manifest.segments), 2)
        self.assertEqual(self.manifest.expected_final_frames, PLANNED_FINAL_FRAMES)
        self.assertEqual(self.manifest.segments[0].generated_span_frames, 362)
        self.assertEqual(self.manifest.segments[1].generated_span_frames, 345)
        self.assertEqual(self.manifest.segments[1].anchor_global_frame, 361)
        self.assertEqual(
            self.manifest.segments[1].requested_total_frames, PLANNED_FINAL_FRAMES
        )

    def test_every_event_has_exactly_one_owner(self):
        """§14.1.3"""
        owners = [self.manifest.owner_of(e.id) for e in self.manifest.events]
        self.assertEqual(owners, [0, 0, 1])
        owned = [i for s in self.manifest.segments for i in s.owned_event_ids]
        self.assertEqual(sorted(owned), ["e0", "e1", "e2"])
        self.assertEqual(len(owned), len(set(owned)))

    def test_persistent_context_survives_into_every_segment(self):
        """§14.1.4"""
        for segment in self.manifest.segments:
            self.assertIn("A tall baker in a blue apron.", segment.prompt)
            self.assertIn("Warm morning light, shot on 35 mm.", segment.prompt)

    def test_completed_one_shot_event_does_not_reappear(self):
        """§14.1.5"""
        self.assertIn("unlocks the door", self.manifest.segments[0].prompt)
        self.assertNotIn("unlocks the door", self.manifest.segments[1].prompt)
        # ... its RESULT is carried instead, as incoming state.
        self.assertIn("The door is unlocked.", self.manifest.segments[1].prompt)
        self.assertIn("The door is unlocked.", self.manifest.segments[1].incoming_state)

    def test_dialogue_is_verbatim_and_only_in_its_owner_segment(self):
        """§14.1.6"""
        line = '"Good morning, everyone."'
        hits = [i for i, p in enumerate(self.manifest.segment_prompts()) if line in p]
        self.assertEqual(hits, [0])

    def test_future_events_are_not_leaked_into_earlier_segments(self):
        self.assertNotIn("pastry case", self.manifest.segments[0].prompt)

    def test_validate_manifest_catches_a_duplicated_owner(self):
        self.manifest.segments[1].owned_event_ids.append("e1")
        with self.assertRaises(VideoChainPlanError):
            validate_manifest(self.manifest)


class ReferenceBindingTest(unittest.TestCase):
    """§14.1.7 / §14.1.7b -- many-to-many binding and token renumbering."""

    def refs(self):
        return [
            ChainReference(id="ref_0", kind="image", label="protagonist.png",
                           token="<Picture 1>"),
            ChainReference(id="ref_1", kind="image", label="shop.png",
                           token="<Picture 2>", segment_indices=[1]),
            ChainReference(id="ref_2", kind="image", label="cat.png",
                           token="<Picture 3>", segment_indices=[0, 2]),
        ]

    def test_unbound_reference_defaults_to_every_segment(self):
        resolved = resolve_reference_bindings([self.refs()[0]], 3)
        self.assertEqual(resolved[0].segment_indices, [0, 1, 2])
        self.assertEqual(resolved[0].binding_source, "default_all")

    def test_binding_round_trips_and_is_many_to_many(self):
        resolved = resolve_reference_bindings(self.refs(), 3)
        per_segment = [segment_reference_ids(resolved, i) for i in range(3)]
        # one reference over several NON-CONTIGUOUS segments ...
        self.assertEqual(resolved[2].segment_indices, [0, 2])
        # ... and one segment holding several references.
        self.assertEqual(per_segment[0], ["ref_0", "ref_2"])
        self.assertEqual(per_segment[1], ["ref_0", "ref_1"])
        self.assertEqual(per_segment[2], ["ref_0", "ref_2"])
        validate_reference_binding(resolved, per_segment)

    def test_inconsistent_inverse_binding_is_a_validation_error(self):
        resolved = resolve_reference_bindings(self.refs(), 3)
        per_segment = [segment_reference_ids(resolved, i) for i in range(3)]
        per_segment[1] = ["ref_0", "ref_2"]
        with self.assertRaises(VideoChainPlanError):
            validate_reference_binding(resolved, per_segment)

    def test_out_of_range_segment_index_is_refused(self):
        with self.assertRaises(VideoChainPlanError):
            resolve_reference_bindings(
                [ChainReference(id="r", kind="image", segment_indices=[7])], 3
            )

    def test_unused_reference_and_bare_segment_only_warn(self):
        warnings = []
        resolve_reference_bindings(
            [ChainReference(id="r", kind="image", label="unused", segment_indices=[])],
            2,
            warnings,
        )
        self.assertTrue(any("not used by any segment" in w for w in warnings))
        self.assertTrue(any("has no reference bound to it" in w for w in warnings))

    def test_tokens_are_renumbered_for_each_segment_actual_order(self):
        resolved = resolve_reference_bindings(self.refs(), 3)
        # Segment 1 carries ref_0 and ref_2, so they become Picture 1 and 2 there.
        self.assertEqual(
            segment_token_map(resolved, 0), {"<Picture 1>": "<Picture 1>",
                                             "<Picture 3>": "<Picture 2>"}
        )
        # Segment 2 carries ref_0 and ref_1 -> Picture 1 and 2.
        self.assertEqual(
            segment_token_map(resolved, 1), {"<Picture 1>": "<Picture 1>",
                                             "<Picture 2>": "<Picture 2>"}
        )

    def test_unbound_tokens_are_dropped_from_the_segment_prompt(self):
        resolved = resolve_reference_bindings(self.refs(), 3)
        text, dropped = rewrite_reference_tokens(
            "<Picture 1> greets <Picture 2> beside <Picture 3>.",
            segment_token_map(resolved, 0),
        )
        self.assertEqual(dropped, ["<Picture 2>"])
        self.assertNotIn("<Picture 3>", text)
        self.assertIn("<Picture 2>", text)  # ref_2 renumbered into slot 2

    def test_manifest_prompts_use_the_segment_local_token_numbering(self):
        events = [
            TimelineEvent(id="e0", kind="shot", start_frame=0, end_frame=362,
                          description="<Picture 3> unlocks the door."),
            TimelineEvent(id="e1", kind="shot", start_frame=362,
                          end_frame=PLANNED_FINAL_FRAMES,
                          description="The cat jumps onto the counter."),
        ]
        # Two segments here, so ref_2's [0, 2] narrows to segment 1 only.
        references = self.refs()
        references[2].segment_indices = [0]
        manifest = plan_video_chain_manifest(
            _plan_request(events=events, references=references)
        )
        self.assertEqual(manifest.segments[0].reference_ids, ["ref_0", "ref_2"])
        self.assertEqual(manifest.segments[1].reference_ids, ["ref_0", "ref_1"])
        # Segment 1 carries ref_0 and ref_2, so ref_2 is its SECOND picture.
        self.assertIn("<Picture 2> unlocks", manifest.segments[0].prompt)
        self.assertNotIn("<Picture 3>", manifest.segments[0].prompt)


class TokenImpliedBindingTest(unittest.TestCase):
    """§5.1 -- a reference token in a segment's text binds that reference to it.

    Deleting the token instead would leave a mutilated sentence ("the woman
    shown in ."), so the binding is widened and the widening is reported.
    """

    def _events_using(self, token):
        return [
            TimelineEvent(id="e0", kind="shot", start_frame=0, end_frame=362,
                          description="The baker unlocks the door."),
            TimelineEvent(id="e1", kind="shot", start_frame=362,
                          end_frame=PLANNED_FINAL_FRAMES,
                          description=f"{token} jumps onto the counter."),
        ]

    def test_derive_token_bindings_unions_with_the_explicit_binding(self):
        resolved = resolve_reference_bindings(
            [ChainReference(id="r", kind="image", token="<Picture 1>",
                            segment_indices=[0])],
            3,
        )
        warnings = []
        derived = derive_token_bindings(
            resolved, ["nothing here", "<Picture 1> waves", "<Picture 1> leaves"], warnings
        )
        self.assertEqual(derived[0].segment_indices, [0, 1, 2])
        self.assertEqual(derived[0].binding_source, "token_implied")
        self.assertTrue(any("segments 2, 3" in w for w in warnings))

    def test_untouched_bindings_keep_their_source_and_warn_nothing(self):
        resolved = resolve_reference_bindings(
            [
                ChainReference(id="a", kind="image", token="<Picture 1>"),
                ChainReference(id="b", kind="image", token="<Picture 2>",
                               segment_indices=[1]),
            ],
            2,
        )
        warnings = []
        derived = derive_token_bindings(
            resolved, ["nothing here", "<Picture 1> and <Picture 2>"], warnings
        )
        self.assertEqual([r.binding_source for r in derived], ["default_all", "explicit"])
        self.assertEqual(derived[1].segment_indices, [1])
        self.assertEqual(warnings, [])

    def test_a_token_in_a_shot_applies_the_reference_to_its_owner_segment(self):
        # The user narrowed ref_1 to segment 1, but segment 2's shot names it.
        references = [
            ChainReference(id="ref_0", kind="image", token="<Picture 1>"),
            ChainReference(id="ref_1", kind="image", label="cat.png",
                           token="<Picture 2>", segment_indices=[0]),
        ]
        manifest = plan_video_chain_manifest(
            _plan_request(events=self._events_using("<Picture 2>"),
                          references=references)
        )
        self.assertEqual(manifest.references[1].segment_indices, [0, 1])
        self.assertEqual(manifest.references[1].binding_source, "token_implied")
        self.assertEqual(manifest.segments[1].reference_ids, ["ref_0", "ref_1"])
        # ... and the token survives instead of leaving "jumps onto the counter."
        self.assertIn("<Picture 2> jumps onto the counter.", manifest.segments[1].prompt)

    def test_widening_an_explicit_binding_is_warned_never_silent(self):
        references = [
            ChainReference(id="ref_1", kind="image", label="cat.png",
                           token="<Picture 1>", segment_indices=[0]),
        ]
        manifest = plan_video_chain_manifest(
            _plan_request(events=self._events_using("<Picture 1>"),
                          references=references)
        )
        self.assertTrue(
            any("was not bound to segment 2" in w for w in manifest.warnings),
            manifest.warnings,
        )

    def test_only_tokens_outside_the_inventory_are_dropped(self):
        references = [
            ChainReference(id="ref_1", kind="image", token="<Picture 1>",
                           segment_indices=[0]),
        ]
        manifest = plan_video_chain_manifest(
            _plan_request(events=self._events_using("<Picture 9>"),
                          references=references)
        )
        dropped = [w for w in manifest.warnings if "removed reference tokens" in w]
        self.assertEqual(len(dropped), 1)
        self.assertIn("<Picture 9>", dropped[0])
        self.assertNotIn("<Picture 9>", manifest.segments[1].prompt)
        # The known reference was still widened rather than dropped anywhere.
        self.assertEqual(manifest.references[0].segment_indices, [0])

    def test_a_segment_with_no_reference_still_warns(self):
        references = [
            ChainReference(id="ref_1", kind="image", token="<Picture 1>",
                           segment_indices=[0]),
        ]
        manifest = plan_video_chain_manifest(
            _plan_request(events=self._events_using("the cat"), references=references)
        )
        self.assertTrue(
            any("Segment 2 has no reference bound to it" in w for w in manifest.warnings)
        )

    def test_token_derived_binding_is_part_of_the_plan_hash(self):
        request = dict(events=self._events_using("<Picture 2>"), seed_policy="derived")
        references = [
            ChainReference(id="ref_0", kind="image", token="<Picture 1>"),
            ChainReference(id="ref_1", kind="image", token="<Picture 2>",
                           segment_indices=[0]),
        ]
        a = plan_video_chain_manifest(_plan_request(references=references, **request))
        b = plan_video_chain_manifest(_plan_request(references=references, **request))
        self.assertEqual(a.plan_hash, b.plan_hash)
        self.assertEqual([s.seed for s in a.segments], [s.seed for s in b.segments])
        # The hash is over the WIDENED binding, not the requested one.
        self.assertEqual(compute_plan_hash(a.to_dict()), a.plan_hash)
        self.assertEqual(a.to_dict()["references"][1]["segment_indices"], [0, 1])
        # A reference the user had already bound everywhere hashes differently
        # only through `binding_source`, which is part of the plan.
        explicit = [
            ChainReference(id="ref_0", kind="image", token="<Picture 1>"),
            ChainReference(id="ref_1", kind="image", token="<Picture 2>",
                           segment_indices=[0, 1]),
        ]
        c = plan_video_chain_manifest(_plan_request(references=explicit, **request))
        self.assertEqual(c.references[1].binding_source, "explicit")
        self.assertEqual(a.segment_prompts(), c.segment_prompts())
        self.assertNotEqual(a.plan_hash, c.plan_hash)


class SeedPolicyTest(unittest.TestCase):
    """§8"""

    def test_fixed_is_the_default_and_repeats_the_root_seed(self):
        manifest = plan_video_chain_manifest(_plan_request())
        self.assertEqual(manifest.seed_policy, "fixed")
        self.assertEqual([s.seed for s in manifest.segments], [1234, 1234])

    def test_explicit_seeds_are_used_verbatim(self):
        manifest = plan_video_chain_manifest(
            _plan_request(seed_policy="explicit", explicit_seeds=[7, 9])
        )
        self.assertEqual([s.seed for s in manifest.segments], [7, 9])
        with self.assertRaises(VideoChainPlanError):
            plan_video_chain_manifest(
                _plan_request(seed_policy="explicit", explicit_seeds=[7])
            )

    def test_derived_seeds_are_stable_and_distinct(self):
        a = plan_video_chain_manifest(_plan_request(seed_policy="derived"))
        b = plan_video_chain_manifest(_plan_request(seed_policy="derived"))
        self.assertEqual([s.seed for s in a.segments], [s.seed for s in b.segments])
        self.assertNotEqual(a.segments[0].seed, a.segments[1].seed)
        for segment in a.segments:
            self.assertTrue(0 <= segment.seed < 2 ** 32)
        self.assertEqual(
            a.segments[1].seed, derive_segment_seed(1234, a.plan_hash, 1)
        )

    def test_random_root_seed_is_fixed_once_at_plan_time(self):
        manifest = plan_video_chain_manifest(
            _plan_request(root_seed=-1, rng=random.Random(0))
        )
        self.assertGreaterEqual(manifest.root_seed, 0)
        self.assertEqual(
            [s.seed for s in manifest.segments],
            [manifest.root_seed, manifest.root_seed],
        )
        self.assertEqual(resolve_root_seed(-1, random.Random(0)), manifest.root_seed)

    def test_unknown_policy_is_refused(self):
        with self.assertRaises(VideoChainPlanError):
            resolve_segment_seeds("random_each_time", 1, "hash", 2)


class PlanHashTest(unittest.TestCase):
    """§5.1 / §14.1.9"""

    def test_same_input_gives_the_same_hash_prompts_and_seeds(self):
        a = plan_video_chain_manifest(_plan_request(seed_policy="derived"))
        b = plan_video_chain_manifest(_plan_request(seed_policy="derived"))
        self.assertEqual(a.plan_hash, b.plan_hash)
        self.assertEqual(a.segment_prompts(), b.segment_prompts())
        self.assertEqual([s.seed for s in a.segments], [s.seed for s in b.segments])

    def test_hash_changes_when_a_planned_field_changes(self):
        base = plan_video_chain_manifest(_plan_request())
        other = plan_video_chain_manifest(_plan_request(root_prompt="something else"))
        self.assertNotEqual(base.plan_hash, other.plan_hash)

    def test_hash_ignores_warnings_and_run_time_fields(self):
        manifest = plan_video_chain_manifest(_plan_request())
        before = manifest.plan_hash
        manifest.warnings.append("a warning added later")
        manifest.segments[1].continuation_state_in = "state-42"
        manifest.segments[1].effective_overlap_frames = 5
        manifest.segments[1].seed = 999
        self.assertEqual(compute_plan_hash(manifest.to_dict()), before)

    def test_canonical_json_is_sorted_compact_and_float_free(self):
        self.assertEqual(
            canonical_json({"b": 1, "a": "é"}), '{"a":"é","b":1}'
        )
        with self.assertRaises(VideoChainPlanError):
            canonical_json({"fps": 24.0})


class LegacyRepeatTest(unittest.TestCase):
    """§14.1.11 -- legacy mode keeps today's request prompt and seed."""

    def test_legacy_repeat_copies_the_root_prompt_and_seed(self):
        manifest = plan_video_chain_manifest(
            _plan_request(context_mode="legacy_repeat", events=None)
        )
        self.assertEqual(len(manifest.segments), 2)
        for segment in manifest.segments:
            self.assertEqual(segment.prompt, "a bakery opens for the day")
            self.assertEqual(segment.seed, 1234)
        self.assertTrue(any("legacy_repeat" in w for w in manifest.warnings))

    def test_legacy_repeat_still_carries_references_to_every_segment(self):
        manifest = plan_video_chain_manifest(
            _plan_request(
                context_mode="legacy_repeat",
                events=None,
                references=[ChainReference(id="ref_0", kind="image", token="<Picture 1>")],
            )
        )
        for segment in manifest.segments:
            self.assertEqual(segment.reference_ids, ["ref_0"])


H3_PROMPT = (
    "integrated_multimodal_description: [Shot 1] A baker unlocks the shop door in "
    "the grey morning light.\n"
    "[Shot 2] At 00:15.000 The baker says \"Good morning, everyone.\" to the queue "
    "outside.\n"
    "[Shot 3] At 00:25.000 The camera pushes in on the pastry case.\n"
    "[Shot 4] At 00:27.000 The baker slides a tray into the case.\n\n"
    "overall_soundscape: Distant traffic and a ticking wall clock.\n\n"
    "non_diegetic_music: N/A"
)


class MiniMaxH3DeterministicPathTest(unittest.TestCase):
    """§6.2 / §17-4 -- shot-atomic parse, reusing the shared H3 validator."""

    def test_parse_extracts_sections_and_shots(self):
        parsed = parse_h3_structured_prompt(H3_PROMPT, "t2va", 30.0)
        self.assertEqual(parsed.family, "base")
        self.assertEqual([s.number for s in parsed.shots], [1, 2, 3, 4])
        self.assertIsNone(parsed.shots[0].start_seconds)
        self.assertAlmostEqual(parsed.shots[1].start_seconds, 15.0)
        self.assertEqual(
            parsed.sections["overall_soundscape"],
            "Distant traffic and a ticking wall clock.",
        )

    def test_prompt_without_shot_markers_is_refused(self):
        plain = (
            "integrated_multimodal_description: a bakery opens\n\n"
            "overall_soundscape: quiet\n\nnon_diegetic_music: N/A"
        )
        with self.assertRaises(VideoChainPlanError):
            parse_h3_structured_prompt(plain, "t2va", 30.0)

    def test_shots_become_a_contiguous_timeline(self):
        parsed = parse_h3_structured_prompt(H3_PROMPT, "t2va", 30.0)
        events = shots_to_events(parsed.shots, 24.0, PLANNED_FINAL_FRAMES)
        self.assertEqual(
            [(e.start_frame, e.end_frame) for e in events],
            [(0, 360), (360, 600), (600, 648), (648, PLANNED_FINAL_FRAMES)],
        )
        validate_timeline(events, PLANNED_FINAL_FRAMES)

    def test_timestamp_past_the_plan_is_refused(self):
        parsed = parse_h3_structured_prompt(H3_PROMPT, "t2va", 30.0)
        with self.assertRaises(VideoChainPlanError):
            shots_to_events(parsed.shots, 24.0, 300)

    def test_end_to_end_plan_rebases_local_shot_numbers_and_timestamps(self):
        # Fixed lengths by name: this checks the re-basing of a segment that does
        # NOT start on a shot, which is what a fixed-length cut produces.
        manifest = plan_h3_chain_from_prompt(
            H3_PROMPT, "t2va", H3_GRID, 24.0, target_frames=700,
            segment_frames=362, root_seed=1234, chain_id="h3-chain",
            allow_boundary_split=True, segment_length_mode="fixed",
        )
        self.assertEqual(len(manifest.segments), 2)
        first, second = manifest.segments
        # Both segments start their own local clock at [Shot 1] with no timestamp.
        self.assertIn("[Shot 1]", first.prompt)
        self.assertIn("[Shot 1]", second.prompt)
        self.assertNotIn("[Shot 3]", second.prompt)
        self.assertNotIn("[Shot 1] At", first.prompt)
        self.assertNotIn("[Shot 1] At", second.prompt)
        # Segment 1 keeps the global clock (its local 0 IS global 0): shot 2 at
        # 00:15.000 stays at 00:15.000.
        self.assertIn("[Shot 2] At 00:15.000", first.prompt)
        # Segment 2 owns shots 3 and 4. Shot 3 becomes its untimed [Shot 1];
        # shot 4 (global frame 648) is re-timed against the shared anchor 361,
        # i.e. local frame 287 -- NOT its global 00:27.000.
        self.assertIn("[Shot 1] The camera pushes in", second.prompt)
        self.assertIn(f"[Shot 2] At {format_timestamp(287, 24.0)}", second.prompt)
        self.assertNotIn("00:27.000", second.prompt)
        # The audio bed is carried by both segments; the H3 layout is preserved.
        for segment in manifest.segments:
            self.assertIn("overall_soundscape: Distant traffic", segment.prompt)
            self.assertIn("non_diegetic_music: N/A", segment.prompt)
        # Dialogue stays verbatim, in one segment only.
        line = '"Good morning, everyone."'
        self.assertEqual(
            [i for i, p in enumerate(manifest.segment_prompts()) if line in p], [0]
        )

    def test_compiled_segments_satisfy_the_shared_h3_validator(self):
        """The compiler must not invent a layout the H3 validator rejects.

        `validate_prompt` is the one definition of the H3 prompt shape
        (minimax_h3_prompt_assistant.py:167-241); a compiled segment is checked
        against its OWN generated-span duration, since that is the clip the
        request actually produces.
        """
        from core.extensions.minimax_h3_prompt_assistant import validate_prompt

        prompt = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n" + H3_PROMPT
        )
        references = [ChainReference(id="r0", kind="image", token="<Picture 1>")]
        manifest = plan_h3_chain_from_prompt(
            prompt, "i2va", H3_GRID, 24.0, target_frames=700, segment_frames=362,
            references=references, allow_boundary_split=True,
        )
        for segment in manifest.segments:
            self.assertEqual(
                validate_prompt(
                    segment.prompt,
                    "i2va",
                    segment.generated_span_frames / 24.0,
                    [{"token": "<Picture 1>"}],
                ),
                [],
                segment.prompt,
            )

    def test_alignment_instruction_uses_the_segment_duration(self):
        manifest = plan_h3_chain_from_prompt(
            H3_PROMPT, "fl2va", H3_GRID, 24.0, target_frames=700,
            segment_frames=362, allow_boundary_split=True,
            segment_length_mode="fixed",
        )
        # Segment 2 generates 345 frames = 14.375 s, not the whole 700-frame clip.
        self.assertIn("14.38-second mark", manifest.segments[1].prompt)

    def test_end_to_end_plan_is_deterministic(self):
        kwargs = dict(
            prompt=H3_PROMPT, mode="t2va", grid=H3_GRID, fps=24.0,
            target_frames=700, segment_frames=362, root_seed=5,
            seed_policy="derived", chain_id="h3-chain", allow_boundary_split=True,
        )
        a = plan_h3_chain_from_prompt(**kwargs)
        b = plan_h3_chain_from_prompt(**kwargs)
        self.assertEqual(a.plan_hash, b.plan_hash)
        self.assertEqual(a.segment_prompts(), b.segment_prompts())
        self.assertEqual([s.seed for s in a.segments], [s.seed for s in b.segments])


REF2VA_PROMPT = (
    "subject_definitions: <Picture 1> is a woman in a red coat. <Picture 2> is a "
    "station platform at dusk.\n\n"
    "summary: The woman waits on the platform and boards the train.\n\n"
    "retention_analysis: The woman shown in <Picture 1> keeps her coat and face "
    "throughout; the platform shown in <Picture 2> keeps its lamps and signage.\n\n"
    "detailed_description: [Shot 1] The woman shown in <Picture 1> walks along the "
    "platform shown in <Picture 2>.\n"
    "[Shot 2] At 00:15.000 She stops beside a bench and checks the departure board.\n"
    "[Shot 3] At 00:25.000 A train pulls in and she steps aboard.\n\n"
    "overall_soundscape: Rain on the canopy and distant announcements.\n\n"
    "non_diegetic_music: N/A"
)


class Ref2vaTokenSurvivalTest(unittest.TestCase):
    """The regression this rule exists for: no "the woman shown in ." sentences."""

    def _manifest(self, first_reference_segments):
        references = [
            ChainReference(id="ref_a", kind="image", label="woman.png",
                           token="<Picture 1>",
                           segment_indices=first_reference_segments),
            ChainReference(id="ref_b", kind="image", label="platform.png",
                           token="<Picture 2>"),
        ]
        return plan_h3_chain_from_prompt(
            REF2VA_PROMPT, "ref2va", H3_GRID, 24.0, target_frames=700,
            segment_frames=362, references=references, root_seed=99,
            chain_id="ref2va-chain", allow_boundary_split=True,
        )

    def test_a_narrowed_reference_is_widened_instead_of_mutilating_the_text(self):
        manifest = self._manifest([0])
        for segment in manifest.segments:
            self.assertNotIn("shown in .", segment.prompt)
            self.assertNotIn("shown in  ", segment.prompt)
            self.assertIn("The woman shown in <Picture 1>", segment.prompt)
            self.assertIn("subject_definitions: <Picture 1> is a woman", segment.prompt)
        self.assertEqual(manifest.references[0].segment_indices, [0, 1])
        self.assertEqual(manifest.references[0].binding_source, "token_implied")
        self.assertEqual([s.reference_ids for s in manifest.segments],
                         [["ref_a", "ref_b"], ["ref_a", "ref_b"]])
        self.assertTrue(
            any("was not bound to segment 2" in w for w in manifest.warnings),
            manifest.warnings,
        )
        self.assertFalse([w for w in manifest.warnings if "removed reference tokens" in w])

    def test_the_default_binding_needs_no_widening(self):
        manifest = self._manifest(None)
        self.assertEqual([r.binding_source for r in manifest.references],
                         ["default_all", "default_all"])
        self.assertFalse([w for w in manifest.warnings if "was not bound" in w])
        self.assertFalse([w for w in manifest.warnings if "removed reference tokens" in w])

    def test_the_widened_plan_is_deterministic(self):
        a, b = self._manifest([0]), self._manifest([0])
        self.assertEqual(a.plan_hash, b.plan_hash)
        self.assertEqual(a.segment_prompts(), b.segment_prompts())


class DriftTest(unittest.TestCase):
    """§4.1"""

    def test_within_tolerance_continues(self):
        check = evaluate_chain_drift(723, 725)
        self.assertEqual(check.drift_frames, 2)
        self.assertTrue(check.within_tolerance)
        self.assertEqual(check.action, "continue")

    def test_over_tolerance_pauses_instead_of_continuing_silently(self):
        check = evaluate_chain_drift(723, 700, tolerance_frames=12)
        self.assertEqual(check.drift_frames, 23)
        self.assertFalse(check.within_tolerance)
        self.assertEqual(check.action, "pause")
        self.assertTrue(check.message)

    def test_default_tolerance_is_frame_based(self):
        self.assertEqual(DEFAULT_CHAIN_DRIFT_TOLERANCE_FRAMES, 12)
        self.assertTrue(evaluate_chain_drift(100, 112).within_tolerance)
        self.assertFalse(evaluate_chain_drift(100, 113).within_tolerance)

    def test_module_default_matches_the_api_default(self):
        # This module keeps its own literal so it stays import-pure and the API
        # layer always passes the real value in; pin the two equal so changing
        # one alone cannot split them.
        from api.param_defaults import VIDEO_CHAIN_DEFAULTS

        self.assertEqual(
            DEFAULT_CHAIN_DRIFT_TOLERANCE_FRAMES,
            VIDEO_CHAIN_DEFAULTS["chain_drift_tolerance_frames"],
        )

    def test_negative_tolerance_is_refused(self):
        with self.assertRaises(VideoChainPlanError):
            evaluate_chain_drift(1, 1, tolerance_frames=-1)


if __name__ == "__main__":
    unittest.main()
