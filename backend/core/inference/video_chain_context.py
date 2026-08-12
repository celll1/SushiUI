"""Video chain context: canonical timeline, frame arithmetic and segment compiler.

Phase A of the long-form video chain design (scratchpad/video_chain_context_design.md).
This module is PURE: no FastAPI, no torch, no filesystem, no network. Everything it
produces is derived deterministically from its inputs so that the same plan input
yields the same prompts, seeds and `plan_hash` (design §14.1.9).

What lives here (design §12 "新規 backend/core/inference/video_chain_context.py"):

* the chain length planner, ported to be VALUE-IDENTICAL to the frontend one
  (`frontend/src/utils/api.ts:1513-1712`: `chainSegmentCap` / `nextVideoChainTotalFrames`
  / `planVideoChain` / `planVideoChainSegments`, anchor = 1 frame). The frontend is the
  behaviour currently shipping, so it -- not a fresh reading of `TemporalSpec` -- is the
  reference this module must match while both exist (design §12 "parity check 用に残す");
* the shared-anchor frame arithmetic of design §4 in both directions;
* the canonical timeline (events) and its validators (design §6.1, §14.1);
* the deterministic MiniMax-H3 structured-prompt parse (design §6.2, §17-4), which
  REUSES `core.extensions.minimax_h3_prompt_assistant` for shot/timestamp validation
  rather than adding a second validator (CLAUDE.md: never build a third resolver);
* the segment compiler (design §6.3 / §6.4) including reference-token renumbering;
* many-to-many reference binding (design §5.1);
* seed policy (design §8) and the canonical `plan_hash` (design §5.1);
* the planned-vs-actual drift helper (design §4.1).

Deliberately NOT here (Phase B+ per the design): continuation state artifacts,
visual-context adapters, local-LLM timeline extraction, HTTP routes/schemas.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Shot sequencing / timestamp / reference-token validation and the canonical H3
# section names live in the prompt assistant and are REUSED, never re-implemented
# (design §6.1, CLAUDE.md "3 つ目の判定機構を作らない").
from core.extensions.minimax_h3_prompt_assistant import (  # noqa: E402
    SECTION_NAMES,
    _alignment_instruction,
    normalize_prompt,
    validate_prompt,
)


MANIFEST_VERSION = 1

# `extend_forward` with no bridge clip shares exactly one frame between the
# preserved prefix and the generated span (design §4; mirrors
# `VIDEO_CHAIN_ANCHOR_FRAMES` in frontend/src/utils/api.ts:1532).
VIDEO_CHAIN_ANCHOR_FRAMES = 1

# Bound identical to the frontend planner's loop guard (api.ts:1635) so a
# pathological arch table can never loop forever AND both planners stop at the
# same segment count.
CHAIN_PLAN_SEGMENT_GUARD = 500

# Design §4.1. Declared here as a module constant so this module stays pure; the
# API layer is expected to supply the real value from
# `backend/api/param_defaults.py` (the single source of truth for API defaults)
# and pass it in as an argument. Do not read a default from anywhere else.
DEFAULT_CHAIN_DRIFT_TOLERANCE_FRAMES = 12

CONTEXT_MODES = ("timeline", "manual", "legacy_repeat")
SEED_POLICIES = ("fixed", "explicit", "derived")
CONTINUATION_MODES = ("boundary_frame",)

EVENT_KINDS = (
    "shot",
    "visual_action",
    "camera",
    "dialogue",
    "physical_sound",
    "music_transition",
    "state_change",
)

REFERENCE_KINDS = ("image", "video", "audio")
_REFERENCE_TOKEN_WORD = {"image": "Picture", "video": "Video", "audio": "Audio"}
_REFERENCE_TOKEN_RE = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>")

# `[Shot N]` optionally followed by the official `At MM:SS.mmm` timestamp. The
# same shapes `minimax_h3_prompt_assistant.validate_prompt` recognises
# (:189-211) -- this only SPLITS on them, the validation of the sequence itself
# is delegated to that module.
_SHOT_RE = re.compile(r"\[Shot\s+(\d+)\]\s*(?:At\s+(\d{2}):(\d{2})\.(\d{3})\s*)?")

_MAX_SEED = 2 ** 32


class VideoChainPlanError(ValueError):
    """A plan input that cannot produce a valid manifest (hard error)."""


# ---------------------------------------------------------------------------
# 1. Grid + chain length planning (frontend parity)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoGridSpec:
    """The subset of an arch's `TemporalSpec` the chain planner needs.

    Field meanings are `backend/core/models/components/wiring.py:177-225`'s, and
    they arrive at a client as `ArchCapabilities.video_constraints[arch]`
    (`VideoConstraints` in frontend/src/utils/api.ts:1363). `max_frames` None
    means the arch has no enforced single-inference ceiling (MiniMax-H3).
    """

    frame_multiple: int
    frame_offset: int
    min_frames: int
    min_decodable_frames: int
    max_frames: Optional[int] = None

    def __post_init__(self) -> None:
        if self.frame_multiple <= 0:
            raise VideoChainPlanError("frame_multiple must be positive")

    @classmethod
    def from_video_constraints(cls, constraints: Dict[str, Any]) -> "VideoGridSpec":
        return cls(
            frame_multiple=int(constraints["frame_multiple"]),
            frame_offset=int(constraints["frame_offset"]),
            min_frames=int(constraints["min_frames"]),
            min_decodable_frames=int(constraints["min_decodable_frames"]),
            max_frames=(
                None if constraints.get("max_frames") is None else int(constraints["max_frames"])
            ),
        )

    @property
    def floor_frames(self) -> int:
        """`TemporalSpec.floor(smoke=False)` (wiring.py:311-313)."""
        return max(self.min_frames, self.min_decodable_frames)

    def is_on_grid(self, frames: int) -> bool:
        if frames < self.floor_frames:
            return False
        if self.max_frames is not None and frames > self.max_frames:
            return False
        return (frames - self.frame_offset) % self.frame_multiple == 0

    def snap_up(self, frames: int) -> int:
        """`TemporalSpec.snap_length` / `snapUpValidVideoFrameCount` (api.ts:1497).

        Rounds UP onto the grid, floored at `max(min_frames, min_decodable_frames)`
        and clamped at `max_frames` when the arch has one.
        """
        lo = self.floor_frames
        k = -(-(frames - self.frame_offset) // self.frame_multiple)
        k_lo = -(-(lo - self.frame_offset) // self.frame_multiple)
        k = max(k, k_lo)
        if self.max_frames is not None:
            k = min(k, (self.max_frames - self.frame_offset) // self.frame_multiple)
        return k * self.frame_multiple + self.frame_offset


def chain_segment_cap(
    spec: VideoGridSpec, segment_frames: Optional[int] = None
) -> Optional[int]:
    """`chainSegmentCap` (api.ts:1556). None == uncapped (nothing to chain)."""
    if segment_frames is not None and segment_frames > 0:
        return int(segment_frames)
    return spec.max_frames


def next_chain_total_frames(
    spec: VideoGridSpec,
    accumulated_frames: int,
    target_frames: int,
    segment_frames: Optional[int] = None,
) -> Optional[int]:
    """`nextVideoChainTotalFrames` (api.ts:1574).

    The `total_frames` a continuation request must ask for, which is also the
    accumulated clip length it will return (preserved prefix + generated span -
    one shared anchor frame). None means "no forward progress / nothing to chain".
    """
    cap = chain_segment_cap(spec, segment_frames)
    if cap is None:
        return None
    remaining = target_frames - accumulated_frames
    if remaining <= 0:
        return None
    requested_generated = min(remaining, cap)
    generated_span = spec.snap_up(requested_generated)
    if generated_span <= VIDEO_CHAIN_ANCHOR_FRAMES:
        return None
    return accumulated_frames + generated_span - VIDEO_CHAIN_ANCHOR_FRAMES


@dataclass(frozen=True)
class ChainLengthPlan:
    """`VideoChainPlan` (api.ts:1613) plus the continuation totals list."""

    cap_frames: int
    segments: int  # total requests, INCLUDING segment 1
    final_frames: int
    continuation_totals: Tuple[int, ...]  # `planVideoChainSegments` (api.ts:1654)


def plan_chain_lengths(
    spec: VideoGridSpec,
    target_frames: int,
    segment_frames: Optional[int] = None,
) -> Optional[ChainLengthPlan]:
    """Both frontend planners in one pass; they share one loop by construction.

    Returns None for the same three "nothing to plan" reasons the frontend keeps
    as separate early-returns (api.ts:1599-1612): uncapped arch, or a target that
    already fits in one segment.

    NOTE (kept deliberately): segment 1 starts at the RAW cap, not `snap_up(cap)`.
    That is what api.ts:1631 does, so a plan built here matches what the queue
    actually enqueues today. When the cap is off-grid the real segment-1 request
    snaps up and the plan drifts; `build_segment_spans` emits a warning for that
    case and §4.1's drift check catches it at run time.
    """
    cap = chain_segment_cap(spec, segment_frames)
    if cap is None:
        return None
    if target_frames <= cap:
        return None

    accumulated = cap
    totals: List[int] = []
    for _ in range(CHAIN_PLAN_SEGMENT_GUARD):
        if accumulated >= target_frames:
            break
        nxt = next_chain_total_frames(spec, accumulated, target_frames, segment_frames)
        if nxt is None:
            break
        totals.append(nxt)
        accumulated = nxt
    return ChainLengthPlan(
        cap_frames=cap,
        segments=1 + len(totals),
        final_frames=accumulated,
        continuation_totals=tuple(totals),
    )


def plan_video_chain(
    spec: VideoGridSpec, target_frames: int, segment_frames: Optional[int] = None
) -> Optional[ChainLengthPlan]:
    """Name-parity wrapper for `planVideoChain` (api.ts:1619)."""
    return plan_chain_lengths(spec, target_frames, segment_frames)


def plan_video_chain_segments(
    spec: VideoGridSpec, target_frames: int, segment_frames: Optional[int] = None
) -> Optional[List[int]]:
    """Name-parity wrapper for `planVideoChainSegments` (api.ts:1654)."""
    plan = plan_chain_lengths(spec, target_frames, segment_frames)
    return None if plan is None else list(plan.continuation_totals)


def effective_segment_frames(
    spec: VideoGridSpec, requested_frames: int, segment_frames: Optional[int] = None
) -> int:
    """`effectiveSegmentFrames` (api.ts:1688): the length of ANY single request."""
    cap = chain_segment_cap(spec, segment_frames)
    if cap is None:
        return requested_frames
    return min(requested_frames, cap)


# ---------------------------------------------------------------------------
# 2. Shared-anchor frame arithmetic (design §4)
# ---------------------------------------------------------------------------


def anchor_global_frame(accumulated_before: int) -> int:
    """`anchor_global_frame = accumulated_frames_before - 1` (design §4)."""
    return accumulated_before - 1


def global_frame(anchor: int, k: int) -> int:
    """local index k (0 = shared anchor) -> global frame."""
    return anchor + k


def local_frame(anchor: int, g: int) -> int:
    """global frame -> local index. Exact inverse of `global_frame`."""
    return g - anchor


def new_output_frames(generated_span_frames: int) -> int:
    return generated_span_frames - VIDEO_CHAIN_ANCHOR_FRAMES


def accumulated_after(accumulated_before: int, generated_span_frames: int) -> int:
    return accumulated_before + generated_span_frames - VIDEO_CHAIN_ANCHOR_FRAMES


@dataclass(frozen=True)
class SegmentSpan:
    """One model invocation's frame geometry. Ranges are half-open (design §5)."""

    index: int
    accumulated_before: int
    generated_span_frames: int
    anchor_global_frame: Optional[int]  # None for segment 0 (no shared anchor)
    owned_start_frame: int
    owned_end_frame: int
    requested_total_frames: int

    @property
    def owned_frames(self) -> int:
        return self.owned_end_frame - self.owned_start_frame

    @property
    def accumulated_after(self) -> int:
        return self.owned_end_frame

    @property
    def _global_origin(self) -> int:
        """Global frame that local index 0 refers to.

        Segment 0 has no shared anchor, so its local 0 IS global 0; a
        continuation's local 0 is the shared anchor frame (design §4: "segment
        prompt のローカル時刻 0 は shared anchor に対応する").
        """
        return 0 if self.anchor_global_frame is None else self.anchor_global_frame

    def global_frame(self, k: int) -> int:
        if not 0 <= k < self.generated_span_frames:
            raise VideoChainPlanError(
                f"local index {k} is outside segment {self.index}'s generated span "
                f"of {self.generated_span_frames} frames"
            )
        return global_frame(self._global_origin, k)

    def local_frame(self, g: int) -> int:
        k = local_frame(self._global_origin, g)
        if not 0 <= k < self.generated_span_frames:
            raise VideoChainPlanError(
                f"global frame {g} is outside segment {self.index}'s generated span"
            )
        return k

    def owns(self, frame: int) -> bool:
        return self.owned_start_frame <= frame < self.owned_end_frame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "accumulated_before": self.accumulated_before,
            "generated_span_frames": self.generated_span_frames,
            "anchor_global_frame": self.anchor_global_frame,
            "owned_start_frame": self.owned_start_frame,
            "owned_end_frame": self.owned_end_frame,
            "requested_total_frames": self.requested_total_frames,
        }


def build_segment_spans(
    spec: VideoGridSpec,
    target_frames: int,
    segment_frames: Optional[int] = None,
    warnings: Optional[List[str]] = None,
) -> List[SegmentSpan]:
    """The full segment geometry of a chain, segment 0 included.

    Built on top of the frontend-parity length planner so the manifest describes
    exactly the requests the existing queue makes.
    """
    if target_frames <= 0:
        raise VideoChainPlanError("target_frames must be positive")

    sink = warnings if warnings is not None else []
    plan = plan_chain_lengths(spec, target_frames, segment_frames)
    if plan is None:
        span = effective_segment_frames(spec, target_frames, segment_frames)
        first_totals: List[int] = []
        first_span = span
    else:
        first_span = plan.cap_frames
        first_totals = list(plan.continuation_totals)

    if first_span <= VIDEO_CHAIN_ANCHOR_FRAMES:
        raise VideoChainPlanError("the first segment must be longer than the shared anchor")
    if not spec.is_on_grid(first_span):
        sink.append(
            f"Segment 1's length {first_span} is not a valid clip length for this "
            f"architecture; the request will be snapped up to {spec.snap_up(first_span)} "
            "frames, so the planned frame ranges will drift (see the drift check)."
        )

    spans: List[SegmentSpan] = [
        SegmentSpan(
            index=0,
            accumulated_before=0,
            generated_span_frames=first_span,
            anchor_global_frame=None,
            owned_start_frame=0,
            owned_end_frame=first_span,
            requested_total_frames=first_span,
        )
    ]
    accumulated = first_span
    for i, total in enumerate(first_totals, start=1):
        span_frames = total - accumulated + VIDEO_CHAIN_ANCHOR_FRAMES
        anchor = anchor_global_frame(accumulated)
        spans.append(
            SegmentSpan(
                index=i,
                accumulated_before=accumulated,
                generated_span_frames=span_frames,
                anchor_global_frame=anchor,
                owned_start_frame=accumulated,
                owned_end_frame=accumulated_after(accumulated, span_frames),
                requested_total_frames=total,
            )
        )
        accumulated = total
    return spans


# ---------------------------------------------------------------------------
# 3. Canonical timeline (design §6.1) and its validators (design §14.1)
# ---------------------------------------------------------------------------


@dataclass
class TimelineEvent:
    """One event of the canonical timeline. `[start_frame, end_frame)`."""

    id: str
    kind: str
    start_frame: int
    end_frame: int
    description: str
    subject_ids: List[str] = field(default_factory=list)
    one_shot: bool = True
    must_complete: bool = True
    resulting_state: str = ""
    source_span: Optional[Tuple[int, int]] = None
    shot_number: Optional[int] = None
    # Dialogue / lyrics / on-screen text that must survive verbatim into exactly
    # one segment prompt (design §6.4, §14.1.6).
    verbatim: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "description": self.description,
            "subject_ids": list(self.subject_ids),
            "one_shot": bool(self.one_shot),
            "must_complete": bool(self.must_complete),
            "resulting_state": self.resulting_state,
            "source_span": list(self.source_span) if self.source_span else None,
            "shot_number": self.shot_number,
            "verbatim": list(self.verbatim),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimelineEvent":
        span = data.get("source_span")
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            description=str(data.get("description", "")),
            subject_ids=[str(s) for s in data.get("subject_ids", [])],
            one_shot=bool(data.get("one_shot", True)),
            must_complete=bool(data.get("must_complete", True)),
            resulting_state=str(data.get("resulting_state", "")),
            source_span=(int(span[0]), int(span[1])) if span else None,
            shot_number=(None if data.get("shot_number") is None else int(data["shot_number"])),
            verbatim=[str(v) for v in data.get("verbatim", [])],
        )


def validate_timeline(events: Sequence[TimelineEvent], total_frames: int) -> None:
    """Design §14.1.8: reject out-of-range timestamps and shot gap / overlap.

    Shot-kind events describe the timeline's own partition, so they must tile
    `[0, total_frames)` exactly. Other event kinds may legitimately overlap
    (a sound continues through a camera move) and are only range-checked.
    """
    seen: set = set()
    for event in events:
        if not event.id:
            raise VideoChainPlanError("every timeline event needs an id")
        if event.id in seen:
            raise VideoChainPlanError(f"duplicate timeline event id: {event.id}")
        seen.add(event.id)
        if event.kind not in EVENT_KINDS:
            raise VideoChainPlanError(f"unknown event kind: {event.kind}")
        if event.start_frame < 0:
            raise VideoChainPlanError(f"event {event.id} starts before frame 0")
        if event.end_frame <= event.start_frame:
            raise VideoChainPlanError(f"event {event.id} has an empty or reversed frame range")
        if event.end_frame > total_frames:
            raise VideoChainPlanError(
                f"event {event.id} ends at frame {event.end_frame}, past the planned "
                f"{total_frames} frames"
            )

    shots = sorted(
        [e for e in events if e.kind == "shot"], key=lambda e: (e.start_frame, e.end_frame)
    )
    if not shots:
        return
    if shots[0].start_frame != 0:
        raise VideoChainPlanError("the first shot must start at frame 0")
    cursor = shots[0].start_frame
    for shot in shots:
        if shot.start_frame < cursor:
            raise VideoChainPlanError(f"shot {shot.id} overlaps the previous shot")
        if shot.start_frame > cursor:
            raise VideoChainPlanError(
                f"there is a gap in the shot timeline before shot {shot.id} "
                f"(frames {cursor}-{shot.start_frame})"
            )
        cursor = shot.end_frame
    if cursor != total_frames:
        raise VideoChainPlanError(
            f"the shot timeline ends at frame {cursor} but the plan covers {total_frames}"
        )


def assign_event_owners(
    events: Sequence[TimelineEvent],
    spans: Sequence[SegmentSpan],
    warnings: Optional[List[str]] = None,
    allow_boundary_split: bool = False,
) -> Dict[str, int]:
    """Design §14.1.3: every event gets EXACTLY ONE owner segment.

    Ownership is decided on the event's start frame inside the segment's OWNED
    (new-output) range, never on the shared anchor -- design §4 warns that
    mixing the two puts a boundary action in two segments.

    An event that crosses a boundary is a hard error unless the caller opted
    into splitting: design §17-4 requires a manual choice or a stop, not a
    silent split.
    """
    sink = warnings if warnings is not None else []
    owners: Dict[str, int] = {}
    if not spans:
        raise VideoChainPlanError("cannot assign event owners without segments")
    last_frame = spans[-1].owned_end_frame
    for event in events:
        owner: Optional[SegmentSpan] = None
        for span in spans:
            if span.owns(event.start_frame):
                owner = span
                break
        if owner is None:
            raise VideoChainPlanError(
                f"event {event.id} starts at frame {event.start_frame}, outside the "
                f"planned {last_frame} frames"
            )
        if event.end_frame > owner.owned_end_frame:
            message = (
                f"event {event.id} crosses the boundary between segment "
                f"{owner.index + 1} and {owner.index + 2}"
            )
            if not allow_boundary_split:
                raise VideoChainPlanError(
                    message
                    + "; split it in the plan editor or choose which segment owns it"
                )
            sink.append(message + "; it is kept whole in the earlier segment")
        owners[event.id] = owner.index
    return owners


# ---------------------------------------------------------------------------
# 4. Deterministic MiniMax-H3 structured-prompt parse (design §6.2, §17-4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedShot:
    number: int
    start_seconds: Optional[float]  # None only for Shot 1
    text: str


@dataclass
class ParsedH3Prompt:
    mode: str
    family: str  # "base" | "ref"
    alignment_instruction: str
    sections: Dict[str, str]
    shots: List[ParsedShot]
    warnings: List[str]

    @property
    def main_section(self) -> str:
        return SECTION_NAMES[self.family][0] if self.family == "base" else "detailed_description"


def h3_family(mode: str) -> str:
    return "ref" if mode.lower() == "ref2va" else "base"


def parse_h3_structured_prompt(
    prompt: str,
    mode: str,
    duration_seconds: float,
    references: Optional[Iterable[Dict[str, str]]] = None,
) -> ParsedH3Prompt:
    """Split an H3 prompt into sections and shots. No interpretation of content.

    Shot sequencing, the "Shot 1 has no timestamp" rule, timestamp ordering and
    unknown reference labels are validated by
    `minimax_h3_prompt_assistant.validate_prompt` (:189-222) -- this function
    only reports what that shared validator said.
    """
    mode = mode.lower()
    family = h3_family(mode)
    if family not in SECTION_NAMES:
        raise VideoChainPlanError(f"unsupported MiniMax-H3 mode: {mode}")
    text = normalize_prompt(prompt)
    warnings = list(validate_prompt(text, mode, duration_seconds, references))

    starts: List[Tuple[int, int, str]] = []
    for name in SECTION_NAMES[family]:
        match = re.search(rf"(?m)^{re.escape(name)}:\s*", text)
        if match is None:
            raise VideoChainPlanError(f"the prompt has no '{name}:' section")
        starts.append((match.start(), match.end(), name))
    starts.sort()
    sections: Dict[str, str] = {}
    for i, (start, body_at, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        sections[name] = text[body_at:end].strip()
    alignment = text[: starts[0][0]].strip()

    main = ParsedH3Prompt(mode, family, alignment, sections, [], warnings).main_section
    shots = _parse_shots(sections.get(main, ""))
    if not shots:
        raise VideoChainPlanError(
            "the prompt has no [Shot N] markers, so it cannot be split "
            "deterministically (design §6.2)"
        )
    return ParsedH3Prompt(mode, family, alignment, sections, shots, warnings)


def _parse_shots(body: str) -> List[ParsedShot]:
    matches = list(_SHOT_RE.finditer(body))
    shots: List[ParsedShot] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        number = int(match.group(1))
        seconds: Optional[float] = None
        if match.group(2) is not None:
            seconds = (
                int(match.group(2)) * 60 + int(match.group(3)) + int(match.group(4)) / 1000.0
            )
        shots.append(ParsedShot(number, seconds, body[match.end() : end].strip()))
    return shots


def shots_to_events(
    shots: Sequence[ParsedShot], fps: float, total_frames: int
) -> List[TimelineEvent]:
    """Shot-atomic canonical timeline (design §17-4).

    A shot is the ATOMIC unit: this never splits a shot into finer events, never
    guesses `one_shot`, and never decides who speaks a line. Frame boundaries come
    only from the `[Shot N]` markers and their `At MM:SS.mmm` timestamps.
    """
    if fps <= 0:
        raise VideoChainPlanError("fps must be positive")
    events: List[TimelineEvent] = []
    starts: List[int] = []
    for i, shot in enumerate(shots):
        if i == 0:
            if shot.start_seconds not in (None, 0.0):
                raise VideoChainPlanError("Shot 1 must not have a timestamp")
            starts.append(0)
            continue
        if shot.start_seconds is None:
            raise VideoChainPlanError(f"Shot {shot.number} has no timestamp")
        frame = int(round(shot.start_seconds * fps))
        if frame >= total_frames:
            raise VideoChainPlanError(
                f"Shot {shot.number}'s timestamp is outside the planned "
                f"{total_frames} frames"
            )
        if frame <= starts[-1]:
            raise VideoChainPlanError("shot timestamps must be strictly increasing")
        starts.append(frame)
    for i, shot in enumerate(shots):
        end = starts[i + 1] if i + 1 < len(starts) else total_frames
        events.append(
            TimelineEvent(
                id=f"shot_{shot.number}",
                kind="shot",
                start_frame=starts[i],
                end_frame=end,
                description=shot.text,
                shot_number=shot.number,
                verbatim=extract_verbatim(shot.text),
            )
        )
    return events


_QUOTED_RE = re.compile(r"[\"“”「『]([^\"“”「」『』]{1,400})[\"“”」』]")


def extract_verbatim(text: str) -> List[str]:
    """Quoted spans (dialogue / lyrics / on-screen text) found in `text`.

    Detection only; the compiler keeps the whole shot text verbatim anyway, so
    this exists so validators can prove each line lands in exactly one segment
    (design §14.1.6).
    """
    return [m.group(0) for m in _QUOTED_RE.finditer(text)]


# ---------------------------------------------------------------------------
# 5. Reference binding (design §5.1)
# ---------------------------------------------------------------------------


# `token_implied` is a binding this module DERIVED from the prompt text: a
# segment whose text uses a reference's token gets that reference (design §5.1,
# see `derive_token_bindings`).
BINDING_SOURCES = ("default_all", "explicit", "token_implied")


@dataclass
class ChainReference:
    id: str
    kind: str  # image / video / audio
    label: str = ""
    token: Optional[str] = None  # the token used in the ROOT prompt, e.g. "<Picture 2>"
    segment_indices: Optional[List[int]] = None  # None => every segment
    binding_source: str = "default_all"  # one of BINDING_SOURCES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "token": self.token,
            "segment_indices": list(self.segment_indices or []),
            "binding_source": self.binding_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChainReference":
        indices = data.get("segment_indices")
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            label=str(data.get("label", "")),
            token=data.get("token"),
            segment_indices=None if indices is None else [int(i) for i in indices],
            binding_source=str(data.get("binding_source", "default_all")),
        )


def resolve_reference_bindings(
    references: Sequence[ChainReference],
    segment_count: int,
    warnings: Optional[List[str]] = None,
    emit_coverage_warnings: bool = True,
) -> List[ChainReference]:
    """Turn `segment_indices=None` into the explicit default-all binding.

    `references[]` is the source of truth; a reference may cover several
    (non-contiguous) segments and a segment may carry several references.

    Callers that afterwards run `derive_token_bindings` pass
    `emit_coverage_warnings=False` and warn once the binding is final, so an
    empty coverage the token pass fills is not reported as empty.
    """
    sink = warnings if warnings is not None else []
    seen: set = set()
    resolved: List[ChainReference] = []
    for ref in references:
        if not ref.id:
            raise VideoChainPlanError("every reference needs an id")
        if ref.id in seen:
            raise VideoChainPlanError(f"duplicate reference id: {ref.id}")
        seen.add(ref.id)
        if ref.kind not in REFERENCE_KINDS:
            raise VideoChainPlanError(f"unknown reference kind: {ref.kind}")
        if ref.segment_indices is None:
            indices = list(range(segment_count))
            source = "default_all"
        else:
            indices = sorted({int(i) for i in ref.segment_indices})
            source = "explicit"
            for index in indices:
                if not 0 <= index < segment_count:
                    raise VideoChainPlanError(
                        f"reference {ref.id} is bound to segment index {index}, which "
                        f"does not exist (the chain has {segment_count} segments)"
                    )
        resolved.append(
            ChainReference(
                id=ref.id,
                kind=ref.kind,
                label=ref.label,
                token=ref.token,
                segment_indices=indices,
                binding_source=source,
            )
        )

    if emit_coverage_warnings:
        sink.extend(reference_coverage_warnings(resolved, segment_count))
    return resolved


def reference_coverage_warnings(
    references: Sequence[ChainReference], segment_count: int
) -> List[str]:
    """Advisory-only coverage report of a FINAL binding (design §5.1)."""
    messages: List[str] = []
    for ref in references:
        if not (ref.segment_indices or []):
            messages.append(f"Reference {ref.label or ref.id} is not used by any segment.")
    if references:
        for index in range(segment_count):
            if not any(index in (ref.segment_indices or []) for ref in references):
                messages.append(
                    f"Segment {index + 1} has no reference bound to it; identity is "
                    "carried only by the boundary frame there."
                )
    return messages


def scan_reference_tokens(text: str) -> List[str]:
    """The reference tokens `text` uses, in first-appearance order."""
    found: List[str] = []
    for match in _REFERENCE_TOKEN_RE.finditer(text):
        if match.group(0) not in found:
            found.append(match.group(0))
    return found


def derive_token_bindings(
    references: Sequence[ChainReference],
    segment_texts: Sequence[str],
    warnings: Optional[List[str]] = None,
) -> List[ChainReference]:
    """A segment whose text uses a reference's token GETS that reference.

    The token's presence is itself a binding: widening the binding is preferred
    over deleting the token, because deleting it leaves a mutilated sentence
    ("the woman shown in ."). This also applies to a reference the user narrowed
    away from that segment in the plan editor -- text wins, and the widening is
    reported, never silent. The final binding is the union of the explicit
    `segment_indices` and the token-derived ones, and it is part of `plan_hash`.

    `segment_texts[i]` must be the text segment `i` will actually carry, before
    token renumbering; anything else would leave droppable tokens behind.
    """
    sink = warnings if warnings is not None else []
    present = [set(scan_reference_tokens(text)) for text in segment_texts]
    resolved: List[ChainReference] = []
    for ref in references:
        if ref.segment_indices is None:
            # Still means "every segment"; nothing to widen.
            resolved.append(ref)
            continue
        indices = sorted({int(i) for i in ref.segment_indices})
        added = (
            []
            if not ref.token
            else [i for i, tokens in enumerate(present) if ref.token in tokens and i not in indices]
        )
        if added and ref.binding_source == "explicit":
            sink.append(
                f"Reference {ref.label or ref.id} ({ref.token}) was not bound to "
                f"segment{'s' if len(added) > 1 else ''} "
                f"{', '.join(str(i + 1) for i in added)}, but that text uses its "
                "token; the reference was applied there so the sentence stays intact."
            )
        resolved.append(
            ChainReference(
                id=ref.id,
                kind=ref.kind,
                label=ref.label,
                token=ref.token,
                segment_indices=sorted(set(indices) | set(added)),
                binding_source="token_implied" if added else ref.binding_source,
            )
        )
    return resolved


def segment_reference_ids(
    references: Sequence[ChainReference], segment_index: int
) -> List[str]:
    """Read-only inverse of `references[].segment_indices` (design §5.1)."""
    return [
        ref.id for ref in references if segment_index in (ref.segment_indices or [])
    ]


def validate_reference_binding(
    references: Sequence[ChainReference], segment_reference_id_lists: Sequence[Sequence[str]]
) -> None:
    """Design §14.1.7b: the forward and inverse bindings must agree exactly."""
    for index, ids in enumerate(segment_reference_id_lists):
        expected = segment_reference_ids(references, index)
        if list(ids) != expected:
            raise VideoChainPlanError(
                f"segment {index}'s reference_ids {list(ids)} do not match the "
                f"binding in references[] {expected}"
            )
    known = {ref.id for ref in references}
    for index, ids in enumerate(segment_reference_id_lists):
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise VideoChainPlanError(
                f"segment {index} refers to unknown reference ids: {unknown}"
            )


def segment_token_map(
    references: Sequence[ChainReference], segment_index: int
) -> Dict[str, str]:
    """Old root-prompt token -> the token for THIS segment's actual ordering.

    Order is manifest order (`references[]`), renumbered per token word from 1,
    because a segment that carries a subset of the references does not carry the
    root prompt's numbering (design §5.1.2, §6.4).
    """
    counters: Dict[str, int] = {}
    mapping: Dict[str, str] = {}
    for ref in references:
        if segment_index not in (ref.segment_indices or []):
            continue
        word = _REFERENCE_TOKEN_WORD[ref.kind]
        counters[word] = counters.get(word, 0) + 1
        if ref.token:
            mapping[ref.token] = f"<{word} {counters[word]}>"
    return mapping


def rewrite_reference_tokens(text: str, token_map: Dict[str, str]) -> Tuple[str, List[str]]:
    """Renumber bound tokens; DROP tokens of references this segment does not get.

    Returns the rewritten text and the dropped tokens, so the caller can warn.
    """
    dropped: List[str] = []

    def _sub(match: "re.Match[str]") -> str:
        token = match.group(0)
        if token in token_map:
            return token_map[token]
        dropped.append(token)
        return ""

    rewritten = _REFERENCE_TOKEN_RE.sub(_sub, text)
    rewritten = re.sub(r"[ \t]{2,}", " ", rewritten)
    rewritten = re.sub(r"(?m)[ \t]+$", "", rewritten)
    return rewritten, dropped


def strip_reference_tokens(text: str) -> str:
    """Text with every reference token removed, for token-insensitive compares."""
    return re.sub(r"[ \t]{2,}", " ", _REFERENCE_TOKEN_RE.sub("", text)).strip()


# ---------------------------------------------------------------------------
# 6. Seeds (design §8) and plan hash (design §5.1)
# ---------------------------------------------------------------------------


def resolve_root_seed(root_seed: int, rng: Optional[random.Random] = None) -> int:
    """`-1` becomes a CONCRETE seed once, at plan time (design §8).

    Nothing downstream may draw a random number again: retry and resume must
    reproduce the same segment seeds from the manifest alone.
    """
    if root_seed is not None and root_seed >= 0:
        return int(root_seed) % _MAX_SEED
    source = rng or random.Random()
    return source.randrange(_MAX_SEED)


def derive_segment_seed(root_seed: int, plan_hash: str, segment_index: int) -> int:
    """Stable 32-bit seed from (root_seed, plan_hash, segment_index) -- design §8."""
    digest = hashlib.sha256(
        f"{int(root_seed)}:{plan_hash}:{int(segment_index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") % _MAX_SEED


def resolve_segment_seeds(
    seed_policy: str,
    root_seed: int,
    plan_hash: str,
    segment_count: int,
    explicit_seeds: Optional[Sequence[int]] = None,
) -> List[int]:
    if seed_policy not in SEED_POLICIES:
        raise VideoChainPlanError(f"unknown seed policy: {seed_policy}")
    if seed_policy == "fixed":
        return [int(root_seed) % _MAX_SEED] * segment_count
    if seed_policy == "explicit":
        if explicit_seeds is None or len(explicit_seeds) != segment_count:
            raise VideoChainPlanError(
                "seed_policy 'explicit' needs exactly one seed per segment"
            )
        return [int(seed) % _MAX_SEED for seed in explicit_seeds]
    return [derive_segment_seed(root_seed, plan_hash, i) for i in range(segment_count)]


# Fields excluded from the hash: the hash itself, advisory text, and everything
# only written at RUN time (design §5.1). `seed` is excluded because the
# `derived` policy computes it FROM the hash -- including it would be circular;
# the inputs that determine it (`seed_policy`, `root_seed`, `explicit_seeds`)
# are hashed instead.
_HASH_EXCLUDED_TOP_LEVEL = frozenset({"plan_hash", "warnings"})
_HASH_EXCLUDED_SEGMENT_FIELDS = frozenset(
    {
        "seed",
        "continuation_state_in",
        "continuation_state_out",
        "effective_overlap_frames",
        "effective_overlap_samples",
        "drift_frames",
        "actual_accumulated_frames",
    }
)


def format_fps(fps: float) -> str:
    """fps as a decimal STRING: the canonical payload must contain no floats."""
    text = f"{float(fps):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def canonical_plan_payload(manifest: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        key: value for key, value in manifest.items() if key not in _HASH_EXCLUDED_TOP_LEVEL
    }
    if "fps" in payload:
        payload["fps"] = format_fps(payload["fps"])
    segments = payload.get("segments")
    if isinstance(segments, list):
        payload["segments"] = [
            {k: v for k, v in segment.items() if k not in _HASH_EXCLUDED_SEGMENT_FIELDS}
            for segment in segments
        ]
    return payload


def canonical_json(payload: Any) -> str:
    """UTF-8, keys sorted ascending, `,`/`:` separators, non-ASCII NOT escaped.

    Floats are refused rather than serialised: their text form is not stable
    across producers, and the design requires frame/seed integers and a
    stringified fps instead (design §5.1).
    """
    _reject_floats(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _reject_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise VideoChainPlanError(f"canonical plan payload contains a float at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_floats(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _reject_floats(item, f"{path}[{i}]")


def compute_plan_hash(manifest: Dict[str, Any]) -> str:
    """THE definition of `plan_hash`. Nothing else may compute one (design §5.1)."""
    payload = canonical_plan_payload(manifest)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 7. Persistent context and the segment compiler (design §6.3 / §6.4)
# ---------------------------------------------------------------------------


@dataclass
class PersistentContext:
    """Everything that is TRUE IN EVERY SEGMENT (design §5, §6.3.1)."""

    subjects: List[str] = field(default_factory=list)
    environment: List[str] = field(default_factory=list)
    visual_style: List[str] = field(default_factory=list)
    camera_rules: List[str] = field(default_factory=list)
    audio_bed: List[str] = field(default_factory=list)
    hard_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subjects": list(self.subjects),
            "environment": list(self.environment),
            "visual_style": list(self.visual_style),
            "camera_rules": list(self.camera_rules),
            "audio_bed": list(self.audio_bed),
            "hard_constraints": list(self.hard_constraints),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersistentContext":
        return cls(
            subjects=[str(v) for v in data.get("subjects", [])],
            environment=[str(v) for v in data.get("environment", [])],
            visual_style=[str(v) for v in data.get("visual_style", [])],
            camera_rules=[str(v) for v in data.get("camera_rules", [])],
            audio_bed=[str(v) for v in data.get("audio_bed", [])],
            hard_constraints=[str(v) for v in data.get("hard_constraints", [])],
        )

    def visual_lines(self) -> List[str]:
        """Persistent lines that belong in the visual description."""
        return [
            line
            for line in (
                list(self.subjects)
                + list(self.environment)
                + list(self.visual_style)
                + list(self.camera_rules)
                + list(self.hard_constraints)
            )
            if line.strip()
        ]

    def all_lines(self) -> List[str]:
        return self.visual_lines() + [line for line in self.audio_bed if line.strip()]


def format_timestamp(frames: int, fps: float) -> str:
    """`MM:SS.mmm`, the only timestamp form H3 accepts (prompt assistant :205-211)."""
    total_ms = int(round(frames * 1000.0 / fps))
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


@dataclass
class SegmentCompileContext:
    """Everything one segment prompt is built from, in design §6.3 order."""

    span: SegmentSpan
    segment_count: int
    fps: float
    persistent_context: PersistentContext
    incoming_state: List[str]
    owned_events: List[TimelineEvent]
    outgoing_state: List[str]
    soundscape: str = ""
    music: str = ""
    extra_sections: Dict[str, str] = field(default_factory=dict)
    token_map: Dict[str, str] = field(default_factory=dict)

    def local_start_frame(self, event: TimelineEvent) -> int:
        return self.span.local_frame(event.start_frame)


def _event_lines(ctx: SegmentCompileContext, shot_numbers: bool) -> List[str]:
    """Owned events, rebased to this segment's local clock (design §6.4).

    The FIRST owned event becomes `[Shot 1]` with no timestamp, because a
    segment's local time 0 is its shared anchor; later ones are renumbered and
    re-timestamped inside the generated span.
    """
    lines: List[str] = []
    for i, event in enumerate(ctx.owned_events):
        body = event.description.strip()
        if not shot_numbers:
            lines.append(body)
            continue
        if i == 0:
            lines.append(f"[Shot 1] {body}")
            continue
        local = ctx.local_start_frame(event)
        lines.append(f"[Shot {i + 1}] At {format_timestamp(local, ctx.fps)} {body}")
    return lines


def _join_sentences(lines: Sequence[str]) -> str:
    return " ".join(line.strip() for line in lines if line and line.strip())


class SegmentPromptFormatter:
    """Architecture adapter: FORMAT only (design §6.5)."""

    def format(self, ctx: SegmentCompileContext) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class PlainSegmentFormatter(SegmentPromptFormatter):
    """Architecture-neutral prompt: one paragraph per design §6.3 stage."""

    def format(self, ctx: SegmentCompileContext) -> str:
        parts: List[str] = []
        persistent = _join_sentences(ctx.persistent_context.visual_lines())
        if persistent:
            parts.append(persistent)
        if ctx.incoming_state:
            parts.append("Continuing state: " + _join_sentences(ctx.incoming_state))
        events = _join_sentences(_event_lines(ctx, shot_numbers=False))
        if events:
            parts.append(events)
        if ctx.outgoing_state:
            parts.append(
                "State at the end of this segment: " + _join_sentences(ctx.outgoing_state)
            )
        audio = _join_sentences(
            list(ctx.persistent_context.audio_bed) + [ctx.soundscape, ctx.music]
        )
        if audio:
            parts.append(audio)
        return "\n\n".join(parts).strip()


class MiniMaxH3SegmentFormatter(SegmentPromptFormatter):
    """MiniMax-H3 section layout (design §6.4).

    The section names, their order and the mode alignment instruction all come
    from `core.extensions.minimax_h3_prompt_assistant` so there is exactly one
    definition of the H3 prompt shape in the codebase.
    """

    def __init__(self, mode: str):
        self.mode = mode.lower()
        self.family = h3_family(self.mode)

    def _base(self, ctx: SegmentCompileContext) -> str:
        preface = _join_sentences(
            ctx.persistent_context.visual_lines()
            + (["Continuing state: " + _join_sentences(ctx.incoming_state)] if ctx.incoming_state else [])
        )
        body_lines = _event_lines(ctx, shot_numbers=True)
        if ctx.outgoing_state:
            body_lines.append(
                "By the end of this segment: " + _join_sentences(ctx.outgoing_state)
            )
        main = _join_sentences(([preface] if preface else []) + body_lines)
        sections = [
            f"integrated_multimodal_description: {main}",
            "overall_soundscape: "
            + (_join_sentences(list(ctx.persistent_context.audio_bed) + [ctx.soundscape]) or "N/A"),
            f"non_diegetic_music: {ctx.music.strip() or 'N/A'}",
        ]
        return "\n\n".join(sections)

    def _ref(self, ctx: SegmentCompileContext) -> str:
        extra = ctx.extra_sections
        subject_definitions = (
            extra.get("subject_definitions", "").strip()
            or _join_sentences(ctx.persistent_context.subjects)
        )
        summary = extra.get("summary", "").strip()
        scope = (
            f"This segment renders only segment {ctx.span.index + 1} of "
            f"{ctx.segment_count} of the overall timeline."
        )
        detail_lines: List[str] = []
        persistent = _join_sentences(ctx.persistent_context.visual_lines())
        if persistent:
            detail_lines.append(persistent)
        if ctx.incoming_state:
            detail_lines.append("Continuing state: " + _join_sentences(ctx.incoming_state))
        detail_lines.extend(_event_lines(ctx, shot_numbers=True))
        if ctx.outgoing_state:
            detail_lines.append(
                "By the end of this segment: " + _join_sentences(ctx.outgoing_state)
            )
        sections = [
            f"subject_definitions: {subject_definitions}",
            f"summary: {_join_sentences([summary, scope])}",
            f"retention_analysis: {extra.get('retention_analysis', '').strip()}",
            f"detailed_description: {_join_sentences(detail_lines)}",
            "overall_soundscape: "
            + (_join_sentences(list(ctx.persistent_context.audio_bed) + [ctx.soundscape]) or "N/A"),
            f"non_diegetic_music: {ctx.music.strip() or 'N/A'}",
        ]
        return "\n\n".join(sections)

    def format(self, ctx: SegmentCompileContext) -> str:
        body = self._ref(ctx) if self.family == "ref" else self._base(ctx)
        duration = ctx.span.generated_span_frames / ctx.fps
        last_shot = max(len(ctx.owned_events), 1)
        instruction = _alignment_instruction(self.mode, duration, last_shot)
        return f"{instruction}\n\n{body}" if instruction else body


def compile_segment_prompt(
    ctx: SegmentCompileContext,
    formatter: SegmentPromptFormatter,
    warnings: Optional[List[str]] = None,
    formatted: Optional[str] = None,
) -> str:
    """Format, then renumber/drop reference tokens for THIS segment (design §5.1.2).

    `formatted` lets a caller that already formatted the segment (to derive
    token bindings from it) reuse that text instead of formatting twice.
    """
    sink = warnings if warnings is not None else []
    prompt = formatter.format(ctx) if formatted is None else formatted
    prompt, dropped = rewrite_reference_tokens(prompt, ctx.token_map)
    if dropped:
        sink.append(
            f"Segment {ctx.span.index + 1}: removed reference tokens that are not "
            f"bound to it ({', '.join(sorted(set(dropped)))})."
        )
    return prompt.strip()


# ---------------------------------------------------------------------------
# 8. Manifest (design §5)
# ---------------------------------------------------------------------------


@dataclass
class SegmentPlan:
    index: int
    anchor_global_frame: Optional[int]
    owned_start_frame: int
    owned_end_frame: int
    generated_span_frames: int
    requested_total_frames: int
    prompt: str
    negative_prompt: str = ""
    incoming_state: List[str] = field(default_factory=list)
    outgoing_state: List[str] = field(default_factory=list)
    owned_event_ids: List[str] = field(default_factory=list)
    reference_ids: List[str] = field(default_factory=list)
    seed: int = 0
    visual_context: Dict[str, Any] = field(default_factory=lambda: {"mode": "initial"})
    continuation_state_in: Optional[str] = None
    continuation_state_out: Optional[str] = None
    requested_overlap_frames: int = 0
    effective_overlap_frames: int = 0
    effective_overlap_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "anchor_global_frame": self.anchor_global_frame,
            "owned_start_frame": self.owned_start_frame,
            "owned_end_frame": self.owned_end_frame,
            "generated_span_frames": self.generated_span_frames,
            "requested_total_frames": self.requested_total_frames,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "incoming_state": list(self.incoming_state),
            "outgoing_state": list(self.outgoing_state),
            "owned_event_ids": list(self.owned_event_ids),
            "reference_ids": list(self.reference_ids),
            "seed": self.seed,
            "visual_context": dict(self.visual_context),
            "continuation_state_in": self.continuation_state_in,
            "continuation_state_out": self.continuation_state_out,
            "requested_overlap_frames": self.requested_overlap_frames,
            "effective_overlap_frames": self.effective_overlap_frames,
            "effective_overlap_samples": self.effective_overlap_samples,
        }


@dataclass
class ChainManifest:
    chain_id: str
    architecture: str
    variant: str
    root_prompt: str
    fps: float
    target_frames: int
    expected_final_frames: int
    context_mode: str
    segments: List[SegmentPlan]
    manifest_version: int = MANIFEST_VERSION
    plan_hash: str = ""
    root_prompt_hash: str = ""
    continuation_mode: str = "boundary_frame"
    seed_policy: str = "fixed"
    root_seed: int = 0
    persistent_context: PersistentContext = field(default_factory=PersistentContext)
    references: List[ChainReference] = field(default_factory=list)
    events: List[TimelineEvent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "chain_id": self.chain_id,
            "plan_hash": self.plan_hash,
            "architecture": self.architecture,
            "variant": self.variant,
            "root_prompt": self.root_prompt,
            "root_prompt_hash": self.root_prompt_hash,
            "fps": self.fps,
            "target_frames": self.target_frames,
            "expected_final_frames": self.expected_final_frames,
            "context_mode": self.context_mode,
            "continuation_mode": self.continuation_mode,
            "seed_policy": self.seed_policy,
            "root_seed": self.root_seed,
            "persistent_context": self.persistent_context.to_dict(),
            "references": [ref.to_dict() for ref in self.references],
            "events": [event.to_dict() for event in self.events],
            "segments": [segment.to_dict() for segment in self.segments],
            "warnings": list(self.warnings),
        }

    def segment_prompts(self) -> List[str]:
        return [segment.prompt for segment in self.segments]

    def owner_of(self, event_id: str) -> Optional[int]:
        for segment in self.segments:
            if event_id in segment.owned_event_ids:
                return segment.index
        return None


@dataclass
class ChainPlanRequest:
    """Plan inputs (design §9 "request の主要項目")."""

    architecture: str
    root_prompt: str
    grid: VideoGridSpec
    fps: float
    target_frames: int
    variant: str = ""
    negative_prompt: str = ""
    segment_frames: Optional[int] = None
    context_mode: str = "timeline"
    seed_policy: str = "fixed"
    root_seed: int = -1
    explicit_seeds: Optional[List[int]] = None
    references: List[ChainReference] = field(default_factory=list)
    persistent_context: Optional[PersistentContext] = None
    events: Optional[List[TimelineEvent]] = None
    soundscape: str = ""
    music: str = ""
    extra_sections: Dict[str, str] = field(default_factory=dict)
    allow_boundary_split: bool = False
    chain_id: Optional[str] = None
    rng: Optional[random.Random] = None


def plan_video_chain_manifest(request: ChainPlanRequest) -> ChainManifest:
    """Build the immutable Chain Manifest (design §5, §6.3, §8).

    Order matters: geometry -> timeline -> ownership -> references -> prompts ->
    `plan_hash` -> seeds. Seeds come last because the `derived` policy is defined
    in terms of the hash, and the hash deliberately excludes the seeds.
    """
    if request.context_mode not in CONTEXT_MODES:
        raise VideoChainPlanError(f"unknown context mode: {request.context_mode}")
    if request.fps <= 0:
        raise VideoChainPlanError("fps must be positive")

    warnings: List[str] = []
    spans = build_segment_spans(
        request.grid, request.target_frames, request.segment_frames, warnings
    )
    segment_count = len(spans)
    final_frames = spans[-1].owned_end_frame

    references = resolve_reference_bindings(
        request.references, segment_count, warnings, emit_coverage_warnings=False
    )
    root_seed = resolve_root_seed(request.root_seed, request.rng)

    if request.context_mode == "legacy_repeat":
        references = derive_token_bindings(
            references, [request.root_prompt] * segment_count, warnings
        )
        segments = _legacy_repeat_segments(request, spans, references, warnings)
        events: List[TimelineEvent] = []
        persistent = PersistentContext()
        warnings.append(
            "legacy_repeat: every segment repeats the whole root prompt, so events "
            "in it can be re-enacted in later segments."
        )
    else:
        persistent = request.persistent_context or PersistentContext()
        events = list(request.events or [])
        validate_timeline(events, final_frames)
        owners = assign_event_owners(
            events, spans, warnings, allow_boundary_split=request.allow_boundary_split
        )
        segments, references = _compile_segments(
            request, spans, references, persistent, events, owners, warnings
        )

    warnings.extend(reference_coverage_warnings(references, segment_count))

    manifest = ChainManifest(
        chain_id=request.chain_id or str(uuid.uuid4()),
        architecture=request.architecture,
        variant=request.variant,
        root_prompt=request.root_prompt,
        root_prompt_hash=sha256_text(request.root_prompt),
        fps=float(request.fps),
        target_frames=int(request.target_frames),
        expected_final_frames=final_frames,
        context_mode=request.context_mode,
        seed_policy=request.seed_policy,
        root_seed=root_seed,
        persistent_context=persistent,
        references=references,
        events=events,
        segments=segments,
        warnings=warnings,
    )

    manifest.plan_hash = compute_plan_hash(manifest.to_dict())
    seeds = resolve_segment_seeds(
        request.seed_policy,
        root_seed,
        manifest.plan_hash,
        segment_count,
        request.explicit_seeds,
    )
    for segment, seed in zip(manifest.segments, seeds):
        segment.seed = seed

    validate_manifest(manifest)
    return manifest


def _visual_context_for(span: SegmentSpan) -> Dict[str, Any]:
    if span.index == 0:
        return {"mode": "initial"}
    return {"mode": "boundary_frame", "shared_context_frames": VIDEO_CHAIN_ANCHOR_FRAMES}


def _legacy_repeat_segments(
    request: ChainPlanRequest,
    spans: Sequence[SegmentSpan],
    references: Sequence[ChainReference],
    warnings: List[str],
) -> List[SegmentPlan]:
    """Design §17-2 / §14.1.11: byte-identical to what ships today."""
    return [
        SegmentPlan(
            index=span.index,
            anchor_global_frame=span.anchor_global_frame,
            owned_start_frame=span.owned_start_frame,
            owned_end_frame=span.owned_end_frame,
            generated_span_frames=span.generated_span_frames,
            requested_total_frames=span.requested_total_frames,
            prompt=request.root_prompt,
            negative_prompt=request.negative_prompt,
            reference_ids=segment_reference_ids(references, span.index),
            visual_context=_visual_context_for(span),
        )
        for span in spans
    ]


def _compile_segments(
    request: ChainPlanRequest,
    spans: Sequence[SegmentSpan],
    references: Sequence[ChainReference],
    persistent: PersistentContext,
    events: Sequence[TimelineEvent],
    owners: Dict[str, int],
    warnings: List[str],
) -> Tuple[List[SegmentPlan], List[ChainReference]]:
    """Compile every segment prompt; returns them with the FINAL reference binding.

    Two passes on purpose: the binding a segment gets depends on the tokens its
    formatted text uses (`derive_token_bindings`), and the per-segment token
    renumbering depends on that binding.
    """
    formatter: SegmentPromptFormatter
    if request.architecture == "minimax_h3":
        formatter = MiniMaxH3SegmentFormatter(request.variant or "t2va")
    else:
        formatter = PlainSegmentFormatter()

    by_segment: Dict[int, List[TimelineEvent]] = {span.index: [] for span in spans}
    for event in sorted(events, key=lambda e: (e.start_frame, e.id)):
        by_segment[owners[event.id]].append(event)

    contexts: List[SegmentCompileContext] = []
    carried_state: List[str] = []
    for span in spans:
        owned = by_segment[span.index]
        incoming = list(carried_state)
        if span.index > 0 and not incoming:
            incoming.append("The scene continues directly from the previous segment.")
            warnings.append(
                f"Segment {span.index + 1}: no incoming state could be derived from a "
                "shot-atomic prompt; describe the boundary state in the plan editor."
            )
        outgoing = [
            event.resulting_state for event in owned if event.resulting_state.strip()
        ][-1:]

        contexts.append(
            SegmentCompileContext(
                span=span,
                segment_count=len(spans),
                fps=request.fps,
                persistent_context=persistent,
                incoming_state=incoming,
                owned_events=owned,
                outgoing_state=outgoing,
                soundscape=request.soundscape,
                music=request.music,
                extra_sections=dict(request.extra_sections),
            )
        )
        if not owned:
            warnings.append(
                f"Segment {span.index + 1} owns no event; it will only continue the "
                "state it inherits."
            )
        # Past events fold into the incoming state; they are never restated
        # (design §6.3 "過去の event は incoming state に畳み込み").
        for event in owned:
            state = event.resulting_state.strip()
            if state and state not in carried_state:
                carried_state.append(state)

    formatted = [formatter.format(ctx) for ctx in contexts]
    references = derive_token_bindings(references, formatted, warnings)

    segments: List[SegmentPlan] = []
    for ctx, text in zip(contexts, formatted):
        span = ctx.span
        ctx.token_map = segment_token_map(references, span.index)
        segments.append(
            SegmentPlan(
                index=span.index,
                anchor_global_frame=span.anchor_global_frame,
                owned_start_frame=span.owned_start_frame,
                owned_end_frame=span.owned_end_frame,
                generated_span_frames=span.generated_span_frames,
                requested_total_frames=span.requested_total_frames,
                prompt=compile_segment_prompt(ctx, formatter, warnings, formatted=text),
                negative_prompt=request.negative_prompt,
                incoming_state=list(ctx.incoming_state),
                outgoing_state=list(ctx.outgoing_state),
                owned_event_ids=[event.id for event in ctx.owned_events],
                reference_ids=segment_reference_ids(references, span.index),
                visual_context=_visual_context_for(span),
            )
        )
    return segments, list(references)


def plan_h3_chain_from_prompt(
    prompt: str,
    mode: str,
    grid: VideoGridSpec,
    fps: float,
    target_frames: int,
    segment_frames: Optional[int] = None,
    references: Optional[Sequence[ChainReference]] = None,
    negative_prompt: str = "",
    seed_policy: str = "fixed",
    root_seed: int = -1,
    explicit_seeds: Optional[List[int]] = None,
    allow_boundary_split: bool = False,
    chain_id: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> ChainManifest:
    """The deterministic path of design §6.2: a structured H3 prompt -> a manifest.

    No LLM is involved. Shots are atomic (design §17-4): this parses `[Shot N]`
    and its timestamp, assigns whole shots to segments and re-bases the local
    clock. It does not split a shot, does not infer `one_shot`, and does not
    decide who speaks a line.

    `overall_soundscape` and `non_diegetic_music` are treated as PERSISTENT audio
    context and repeated in every segment: a rule-only pass cannot tell which
    part of an ambience belongs to which segment, and dropping it would silence
    later segments. Splitting them is a plan-editor decision.
    """
    spans = build_segment_spans(grid, target_frames, segment_frames)
    final_frames = spans[-1].owned_end_frame

    reference_inventory = [
        {"token": ref.token or "", "label": ref.label} for ref in (references or [])
    ]
    parsed = parse_h3_structured_prompt(
        prompt, mode, target_frames / fps, reference_inventory or None
    )
    events = shots_to_events(parsed.shots, fps, final_frames)

    # `subject_definitions` is NOT copied into `persistent_context.subjects`: the
    # full-reference layout already has a dedicated section that every compiled
    # segment carries, and duplicating it would restate the same text twice in
    # one prompt.
    persistent = PersistentContext()
    extra = {
        name: parsed.sections.get(name, "")
        for name in ("subject_definitions", "summary", "retention_analysis")
        if name in parsed.sections
    }

    request = ChainPlanRequest(
        architecture="minimax_h3",
        variant=mode.lower(),
        root_prompt=prompt,
        negative_prompt=negative_prompt,
        grid=grid,
        fps=fps,
        target_frames=target_frames,
        segment_frames=segment_frames,
        context_mode="timeline",
        seed_policy=seed_policy,
        root_seed=root_seed,
        explicit_seeds=explicit_seeds,
        references=list(references or []),
        persistent_context=persistent,
        events=events,
        soundscape=parsed.sections.get("overall_soundscape", ""),
        music=parsed.sections.get("non_diegetic_music", ""),
        extra_sections=extra,
        allow_boundary_split=allow_boundary_split,
        chain_id=chain_id,
        rng=rng,
    )
    manifest = plan_video_chain_manifest(request)
    manifest.warnings.extend(
        f"Root prompt: {warning}" for warning in parsed.warnings
    )
    return manifest


# ---------------------------------------------------------------------------
# 9. Manifest validators (design §14.1)
# ---------------------------------------------------------------------------


def validate_manifest(manifest: ChainManifest) -> None:
    """Design §14.1 items 3, 4, 5, 6, 7, 7b, 8 as they apply to a built manifest."""
    spans = [
        SegmentSpan(
            index=s.index,
            accumulated_before=s.owned_start_frame if s.index > 0 else 0,
            generated_span_frames=s.generated_span_frames,
            anchor_global_frame=s.anchor_global_frame,
            owned_start_frame=s.owned_start_frame,
            owned_end_frame=s.owned_end_frame,
            requested_total_frames=s.requested_total_frames,
        )
        for s in manifest.segments
    ]
    if not spans:
        raise VideoChainPlanError("a manifest needs at least one segment")

    # 14.1.3 exactly one owner
    owned_ids: List[str] = []
    for segment in manifest.segments:
        owned_ids.extend(segment.owned_event_ids)
    if len(owned_ids) != len(set(owned_ids)):
        raise VideoChainPlanError("an event is owned by more than one segment")
    event_ids = {event.id for event in manifest.events}
    missing = sorted(event_ids - set(owned_ids))
    if missing:
        raise VideoChainPlanError(f"events without an owner segment: {missing}")
    unknown = sorted(set(owned_ids) - event_ids)
    if unknown:
        raise VideoChainPlanError(f"segments own unknown events: {unknown}")

    # 14.1.7b reference binding round-trip
    validate_reference_binding(
        manifest.references, [segment.reference_ids for segment in manifest.segments]
    )

    if manifest.context_mode == "legacy_repeat":
        return

    # 14.1.4 persistent context survives into every segment
    stripped_prompts = [strip_reference_tokens(s.prompt) for s in manifest.segments]
    for line in manifest.persistent_context.all_lines():
        needle = strip_reference_tokens(line)
        if not needle:
            continue
        for index, prompt in enumerate(stripped_prompts):
            if needle not in prompt:
                raise VideoChainPlanError(
                    f"persistent context is missing from segment {index + 1}: {line!r}"
                )

    # 14.1.5 a completed one-shot event never reappears later
    for event in manifest.events:
        if not event.one_shot:
            continue
        owner = manifest.owner_of(event.id)
        needle = strip_reference_tokens(event.description).strip()
        if owner is None or not needle:
            continue
        for index in range(owner + 1, len(manifest.segments)):
            if needle and needle in stripped_prompts[index]:
                raise VideoChainPlanError(
                    f"one-shot event {event.id} is repeated in segment {index + 1}"
                )

    # 14.1.6 dialogue / lyrics / on-screen text appear in exactly one segment
    for event in manifest.events:
        for line in event.verbatim:
            hits = [i for i, prompt in enumerate(manifest.segment_prompts()) if line in prompt]
            if len(hits) != 1:
                raise VideoChainPlanError(
                    f"verbatim line {line!r} appears in {len(hits)} segments; it must "
                    "appear in exactly one"
                )

    # 14.1.1 / 14.1.2 local timestamps stay inside the generated span
    for segment, span in zip(manifest.segments, spans):
        for event_id in segment.owned_event_ids:
            event = next(e for e in manifest.events if e.id == event_id)
            span.local_frame(event.start_frame)  # raises when out of range


# ---------------------------------------------------------------------------
# 10. Planned vs actual drift (design §4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftCheck:
    planned_accumulated_frames: int
    actual_accumulated_frames: int
    drift_frames: int
    tolerance_frames: int
    within_tolerance: bool
    action: str  # "continue" | "pause"
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "planned_accumulated_frames": self.planned_accumulated_frames,
            "actual_accumulated_frames": self.actual_accumulated_frames,
            "drift_frames": self.drift_frames,
            "tolerance_frames": self.tolerance_frames,
            "within_tolerance": self.within_tolerance,
            "action": self.action,
            "message": self.message,
        }


def evaluate_chain_drift(
    planned_accumulated_frames: int,
    actual_accumulated_frames: int,
    tolerance_frames: int = DEFAULT_CHAIN_DRIFT_TOLERANCE_FRAMES,
) -> DriftCheck:
    """Design §4.1. Over tolerance the chain PAUSES; it never continues silently.

    `tolerance_frames` is an argument on purpose: the API layer supplies it from
    `backend/api/param_defaults.py`, this module only carries a constant so it
    stays usable (and testable) standalone.
    """
    if tolerance_frames < 0:
        raise VideoChainPlanError("drift tolerance cannot be negative")
    drift = abs(int(actual_accumulated_frames) - int(planned_accumulated_frames))
    within = drift <= tolerance_frames
    return DriftCheck(
        planned_accumulated_frames=int(planned_accumulated_frames),
        actual_accumulated_frames=int(actual_accumulated_frames),
        drift_frames=drift,
        tolerance_frames=int(tolerance_frames),
        within_tolerance=within,
        action="continue" if within else "pause",
        message=(
            ""
            if within
            else (
                f"The chain is {drift} frames away from the plan (tolerance "
                f"{tolerance_frames}); continue or stop."
            )
        ),
    )
