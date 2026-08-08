"""ref2va references on video outpaint: the decision table and the row order.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_outpaint_reference_gate_test.py -v

WHY THIS FILE EXISTS
--------------------
`minimax_h3_outpaint_refs_design.md` §3 fixes a decision table for
`/generate/outpaint/video` + MiniMax-H3 references, and §1 fixes a row order
for the reference tuple the merged builder consumes. Both are pure arithmetic
(no model, no GPU), so both are pinned here rather than left to be re-derived
from routes.py/pipeline_backends/minimax_h3.py by inspection:

* `resolve_minimax_h3_outpaint_reference_gate` (api.generation_utils) is the
  ONE function the route (fast 400) and the backend (defensive re-check)
  both call -- see the decision table below.
* `build_outpaint_references` (core.pipeline_backends.minimax_h3) builds the
  reference tuple `_generate_vidoutpaint_minimax_h3` hands to the merged
  builder: row 0 is always the source clip, TAIL-truncated (the frames
  nearest the join, not the ones `normalize_reference_video`'s own
  head-truncation would keep), and rows 1.. are the image references in
  request order.
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.error_handlers import ValidationError  # noqa: E402
from api.generation_utils import resolve_minimax_h3_outpaint_reference_gate  # noqa: E402
from core.pipeline_backends.minimax_h3 import build_outpaint_references  # noqa: E402
from core.models.minimax_h3.h3_references import MIN_REFERENCE_VIDEO_FRAMES  # noqa: E402


# ---------------------------------------------------------------------------
# The decision table (minimax_h3_outpaint_refs_design.md §3)
# ---------------------------------------------------------------------------

def _gate(variant, *, has_reference_images, placement, generated_frames=None):
    return resolve_minimax_h3_outpaint_reference_gate(
        variant, has_reference_images=has_reference_images, placement=placement,
        generated_frames=generated_frames)


def test_fl2va_no_refs_is_allowed():
    """H3 fl2va, no reference_images -> ALLOW."""
    assert _gate("fl2va", has_reference_images=False, placement="extend_forward") is None
    assert _gate("fl2va", has_reference_images=False, placement="extend_backward") is None
    assert _gate("fl2va", has_reference_images=False, placement="bridge") is None


def test_fl2va_with_refs_is_refused():
    """H3 fl2va, reference_images present -> 400, on every placement."""
    for placement in ("extend_forward", "extend_backward", "bridge"):
        with pytest.raises(ValidationError, match="fl2va"):
            _gate("fl2va", has_reference_images=True, placement=placement)


def test_ref2va_extend_forward_is_allowed_with_or_without_refs():
    """H3 ref2va + extend_forward -> ALLOW regardless of reference_images.

    The source clip is ALWAYS auto-referenced on this row, so the no-image
    case is a real arm (A-V8's R), not a no-op -- this is the row the design
    calls out explicitly as the deliberate asymmetry against the reverted
    surface.
    """
    assert _gate("ref2va", has_reference_images=False, placement="extend_forward",
                 generated_frames=124) is None
    assert _gate("ref2va", has_reference_images=True, placement="extend_forward",
                 generated_frames=124) is None


def test_ref2va_extend_backward_is_refused_regardless_of_refs():
    """H3 ref2va + extend_backward -> 400, with or without reference_images."""
    for has_refs in (False, True):
        with pytest.raises(ValidationError, match="extend_forward"):
            _gate("ref2va", has_reference_images=has_refs, placement="extend_backward",
                  generated_frames=124)


def test_ref2va_bridge_is_refused_regardless_of_refs():
    """H3 ref2va + bridge -> 400, with or without reference_images."""
    for has_refs in (False, True):
        with pytest.raises(ValidationError, match="extend_forward"):
            _gate("ref2va", has_reference_images=has_refs, placement="bridge",
                  generated_frames=124)


def test_ref2va_extend_forward_below_the_reference_floor_is_refused():
    """Generated span < MIN_REFERENCE_VIDEO_FRAMES -> 400, stated with the arithmetic."""
    with pytest.raises(ValidationError) as excinfo:
        _gate("ref2va", has_reference_images=False, placement="extend_forward",
              generated_frames=MIN_REFERENCE_VIDEO_FRAMES - 1)
    assert str(MIN_REFERENCE_VIDEO_FRAMES) in excinfo.value.detail


def test_ref2va_extend_forward_at_exactly_the_floor_is_allowed():
    """NEGATIVE CONTROL for the above: the floor itself is not refused (< not <=)."""
    assert _gate("ref2va", has_reference_images=False, placement="extend_forward",
                 generated_frames=MIN_REFERENCE_VIDEO_FRAMES) is None


def test_unidentified_variant_refuses_references_but_allows_a_plain_request():
    """A variant this repo cannot name: refuse references (mismatch cannot be
    detected from the weights), but do not block a request with none -- that
    would refuse ordinary fl2va-shaped requests whenever detection fails.
    """
    assert _gate("", has_reference_images=False, placement="extend_forward") is None
    with pytest.raises(ValidationError):
        _gate("", has_reference_images=True, placement="extend_forward")


def test_the_gate_never_names_ref2vid_as_a_destination():
    """No row of this table may reroute to /generate/ref2vid (the reverted
    surface's failure mode) -- checked on every refusing row.
    """
    refusals = []
    for placement in ("extend_backward", "bridge"):
        try:
            _gate("ref2va", has_reference_images=False, placement=placement, generated_frames=124)
        except ValidationError as exc:
            refusals.append(str(exc))
    try:
        _gate("fl2va", has_reference_images=True, placement="extend_forward")
    except ValidationError as exc:
        refusals.append(str(exc))
    assert len(refusals) == 3
    assert not any("ref2vid" in message.lower() for message in refusals)


# ---------------------------------------------------------------------------
# The reference row order (minimax_h3_outpaint_refs_design.md §1)
# ---------------------------------------------------------------------------

def _head(num_frames: int, height: int = 4, width: int = 4) -> np.ndarray:
    """A synthetic preserved clip whose frame VALUE is its own index, so a
    slice's identity (which frames survived, and from which end) is checkable
    without decoding real pixels.
    """
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    for i in range(num_frames):
        frames[i] = i % 256
    return frames


def _solid_image(value: int) -> Image.Image:
    return Image.new("RGB", (8, 8), (value, value, value))


def test_row_zero_is_always_the_source_clip():
    head = _head(30)
    references = build_outpaint_references(head, generated_frames=10, frame_rate=24.0, reference_images=())
    assert len(references) == 1
    assert references[0].kind == "video"
    assert references[0].fps == 24.0


def test_source_reference_is_tail_truncated_not_head_truncated():
    """THE arithmetic the design calls out: `normalize_reference_video` keeps
    the HEAD of whatever it is given, so the orchestration must hand it the
    source's own TAIL -- the frames nearest the join -- itself.

    NEGATIVE CONTROL: a head-truncating mutant (`head[:n]`) passes every
    other test in this file but fails this one, because it keeps frames
    [0..9] instead of [20..29].
    """
    head = _head(30)
    references = build_outpaint_references(head, generated_frames=10, frame_rate=24.0, reference_images=())
    kept = references[0].frames
    assert kept.shape[0] == 10
    assert int(kept[0, 0, 0, 0]) == 20   # first kept frame is head's frame 20
    assert int(kept[-1, 0, 0, 0]) == 29  # last kept frame is head's own last frame


def test_source_reference_is_never_longer_than_the_head():
    """A short preserved clip (head shorter than the generated span) hands
    over the WHOLE head, not a padded or repeated version of it -- the
    22-frame floor is enforced upstream (the gate), not by inventing frames
    here.
    """
    head = _head(5)
    references = build_outpaint_references(head, generated_frames=124, frame_rate=24.0, reference_images=())
    assert references[0].frames.shape[0] == 5


def test_image_references_follow_the_source_in_request_order():
    head = _head(30)
    images = [_solid_image(10), _solid_image(20), _solid_image(30)]
    references = build_outpaint_references(head, generated_frames=10, frame_rate=24.0, reference_images=images)
    assert len(references) == 4
    assert references[0].kind == "video"
    for i, image in enumerate(images):
        assert references[i + 1].kind == "image"
        assert references[i + 1].image is image
        assert references[i + 1].label == f"reference {i + 1}"


def test_no_image_references_leaves_only_the_source_row():
    head = _head(30)
    references = build_outpaint_references(head, generated_frames=10, frame_rate=24.0, reference_images=[])
    assert len(references) == 1
