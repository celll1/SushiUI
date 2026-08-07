"""Temporal-outpaint placement: what each architecture's conditioning can anchor.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_outpaint_placement_test.py -v

WHY THIS FILE EXISTS
--------------------
``POST /generate/outpaint/video`` serves two architectures whose CONDITIONING
differs in kind, not in degree:

* LTX-2.3's ``LTX2VideoCondition.index`` addresses an arbitrary latent frame, so
  the input clip goes anywhere in the output timeline and the whole timeline is
  generated (the input is pasted back afterwards).
* MiniMax-H3's outpaint path hands the model the FIRST and/or the LAST frame of
  the span it generates and has no denoising-strength video-to-video path. Only
  the missing span is generated, anchored on a boundary frame, and the result is
  concatenated with the untouched input.

  This is the ENDPOINT's scope, not an architectural limit, and the difference
  is asserted below rather than left to a comment: MiniMax-H3 does have
  index-addressable conditioning (``/generate/img2vid`` places keyframes at
  arbitrary pixel frames with it). What is unmeasured is the outpaint shape --
  a preserved clip anchored mid-span with exact preservation around it -- so
  the refusal stays and the reason names that instead.

Two things follow that are easy to get wrong and cheap to pin here:

1. a mid-timeline placement must be REFUSED with that reason, not approximated
   by the nearest boundary — an approximation would silently produce a video
   whose preserved clip is not where the client asked for it;
2. the ``17n + 5`` rule binds the GENERATED span, not the output timeline. The
   preserved frames are pasted, never sampled, so they are exempt; and the
   anchor frame is the same instant as the preserved frame it was taken from,
   so an extend is ``preserved + generated - 1`` frames long, not the sum.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.error_handlers import ValidationError  # noqa: E402
from api.generation_utils import plan_video_outpaint_placement  # noqa: E402
from core.models.components.wiring import (  # noqa: E402
    LTX2_TEMPORAL,
    MINIMAX_H3_TEMPORAL,
)


def _params(total, offset=0):
    return {"total_frames": total, "input_offset_frames": offset}


# --------------------------------------------------------------------------
# The declared contract
# --------------------------------------------------------------------------

def test_declared_placements():
    assert LTX2_TEMPORAL.outpaint_placements == ("free",)
    assert MINIMAX_H3_TEMPORAL.outpaint_placements == (
        "extend_forward", "extend_backward", "bridge")


def test_capability_payload_carries_the_placements():
    """A client builds its placement control from this, not from an arch check."""
    from api.arch_capabilities import video_constraints_payload

    payload = video_constraints_payload()
    assert payload["ltx2"]["outpaint_placements"] == ["free"]
    assert payload["minimax_h3"]["outpaint_placements"] == [
        "extend_forward", "extend_backward", "bridge"]


# --------------------------------------------------------------------------
# LTX-2.3 / unknown archs: unchanged behaviour
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["ltx2", None, "sdxl"])
def test_free_placement_archs_are_not_constrained(arch):
    for offset in (0, 1, 57, 120):
        assert plan_video_outpaint_placement(
            _params(121, offset), arch, head_frames=25) == {"placement": "free"}


def test_bridge_clip_is_refused_where_there_is_no_bridge_placement():
    with pytest.raises(ValidationError):
        plan_video_outpaint_placement(_params(121, 0), "ltx2", head_frames=25, tail_frames=25)


# --------------------------------------------------------------------------
# MiniMax-H3: the three placements it can anchor
# --------------------------------------------------------------------------

def test_extend_forward_solves_for_the_generated_span():
    """A 124-frame clip + a 124-frame generated span = 247 output frames.

    Not 248: the generated span's frame 0 IS the anchor, i.e. the same instant
    as the last preserved frame, so it is emitted once.
    """
    plan = plan_video_outpaint_placement(_params(247, 0), "minimax_h3", head_frames=124)
    assert plan["placement"] == "extend_forward"
    assert plan["generated_frames"] == 124
    assert plan["total_frames"] == 247
    assert plan["shared_anchor_frames"] == 1
    assert plan["head_frames"] == 124 and plan["tail_frames"] == 0


def test_unreachable_total_rounds_the_generated_span_up():
    """248 needs a 125-frame span, which is off the grid; 141 is the next one."""
    plan = plan_video_outpaint_placement(_params(248, 0), "minimax_h3", head_frames=124)
    assert plan["generated_frames"] == 141          # 17*8 + 5
    assert plan["total_frames"] == 124 + 141 - 1    # 264
    assert plan["requested_total_frames"] == 248


def test_a_too_short_total_still_generates_the_production_floor():
    """The generated span cannot go below the trained range, so it is raised."""
    plan = plan_video_outpaint_placement(_params(130, 0), "minimax_h3", head_frames=124)
    assert plan["generated_frames"] == MINIMAX_H3_TEMPORAL.min_frames == 124
    assert plan["total_frames"] == 247


def test_extend_backward_is_the_flush_end_placement():
    plan = plan_video_outpaint_placement(
        _params(247, 247 - 124), "minimax_h3", head_frames=124)
    assert plan["placement"] == "extend_backward"
    assert plan["generated_frames"] == 124
    assert plan["total_frames"] == 247


def test_bridge_preserves_both_clips_and_shares_two_anchors():
    plan = plan_video_outpaint_placement(
        _params(370, 0), "minimax_h3", head_frames=124, tail_frames=124)
    assert plan["placement"] == "bridge"
    assert plan["shared_anchor_frames"] == 2
    assert plan["generated_frames"] == 124
    assert plan["total_frames"] == 124 + 124 + 124 - 2  # 370


def test_bridge_requires_the_head_clip_at_the_start():
    with pytest.raises(ValidationError):
        plan_video_outpaint_placement(
            _params(370, 10), "minimax_h3", head_frames=124, tail_frames=124)


# --------------------------------------------------------------------------
# MiniMax-H3: what it refuses, and why the message matters
# --------------------------------------------------------------------------

@pytest.mark.parametrize("offset", [1, 8, 60, 122])
def test_mid_timeline_placement_is_refused_with_the_endpoints_reason(offset):
    with pytest.raises(ValidationError) as excinfo:
        plan_video_outpaint_placement(_params(247, offset), "minimax_h3", head_frames=124)
    message = f"{excinfo.value} {getattr(excinfo.value, 'detail', '')}".lower()
    assert "boundary frames" in message
    assert "unmeasured" in message
    # The two offsets that WOULD work are named, so the client can fix it.
    assert "0" in message and str(247 - 124) in message


def test_the_refusal_does_not_claim_the_architecture_cannot_address_a_frame():
    """NEGATIVE CONTROL for the two statements C2 corrected.

    The refusal used to read "mid-timeline placement requires index-addressable
    conditioning this architecture does not have". The refusal is right and that
    reason is not: `h3_pipeline_ops.build_packed_layout` takes an integer pixel
    frame per anchor, and `/generate/img2vid` places keyframes with it. A future
    edit that restores the old sentence -- in the message or in the TemporalSpec
    comment that mirrors it -- fails here.
    """
    import inspect

    from core.models.components import wiring

    with pytest.raises(ValidationError) as excinfo:
        plan_video_outpaint_placement(_params(247, 60), "minimax_h3", head_frames=124)
    message = f"{excinfo.value} {getattr(excinfo.value, 'detail', '')}".lower()
    assert "does not have" not in message
    assert "no index-addressable" not in message

    spec_source = inspect.getsource(wiring).lower()
    assert "has no index-addressable conditioning" not in spec_source

    # ... and the capability the reason must not deny is real.
    from core.models.minimax_h3 import h3_pipeline_ops as ops
    assert ops._anchor_rotary_time(60, 16, 37) == 16.0 + ops.ROPE_FRAME_RESCALE * 60


def test_an_empty_input_is_refused():
    with pytest.raises(ValidationError):
        plan_video_outpaint_placement(_params(247, 0), "minimax_h3", head_frames=0)


# --------------------------------------------------------------------------
# The invariant the whole feature rests on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("head", [22, 47, 124, 200, 333])
@pytest.mark.parametrize("total", [124, 200, 247, 248, 400, 1000])
def test_the_generated_span_is_always_a_length_the_model_can_generate(head, total):
    plan = plan_video_outpaint_placement(_params(total, 0), "minimax_h3", head_frames=head)
    generated = plan["generated_frames"]
    assert MINIMAX_H3_TEMPORAL.is_valid_length(generated)
    assert MINIMAX_H3_TEMPORAL.min_frames <= generated <= MINIMAX_H3_TEMPORAL.max_frames
    # And the output length is exactly what the concatenation will produce.
    assert plan["total_frames"] == head + generated - 1
