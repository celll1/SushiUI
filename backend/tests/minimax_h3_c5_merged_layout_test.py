"""MiniMax-H3 C5: the merged builder (anchors x references).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_c5_merged_layout_test.py -v

Regression bar (per the C5 task): a request carrying only one track must
reproduce EXACTLY what the pre-merge code produced for that track. Since
``build_ref2va_packed_layout``'s new ``keyframe_anchors`` parameter defaults to
``()``, the refs-only regression is structural (the anchor loop never runs, so
every line below it is untouched code) rather than a captured digest; the
anchors-only path is untouched entirely (``build_packed_layout`` was not
edited). ``minimax_h3_layout_test.py``'s existing 108 cases (string-anchor
digests, ref2va cross-check cases, K0.3 invariants) already re-run unchanged
and are the anchors-only / refs-only regression evidence for those two
builders; this file adds only the merged (both-tracks) behaviour.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402


_TARGET = (7, 24, 40, 37)   # T=22 @ 384x640


def test_refs_only_call_is_bitwise_unchanged_by_the_new_parameter():
    """The default ``keyframe_anchors=()`` reproduces the pre-C5 layout exactly."""
    tags = [ops.TEXT_TAG] * 40
    without_param = ops.build_ref2va_packed_layout(
        tags, [("image", False)], [(1, 64, 64)], [], *_TARGET)
    with_empty_default = ops.build_ref2va_packed_layout(
        tags, [("image", False)], [(1, 64, 64)], [], *_TARGET, keyframe_anchors=())
    for key in ("sequence_length", "num_condition_video_rows", "num_condition_audio_rows"):
        assert without_param[key] == with_empty_default[key]
    for key in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
        assert torch.equal(without_param[key], with_empty_default[key])


def test_anchors_only_builder_is_untouched():
    """``build_packed_layout`` was not edited for C5; its own suite is the proof.

    Sanity check here: it still takes no ``keyframe_anchors``-adjacent argument
    referencing references, and a plain fl2va call is identical to calling it
    with the merged builder's sibling defaults.
    """
    import inspect
    assert "reference_blocks" not in inspect.signature(ops.build_packed_layout).parameters


def test_anchors_land_after_the_reference_blocks():
    """One image reference + two anchors: anchors occupy the NEXT block."""
    tags = [ops.TEXT_TAG] * 40
    layout = ops.build_ref2va_packed_layout(
        tags, [("image", False)], [(1, 64, 64)], [], *_TARGET,
        keyframe_anchors=("first", "last"))
    rows_per_frame = (24 // 2) * (40 // 2)   # TARGET grid, what an anchor costs
    ref_rows = 1 * (64 // 2) * (64 // 2)     # the reference's OWN grid (32x32 -> 1024)
    cond = layout["video_indices"][:layout["num_condition_video_rows"]]
    assert layout["num_condition_video_rows"] == ref_rows + 2 * rows_per_frame
    # The reference block comes first (row 40), the anchors follow it.
    assert int(cond[0]) == 40
    assert int(cond[ref_rows]) == 40 + ref_rows


def test_merged_layout_adds_exactly_rows_per_frame_per_anchor():
    """An anchor costs ``rows_per_frame`` rows wherever it sits, refs included."""
    tags = [ops.TEXT_TAG] * 40
    refs_only = ops.build_ref2va_packed_layout(
        tags, [("image", False)], [(1, 64, 64)], [], *_TARGET)
    rows_per_frame = (24 // 2) * (40 // 2)
    for anchors in (("first",), ("first", "last"), (0, 11, 21)):
        combined = ops.build_ref2va_packed_layout(
            tags, [("image", False)], [(1, 64, 64)], [], *_TARGET, keyframe_anchors=anchors)
        assert combined["sequence_length"] == refs_only["sequence_length"] + len(anchors) * rows_per_frame
        assert combined["num_condition_video_rows"] == (
            refs_only["num_condition_video_rows"] + len(anchors) * rows_per_frame)


def test_anchor_time_is_computed_from_the_post_reference_origin():
    """A ``"first"`` anchor sits at the SAME rotary time the target's own frame 0 does.

    That is the post-reference origin (§1.1 of the conditioning design doc),
    not ``num_text_tokens`` -- an image reference alone advances the shared
    clock by 1.0, so the origin here is 41.0, not 40 (the text length).
    """
    tags = [ops.TEXT_TAG] * 40
    layout = ops.build_ref2va_packed_layout(
        tags, [("image", False)], [(1, 64, 64)], [], *_TARGET, keyframe_anchors=("first",))
    rows_per_frame = (24 // 2) * (40 // 2)
    ref_rows = 1 * (64 // 2) * (64 // 2)
    anchor_rows = layout["video_indices"][ref_rows:ref_rows + rows_per_frame]
    anchor_time = layout["position_ids"][anchor_rows, 0]
    generated_rows = layout["video_indices"][layout["num_condition_video_rows"]:]
    target_frame0_time = layout["position_ids"][generated_rows[:rows_per_frame], 0]
    assert torch.equal(anchor_time, anchor_time[:1].expand_as(anchor_time))
    assert float(anchor_time[0]) == 41.0
    assert torch.allclose(anchor_time, target_frame0_time)


def test_integer_anchor_with_a_reference_present_matches_the_pure_anchor_offset():
    """An integer anchor's OWN advance (``5/3 * f``) is unaffected by a reference.

    The origin shifts (post-reference), but the spacing between two anchors --
    or an anchor and the target's own frame N -- is the same ``ROPE_FRAME_RESCALE``
    step whether or not a reference precedes them.
    """
    tags = [ops.TEXT_TAG] * 40
    layout = ops.build_ref2va_packed_layout(
        tags, [("image", False)], [(1, 64, 64)], [], *_TARGET, keyframe_anchors=(0, 5))
    rows_per_frame = (24 // 2) * (40 // 2)
    ref_rows = 1 * (64 // 2) * (64 // 2)
    t0 = layout["position_ids"][layout["video_indices"][ref_rows], 0]
    t5 = layout["position_ids"][layout["video_indices"][ref_rows + rows_per_frame], 0]
    assert float(t5 - t0) == pytest.approx(5 * ops.ROPE_FRAME_RESCALE, abs=1e-4)


def test_multiple_references_and_multiple_anchors_cover_disjoint_and_ascend():
    """The K0.3-style invariants still hold on the fully merged shape."""
    tags = [ops.TEXT_TAG] * 120
    layout = ops.build_ref2va_packed_layout(
        tags,
        [("image", False), ("image", False), ("audio", True)],
        [(1, 64, 64), (1, 32, 96)], [40],
        *_TARGET, keyframe_anchors=("first", 11, "last"))
    seq_len = layout["sequence_length"]
    blocks = [layout["text_indices"], layout["audio_indices"], layout["video_indices"]]
    for block in blocks:
        assert torch.equal(block, block.sort().values)
    assert torch.equal(torch.cat(blocks).sort().values, torch.arange(seq_len))

    tags_out = layout["token_tags"]
    assert (tags_out[layout["text_indices"]] == ops.TEXT_TAG).all()
    assert (tags_out[layout["audio_indices"]] == ops.AUDIO_TAG).all()
    assert (tags_out[layout["video_indices"]] == ops.VIDEO_TAG).all()
