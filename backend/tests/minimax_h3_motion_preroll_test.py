"""`continuation_mode: motion_preroll` -- the chain continuation with anchors.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_motion_preroll_test.py -v

WHAT THIS MODE IS, AND WHAT IS NEW
----------------------------------
Design §7.3::

    preserved prefix | context pre-roll | new suffix
                      \\-- regenerated, discarded --/\\-- appended --/

The overlap arithmetic is `pinned_tail`'s, unchanged: the shared frames come off
the generated span (`continuation_generated_span` /
`plan_video_outpaint_placement`), and the preserved clip is concatenated over
them. What differs is the conditioning -- nothing is pinned; several of the
predecessor's frames are placed as keyframe ANCHORS at their own indices inside
the generated span, through the same `plan_keyframe_placements` /
`build_packed_layout` path `/generate/img2vid` uses.

What this file pins:

* the preserved prefix is EXACT (the gate): regenerating the pre-roll does not
  touch a frame of it;
* the regenerated pre-roll is discarded and only the new suffix is appended;
* the manifest's `owned_end_frame` is the length the generation returns -- the
  same defect commit 17cfdb7a fixed for `pinned_tail`, with the extra frames
  the discard removes;
* the anchors are uniform, and are a deterministic function of (pre-roll length,
  anchor count) alone, so a manifest fixes them;
* `pinned_tail` and `motion_preroll` cannot be combined (they claim the same
  conditioning prefix) and neither is silently disabled;
* an arch/variant that does not advertise the mode is a 400;
* the row invariant in this configuration: anchors reserve condition rows, those
  rows are never written, and the free rows keep the anchor-free noise.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.arch_capabilities import (  # noqa: E402
    MINIMAX_H3_MOTION_PREROLL_MAX_ANCHORS,
    MINIMAX_H3_MOTION_PREROLL_MAX_FRAMES,
    MINIMAX_H3_MOTION_PREROLL_MIN_ANCHORS,
    MINIMAX_H3_MOTION_PREROLL_MIN_FRAMES,
    chain_context_for,
    video_constraints_payload,
)
from api.generation_utils import (  # noqa: E402
    plan_video_continuation_context,
    plan_video_outpaint_placement,
)
from core.inference.video_chain_context import (  # noqa: E402
    VideoChainPlanError,
    VideoGridSpec,
    build_segment_spans,
    motion_preroll_anchor_frames,
)
from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402

PREROLL = 9
ANCHORS = 3
SEGMENT = 124            # MiniMax-H3's shortest clip: one segment per request
TARGET = 500


def _grid() -> VideoGridSpec:
    return VideoGridSpec.from_video_constraints(video_constraints_payload()["minimax_h3"])


# --------------------------------------------------------------------------
# 1. The anchors: uniform, deterministic, and inside the pre-roll
# --------------------------------------------------------------------------

def test_the_anchors_are_uniform_and_span_the_whole_preroll():
    """First anchor at the oldest pre-roll frame, last at the boundary frame.

    The last anchor is `preroll - 1`, which is the same instant a
    `boundary_frame` continuation anchors -- the mode ADDS earlier frames, it
    does not move the seam.
    """
    assert motion_preroll_anchor_frames(9, 3) == (0, 4, 8)
    assert motion_preroll_anchor_frames(5, 2) == (0, 4)
    assert motion_preroll_anchor_frames(17, 4) == (0, 5, 11, 16)
    for preroll in range(MINIMAX_H3_MOTION_PREROLL_MIN_FRAMES,
                         MINIMAX_H3_MOTION_PREROLL_MAX_FRAMES + 1):
        for count in range(MINIMAX_H3_MOTION_PREROLL_MIN_ANCHORS,
                           min(MINIMAX_H3_MOTION_PREROLL_MAX_ANCHORS, preroll) + 1):
            frames = motion_preroll_anchor_frames(preroll, count)
            assert len(frames) == count
            assert len(set(frames)) == count, "two anchors landed on one frame"
            assert frames[0] == 0 and frames[-1] == preroll - 1
            assert list(frames) == sorted(frames)
            gaps = [b - a for a, b in zip(frames, frames[1:])]
            # Uniform: every gap is within one frame of every other.
            assert max(gaps) - min(gaps) <= 1


def test_the_anchor_positions_are_a_function_of_the_manifest_values_alone():
    """Determinism: same (pre-roll, count) -> same frames, every time.

    This is what lets the manifest fix the anchors and the generation re-derive
    them from the two numbers the request carries.
    """
    for _ in range(3):
        assert motion_preroll_anchor_frames(PREROLL, ANCHORS) == (0, 4, 8)
    resolved = plan_video_continuation_context(
        "motion_preroll", PREROLL, "minimax_h3", "fl2va", ANCHORS)
    assert resolved == {
        "mode": "motion_preroll", "overlap_frames": PREROLL,
        "anchor_count": ANCHORS,
        "anchor_local_frames": motion_preroll_anchor_frames(PREROLL, ANCHORS),
    }


def test_a_preroll_too_small_for_its_anchors_is_refused():
    with pytest.raises(VideoChainPlanError):
        motion_preroll_anchor_frames(2, 3)
    with pytest.raises(VideoChainPlanError):
        motion_preroll_anchor_frames(9, 1)


# --------------------------------------------------------------------------
# 2. The bounds, and every refusal that is a refusal
# --------------------------------------------------------------------------

def test_the_capability_advertises_the_mode_and_its_own_bounds():
    entry = chain_context_for("minimax_h3", "fl2va")
    assert "motion_preroll" in entry["chain_continuation_modes"]
    assert entry["chain_supports_sparse_motion_anchors"] is True
    assert entry["chain_motion_preroll_min_frames"] == MINIMAX_H3_MOTION_PREROLL_MIN_FRAMES
    assert entry["chain_motion_preroll_max_frames"] == MINIMAX_H3_MOTION_PREROLL_MAX_FRAMES
    assert entry["chain_motion_preroll_min_anchors"] == MINIMAX_H3_MOTION_PREROLL_MIN_ANCHORS
    assert entry["chain_motion_preroll_max_anchors"] == MINIMAX_H3_MOTION_PREROLL_MAX_ANCHORS
    # A pre-roll is NOT a pin: its floor is the structural one (two anchors need
    # two frames), not the pin's measured 5, and it needs no VAE alignment.
    assert entry["chain_motion_preroll_min_frames"] < entry["chain_context_min_frames"]


@pytest.mark.parametrize("preroll", [0, 1, 18, 33])
def test_a_preroll_outside_the_served_range_is_refused_not_clamped(preroll):
    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError) as error:
        plan_video_continuation_context("motion_preroll", preroll, "minimax_h3", "fl2va", 2)
    detail = str(error.value.detail)
    assert "2..17" in detail
    assert "refused rather than clamped" in detail


@pytest.mark.parametrize("preroll", [2, 3, 4, 6, 7, 10, 16, 17])
def test_every_in_range_preroll_is_accepted_including_unaligned_ones(preroll):
    """NEGATIVE CONTROL, and the difference from `pinned_tail`.

    3, 4, 6, 7, 10, 16 are NOT video-VAE group boundaries and a pin of that
    length is a 400 -- but an anchor addresses a pixel frame directly, so a
    pre-roll of that length is legal here. The two modes bound the same request
    field differently and this asserts they really do.
    """
    from api.error_handlers import ValidationError

    resolved = plan_video_continuation_context(
        "motion_preroll", preroll, "minimax_h3", "fl2va", 2)
    assert resolved["overlap_frames"] == preroll
    if preroll not in (5, 9, 13, 17):
        with pytest.raises(ValidationError):
            plan_video_continuation_context("pinned_tail", preroll, "minimax_h3", "fl2va")


@pytest.mark.parametrize("count", [0, 1, 5, 9])
def test_an_anchor_count_outside_the_served_range_is_refused(count):
    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError) as error:
        plan_video_continuation_context("motion_preroll", PREROLL, "minimax_h3", "fl2va", count)
    assert "2..4" in str(error.value.detail)


def test_more_anchors_than_preroll_frames_is_refused():
    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError) as error:
        plan_video_continuation_context("motion_preroll", 3, "minimax_h3", "fl2va", 4)
    assert "anchors" in str(error.value.detail)


@pytest.mark.parametrize("mode", ["boundary_frame", "pinned_tail"])
def test_an_anchor_count_with_a_pinning_or_anchorless_mode_is_refused(mode):
    """THE EXCLUSIVITY. An anchor reserves conditioning rows ahead of the clip
    and a pin re-uses that same prefix for rows OF the clip
    (`h3_pipeline_ops._validated_pinned_frames`), so the two cannot both be
    asked for -- and the request is refused rather than run with the anchors
    (or the pin) quietly dropped."""
    from api.error_handlers import ValidationError

    overlap = 0 if mode == "boundary_frame" else 9
    with pytest.raises(ValidationError) as error:
        plan_video_continuation_context(mode, overlap, "minimax_h3", "fl2va", ANCHORS)
    detail = str(error.value.detail)
    assert "mutually exclusive" in detail
    assert "refused rather than run with one of them dropped" in detail
    # NEGATIVE CONTROL: the same call without a count is unchanged.
    assert plan_video_continuation_context(mode, overlap, "minimax_h3", "fl2va")["mode"] == mode


@pytest.mark.parametrize("arch,variant", [
    ("minimax_h3", "ref2va"),   # its reference block owns the same prefix
    ("ltx2", None),             # conditions on the whole preserved clip already
    ("sdxl", None),             # not a video architecture at all
])
def test_motion_preroll_is_refused_where_it_is_not_advertised(arch, variant):
    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError) as error:
        plan_video_continuation_context("motion_preroll", PREROLL, arch, variant, ANCHORS)
    assert "chain_context" in str(error.value.detail)
    entry = chain_context_for(arch, variant)
    if entry is not None:
        assert entry["chain_supports_sparse_motion_anchors"] is False


# --------------------------------------------------------------------------
# 3. The discard arithmetic: what the plan promises is what comes back
# --------------------------------------------------------------------------

def test_the_generated_span_absorbs_the_preroll_and_the_output_length_holds():
    params = {"total_frames": 500, "input_offset_frames": 0}
    anchor = plan_video_outpaint_placement(params, "minimax_h3", head_frames=362)
    preroll = plan_video_outpaint_placement(params, "minimax_h3", head_frames=362,
                                            overlap_frames=PREROLL)
    assert anchor["shared_anchor_frames"] == 1
    assert preroll["shared_anchor_frames"] == PREROLL
    for plan in (anchor, preroll):
        # `total = head + generated - shared`: the shared frames are generated
        # and then dropped, which is the whole cost of the mode.
        assert plan["total_frames"] == (
            plan["head_frames"] + plan["generated_frames"] - plan["shared_anchor_frames"])


@pytest.mark.parametrize("preroll", [2, 5, 9, 17])
def test_the_manifests_owned_end_is_what_the_generation_returns(preroll):
    """The 17cfdb7a defect, re-checked for this mode.

    One request at a time, the placement planner (which is what the GENERATION
    solves) is fed the manifest's own `requested_total_frames` and must answer
    the manifest's own `owned_end_frame`.
    """
    spans = build_segment_spans(_grid(), TARGET, SEGMENT, None, preroll)
    accumulated = spans[0].owned_end_frame
    for span in spans[1:]:
        placement = plan_video_outpaint_placement(
            {"total_frames": span.requested_total_frames, "input_offset_frames": 0},
            "minimax_h3", head_frames=accumulated, overlap_frames=preroll,
        )
        assert placement["generated_frames"] == span.generated_span_frames
        assert placement["total_frames"] == span.owned_end_frame
        # The discard, stated as arithmetic: the segment ADDS fewer frames than
        # it generates, by exactly the pre-roll.
        assert span.owned_frames == span.generated_span_frames - preroll
        accumulated = span.owned_end_frame


# --------------------------------------------------------------------------
# 4. The row invariant, in THIS configuration (anchors, nothing pinned)
# --------------------------------------------------------------------------

GEOMETRY = dict(num_text_tokens=3, num_latent_frames=5, latent_height=4,
                latent_width=4, num_audio_latents=10)
CHANNELS = 24
LAYOUT_ANCHORS = ("first", 3, 7)


class _StubScheduler:
    """Steps by `sample - 0.25 * velocity` (the sibling files' stub), so a row
    that WAS written cannot look unwritten."""

    def __init__(self, timesteps):
        self.timesteps = torch.tensor(timesteps, dtype=torch.float32)

    def set_shift(self, shift):
        pass

    def set_timesteps(self, steps, device=None):
        pass

    def set_begin_index(self, index):
        pass

    def step(self, velocity, timestep, sample, return_dict=False):
        return (sample - 0.25 * velocity,)


def _anchor_layout():
    return ops.build_packed_layout(**GEOMETRY, keyframe_anchors=LAYOUT_ANCHORS)


def test_several_anchors_reserve_their_own_rows_ahead_of_the_clip():
    layout = _anchor_layout()
    assert layout["num_condition_video_rows"] == len(LAYOUT_ANCHORS) * layout["rows_per_frame"]
    # The clip's own rows are untouched by the anchors: an anchor's rows are
    # EXTRA, which is why this composes with nothing being pinned and why the
    # cost is linear in the anchor count.
    assert layout["video_indices"].numel() == (
        layout["num_condition_video_rows"] + GEOMETRY["num_latent_frames"] * layout["rows_per_frame"]
    )
    unique, index = ops.build_row_timesteps(layout, 0.1, 0.1)
    row_time = unique[index]
    n_cond = layout["num_condition_video_rows"]
    assert bool((row_time[layout["video_indices"][:n_cond]] == ops.VISUAL_COND_TIMESTEP).all())
    assert bool((row_time[layout["video_indices"][n_cond:]] == 0.1).all())


def test_the_anchor_rows_survive_every_step_and_only_the_clips_rows_are_written():
    layout = _anchor_layout()
    n_cond = layout["num_condition_video_rows"]
    video_rows = torch.randn(layout["video_indices"].numel(), CHANNELS * 4,
                             generator=torch.Generator().manual_seed(21))
    audio_rows = torch.randn(layout["audio_indices"].numel(), 32,
                             generator=torch.Generator().manual_seed(22))
    before = video_rows.clone()

    steps = [0.9, 0.6, 0.3]
    out_video, _out_audio = ops.denoise(
        lambda **kw: (torch.full((1, video_rows.shape[0], video_rows.shape[1]), 2.0),
                      torch.full((1, audio_rows.shape[0], audio_rows.shape[1]), 3.0)),
        _StubScheduler(steps), _StubScheduler(steps),
        prompt_embeds=torch.zeros(1, 3, 8), layout=layout,
        video_rows=video_rows, audio_rows=audio_rows,
        num_inference_steps=len(steps), device="cpu",
    )
    assert torch.equal(out_video[:n_cond], before[:n_cond])
    # NEGATIVE CONTROL: the clip's rows moved by exactly the three steps.
    assert torch.allclose(out_video[n_cond:], before[n_cond:] - len(steps) * 0.5)


# --------------------------------------------------------------------------
# 5. The orchestration: the gate (exact prefix) and the discard
# --------------------------------------------------------------------------

GENERATED_VALUE = 7
HEAD_VALUE = 200


def _runner(width=64, height=32):
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    captured = {}

    class Runner(MiniMaxH3Mixin):
        minimax_h3_components = {"variant": "fl2va", "audio_sample_rate": 32000,
                                 "audio_latent_rate": 40.0, "fps": 24.0}
        current_model_info = {"type": "minimax_h3", "variant": "fl2va"}

        def _generate_minimax_h3(self, params, **kwargs):
            captured.update(kwargs)
            captured["params"] = params
            frames = np.full((int(params["num_frames"]), height, width, 3),
                             GENERATED_VALUE, dtype=np.uint8)
            return frames, None, None, 4242

    return Runner(), captured


def _head_clip(width=64, height=32, frames=362):
    clip = np.full((frames, height, width, 3), HEAD_VALUE, dtype=np.uint8)
    for index in range(frames):
        clip[index, 0, 0, 0] = index % 251     # per-frame marks
    return clip


def _outpaint_params(**overrides):
    params = {"width": 64, "height": 32, "frame_rate": 24.0, "total_frames": 700,
              "input_offset_frames": 0, "audio_enable": False,
              "outpaint_video_audio_mode": "regenerate",
              "continuation_mode": "motion_preroll",
              "continuation_overlap_frames": PREROLL,
              "continuation_anchor_count": ANCHORS}
    params.update(overrides)
    return params


def test_the_continuation_anchors_the_preroll_and_pins_nothing():
    runner, captured = _runner()
    head = _head_clip()
    params = _outpaint_params()
    runner._generate_vidoutpaint_minimax_h3(params, head, 24.0, None)

    # Nothing is pinned: a pin would claim the prefix the anchors hold.
    assert captured["pinned_video_frames"] == ()
    assert captured["pinned_video_source"] is None
    assert captured["input_audio"] is None
    anchors = [anchor for anchor, _image in captured["keyframes"]]
    # Local 0 is the pre-roll's oldest frame ("first"), the rest are interior
    # indices of the generated span.
    assert anchors == ["first", 4, 8]
    # Each anchor is the preserved clip's frame at the SAME instant.
    for anchor, image in captured["keyframes"]:
        local = 0 if anchor == "first" else int(anchor)
        source = head[head.shape[0] - PREROLL + local]
        assert np.array_equal(np.asarray(image), source)
    assert params["continuation_anchor_count"] == ANCHORS
    assert params["continuation_anchor_frames"] == [0, 4, 8]
    assert params["continuation_effective_overlap_frames"] == PREROLL
    # No audio was pinned, so no audio overlap is claimed.
    assert params["continuation_effective_overlap_samples"] == 0


def test_the_preserved_prefix_is_exact_and_only_the_new_suffix_is_appended():
    """THE GATE (design §15 Gate 2, first bullet).

    The pre-roll is regenerated -- the model returns its own version of those
    frames -- and not one pixel of the preserved clip may move because of it.
    The generated pre-roll is dropped and the rest appended.
    """
    runner, captured = _runner()
    head = _head_clip()
    params = _outpaint_params()
    frames, _audio, _rate, seed = runner._generate_vidoutpaint_minimax_h3(
        params, head, 24.0, None)

    generated = params["outpaint_generated_frames"]
    assert seed == 4242
    # 1. bit-exact prefix, every frame of it.
    assert np.array_equal(frames[:head.shape[0]], head)
    # 2. the discard: the generated clip is `generated` frames, of which the
    #    first PREROLL never reach the output.
    assert captured["params"]["num_frames"] == generated
    assert frames.shape[0] == head.shape[0] + generated - PREROLL
    assert (frames[head.shape[0]:] == GENERATED_VALUE).all()
    # 3. and the model's version of the pre-roll is nowhere in the output: the
    #    frames at those instants are the preserved clip's own.
    assert np.array_equal(frames[head.shape[0] - PREROLL:head.shape[0]], head[-PREROLL:])
    assert not (frames[head.shape[0] - PREROLL:head.shape[0]] == GENERATED_VALUE).all()


def test_the_recorded_output_length_matches_the_frames_returned():
    """`owned_end_frame`'s generation-side counterpart: `params["total_frames"]`
    is written back in place, and it is what the gallery row records."""
    runner, _captured = _runner()
    head = _head_clip()
    params = _outpaint_params()
    frames, _audio, _rate, _seed = runner._generate_vidoutpaint_minimax_h3(
        params, head, 24.0, None)
    assert params["total_frames"] == frames.shape[0]
    assert params["num_frames"] == frames.shape[0]
    assert params["outpaint_effective_preserved_frames"] == head.shape[0]


def test_the_boundary_frame_default_is_untouched_by_this_mode():
    """NEGATIVE CONTROL: an ordinary extend still sends one anchor and shares one
    frame."""
    runner, captured = _runner()
    head = _head_clip()
    params = _outpaint_params(continuation_mode="boundary_frame",
                              continuation_overlap_frames=0,
                              continuation_anchor_count=0)
    frames, _audio, _rate, _seed = runner._generate_vidoutpaint_minimax_h3(
        params, head, 24.0, None)
    assert [anchor for anchor, _image in captured["keyframes"]] == ["first"]
    assert captured["pinned_video_frames"] == ()
    assert frames.shape[0] == head.shape[0] + params["outpaint_generated_frames"] - 1
    assert params["continuation_effective_overlap_frames"] == 0
    assert params["continuation_anchor_count"] == 0


# --------------------------------------------------------------------------
# 6. The route: refused before the upload is even decoded
# --------------------------------------------------------------------------

class _StubPipelineManager:
    def __init__(self, arch="minimax_h3", variant="fl2va"):
        self.is_minimax_h3_model = arch == "minimax_h3"
        self.is_ltx2_model = arch == "ltx2"
        self.current_model_info = {"type": arch, "variant": variant}
        self.minimax_h3_components = None


def _post(monkeypatch, manager, **fields):
    """POST /generate/outpaint/video on a bare app, with a junk clip.

    The clip is never decoded: every assertion below is about a refusal that
    happens BEFORE `video.read()`.
    """
    import asyncio

    import httpx
    from fastapi import FastAPI

    import api.routes as routes
    from api.error_handlers import register_error_handlers

    monkeypatch.setattr(routes, "pipeline_manager", manager)
    app = FastAPI()
    register_error_handlers(app)
    app.post("/generate/outpaint/video")(routes.generate_outpaint_video)
    app.dependency_overrides[routes.get_gallery_db] = lambda: None

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/generate/outpaint/video",
                data={"prompt": "a cat", **{k: str(v) for k, v in fields.items()}},
                files={"video": ("clip.mp4", b"not a real clip", "video/mp4")},
            )
            return response.status_code, response.json()

    return asyncio.run(run())


@pytest.mark.parametrize("preroll", [0, 1, 18])
def test_the_route_refuses_an_out_of_range_preroll_before_decoding(monkeypatch, preroll):
    status, payload = _post(monkeypatch, _StubPipelineManager(),
                            continuation_mode="motion_preroll",
                            continuation_overlap_frames=preroll,
                            continuation_anchor_count=ANCHORS)
    assert status == 400, payload
    assert "2..17" in payload["detail"]


@pytest.mark.parametrize("count", [0, 1, 5])
def test_the_route_refuses_an_out_of_range_anchor_count(monkeypatch, count):
    status, payload = _post(monkeypatch, _StubPipelineManager(),
                            continuation_mode="motion_preroll",
                            continuation_overlap_frames=PREROLL,
                            continuation_anchor_count=count)
    assert status == 400, payload
    assert "2..4" in payload["detail"]


def test_the_route_refuses_anchors_on_a_pinned_tail(monkeypatch):
    status, payload = _post(monkeypatch, _StubPipelineManager(),
                            continuation_mode="pinned_tail",
                            continuation_overlap_frames=9,
                            continuation_anchor_count=ANCHORS)
    assert status == 400, payload
    assert "mutually exclusive" in payload["detail"]


@pytest.mark.parametrize("arch,variant", [("minimax_h3", "ref2va"), ("ltx2", None)])
def test_the_route_refuses_a_variant_that_does_not_advertise_the_mode(
        monkeypatch, arch, variant):
    status, payload = _post(monkeypatch, _StubPipelineManager(arch, variant),
                            continuation_mode="motion_preroll",
                            continuation_overlap_frames=PREROLL,
                            continuation_anchor_count=ANCHORS)
    assert status == 400, payload
    assert "chain_context" in payload["detail"]
    assert "refused rather than downgraded" in payload["detail"]


def test_the_route_takes_the_anchor_count_default_from_param_defaults():
    import inspect

    from api.param_defaults import VIDEO_CHAIN_DEFAULTS
    from api.routes import generate_outpaint_video

    parameters = inspect.signature(generate_outpaint_video).parameters
    assert parameters["continuation_anchor_count"].default.default == (
        VIDEO_CHAIN_DEFAULTS["requested_anchor_count"])
