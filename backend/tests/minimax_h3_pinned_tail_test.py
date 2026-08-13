"""`continuation_mode: pinned_tail` -- the chain continuation that pins a tail.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_pinned_tail_test.py -v

WHAT IS NEW HERE AND WHAT IS NOT
--------------------------------
Nothing in the sampler is new: a `pinned_tail` continuation is
`/generate/inpaint/video`'s pin mechanism (`pinned_video_frames` +
`pinned_audio_latents`) pointed at the HEAD of the generated span, so the
predecessor's decoded tail becomes the new clip's own leading latent frames.
`minimax_h3_temporal_inpaint_test.py` and `minimax_h3_audio_pin_test.py` already
pin the permutation, the timestep plan and the substitute/un-permute round trip.

What this file adds is the row invariants IN THIS CONFIGURATION -- video and
audio pinned TOGETHER, both as a leading prefix -- plus the arithmetic that
decides which rows those are:

* protected rows are unchanged after EVERY denoise step, and only free rows are
  written (the pin is the prefix count, and it must hold for both tracks at
  once);
* the free rows carry the same noise a continuation-off run at that seed draws:
  the substitution follows the draw, on both tracks;
* the video pin and the audio pin describe the SAME physical time -- the audio
  set is the whole latents INSIDE the video overlap, never past it;
* the overlap lengths are the cumulative sums of the arch's CYCLING
  `latent_chunk_pattern` (1, 5, 9, 13, 17, 18, ... -- not 1, 5, 17, 33), and an
  unaligned request is refused rather than snapped.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.generation_utils import (  # noqa: E402
    latent_frame_spans,
    plan_audio_pin_latents,
    plan_video_continuation_context,
    plan_video_outpaint_placement,
    video_continuation_overlap_lengths,
)
from core.models.components.wiring import temporal_spec_for_arch  # noqa: E402
from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402
from core.models.minimax_h3.loader import minimax_h3_latent_frames  # noqa: E402

SPEC = temporal_spec_for_arch("minimax_h3")
GENERATED = 124          # the shortest span this arch generates
OVERLAP = 5              # 1 + 4: the second cumulative sum of (1, 4, 4, 4, 4)


def _pinned_video_frames(generated_frames=GENERATED, overlap=OVERLAP):
    spans = latent_frame_spans(SPEC, minimax_h3_latent_frames(generated_frames))
    return tuple(i for i, (_lo, hi) in enumerate(spans) if hi <= overlap)


# --------------------------------------------------------------------------
# 1. The addressable overlap lengths -- ONE enumerator, and it CYCLES
# --------------------------------------------------------------------------

def test_the_overlap_lengths_are_the_cycling_cumulative_sums():
    """1, 5, 9, 13, 17 -- not 1, 5, 17, 33.

    `ROPE_FRAMES_PER_LATENT` repeats after five latent frames, so the naive
    doubling extrapolation names lengths that split a latent frame in half. The
    list is derived from the shipped pattern, and cross-checked against the
    rotary clock's own frame count so the two cannot drift onto two chunkings.
    """
    assert SPEC.latent_chunk_pattern == (1, 4, 4, 4, 4)
    assert video_continuation_overlap_lengths(SPEC, 17) == (1, 5, 9, 13, 17)
    # Past one cycle the next boundary is 18, which is what makes 33 wrong.
    assert video_continuation_overlap_lengths(SPEC, 40)[:8] == (1, 5, 9, 13, 17, 18, 22, 26)
    assert 33 not in video_continuation_overlap_lengths(SPEC, 40)
    for length in video_continuation_overlap_lengths(SPEC, 40):
        assert ops._clip_pixel_frames(len(_pinned_video_frames(GENERATED, length))) == length


def test_an_unaligned_or_oversized_overlap_is_refused_not_snapped():
    from api.error_handlers import ValidationError

    for bad in (0, 2, 4, 16, 33, 18):
        with pytest.raises(ValidationError) as error:
            plan_video_continuation_context("pinned_tail", bad, "minimax_h3", "fl2va")
        # The refusal names the lengths that work, so the caller is not left
        # guessing which nearby value would have been snapped to.
        assert "1, 5, 9, 13, 17" in str(error.value.detail)
    # NEGATIVE CONTROL: every advertised length is accepted at the same call.
    for good in (1, 5, 9, 13, 17):
        assert plan_video_continuation_context(
            "pinned_tail", good, "minimax_h3", "fl2va")["overlap_frames"] == good


@pytest.mark.parametrize("arch,variant", [
    ("minimax_h3", "ref2va"),   # references claim the prefix a pin needs
    ("ltx2", None),             # conditions on the whole preserved clip already
    ("sdxl", None),             # not a video architecture at all
])
def test_pinned_tail_is_refused_where_it_is_not_advertised(arch, variant):
    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError) as error:
        plan_video_continuation_context("pinned_tail", 5, arch, variant)
    assert "chain_context" in str(error.value.detail)


def test_boundary_frame_is_unchanged_and_refuses_a_meaningless_overlap():
    from api.error_handlers import ValidationError

    for arch, variant in (("minimax_h3", "fl2va"), ("minimax_h3", "ref2va"), ("ltx2", None)):
        assert plan_video_continuation_context("boundary_frame", 0, arch, variant) == {
            "mode": "boundary_frame", "overlap_frames": 1}
    with pytest.raises(ValidationError):
        plan_video_continuation_context("boundary_frame", 5, "minimax_h3", "fl2va")


# --------------------------------------------------------------------------
# 2. The placement: the OUTPUT length is what stays fixed
# --------------------------------------------------------------------------

def test_the_generated_span_absorbs_the_overlap_and_the_output_length_holds():
    """A wider overlap lengthens the span, not the answer.

    `total = head + generated - shared`, so asking for the same `total_frames`
    with a 5-frame overlap has to solve for a span 4 frames longer (then rounded
    up to 17n+5, which the endpoint already warns about).
    """
    params = {"total_frames": 500, "input_offset_frames": 0}
    anchor = plan_video_outpaint_placement(params, "minimax_h3", head_frames=362)
    pinned = plan_video_outpaint_placement(params, "minimax_h3", head_frames=362,
                                           overlap_frames=OVERLAP)
    assert anchor["shared_anchor_frames"] == 1
    assert pinned["shared_anchor_frames"] == OVERLAP
    assert pinned["generated_frames"] >= anchor["generated_frames"] + OVERLAP - 1
    for plan in (anchor, pinned):
        assert plan["total_frames"] == (
            plan["head_frames"] + plan["generated_frames"] - plan["shared_anchor_frames"])
    # NEGATIVE CONTROL: the default argument reproduces the shipped plan exactly.
    assert plan_video_outpaint_placement(params, "minimax_h3", head_frames=362,
                                         overlap_frames=1) == anchor


def test_an_overlap_is_refused_on_the_placements_that_cannot_carry_it():
    from api.error_handlers import ValidationError

    for kwargs in (
        dict(head_frames=362, tail_frames=124),                      # bridge
        dict(head_frames=362),                                       # extend_backward
    ):
        params = {"total_frames": 500,
                  "input_offset_frames": 0 if "tail_frames" in kwargs else 138}
        with pytest.raises(ValidationError, match="extend-forward"):
            plan_video_outpaint_placement(params, "minimax_h3", overlap_frames=5, **kwargs)
    # ... and a clip shorter than the overlap has no tail to pin.
    with pytest.raises(ValidationError, match="shorter than the requested overlap"):
        plan_video_outpaint_placement({"total_frames": 500, "input_offset_frames": 0},
                                      "minimax_h3", head_frames=3, overlap_frames=5)


# --------------------------------------------------------------------------
# 3. Video and audio pins describe the SAME physical time
# --------------------------------------------------------------------------

def test_the_video_and_audio_pins_cover_the_same_overlap():
    """The audio set is the whole latents INSIDE the video overlap.

    The audio grid (40/s) is finer than the video one (24/s in groups of up to
    4), and `plan_audio_pin_latents` snaps a partially covered latent to FREE --
    so the audio pin can only ever be a subset of the video overlap in time,
    never an overhang into frames the model is supposed to generate.
    """
    spans = latent_frame_spans(SPEC, minimax_h3_latent_frames(GENERATED))
    for overlap in (1, 5, 9, 13, 17):
        video = _pinned_video_frames(GENERATED, overlap)
        assert spans[video[-1]][1] == overlap
        num_audio = ops.audio_latent_frames(GENERATED)
        free, pinned = plan_audio_pin_latents(overlap, GENERATED, num_audio)
        assert pinned == tuple(range(len(pinned))), "the audio pin is not a leading prefix"
        assert set(pinned).isdisjoint(free)
        # Both in SECONDS, from their own clocks.
        video_end = overlap / 24.0
        audio_end = len(pinned) / 40.0
        assert audio_end <= video_end + 1e-9
        assert video_end - audio_end < 1 / 40.0, "a whole audio latent inside the overlap is free"


# --------------------------------------------------------------------------
# 4. The row invariants, in THIS configuration (both tracks pinned at once)
# --------------------------------------------------------------------------

# The chain shape at a size a CPU can run: a leading prefix of video latent
# frames and of audio latents, pinned together.
GEOMETRY = dict(num_text_tokens=3, num_latent_frames=5, latent_height=4,
                latent_width=4, num_audio_latents=10)
PIN_VIDEO = (0, 1)
PIN_AUDIO = (0, 1, 2)
CHANNELS = 24


def _pinned_layout():
    return ops.build_packed_layout(
        **GEOMETRY, pinned_video_frames=PIN_VIDEO, pinned_audio_latents=PIN_AUDIO)


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


def test_both_pinned_prefixes_ride_the_conditioning_timestep():
    layout = _pinned_layout()
    unique, index = ops.build_row_timesteps(layout, 0.1, 0.1)
    row_time = unique[index]
    n_video = layout["num_condition_video_rows"]
    n_audio = layout["num_condition_audio_rows"]
    assert n_video == len(PIN_VIDEO) * layout["rows_per_frame"]
    assert n_audio == len(PIN_AUDIO) * ops.AUDIO_CHANNELS
    assert bool((row_time[layout["video_indices"][:n_video]] == ops.VISUAL_COND_TIMESTEP).all())
    assert bool((row_time[layout["audio_indices"][:n_audio]] == ops.AUDIO_COND_TIMESTEP).all())
    # NEGATIVE CONTROL: the free rows of both tracks are on the sampling clock.
    assert bool((row_time[layout["video_indices"][n_video:]] == 0.1).all())
    assert bool((row_time[layout["audio_indices"][n_audio:]] == 0.1).all())


def test_the_pinned_rows_of_both_tracks_survive_every_step_and_only_free_rows_are_written():
    """THE invariant: the prefix count protects both tracks, at every step.

    Multi-step deliberately -- a single step cannot distinguish "never written"
    from "written back to the same value on the only step there was".
    """
    layout = _pinned_layout()
    n_video = layout["num_condition_video_rows"]
    n_audio = layout["num_condition_audio_rows"]
    video_rows = torch.randn(layout["video_indices"].numel(), CHANNELS * 4,
                             generator=torch.Generator().manual_seed(11))
    audio_rows = torch.randn(layout["audio_indices"].numel(), 32,
                             generator=torch.Generator().manual_seed(12))
    before_video, before_audio = video_rows.clone(), audio_rows.clone()

    steps = [0.9, 0.6, 0.3]
    out_video, out_audio = ops.denoise(
        lambda **kw: (torch.full((1, video_rows.shape[0], video_rows.shape[1]), 2.0),
                      torch.full((1, audio_rows.shape[0], audio_rows.shape[1]), 3.0)),
        _StubScheduler(steps), _StubScheduler(steps),
        prompt_embeds=torch.zeros(1, 3, 8), layout=layout,
        video_rows=video_rows, audio_rows=audio_rows,
        num_inference_steps=len(steps), device="cpu",
    )
    assert torch.equal(out_video[:n_video], before_video[:n_video])
    assert torch.equal(out_audio[:n_audio], before_audio[:n_audio])
    # NEGATIVE CONTROL: the free rows moved by exactly the three steps, so the
    # assertions above are about protection and not about a loop that did
    # nothing.
    assert torch.allclose(out_video[n_video:], before_video[n_video:] - len(steps) * 0.5)
    assert torch.allclose(out_audio[n_audio:], before_audio[n_audio:] - len(steps) * 0.75)


def test_the_free_rows_of_both_tracks_keep_the_continuation_off_noise():
    """The substitution FOLLOWS the draw, so a seed means the same thing.

    Drawn once with a pin and once without, from the same seed: every row that
    is not pinned must be bit-identical, or turning the continuation on silently
    changes what every other frame of the segment is.
    """
    shape = (1, CHANNELS, GEOMETRY["num_latent_frames"],
             GEOMETRY["latent_height"], GEOMETRY["latent_width"])
    patch = (1, 2, 2)

    def draw():
        generator = torch.Generator(device="cpu").manual_seed(7)
        _cond, video_noise, audio_rows = ops.draw_noise(
            generator, video_latent_shape=shape,
            num_audio_latents=GEOMETRY["num_audio_latents"], condition_shapes=(),
            device="cpu", audio_latent_channels=32)
        return ops.patchify_video_latents(video_noise, patch)[0], audio_rows

    plain_video, plain_audio = draw()
    video_rows, audio_rows = draw()
    layout = _pinned_layout()

    # Video: substitute in FRAME-MAJOR space, then apply the layout permutation
    # (the identity for a leading prefix -- asserted, so this test would notice
    # if the pin stopped being one).
    rows_per_frame = int(layout["rows_per_frame"])
    pin_rows = tuple(range(len(PIN_VIDEO) * rows_per_frame))
    source_video = torch.full_like(video_rows, 1000.0)

    class _Scheduler:
        @staticmethod
        def scale_noise(source, timestep, noise):
            return source
    video_rows = ops.pin_video_rows(video_rows, source_video, pin_rows, _Scheduler(),
                                    ops.VISUAL_COND_TIMESTEP)
    assert torch.equal(layout["video_row_permutation"],
                       torch.arange(video_rows.shape[0])), "a tail pin is not a leading prefix"
    video_rows = video_rows[layout["video_row_permutation"]]

    source_audio = torch.full_like(audio_rows, 2000.0)
    audio_rows = ops.substitute_and_permute_audio_rows(
        audio_rows, source_audio, PIN_AUDIO, GEOMETRY["num_audio_latents"],
        layout["audio_row_permutation"])

    n_video = layout["num_condition_video_rows"]
    n_audio = layout["num_condition_audio_rows"]
    assert torch.equal(video_rows[n_video:], plain_video[n_video:])
    # Audio rows are CHANNEL-major, so the free set is not a suffix of the
    # original block: compare after un-permuting, on the rows that were not
    # pinned.
    restored = audio_rows[layout["audio_row_order"]]
    pin_indices = set(ops.audio_pin_row_indices(PIN_AUDIO, GEOMETRY["num_audio_latents"]))
    free_indices = [i for i in range(restored.shape[0]) if i not in pin_indices]
    assert torch.equal(restored[free_indices], plain_audio[free_indices])
    # NEGATIVE CONTROL: the pinned rows really were replaced (so "identical
    # free rows" is not just "nothing happened").
    assert bool((restored[sorted(pin_indices)] == 2000.0).all())
    assert bool((video_rows[:n_video] == 1000.0).all())


# --------------------------------------------------------------------------
# 5. The orchestration: what is pinned, and what the output is made of
# --------------------------------------------------------------------------

GENERATED_VALUE = 7
HEAD_VALUE = 200


def _runner(width=64, height=32):
    """A mixin instance whose generate step returns a constant, capturing kwargs
    (the same harness `minimax_h3_temporal_inpaint_route_test` uses)."""
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
              "continuation_mode": "pinned_tail", "continuation_overlap_frames": OVERLAP}
    params.update(overrides)
    return params


def test_the_continuation_pins_the_head_clips_own_tail_and_no_anchor():
    runner, captured = _runner()
    head = _head_clip()
    params = _outpaint_params()
    frames, _audio, _rate, seed = runner._generate_vidoutpaint_minimax_h3(
        params, head, 24.0, None)

    generated = params["outpaint_generated_frames"]
    assert seed == 4242
    assert captured["pinned_video_frames"] == _pinned_video_frames(generated, OVERLAP)
    # An anchor keyframe would claim the same conditioning prefix.
    assert captured["keyframes"] == ()
    # The pin source's first OVERLAP frames ARE the preserved clip's last ones.
    source = captured["pinned_video_source"]
    assert source.shape[0] == generated
    assert np.array_equal(source[:OVERLAP], head[-OVERLAP:])
    assert np.array_equal(source[OVERLAP], head[-1])   # held, not black
    # The output keeps the head pixel-exact and drops the re-rendered overlap.
    assert np.array_equal(frames[:head.shape[0]], head)
    assert (frames[head.shape[0]:] == GENERATED_VALUE).all()
    assert frames.shape[0] == head.shape[0] + generated - OVERLAP
    assert params["continuation_effective_overlap_frames"] == OVERLAP


def test_the_default_mode_reproduces_the_boundary_anchor_request():
    """NEGATIVE CONTROL: nothing about an ordinary extend changed."""
    runner, captured = _runner()
    head = _head_clip()
    params = _outpaint_params(continuation_mode="boundary_frame",
                              continuation_overlap_frames=0)
    frames, _audio, _rate, _seed = runner._generate_vidoutpaint_minimax_h3(
        params, head, 24.0, None)
    assert captured["pinned_video_frames"] == ()
    assert captured["pinned_video_source"] is None
    assert [anchor for anchor, _image in captured["keyframes"]] == ["first"]
    assert frames.shape[0] == head.shape[0] + params["outpaint_generated_frames"] - 1
    assert params["continuation_effective_overlap_frames"] == 0


def test_a_missing_input_track_pins_video_only_and_says_so():
    """A clip with no audio to pin is disclosed, never assumed."""
    runner, captured = _runner()
    warnings = []
    audio, latents = runner._minimax_h3_pinned_tail_audio(
        None, {"outpaint_video_audio_mode": "regenerate", "audio_enable": False},
        head_frames=362, generated_frames=GENERATED, overlap_frames=OVERLAP,
        source_fps=24.0, trim_start=0, frame_rate=24.0,
        warn=lambda message, code: warnings.append(code))
    assert (audio, latents) == (None, ())
    assert warnings == ["minimax_h3_pinned_tail_video_only"]
    assert captured == {}


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
    happens BEFORE `video.read()`, which is exactly where this endpoint's other
    cheap gates live.
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


@pytest.mark.parametrize("overlap", [0, 2, 16, 33])
def test_the_route_refuses_an_unaligned_overlap_before_decoding(monkeypatch, overlap):
    status, payload = _post(monkeypatch, _StubPipelineManager(),
                            continuation_mode="pinned_tail",
                            continuation_overlap_frames=overlap)
    assert status == 400, payload
    assert "1, 5, 9, 13, 17" in payload["detail"]


@pytest.mark.parametrize("arch,variant", [("minimax_h3", "ref2va"), ("ltx2", None)])
def test_the_route_refuses_a_mode_the_loaded_variant_does_not_advertise(
        monkeypatch, arch, variant):
    status, payload = _post(monkeypatch, _StubPipelineManager(arch, variant),
                            continuation_mode="pinned_tail",
                            continuation_overlap_frames=5)
    assert status == 400, payload
    assert "chain_context" in payload["detail"]
    assert "refused rather than downgraded" in payload["detail"]


def test_the_route_takes_the_continuation_fields_from_param_defaults():
    import inspect

    from api.param_defaults import VIDEO_CHAIN_DEFAULTS
    from api.routes import generate_outpaint_video

    parameters = inspect.signature(generate_outpaint_video).parameters
    assert parameters["continuation_mode"].default.default == (
        VIDEO_CHAIN_DEFAULTS["continuation_mode"])
    assert parameters["continuation_overlap_frames"].default.default == (
        VIDEO_CHAIN_DEFAULTS["requested_overlap_frames"])
