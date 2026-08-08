"""ia2v: an uploaded audio track pinned across the clip the video is generated for.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_ia2v_test.py -v

WHY THIS FILE EXISTS
--------------------
`POST /generate/img2vid` grew `input_audio`. The mechanism is one flag and one
substitution, and every part of it is a property something else already relies
on, so the interesting assertions are the ones that fail if the obvious
"simplification" is made:

* the pinned rows are the clip's OWN audio rows, so the layout gains no rows and
  only `num_condition_audio_rows` changes. Every other tensor must stay bitwise
  identical, or a request that sends no audio changes too;
* the audio noise IS STILL DRAWN and then discarded. Skipping the draw would
  leave the generator in a different state, and the K0.6 order is a recorded
  contract, not an implementation detail;
* the required track length is the audio GRID's, not the clip's: `round(T/24*40)`
  latents x 800 samples can be longer than `round(T/24*32000)` samples of video
  (124 frames: 165 600 against 165 333). Taking the clip length as the
  requirement would silently under-feed the encoder;
* the muxed track is the SOURCE, sample for sample. The pinned rows are never
  written, so decoding them would return a VAE round trip of the input.

Each block below carries a negative control.
"""

import hashlib
import inspect
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.param_defaults import IMG2VID_DEFAULTS  # noqa: E402
from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402
from core.models.minimax_h3 import h3_references as refs  # noqa: E402


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The production point the C0 measurement pack ran at, in latent terms.
NUM_TEXT_TOKENS = 137
LATENT_FRAMES = 37          # 124 pixel frames
LATENT_HEIGHT, LATENT_WIDTH = 24, 40
NUM_AUDIO_LATENTS = ops.audio_latent_frames(124)


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _layout(pin: bool, anchors=("first",)):
    return ops.build_packed_layout(
        NUM_TEXT_TOKENS, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, NUM_AUDIO_LATENTS,
        keyframe_anchors=anchors, pin_target_audio=pin,
    )


# --------------------------------------------------------------------------
# How long the track has to be
# --------------------------------------------------------------------------

def test_the_required_length_is_the_audio_grid_not_the_clip_duration():
    """124 frames need 165 600 samples, which is MORE than the video's 165 333.

    NEGATIVE CONTROL: the required count is not the clip count. `round(T/24*40)`
    rounds UP at 124 frames (206.67 -> 207 latents), and 207 latents x 800
    samples outruns the video by a third of a frame. Requiring only the clip
    duration would hand the audio VAE 165 333 samples and get 206 latents back,
    one row short of what the layout reserves.
    """
    required, grid, clip = refs.pinned_audio_sample_counts(124)
    assert grid == NUM_AUDIO_LATENTS * 800 == 165600
    assert clip == 165333
    assert required == 165600
    assert required != clip


def test_the_required_length_is_the_clip_when_the_grid_rounds_the_other_way():
    """141 frames: the two agree exactly. 5 frames: the grid is SHORTER.

    NEGATIVE CONTROL for the mirror mistake -- taking the grid alone. At 5
    frames `round(5/24*40)` is 8 latents = 6 400 samples while the video runs
    6 667, so the mux would come up short.
    """
    required, grid, clip = refs.pinned_audio_sample_counts(141)
    assert grid == clip == required == 188000

    required, grid, clip = refs.pinned_audio_sample_counts(5)
    assert grid == 6400
    assert clip == 6667
    assert required == clip


# --------------------------------------------------------------------------
# Preparing the track
# --------------------------------------------------------------------------

def _tone(samples: int, channels: int = 2, sample_rate: int = 32000) -> torch.Tensor:
    t = torch.arange(samples, dtype=torch.float32) / sample_rate
    wave = torch.sin(2 * torch.pi * 440.0 * t) * 0.5
    return wave.expand(channels, -1).contiguous()


def test_a_track_at_the_models_rate_is_carried_through_sample_for_sample():
    """No resample, no gain, no fade: the first N samples of what was sent.

    NEGATIVE CONTROL: not merely the right LENGTH -- the values are compared
    against the source, so a normalisation or a resample-through-the-same-rate
    would fail even though the shape stayed correct.
    """
    required, _grid, _clip = refs.pinned_audio_sample_counts(124)
    source = _tone(required + 5000)
    prepared = refs.prepare_pinned_audio(source, 32000, num_frames=124)
    assert prepared.shape == (2, required)
    assert torch.equal(prepared, source[:, :required])


def test_a_mono_track_is_duplicated_into_both_channels():
    """The packed layout is channel-major STEREO, so mono has to become two."""
    required, _grid, _clip = refs.pinned_audio_sample_counts(124)
    source = _tone(required, channels=1)
    prepared = refs.prepare_pinned_audio(source, 32000, num_frames=124)
    assert prepared.shape == (2, required)
    assert torch.equal(prepared[0], prepared[1])
    assert torch.equal(prepared[0], source[0, :required])


def test_a_short_track_is_refused_with_both_durations():
    """A 400, not a pad.

    NEGATIVE CONTROL: the message has to carry BOTH numbers -- what was supplied
    and what is needed -- because "too short" without them is unactionable when
    the required length is the snapped clip's audio grid rather than anything
    the client computed.
    """
    required, _grid, _clip = refs.pinned_audio_sample_counts(124)
    source = _tone(required - 1)
    with pytest.raises(ValueError) as error:
        refs.prepare_pinned_audio(source, 32000, num_frames=124)
    message = str(error.value)
    assert "5.175" in message                      # required seconds
    assert "%.3f" % ((required - 1) / 32000) in message   # supplied seconds
    assert "124 frames" in message


def test_a_track_one_sample_long_enough_is_accepted():
    """The boundary is exactly the requirement, not the requirement plus slack."""
    required, _grid, _clip = refs.pinned_audio_sample_counts(124)
    prepared = refs.prepare_pinned_audio(_tone(required), 32000, num_frames=124)
    assert prepared.shape[-1] == required


def test_a_track_at_another_rate_is_resampled_to_exactly_the_needed_length():
    """44.1 kHz in, 32 kHz out, and the length is the grid's, not the source's."""
    required, _grid, _clip = refs.pinned_audio_sample_counts(124)
    source = _tone(int(6.0 * 44100), sample_rate=44100)
    prepared = refs.prepare_pinned_audio(source, 44100, num_frames=124)
    assert prepared.shape == (2, required)
    assert torch.isfinite(prepared).all()


def test_the_mux_slice_is_a_prefix_of_the_prepared_track():
    """What the route hands back is the SOURCE window, not a decode of it.

    `trim_audio_to_video` is the one trim both audio paths use, so the ia2v
    track handed to the muxer is exactly the first `clip` samples of what was
    uploaded. The exactness lives HERE, at the handoff, and not in the mp4:
    `save_video_with_metadata` encodes audio as AAC, so the file is a lossy
    encoding of these samples the same way it is for a generated soundtrack.
    """
    required, _grid, clip = refs.pinned_audio_sample_counts(124)
    source = _tone(required)
    prepared = refs.prepare_pinned_audio(source, 32000, num_frames=124)
    muxed = ops.trim_audio_to_video(prepared, 124)
    assert muxed.shape[-1] == clip
    assert torch.equal(muxed, source[:, :clip])


# --------------------------------------------------------------------------
# The layout, and what pinning is allowed to change
# --------------------------------------------------------------------------

def test_pinning_changes_exactly_one_number_in_the_layout():
    """Every tensor is bitwise identical; only the audio condition count moves.

    NEGATIVE CONTROL: the free layout still reports 0 pinned audio rows, so this
    is not asserting that both builds are the same dict.
    """
    free = _layout(pin=False)
    pinned = _layout(pin=True)

    assert free["num_condition_audio_rows"] == 0
    assert pinned["num_condition_audio_rows"] == NUM_AUDIO_LATENTS * ops.AUDIO_CHANNELS
    assert pinned["num_condition_audio_rows"] == int(pinned["audio_indices"].numel())

    for key in ("sequence_length", "num_condition_video_rows", "rows_per_frame"):
        assert free[key] == pinned[key], key
    for key in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
        assert _digest(free[key]) == _digest(pinned[key]), key


def test_pinning_defaults_to_off_so_every_existing_request_is_unchanged():
    """The flag is keyword-only with a False default: an old call site is safe."""
    signature = inspect.signature(ops.build_packed_layout)
    parameter = signature.parameters["pin_target_audio"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False
    assert ops.build_packed_layout(
        NUM_TEXT_TOKENS, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, NUM_AUDIO_LATENTS,
        keyframe_anchors=("first",),
    )["num_condition_audio_rows"] == 0


def test_every_audio_row_is_pinned_clean_and_none_is_left_to_write():
    """t = 1.0 is exactly clean under `x_t = t*x0 + (1-t)*noise`.

    NEGATIVE CONTROL: the free layout puts every audio row on the audio SCHEDULE
    instead, and leaves the whole block for `denoise` to write.
    """
    pinned = _layout(pin=True)
    free = _layout(pin=False)

    steps, index = ops.build_row_timesteps(pinned, video_timestep=0.5, audio_timestep=0.4)
    row_timesteps = steps[index]
    audio_rows = row_timesteps[pinned["audio_indices"]]
    assert torch.all(audio_rows == ops.AUDIO_COND_TIMESTEP)
    assert ops.AUDIO_COND_TIMESTEP in [float(v) for v in steps]
    # The slice `denoise` writes is empty, which is why it needs no branch.
    assert pinned["audio_indices"][pinned["num_condition_audio_rows"]:].numel() == 0

    free_steps, free_index = ops.build_row_timesteps(free, video_timestep=0.5, audio_timestep=0.4)
    free_audio = free_steps[free_index][free["audio_indices"]]
    assert torch.all(free_audio == 0.4)
    assert free["audio_indices"][free["num_condition_audio_rows"]:].numel() == \
        NUM_AUDIO_LATENTS * ops.AUDIO_CHANNELS


def test_pinning_half_the_rows_would_pin_one_channel_not_half_the_timeline():
    """WHY partial-timeline placement is refused rather than approximated.

    The count is a PREFIX and the rows are CHANNEL-MAJOR: the first half of
    `audio_indices` is channel 0's entire timeline. A "pin seconds 0-3" feature
    built on this count would pin the left channel of the whole clip. This is
    the measurement behind that refusal, kept as a test so the refusal's reason
    cannot quietly stop being true.
    """
    pinned = _layout(pin=True)
    half = NUM_AUDIO_LATENTS
    times = pinned["position_ids"][pinned["audio_indices"], 0]
    assert float(times[0]) == pytest.approx(NUM_TEXT_TOKENS)
    assert float(times[half - 1]) == pytest.approx(NUM_TEXT_TOKENS + NUM_AUDIO_LATENTS - 1)
    # ... and the second half is the SAME time range again (the other channel).
    assert torch.equal(times[:half], times[half:])


# --------------------------------------------------------------------------
# The noise draw
# --------------------------------------------------------------------------

def _rows_for_one_request(seed: int, *, pin: bool):
    """The backend's own pre-denoise sequence, mirrored: draw, THEN substitute.

    A mirror is documentation, not a defence -- it cannot fail if the backend
    stops looking like it. What defends the property is
    `test_the_draw_is_structurally_unconditional` below, which reads the shipped
    function's AST. This one exists to state, numerically and readably, what the
    property IS.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    condition_noises, video_noise, audio_rows = ops.draw_noise(
        generator,
        video_latent_shape=(1, 24, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH),
        num_audio_latents=NUM_AUDIO_LATENTS,
        condition_shapes=((1, 24, 1, LATENT_HEIGHT, LATENT_WIDTH),),
        device="cpu",
    )
    if pin:
        # The encoded track replaces the drawn rows AFTER the draw, which is the
        # whole trick: the generator has already advanced past them.
        audio_rows = torch.full_like(audio_rows, 0.25)
    trailing = torch.randn(4, generator=generator, device="cpu")
    return condition_noises, video_noise, audio_rows, trailing


def test_the_video_noise_is_identical_with_and_without_a_pinned_track():
    """Same seed, same video noise, and the generator ends in the same state."""
    free_cond, free_video, free_audio, free_trailing = _rows_for_one_request(4242, pin=False)
    pin_cond, pin_video, pin_audio, pin_trailing = _rows_for_one_request(4242, pin=True)

    assert _digest(free_video) == _digest(pin_video)
    assert _digest(free_cond[0]) == _digest(pin_cond[0])
    assert _digest(free_trailing) == _digest(pin_trailing)
    # Not vacuous: the audio rows genuinely differ between the two runs.
    assert _digest(free_audio) != _digest(pin_audio)


def _generate_ast():
    """The shipped `_generate_minimax_h3` as an AST, with parent links."""
    import ast
    import textwrap

    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    source = textwrap.dedent(inspect.getsource(MiniMaxH3Mixin._generate_minimax_h3))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    return ast, tree, source


def test_the_draw_is_structurally_unconditional():
    """THE defence of the same-seed contract, read off the shipped function's AST.

    The property "a pinned run and a free-audio run share their video noise"
    holds because `draw_noise` is called ONCE, on a path no branch guards, with
    arguments that do not mention the track -- and because nothing else in the
    function draws from the request generator.

    THE MUTANT THIS EXISTS FOR: wrap the `draw_noise` call in
    `if pinned_audio_rows is None:` and draw the condition/video noise
    separately in the `else`. That keeps the call count at one, keeps the
    substitution after the draw, and silently changes the video of every ia2v
    request at a fixed seed. It is caught here by the ancestor walk (the call
    acquires an `If` parent) and by the RNG-call check (the `else` branch has to
    call `randn` itself).
    """
    ast, tree, _source = _generate_ast()

    draws = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "draw_noise"]
    assert len(draws) == 1, "exactly one draw, or the order is no longer one thing"
    draw = draws[0]

    # 1. No conditional anywhere between the function body and the call.
    ancestors, node = [], draw
    while hasattr(node, "parent"):
        node = node.parent
        ancestors.append(node)
    guards = [a for a in ancestors if isinstance(a, (ast.If, ast.IfExp, ast.Try, ast.While))]
    assert not guards, (
        "ops.draw_noise is inside a "
        + ", ".join(type(a).__name__ for a in guards)
        + ": the draw must happen for every request, pinned or not, or the same "
          "seed stops meaning the same video noise")

    # 2. The call's arguments do not mention the track, so it cannot be shaped
    #    differently for an ia2v request.
    argument_names = {n.id for n in ast.walk(draw) if isinstance(n, ast.Name)}
    assert "input_audio" not in argument_names
    assert "pinned_audio_rows" not in argument_names

    # 3. Nothing else in the function draws from the request generator -- that
    #    is how a mutant would replace the skipped draw.
    other_rng = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr in ("randn", "randn_like", "rand", "normal_")]
    assert not other_rng, (
        "the request generator is drawn from outside ops.draw_noise "
        f"(line(s) {[n.lineno for n in other_rng]}), so the recorded draw order is "
        "no longer the only thing that decides the noise")

    # 4. The substitution follows the draw (the "discard" half of the contract).
    substitutions = [node for node in ast.walk(tree)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "audio_rows"
                             for t in node.targets)
                     and "pinned_audio_rows" in {n.id for n in ast.walk(node.value)
                                                 if isinstance(n, ast.Name)}]
    assert len(substitutions) == 1
    assert substitutions[0].lineno > draw.lineno


def test_the_backend_muxes_the_source_and_does_not_decode_the_pinned_rows():
    """The ia2v branch of the decode stage returns the uploaded samples."""
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    source = inspect.getsource(MiniMaxH3Mixin._generate_minimax_h3)
    branch = source[source.index("if audio_enable and input_audio is not None:"):]
    branch = branch[:branch.index("elif audio_enable:")]
    assert "trim_audio_to_video(" in branch
    assert "input_audio" in branch
    assert "decode_audio(" not in branch


# --------------------------------------------------------------------------
# Reaching the backend at all
# --------------------------------------------------------------------------

def test_an_imageless_request_is_a_real_request_when_a_track_is_sent():
    """No keyframes + a pinned track = pure a2v, which is measured working.

    NEGATIVE CONTROL: with no track either, the same call still refuses.
    """
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    captured = {}

    class Runner(MiniMaxH3Mixin):
        def _generate_minimax_h3(self, params, **kwargs):
            captured.update(kwargs)
            return "generated"

    runner = Runner()
    track = torch.zeros(2, 8)
    assert runner._generate_img2vid_minimax_h3({}, None, input_audio=track) == "generated"
    assert captured["keyframes"] == ()
    assert captured["input_audio"] is track

    with pytest.raises(RuntimeError):
        runner._generate_img2vid_minimax_h3({}, None)


def test_a_pinned_track_and_references_are_refused_as_two_mechanisms():
    """ref2va reaches a track through its own block, at another rotary offset."""
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    class Runner(MiniMaxH3Mixin):
        minimax_h3_components = {"variant": "ref2va"}

    with pytest.raises(RuntimeError) as error:
        Runner()._generate_minimax_h3({}, references=("a reference",),
                                      input_audio=torch.zeros(2, 8))
    assert "ia2v" in str(error.value)


# --------------------------------------------------------------------------
# The API surface
# --------------------------------------------------------------------------

def test_the_default_lives_in_param_defaults():
    assert "input_audio" in IMG2VID_DEFAULTS
    assert IMG2VID_DEFAULTS["input_audio"] is None


def test_the_route_declares_the_field_and_makes_the_image_optional():
    """`image` became `File(None)`; the "at least one medium" rule is explicit."""
    import api.routes as routes

    signature = inspect.signature(routes.generate_img2vid)
    assert "input_audio" in signature.parameters
    assert signature.parameters["input_audio"].default.default is None
    assert signature.parameters["image"].default.default is None

    source = inspect.getsource(routes.generate_img2vid)
    assert '"input_audio": recover_upload_filename(_input_audio.filename) if _input_audio is not None else None' in source
    assert "img2vid needs something to condition on" in source
    # LTX-2.3 keeps the requirement, with the reason.
    assert "LTX-2.3 image-to-video needs an input image" in source
    # The prepared waveform reaches the pipeline.
    assert "input_audio=input_audio_prepared" in source


def test_the_track_is_prepared_after_the_clip_length_is_snapped():
    """Its required length is a function of the SNAPPED length, like placement."""
    import api.routes as routes

    source = inspect.getsource(routes.generate_img2vid)
    assert source.index("validate_video_geometry(params, _vid_arch)") < \
        source.index("prepare_pinned_audio(")


def test_one_warning_covers_every_undocumented_shape_of_the_request():
    """Placement and audio conditioning share the entry and its code."""
    import api.routes as routes

    source = inspect.getsource(routes.generate_img2vid)
    assert source.count('code="minimax_h3_undocumented_conditioning"') == 1
    assert "an input audio track pinned clean across the whole clip" in source
    # audio_enable=false + input_audio is legal and named, under its own code.
    assert 'code="minimax_h3_input_audio_not_muxed"' in source


def test_the_capability_key_exists_and_gates_the_right_architecture():
    from api.arch_capabilities import (ARCH_UNSUPPORTED, FEATURE_LABELS, FEATURE_PARAMS,
                                       arch_supports_feature)

    assert FEATURE_PARAMS["audio_conditioning"] == ["input_audio"]
    assert "input_audio" in FEATURE_LABELS["audio_conditioning"]
    assert arch_supports_feature("minimax_h3", "audio_conditioning")
    assert not arch_supports_feature("ltx2", "audio_conditioning")
    assert "audio" in ARCH_UNSUPPORTED["ltx2"]["audio_conditioning"]


def test_openapi_documents_the_field_and_stops_requiring_an_image():
    import yaml

    with open(os.path.join(REPO_ROOT, "openapi.yaml"), encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    schema = spec["components"]["schemas"]["Img2VidRequest"]["allOf"][1]
    assert "required" not in schema
    description = schema["properties"]["input_audio"]["description"].lower()
    assert "entire clip" in description
    assert "partial-timeline placement is not supported" in description
    assert "400" in description                       # the short-track refusal
    assert "sample for sample" in description         # the mux contract


def test_the_frontend_sender_appends_the_track():
    path = os.path.join(REPO_ROOT, "frontend", "src", "utils", "api.ts")
    with open(path, encoding="utf-8") as handle:
        api_ts = handle.read()
    sender = api_ts[api_ts.index("export const generateImg2Vid"):]
    sender = sender[:sender.index("export const generateRef2Vid")]
    assert 'formData.append("input_audio"' in sender
    assert "input_audio?: File | null" in api_ts


def test_the_panel_carries_the_track_onto_the_item_and_back_out_at_dequeue():
    """Two sites, and the second is the one that is easy to miss.

    The track is a File, so it rides on the QUEUE ITEM (`inputAudio`) like every
    other upload rather than sitting in the persisted params blob -- and it has
    to be merged back into the params object at DEQUEUE time, because that
    object is what `generateImg2Vid` reads. Missing the second site is the
    classic "main generation works, the queued one sends null".
    """
    path = os.path.join(REPO_ROOT, "frontend", "src", "components", "generation",
                        "Img2ImgPanel.tsx")
    with open(path, encoding="utf-8") as handle:
        panel = handle.read()

    enqueue = panel[panel.index("const videoParams: Img2VidParams = {"):]
    enqueue = enqueue[:enqueue.index("return;")]
    assert "inputAudio:" in enqueue
    assert "inputAudioTrack" in enqueue

    dequeue = panel[panel.index('if (nextItem.type === "img2vid")'):]
    dequeue = dequeue[:dequeue.index("generateImg2Vid(")]
    assert "input_audio: nextItem.inputAudio" in dequeue

    # The lane is gated on the capability, not on an arch string compare.
    assert 'archSupportsFeature(\n    archCapabilities, loadedArch, "audio_conditioning")' in panel


def test_the_timeline_draws_the_lane_full_width_with_no_offset_handles():
    """Whole-clip is the only supported placement, so nothing suggests another."""
    path = os.path.join(REPO_ROOT, "frontend", "src", "components", "common",
                        "MiniMaxH3KeyframeTimeline.tsx")
    with open(path, encoding="utf-8") as handle:
        timeline = handle.read()
    flat = " ".join(timeline.split())
    assert "onInputAudioChange" in timeline
    assert "conditions the entire clip" in flat
    assert "partial-timeline placement is not supported" in flat
    # The measured scope is stated where the control is, not only in the docs.
    assert "Speech, pitch and timbre were not measured." in timeline
    # ... and the mux claim matches what the file actually is. "carries this
    # file's audio unchanged" was the shipped over-claim: the mp4's audio is
    # AAC, and with audio_enable off there is no audio track at all.
    assert "unchanged" not in flat
    assert "AAC encode" in flat
    assert "audioEnabled" in timeline
    assert "Audio output is off" in flat
