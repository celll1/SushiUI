"""Temporal inpaint: the route, the span planner and the pin/paste orchestration.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_temporal_inpaint_route_test.py -v

WHY THIS FILE EXISTS
--------------------
`POST /generate/inpaint/video` regenerates one time range of a clip and keeps
the rest. Two mechanisms do that, and each one is invisible in the output of the
other's absence unless it is asserted directly:

* the PIN -- the kept latent frames lead the packed video block and are never
  denoised, which is what makes the generated range continue this clip rather
  than a new one. Skipping it still returns a video of the right length;
* the PASTE -- the source pixels go back over the preserved region after
  decode. Skipping it still returns preserved-looking frames, at a VAE round
  trip of 3.20 RMS plus up to 2.97 of decoder bleed near the boundary
  (`scratchpad/minimax_h3_ti_probe_results.md`), which is exactly the difference
  between "preserved" being true and being nearly true.

So the two blocks below carry the negative controls the brief asks for: one test
fails if the paste is skipped, one fails if the pin is not applied. The rest
covers the arithmetic (the outward snap and its refusals) and the same-seed
noise contract, defended structurally the way `minimax_h3_ia2v_test` defends
ia2v's.
"""

import ast
import inspect
import os
import sys
import textwrap

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.generation_utils import latent_frame_spans, plan_video_inpaint_span  # noqa: E402
from api.param_defaults import (  # noqa: E402
    INPAINT_VIDEO_DEFAULTS,
    inpaint_video_defaults_for_arch,
)
from core.models.components.wiring import temporal_spec_for_arch  # noqa: E402
from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402

SPEC = temporal_spec_for_arch("minimax_h3")
CLIP = 124          # 17*7 + 5, the production floor and every example request
LATENTS = 37        # SPEC.latent_frames(124)


def _plan(start, end, clip_frames=CLIP, arch="minimax_h3"):
    return plan_video_inpaint_span(
        {"regenerate_start_frame": start, "regenerate_end_frame": end},
        arch, clip_frames=clip_frames)


# --------------------------------------------------------------------------
# The addressable unit
# --------------------------------------------------------------------------

def test_the_latent_spans_tile_the_clip_exactly():
    """The (1,4,4,4,4) chunking, against the module that already relies on it."""
    spans = latent_frame_spans(SPEC, LATENTS)
    assert len(spans) == LATENTS
    assert spans[:6] == [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 18)]
    assert spans[0][0] == 0 and spans[-1][1] == CLIP
    assert all(spans[i][1] == spans[i + 1][0] for i in range(LATENTS - 1))
    # The same total `h3_pipeline_ops` computes from ROPE_FRAMES_PER_LATENT, so
    # the planner and the rotary clock cannot drift onto two chunkings.
    assert spans[-1][1] == ops._clip_pixel_frames(LATENTS)


def test_a_requested_range_is_expanded_outward_and_never_shrunk():
    """At a group boundary "regenerate this" wins over "keep that"."""
    plan = _plan(40, 85)
    assert plan["snapped"] is True
    assert plan["start_frame"] <= 40 and plan["end_frame"] >= 85
    spans = latent_frame_spans(SPEC, LATENTS)
    first, last = plan["regenerate_latent_frames"][0], plan["regenerate_latent_frames"][-1]
    assert spans[first][0] == plan["start_frame"]
    assert spans[last][1] == plan["end_frame"]
    # NEGATIVE CONTROL: a range that already sits on boundaries does not move,
    # so the assertion above is about the snap and not about any range at all.
    exact = _plan(spans[10][0], spans[20][1])
    assert exact["snapped"] is False
    assert (exact["start_frame"], exact["end_frame"]) == (spans[10][0], spans[20][1])


def test_the_pinned_frames_are_exactly_the_complement_of_the_regenerated_ones():
    plan = _plan(40, 85)
    pinned = set(plan["pinned_latent_frames"])
    regenerate = set(plan["regenerate_latent_frames"])
    assert pinned | regenerate == set(range(LATENTS))
    assert not (pinned & regenerate)
    assert plan["regenerate_latent_frames"] == tuple(sorted(regenerate))
    # Contiguous: one span, which is the shape that was measured.
    assert regenerate == set(range(min(regenerate), max(regenerate) + 1))


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------

def test_an_invalid_clip_length_is_refused_rather_than_snapped():
    """Snapping a clip length means deleting frames the caller said to keep."""
    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError) as error:
        _plan(10, 20, clip_frames=130)
    detail = str(error.value.detail)
    # The trim named must REACH a valid length: 130 - 6 = 124.
    assert "Trim 6 more frame(s)" in detail and "124" in detail, detail
    # Too long is the same rule from the other side; too short says so instead,
    # because no trim reaches the floor from below.
    with pytest.raises(ValidationError) as error:
        _plan(10, 20, clip_frames=400)
    # Derived from the spec, not hardcoded: this assertion was written against
    # the old 345 cap and silently became a statement about a number the model
    # no longer has when the cap was corrected to 362.
    _trim_to_cap = 400 - SPEC.max_frames
    assert f"Trim {_trim_to_cap} more frame(s)" in str(error.value.detail), error.value.detail
    with pytest.raises(ValidationError) as error:
        _plan(10, 20, clip_frames=121)
    assert "shorter" in str(error.value), str(error.value)
    # NEGATIVE CONTROL: the two lengths either side of 130 are accepted, so this
    # is the grid rule and not a blanket refusal.
    assert _plan(10, 20, clip_frames=124)["clip_frames"] == 124
    assert _plan(10, 20, clip_frames=141)["clip_frames"] == 141


def test_a_whole_clip_range_is_refused_because_nothing_is_preserved():
    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError) as error:
        _plan(0, CLIP)
    assert "txt2vid" in str(error.value.detail)
    # NEGATIVE CONTROL: leaving one latent frame is enough.
    spans = latent_frame_spans(SPEC, LATENTS)
    assert _plan(spans[1][0], CLIP)["pinned_latent_frames"] == (0,)


def test_a_whole_clip_range_can_be_planned_for_spatial_token_preservation():
    plan = plan_video_inpaint_span(
        {"regenerate_start_frame": 0, "regenerate_end_frame": CLIP},
        "minimax_h3",
        clip_frames=CLIP,
        allow_full_range=True,
    )
    assert plan["regenerate_latent_frames"] == tuple(range(LATENTS))
    assert plan["pinned_latent_frames"] == ()


@pytest.mark.parametrize("start,end", [(0, 0), (50, 50), (85, 40), (-1, 40), (10, CLIP + 1)])
def test_an_empty_or_out_of_range_request_is_refused(start, end):
    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError):
        _plan(start, end)


def test_an_architecture_without_a_declared_chunking_is_refused():
    """LTX-2.3 declares no latent chunk pattern, so nothing may address it."""
    from api.error_handlers import ValidationError

    assert temporal_spec_for_arch("ltx2").latent_chunk_pattern == ()
    with pytest.raises(ValidationError) as error:
        _plan(10, 20, clip_frames=121, arch="ltx2")
    assert "temporal inpaint" in str(error.value).lower()


# --------------------------------------------------------------------------
# The orchestration: the pin and the paste, each with its own control
# --------------------------------------------------------------------------

GENERATED_VALUE = 7      # what the fake model "generates"
SOURCE_VALUE = 200       # what the source clip carries


def _runner(width=64, height=32):
    """A mixin instance whose generate step returns a constant, capturing kwargs."""
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    captured = {}

    class Runner(MiniMaxH3Mixin):
        minimax_h3_components = {"variant": "fl2va", "audio_sample_rate": 32000}
        current_model_info = {"type": "minimax_h3", "variant": "fl2va"}

        def _generate_minimax_h3(self, params, **kwargs):
            captured.update(kwargs)
            captured["params"] = params
            frames = np.full((int(params["num_frames"]), height, width, 3),
                             GENERATED_VALUE, dtype=np.uint8)
            return frames, None, None, 4242

    return Runner(), captured


def _source_clip(width=64, height=32, frames=CLIP):
    clip = np.full((frames, height, width, 3), SOURCE_VALUE, dtype=np.uint8)
    # Per-frame marks, so a paste that used the wrong frames is visible.
    for index in range(frames):
        clip[index, 0, 0, 0] = index % 251
    return clip


def test_the_preserved_region_is_the_input_pixels_and_the_range_is_not():
    """THE PASTE. Fails if the source pixels are not put back after decode.

    The fake model returns a constant that is nothing like the source, so a
    missing paste shows up as the preserved region carrying that constant.
    """
    runner, _captured = _runner()
    clip = _source_clip()
    params = {"width": 64, "height": 32, "frame_rate": 24.0,
              "regenerate_start_frame": 40, "regenerate_end_frame": 85,
              "inpaint_video_audio_mode": "regenerate"}

    frames, _audio, _rate, seed = runner._generate_vidinpaint_minimax_h3(
        params, clip, 24.0, None)

    start = params["inpaint_video_effective_start_frame"]
    end = params["inpaint_video_effective_end_frame"]
    assert seed == 4242 and frames.shape == clip.shape and frames.dtype == np.uint8
    assert np.array_equal(frames[:start], clip[:start])
    assert np.array_equal(frames[end:], clip[end:])
    # NEGATIVE CONTROL: the regenerated range is NOT pasted over -- otherwise
    # the assertions above would hold for a function that returned the input.
    assert (frames[start:end] == GENERATED_VALUE).all()
    assert params["inpaint_video_preserved_frames"] == CLIP - (end - start)


def test_the_kept_latent_frames_are_pinned_and_the_clip_is_the_pin_source():
    """THE PIN. Fails if the generation is asked for without one."""
    runner, captured = _runner()
    clip = _source_clip()
    params = {"width": 64, "height": 32, "frame_rate": 24.0,
              "regenerate_start_frame": 40, "regenerate_end_frame": 85,
              "inpaint_video_audio_mode": "regenerate"}
    runner._generate_vidinpaint_minimax_h3(params, clip, 24.0, None)

    plan = _plan(40, 85)
    assert captured["pinned_video_frames"] == plan["pinned_latent_frames"]
    assert len(captured["pinned_video_frames"]) > 0
    # The pin's source is the PREPROCESSED clip -- the same pixels the paste
    # uses, or the model would condition on one clip and the output carry
    # another.
    assert np.array_equal(captured["pinned_video_source"], clip)
    assert captured["params"]["num_frames"] == CLIP
    # Not an anchor request: the two claim the same conditioning prefix.
    assert captured.get("keyframes", ()) == ()


def test_the_backend_refuses_a_pin_next_to_keyframes_or_references():
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    class Runner(MiniMaxH3Mixin):
        minimax_h3_components = {"variant": "fl2va"}

    source = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError) as error:
        Runner()._generate_minimax_h3({}, keyframes=(("first", None),),
                                      pinned_video_frames=(0,), pinned_video_source=source)
    assert "prefix" in str(error.value)
    # And a pin with no source clip is refused rather than silently unpinned.
    with pytest.raises(RuntimeError):
        Runner()._generate_minimax_h3({}, pinned_video_frames=(0,))


def test_the_layout_pins_exactly_the_kept_rows_at_the_conditioning_timestep():
    """The route's frame set, through the shipped layout builder and timesteps."""
    import torch

    plan = _plan(40, 85)
    layout = ops.build_packed_layout(
        137, LATENTS, 24, 40, ops.audio_latent_frames(CLIP),
        pinned_video_frames=plan["pinned_latent_frames"])
    rows_per_frame = int(layout["rows_per_frame"])
    assert layout["num_condition_video_rows"] == len(plan["pinned_latent_frames"]) * rows_per_frame

    unique, index = ops.build_row_timesteps(layout, 0.1, 0.1)
    row_time = unique[index]
    pinned_rows = layout["video_indices"][:layout["num_condition_video_rows"]]
    free_rows = layout["video_indices"][layout["num_condition_video_rows"]:]
    assert bool((row_time[pinned_rows] == ops.VISUAL_COND_TIMESTEP).all())
    assert bool((row_time[free_rows] == 0.1).all())
    # The pinned rows are the KEPT frames' rows, read back through the
    # permutation rather than through the same arithmetic that built it.
    order = layout["video_row_order"]
    frame_of_row = torch.arange(LATENTS * rows_per_frame) // rows_per_frame
    pinned_frames = set(frame_of_row[torch.argsort(order)[:len(pinned_rows)]].tolist())
    assert pinned_frames == set(plan["pinned_latent_frames"])


# --------------------------------------------------------------------------
# The same-seed noise contract, structurally
# --------------------------------------------------------------------------

def test_the_pin_substitution_follows_the_draw():
    """A pinned run and a t2va run share their generated frames' noise.

    THE MUTANT THIS EXISTS FOR: draw the video noise inside a branch that knows
    about the pin, or substitute the source rows before the draw. Either keeps
    the output plausible and silently changes what a seed means.
    """
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    source = textwrap.dedent(inspect.getsource(MiniMaxH3Mixin._generate_minimax_h3))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    draws = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "draw_noise"]
    assert len(draws) == 1
    draw = draws[0]

    ancestors, node = [], draw
    while hasattr(node, "parent"):
        node = node.parent
        ancestors.append(node)
    assert not [a for a in ancestors if isinstance(a, (ast.If, ast.IfExp, ast.Try, ast.While))]
    assert "pinned_video_rows" not in {n.id for n in ast.walk(draw) if isinstance(n, ast.Name)}

    # The substitution reads the drawn rows and is guarded by the pin, i.e. it
    # happens after the draw rather than replacing it.
    substitutions = [node for node in ast.walk(tree)
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "pin_video_rows"]
    assert substitutions and all(s.lineno > draw.lineno for s in substitutions)


def test_the_decode_takes_every_video_row_when_the_layout_permuted_them():
    """With pinned frames the conditioning prefix IS clip content."""
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    source = inspect.getsource(MiniMaxH3Mixin._generate_minimax_h3)
    branch = source[source.index("video_row_order = layout[\"video_row_order\"]"):]
    branch = branch[:branch.index("decode_start")]
    assert "video_rows[n_cond_video:]" in branch, "the unpinned path must be unchanged"
    assert "video_rows[video_row_order" in branch, "the pinned path must un-permute"


# --------------------------------------------------------------------------
# Defaults and route shape
# --------------------------------------------------------------------------

def test_the_audio_default_is_per_architecture():
    assert INPAINT_VIDEO_DEFAULTS["inpaint_video_audio_mode"] == "regenerate"
    assert inpaint_video_defaults_for_arch("minimax_h3")["inpaint_video_audio_mode"] == "preserve_input"
    assert inpaint_video_defaults_for_arch(None)["inpaint_video_audio_mode"] == "regenerate"


def test_the_endpoint_advertises_no_clip_length():
    """The output is as long as the trimmed input; there is no length to set."""
    for key in ("num_frames", "total_frames"):
        assert key not in INPAINT_VIDEO_DEFAULTS
        assert key not in inpaint_video_defaults_for_arch("minimax_h3")
    # NEGATIVE CONTROL: the shared video keys the endpoint DOES take are still
    # resolved per architecture, so the pop above is not stripping the overlay.
    assert inpaint_video_defaults_for_arch("minimax_h3")["width"] == 1344


def test_the_route_takes_a_required_range_and_a_sentinel_audio_mode():
    from api.routes import generate_inpaint_video

    parameters = inspect.signature(generate_inpaint_video).parameters
    for key in ("regenerate_start_frame", "regenerate_end_frame"):
        # `Form(...)` records the ellipsis as Pydantic's undefined sentinel,
        # which is the same thing `prompt` carries.
        assert parameters[key].default.default is parameters["prompt"].default.default, (
            f"{key} must be required")
    assert parameters["inpaint_video_audio_mode"].default.default is None, (
        "the audio mode must be a Form(None) sentinel so an omitted field can reach the "
        "per-architecture overlay")
    # No clip-length field of any name.
    assert not {"num_frames", "total_frames"} & set(parameters)


def test_an_inpaint_request_on_a_video_model_names_the_counterpart_route():
    from api.routes import _VIDEO_ROUTE_FOR_IMAGE_ROUTE, _video_route_hint

    assert _VIDEO_ROUTE_FOR_IMAGE_ROUTE["/generate/inpaint"] == "/generate/inpaint/video"
    assert _video_route_hint("/generate/inpaint", "minimax_h3") == "use /generate/inpaint/video"
    # NEGATIVE CONTROL: the route exists but does not serve LTX-2.3, so that
    # architecture must not be pointed at it.
    assert "/generate/inpaint/video" not in _video_route_hint("/generate/inpaint", "ltx2")


def test_ltx2_declares_temporal_inpaint_unsupported():
    from api.arch_capabilities import arch_supports_feature

    assert arch_supports_feature("minimax_h3", "temporal_inpaint")
    assert not arch_supports_feature("ltx2", "temporal_inpaint")
