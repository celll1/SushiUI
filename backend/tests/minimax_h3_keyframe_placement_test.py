"""Keyframe placement: what a request means, and what reaches the layout.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_keyframe_placement_test.py -v

WHY THIS FILE EXISTS
--------------------
`POST /generate/img2vid` grew three fields (`input_image_frame_index`,
`keyframe_images`, `keyframe_frame_indices`) whose whole content is arithmetic
the client cannot do:

* `-1` resolves against the clip length AFTER the server snapped it to the
  `17n + 5` grid. A request that means "the last frame" is only correct because
  the resolution happens on this side of the snap;
* frame 0 and the last frame resolve to the `"first"` / `"last"` STRING anchors,
  which is what makes every request expressible before this phase produce a
  byte-identical layout (`minimax_h3_layout_test` pins those layouts; this file
  pins that the API still reaches them);
* anchors are packed in ascending frame order, so upload order is not part of
  the request.

Each block below has a negative control: an assertion that fails if the
behaviour is replaced by the plausible thing that is not it.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.error_handlers import ValidationError  # noqa: E402
from api.generation_utils import plan_keyframe_placements  # noqa: E402
from api.param_defaults import IMG2VID_DEFAULTS  # noqa: E402


def _anchors(plan):
    return [entry["anchor"] for entry in plan["anchors"]]


def _frames(plan):
    return [entry["frame"] for entry in plan["anchors"]]


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def test_the_legacy_request_still_resolves_to_the_two_string_anchors():
    """`image` + `last_frame_image` is `("first", "last")`, as it always was.

    NEGATIVE CONTROL: not `(0, 123)`. The two integers place the anchors at the
    same float32 positions but take the integer branch of
    `_anchor_rotary_time`, whose `"last"` differs from `(5/3)*(T-1)` in the last
    float64 ulp -- the exact edit `minimax_h3_layout_test` exists to catch. If
    this resolver starts emitting integers for the ends, that guard is bypassed
    from the API side.
    """
    plan = plan_keyframe_placements([("image", 0), ("last_frame_image", -1)], 124)
    assert _anchors(plan) == ["first", "last"]
    assert _frames(plan) == [0, 123]
    assert plan["undocumented"] == []
    assert not any(isinstance(anchor, int) for anchor in _anchors(plan))


def test_a_lone_first_frame_request_is_unchanged():
    plan = plan_keyframe_placements([("image", 0)], 124)
    assert _anchors(plan) == ["first"]
    assert plan["undocumented"] == []


@pytest.mark.parametrize("num_frames", [124, 141, 345])
def test_minus_one_resolves_against_the_snapped_clip_length(num_frames):
    """THE reason the sentinel exists.

    `num_frames` is snapped server-side, so a client that computed
    `num_frames - 1` itself would name the wrong frame whenever its request was
    snapped. Checked at three grid lengths, and with the negative control that
    the resolution is not a constant.
    """
    plan = plan_keyframe_placements([("image", 0), ("keyframe_images[0]", -1)], num_frames)
    assert _frames(plan) == [0, num_frames - 1]
    assert _anchors(plan) == ["first", "last"]


def test_a_request_snapped_from_130_to_141_lands_its_end_anchor_on_140():
    """The concrete case: the client asked for 130 frames, the server snapped.

    130 is off the 17n+5 grid; `MINIMAX_H3_TEMPORAL.snap_length` rounds it up to
    141, so a client that had resolved -1 itself would have asked for frame 129
    -- an INTERIOR frame of the clip that actually gets generated, silently
    turning an end anchor into a mid one 11 frames early.
    """
    from core.models.components.wiring import MINIMAX_H3_TEMPORAL

    snapped = MINIMAX_H3_TEMPORAL.snap_length(130)
    assert snapped == 141
    plan = plan_keyframe_placements([("image", 0), ("last_frame_image", -1)], snapped)
    assert _frames(plan) == [0, 140]
    # The control: the index the client could have computed is NOT the answer.
    assert _frames(plan)[1] != 130 - 1


def test_an_intermediate_index_stays_an_integer_anchor():
    plan = plan_keyframe_placements([("image", 60)], 124)
    assert _anchors(plan) == [60]
    assert _frames(plan) == [60]
    # ... and it is flagged as outside the model card, once.
    assert len(plan["undocumented"]) == 1
    assert "60" in plan["undocumented"][0]


def test_the_last_index_sent_explicitly_is_the_last_anchor():
    """An explicit `T-1` means the same frame as `-1`, so it takes the same path."""
    plan = plan_keyframe_placements([("keyframe_images[0]", 123)], 124)
    assert _anchors(plan) == ["last"]
    # NEGATIVE CONTROL: one frame earlier is NOT the string anchor.
    assert _anchors(plan_keyframe_placements([("keyframe_images[0]", 122)], 124)) == [122]


def test_anchors_are_packed_in_ascending_frame_order_not_upload_order():
    plan = plan_keyframe_placements([
        ("image", 60),
        ("keyframe_images[0]", 0),
        ("keyframe_images[1]", 90),
        ("last_frame_image", -1),
    ], 124)
    assert _frames(plan) == [0, 60, 90, 123]
    assert [entry["source"] for entry in plan["anchors"]] == [
        "keyframe_images[0]", "image", "keyframe_images[1]", "last_frame_image"]
    assert _anchors(plan) == ["first", 60, 90, "last"]


def test_reordering_the_uploads_is_the_same_request():
    """Upload order is not semantic here (it IS on /generate/ref2vid)."""
    forward = plan_keyframe_placements(
        [("image", 0), ("keyframe_images[0]", 60), ("keyframe_images[1]", 90)], 124)
    reversed_ = plan_keyframe_placements(
        [("image", 0), ("keyframe_images[0]", 90), ("keyframe_images[1]", 60)], 124)
    assert _frames(forward) == _frames(reversed_)
    assert _anchors(forward) == _anchors(reversed_)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------

@pytest.mark.parametrize("index", [124, 200, -2, -5])
def test_an_index_outside_the_clip_is_refused_with_the_frame_math(index):
    with pytest.raises(ValidationError) as excinfo:
        plan_keyframe_placements([("image", index)], 124)
    message = f"{excinfo.value} {getattr(excinfo.value, 'detail', '')}"
    assert "124" in message and "0..123" in message


def test_two_anchors_on_one_frame_are_refused_naming_both():
    with pytest.raises(ValidationError) as excinfo:
        plan_keyframe_placements(
            [("image", 123), ("last_frame_image", -1)], 124)
    message = f"{excinfo.value} {getattr(excinfo.value, 'detail', '')}"
    assert "last_frame_image" in message and "image" in message and "123" in message


def test_a_zero_length_clip_is_refused_rather_than_placing_on_frame_minus_one():
    with pytest.raises(ValidationError):
        plan_keyframe_placements([("image", 0)], 0)


# --------------------------------------------------------------------------
# The undocumented-shape flag
# --------------------------------------------------------------------------

def test_only_shapes_outside_the_model_card_are_flagged():
    """Zero, one and two end anchors are documented; nothing else is claimed."""
    for requests in ([("image", 0)],
                     [("image", 0), ("last_frame_image", -1)],
                     [("image", -1)]):
        assert plan_keyframe_placements(requests, 124)["undocumented"] == []
    # Three anchors, all at documented POSITIONS, is still three anchors.
    flagged = plan_keyframe_placements(
        [("image", 0), ("keyframe_images[0]", 60), ("last_frame_image", -1)], 124)
    assert len(flagged["undocumented"]) == 2      # intermediate frame + count


# --------------------------------------------------------------------------
# Threading: the route and the sender must carry every field
# --------------------------------------------------------------------------

def test_the_defaults_live_in_param_defaults():
    assert IMG2VID_DEFAULTS["input_image_frame_index"] == 0
    assert IMG2VID_DEFAULTS["keyframe_images"] is None
    assert IMG2VID_DEFAULTS["keyframe_frame_indices"] is None


def test_the_route_declares_the_fields_and_reads_their_defaults_from_the_map():
    """Failure pattern 2 (missing `Form(...)`) and the hardcoded-default trap.

    A literal default here would be a second source of truth; the assertion is
    on the SOURCE so it catches `Form(0)` as well as a missing parameter.
    """
    from api import routes

    signature = inspect.signature(routes.generate_img2vid)
    for name in ("input_image_frame_index", "keyframe_images", "keyframe_frame_indices"):
        assert name in signature.parameters, name

    source = inspect.getsource(routes.generate_img2vid)
    assert 'Form(IMG2VID_DEFAULTS["input_image_frame_index"])' in source
    assert "Form(0)" not in source


def test_the_route_puts_the_fields_in_the_params_dict():
    """Failure pattern 3: received as a Form field, never reaching the backend."""
    from api import routes

    source = inspect.getsource(routes.generate_img2vid)
    assert '"input_image_frame_index": input_image_frame_index,' in source
    assert '"keyframe_images":' in source
    assert '"keyframe_frame_indices":' in source
    # ... and the resolved plan really is handed to the generator.
    assert "keyframes=keyframe_plan" in source


def test_the_pipeline_and_the_backend_accept_the_resolved_plan():
    from core.pipeline import DiffusionPipelineManager
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    assert "keyframes" in inspect.signature(
        DiffusionPipelineManager.generate_img2vid).parameters
    assert "keyframes" in inspect.signature(
        MiniMaxH3Mixin._generate_img2vid_minimax_h3).parameters


def test_the_frontend_sender_appends_both_lists_in_one_loop():
    """Failure pattern 1: a field in the JS object that never reaches the wire.

    The two lists are POSITIONAL, so they are checked together: appending the
    images without their indices (or vice versa) is a 400 the user would see as
    "the second keyframe was ignored".
    """
    api_ts = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "frontend", "src", "utils", "api.ts")
    with open(api_ts, encoding="utf-8") as handle:
        source = handle.read()
    sender = source.split("export const generateImg2Vid")[1].split("export const ")[0]
    assert 'formData.append("input_image_frame_index"' in sender
    assert 'formData.append("keyframe_images"' in sender
    assert 'formData.append("keyframe_frame_indices"' in sender


def test_the_panel_carries_the_fields_into_the_queued_item():
    """Failure patterns 4 and 6: the dequeue object, and DEFAULT_PARAMS.

    The queued item's params object is what `generateImg2Vid` is handed, so a
    field left in the panel's `params` alone never reaches the request.
    """
    panel = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "frontend", "src", "components", "generation", "Img2ImgPanel.tsx")
    with open(panel, encoding="utf-8") as handle:
        source = handle.read()
    assert "input_image_frame_index: 0," in source          # DEFAULT_PARAMS
    assert "keyframes: []," in source                       # DEFAULT_PARAMS
    assert "input_image_frame_index: params.input_image_frame_index ?? 0," in source
    assert "keyframes: params.keyframes ?? []," in source


def test_the_capability_key_exists_and_gates_the_right_architecture():
    from api.arch_capabilities import FEATURE_PARAMS, arch_supports_feature

    assert set(FEATURE_PARAMS["keyframe_placement"]) == {
        "input_image_frame_index", "keyframe_images", "keyframe_frame_indices"}
    assert arch_supports_feature("minimax_h3", "keyframe_placement")
    assert not arch_supports_feature("ltx2", "keyframe_placement")
    # It is a DIFFERENT claim from the last-frame slot, and stays so.
    assert not arch_supports_feature("ltx2", "last_frame_image")


def test_openapi_documents_the_three_fields_on_the_img2vid_schema():
    import yaml

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, "openapi.yaml"), encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    schema = spec["components"]["schemas"]["Img2VidRequest"]["allOf"][1]["properties"]
    for field in ("input_image_frame_index", "keyframe_images", "keyframe_frame_indices"):
        assert field in schema, field
        assert schema[field].get("description")
    # The alias is documented AS an alias rather than as a separate mechanism.
    assert "-1" in schema["last_frame_image"]["description"]


# --------------------------------------------------------------------------
# The geometry rule (see `_minimax_h3_fit_keyframe`)
# --------------------------------------------------------------------------

def test_only_a_frame_zero_anchor_is_stretched():
    """The rule follows the FRAME, not the position in the list."""
    from PIL import Image as PILImage
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    fit = MiniMaxH3Mixin._minimax_h3_fit_keyframe
    width, height = 640, 384
    square = PILImage.new("RGB", (512, 512))
    for y in range(512):
        for x in range(0, 512, 64):
            square.putpixel((x, y), (y // 2, y // 2, y // 2))

    stretched = fit(square, width, height, "first")
    assert list(fit(square, width, height, 0).getdata()) == list(stretched.getdata())
    for follower in ("last", 1, 60, 123):
        cropped = fit(square, width, height, follower)
        assert list(cropped.getdata()) != list(stretched.getdata()), follower


def test_the_outpaint_paths_own_anchors_are_unaffected_by_the_rule_change():
    """The one live path that feeds a LONE non-zero anchor, pinned.

    `_generate_vidoutpaint_minimax_h3`'s `extend_backward` sends
    `("last", head[0])` as its only anchor. Under the old rule ("the packed-first
    anchor is stretched") that image was stretched; under the new one it is a
    follower. It cannot matter, and this asserts WHY rather than trusting it:
    every frame the outpaint path sends comes out of
    `center_crop_resize_frames(..., width, height)`, so it is already exactly the
    canvas size and returns at the identity check before either branch runs.
    """
    import numpy as np
    from PIL import Image as PILImage
    from core.inference.outpaint_utils import center_crop_resize_frames
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    fit = MiniMaxH3Mixin._minimax_h3_fit_keyframe
    width, height = 640, 384
    source = (np.random.default_rng(0).integers(0, 255, (3, 300, 700, 3))).astype(np.uint8)
    head = center_crop_resize_frames(source, width, height)
    assert head.shape[1:3] == (height, width)

    for frame_index in (0, -1):
        image = PILImage.fromarray(head[frame_index])
        old_rule = fit(image, width, height, 0)          # packed-first == stretch
        new_rule_first = fit(image, width, height, "first")
        new_rule_last = fit(image, width, height, "last")
        assert list(new_rule_first.getdata()) == list(old_rule.getdata())
        assert list(new_rule_last.getdata()) == list(old_rule.getdata())
