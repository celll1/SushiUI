"""A modality refusal must name the endpoint the CALLER was trying to use.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/modality_refusal_endpoint_hint_test.py -v

WHY THIS FILE EXISTS
--------------------
The four still-image routes refuse a request when a video (LTX-2.3,
MiniMax-H3) or audio (ACE-Step) model is loaded. That refusal used to suggest
``/generate/txt2vid`` no matter which route was hit, so a caller that had just
uploaded an input image to ``/generate/img2img`` was told to use the
text-only endpoint: wrong twice over, since txt2vid neither accepts nor uses
the image. The suggestion now follows the request shape --

    /generate/txt2img  -> /generate/txt2vid   / /generate/txt2aud
    /generate/img2img  -> /generate/img2vid   / /generate/aud2aud
    /generate/outpaint -> /generate/outpaint/video / /generate/outpaint/audio
    /generate/inpaint  -> (no video counterpart; both alternatives named)

-- and, just as importantly, each route actually PASSES its own endpoint. A
correct mapping wired to a default argument would still refuse img2img with
"use /generate/txt2vid", so the wiring is asserted from the route source, and
every assertion has a negative control proving it depends on the thing it
claims to test.
"""

import inspect
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import routes  # noqa: E402
from api.error_handlers import ValidationError as CustomValidationError  # noqa: E402


@pytest.fixture
def no_model(monkeypatch):
    """Neither a video nor an audio model is loaded (the passing baseline)."""
    for flag in ("is_ltx2_model", "is_minimax_h3_model", "is_acestep_model"):
        monkeypatch.setattr(routes.pipeline_manager, flag, False, raising=False)


@pytest.fixture
def h3_loaded(monkeypatch, no_model):
    monkeypatch.setattr(routes.pipeline_manager, "is_minimax_h3_model", True, raising=False)


@pytest.fixture
def ltx2_loaded(monkeypatch, no_model):
    monkeypatch.setattr(routes.pipeline_manager, "is_ltx2_model", True, raising=False)


@pytest.fixture
def acestep_loaded(monkeypatch, no_model):
    monkeypatch.setattr(routes.pipeline_manager, "is_acestep_model", True, raising=False)


# --------------------------------------------------------------------------
# NEGATIVE CONTROL: with no video/audio model loaded nothing is refused, so a
# refusal below is caused by the loaded-model flag and not by the call itself.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("endpoint", [
    "/generate/txt2img", "/generate/img2img", "/generate/inpaint", "/generate/outpaint",
])
def test_image_model_is_not_refused(no_model, endpoint):
    routes._reject_if_video_model(endpoint)
    routes._reject_if_audio_model(endpoint)


@pytest.mark.parametrize("arch_fixture", ["h3_loaded", "ltx2_loaded"])
@pytest.mark.parametrize("endpoint,expected", [
    ("/generate/txt2img", "use /generate/txt2vid"),
    ("/generate/img2img", "use /generate/img2vid"),
    ("/generate/outpaint", "use /generate/outpaint/video"),
])
def test_video_refusal_names_the_matching_video_route(request, arch_fixture, endpoint, expected):
    request.getfixturevalue(arch_fixture)
    with pytest.raises(CustomValidationError) as excinfo:
        routes._reject_if_video_model(endpoint)
    message = str(excinfo.value)
    assert expected in message, message
    # The endpoint the caller actually hit is named in the detail, so the
    # message says which request was refused, not only what to do instead.
    assert endpoint in str(getattr(excinfo.value, "detail", "")), excinfo.value.detail


def test_img2img_refusal_does_not_suggest_the_text_only_route(h3_loaded):
    """The exact regression: an image-carrying request pointed at txt2vid."""
    with pytest.raises(CustomValidationError) as excinfo:
        routes._reject_if_video_model("/generate/img2img")
    assert "/generate/txt2vid" not in str(excinfo.value)


def test_inpaint_has_no_video_counterpart_but_still_gets_alternatives(h3_loaded):
    """There is no video inpainting route; name what CAN be done instead."""
    with pytest.raises(CustomValidationError) as excinfo:
        routes._reject_if_video_model("/generate/inpaint")
    message = str(excinfo.value)
    assert "/generate/img2vid" in message, message
    assert "/generate/outpaint/video" in message, message


@pytest.mark.parametrize("endpoint,expected", [
    ("/generate/txt2img", "use /generate/txt2aud"),
    ("/generate/img2img", "use /generate/aud2aud"),
    ("/generate/inpaint", "use /generate/aud2aud"),
    ("/generate/outpaint", "use /generate/outpaint/audio"),
])
def test_audio_refusal_names_the_matching_audio_route(acestep_loaded, endpoint, expected):
    with pytest.raises(CustomValidationError) as excinfo:
        routes._reject_if_audio_model(endpoint)
    assert expected in str(excinfo.value), str(excinfo.value)


# --------------------------------------------------------------------------
# Wiring: a correct mapping is useless if the routes never pass their endpoint.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("func_name,endpoint", [
    ("generate_txt2img", "/generate/txt2img"),
    ("generate_img2img", "/generate/img2img"),
    ("generate_inpaint", "/generate/inpaint"),
    ("generate_outpaint", "/generate/outpaint"),
])
def test_route_passes_its_own_endpoint_to_both_guards(func_name, endpoint):
    source = inspect.getsource(getattr(routes, func_name))
    for guard in ("_reject_if_video_model", "_reject_if_audio_model"):
        calls = re.findall(rf"{guard}\((.*?)\)", source)
        assert calls, f"{func_name} does not call {guard}"
        assert calls[0].strip().strip("\"'") == endpoint, (
            f"{func_name} calls {guard}({calls[0]}), expected the literal {endpoint!r}"
        )
