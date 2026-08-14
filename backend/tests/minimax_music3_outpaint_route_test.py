"""Route-level test for `POST /generate/outpaint/audio`'s MiniMax Music 3 branch.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_outpaint_route_test.py -v

WHY THIS FILE EXISTS
--------------------
`backend/tests/minimax_music3_extend_test.py` exhaustively covers
`MiniMaxMusic3Mixin._generate_audoutpaint_minimax_music3` (the backend
mechanism) in isolation, and `minimax_music3_api_defaults_test.py` covers
`outpaint_audio_defaults_for_arch` (the overlay resolution) in isolation --
but nothing exercised `routes.generate_outpaint_audio` ITSELF: the client-
facing wiring that reads the multipart form, resolves the `Form(None)`
sentinels from the overlay, matches the uploaded clip to a gallery row by
content hash, and threads the result into `pipeline_manager.
generate_aud_outpaint`. The frontend panel always hardcodes
`placement: "extend_forward"` and always uploads a real gallery file, so no
CLIENT path ever reaches an omitted `placement` or a non-matching upload --
which is exactly how `num_inference_steps`/`flow_guidance_scale`'s missing
`ge=1`/`gt=0` bounds went unexercised at the route layer even though the
sibling txt2aud route had the same bound (a zero step count reaches
`FlowMatchEulerDiscreteScheduler.set_timesteps`'s division only after the
whole autoregressive-stage resume replay has already run). This file is the
route-level insurance an out-of-band client (or a future frontend bug) would
actually need.

Most tests below call `routes.generate_outpaint_audio` directly (mirrors
`gallery_delete_test.py`'s `asyncio.run(routes.delete_image(...))` and
`studio_render_e2e_test.py`'s `UploadFile(file=io.BytesIO(...))` pattern)
rather than through a mounted ASGI app: `Form(...)`/`File(...)` defaults are
ordinary Python default values when the function is called directly, so
every branch below (sentinel resolution, gallery-hash refusal, the causal-LM
placement refusal surfacing as a 400) is reachable without spinning up a
client. That does NOT exercise FastAPI's OWN `Form(..., ge=1)` bound
enforcement, which only fires inside the ASGI request-parsing stack -- the
one ASGI-mounted test at the bottom of this file (mirroring `minimax_h3_
hybrid_variant_gate_test.py`'s `_app`/`_post` helpers) covers that
separately, since it is the actual property item 2 fixed.
"""

import asyncio
import hashlib
import io
import os
import sys

import pytest
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import routes  # noqa: E402
from api.error_handlers import GenerationError, ValidationError  # noqa: E402
from api.param_defaults import OUTPAINT_AUDIO_DEFAULTS  # noqa: E402
from database.models import GalleryBase, GeneratedImage  # noqa: E402


class _StubGenerationFailed(Exception):
    """Deliberately raised by the stub `generate_aud_outpaint` instead of
    doing any real generation work, so a test never needs the backend call
    to actually succeed (write a FLAC, a sidecar, a DB row, ...) to assert on
    the WIRING (what params reached the call). NOT asserted on directly --
    `generate_aud_outpaint` runs inside `_run_generation_in_executor`
    (a thread-pool executor), and the route's own outermost `except
    Exception as e: raise GenerationError(...)` re-wraps whatever surfaces
    from there, so the ORIGINAL exception object/type does not survive back
    to the caller. `_StubPipelineManager.captured_params` (set before the
    raise) is what a test reads instead."""


class _StubPipelineManager:
    """Same shape as `minimax_h3_hybrid_variant_gate_test.py`'s stub, plus
    the extra attributes this route's non-refusal path reads before it
    reaches the generation call (`current_pipeline_kind` for the VRAM-peak
    lookup, `reset_cancel_flag`)."""

    def __init__(self):
        self.is_minimax_music3_model = True
        self.is_acestep_model = False
        self.is_minimax_h3_model = False
        self.current_model_info = {"type": "minimax_music3"}
        self.current_pipeline_kind = "minimax_music3"
        self.captured_params = None

    def reset_cancel_flag(self):
        pass

    def generate_aud_outpaint(self, params, reference_audio_source, progress_callback=None):
        self.captured_params = dict(params)
        raise _StubGenerationFailed("stub: no real generation performed")


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/minimax_music3_outpaint_route_test.db")
    GalleryBase.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _seed_matching_row(session, content: bytes, filename="txt2aud_20260101_000000_1.flac"):
    row = GeneratedImage(
        filename=filename,
        prompt="p",
        generation_type="txt2aud",
        parameters={"is_audio": True},
        image_hash=hashlib.sha256(content).hexdigest(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _call(monkeypatch, tmp_path, manager, session, content: bytes, **overrides):
    monkeypatch.setattr(routes, "pipeline_manager", manager)
    monkeypatch.setattr(routes.settings, "outputs_dir", str(tmp_path))
    kwargs = dict(
        prompt=OUTPAINT_AUDIO_DEFAULTS["prompt"],
        lyrics=OUTPAINT_AUDIO_DEFAULTS["lyrics"],
        seed=OUTPAINT_AUDIO_DEFAULTS["seed"],
        inference_steps=OUTPAINT_AUDIO_DEFAULTS["inference_steps"],
        guidance_scale=OUTPAINT_AUDIO_DEFAULTS["guidance_scale"],
        shift=OUTPAINT_AUDIO_DEFAULTS["shift"],
        vocal_language=OUTPAINT_AUDIO_DEFAULTS["vocal_language"],
        total_duration=OUTPAINT_AUDIO_DEFAULTS["total_duration"],
        input_offset_sec=OUTPAINT_AUDIO_DEFAULTS["input_offset_sec"],
        input_trim_start_sec=OUTPAINT_AUDIO_DEFAULTS["input_trim_start_sec"],
        input_trim_end_sec=OUTPAINT_AUDIO_DEFAULTS["input_trim_end_sec"],
        placement=None,
        extend_duration_sec=None,
        num_inference_steps=None,
        flow_guidance_scale=None,
        loras="[]",
        unet_quantization=OUTPAINT_AUDIO_DEFAULTS["unet_quantization"],
        quantized_gemm_mode=OUTPAINT_AUDIO_DEFAULTS["quantized_gemm_mode"],
        reference_audio=UploadFile(file=io.BytesIO(content), filename="song.flac"),
        db=session,
    )
    kwargs.update(overrides)
    return asyncio.run(routes.generate_outpaint_audio(**kwargs))


# ---------------------------------------------------------------------------
# Gallery-hash refusal (no GPU/executor work reached -- raised before the
# route's own try: block).
# ---------------------------------------------------------------------------

def test_upload_with_no_matching_gallery_row_is_refused(tmp_path, monkeypatch):
    session = _session(tmp_path)
    manager = _StubPipelineManager()
    with pytest.raises(ValidationError) as exc_info:
        _call(monkeypatch, tmp_path, manager, session, b"not a real gallery file",
              placement="extend_forward")
    assert exc_info.value.status_code == 400
    assert "gallery" in exc_info.value.message.lower()
    session.close()


# ---------------------------------------------------------------------------
# Sentinel resolution: an omitted extend_duration_sec/num_inference_steps/
# flow_guidance_scale reaches `generate_aud_outpaint` filled from
# OUTPAINT_AUDIO_ARCH_OVERLAYS["minimax_music3"], not as None.
# ---------------------------------------------------------------------------

def test_omitted_sentinels_resolve_from_the_overlay_before_reaching_the_backend(tmp_path, monkeypatch):
    content = b"a fake but stable song file"
    session = _session(tmp_path)
    _seed_matching_row(session, content)
    manager = _StubPipelineManager()

    with pytest.raises(GenerationError):
        _call(monkeypatch, tmp_path, manager, session, content, placement="extend_forward")

    params = manager.captured_params
    assert params is not None, "generate_aud_outpaint was never reached"
    assert params["extend_duration_sec"] == 30.0
    assert params["num_inference_steps"] == 30
    assert params["flow_guidance_scale"] == 1.7
    assert params["placement"] == "extend_forward"
    session.close()


def test_explicitly_supplied_sentinels_are_not_overridden_by_the_overlay(tmp_path, monkeypatch):
    content = b"a fake but stable song file, take 2"
    session = _session(tmp_path)
    _seed_matching_row(session, content)
    manager = _StubPipelineManager()

    with pytest.raises(GenerationError):
        _call(monkeypatch, tmp_path, manager, session, content, placement="extend_forward",
              extend_duration_sec=12.5, num_inference_steps=7, flow_guidance_scale=2.2)

    params = manager.captured_params
    assert params is not None, "generate_aud_outpaint was never reached"
    assert params["extend_duration_sec"] == 12.5
    assert params["num_inference_steps"] == 7
    assert params["flow_guidance_scale"] == 2.2
    session.close()


# ---------------------------------------------------------------------------
# Omitted `placement` reaches the backend as `None` (never silently filled
# with "extend_forward"), and the backend's own causal-LM refusal for it
# surfaces as a 400 through this route's exception handling -- not a 500.
# ---------------------------------------------------------------------------

def test_omitted_placement_reaches_the_backend_as_none_and_its_refusal_is_a_400(tmp_path, monkeypatch):
    content = b"a fake but stable song file, take 3"
    session = _session(tmp_path)
    _seed_matching_row(session, content)

    class _RefusingStubPipelineManager(_StubPipelineManager):
        def generate_aud_outpaint(self, params, reference_audio_source, progress_callback=None):
            # Mirrors `_generate_audoutpaint_minimax_music3`'s own real
            # refusal shape for an omitted/unsupported placement -- this
            # test is about the ROUTE's wiring (does `placement=None` really
            # reach here, and does the raised ValidationError really become
            # a 400 through this route), not a second copy of the backend's
            # own placement-refusal test (`minimax_music3_extend_test.py`
            # already owns that).
            if params.get("placement") != "extend_forward":
                raise ValidationError(
                    "MiniMax Music 3 extend only supports placement='extend_forward'",
                    detail=f"got {params.get('placement')!r}; the autoregressive stage is causal.",
                )
            self.captured_params = dict(params)  # pragma: no cover -- not reached here
            raise _StubGenerationFailed("stub: no real generation performed")  # pragma: no cover

    manager = _RefusingStubPipelineManager()
    with pytest.raises(ValidationError) as exc_info:
        _call(monkeypatch, tmp_path, manager, session, content, placement=None)

    assert exc_info.value.status_code == 400
    assert "extend_forward" in exc_info.value.message
    session.close()


# ---------------------------------------------------------------------------
# FastAPI's OWN `Form(..., ge=1)`/`Form(..., gt=0)` enforcement for
# `num_inference_steps`/`flow_guidance_scale` -- item 2's actual fix. This
# only fires at the ASGI request-parsing boundary (a direct Python call, like
# every test above, bypasses it entirely: a keyword argument overrides the
# `Form(...)` default object outright), so this one test is mounted for real
# through an ASGI app, mirroring `minimax_h3_hybrid_variant_gate_test.py`'s
# `_app`/`_post` helpers.
#
# FastAPI's OWN request-body validation errors are HTTP 422 (`Request
# ValidationError`, registered separately from this repo's `ErrorResponse`
# 4xx/5xx family), not 400 -- what matters for the bug these bounds close is
# NOT the exact status code, it is that it is a clean, immediate 4xx returned
# from request PARSING, never a 500 with a traceback after the entire
# autoregressive-stage resume replay has already run for minutes on the GPU.
# ---------------------------------------------------------------------------

def _app(monkeypatch, manager, session):
    from fastapi import FastAPI

    from api.error_handlers import register_error_handlers

    monkeypatch.setattr(routes, "pipeline_manager", manager)
    app = FastAPI()
    register_error_handlers(app)
    app.post("/generate/outpaint/audio")(routes.generate_outpaint_audio)
    app.dependency_overrides[routes.get_gallery_db] = lambda: session
    return app


def _post_multipart(app, content: bytes, **data):
    import httpx

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/generate/outpaint/audio",
                data={"placement": "extend_forward", **data},
                files={"reference_audio": ("song.flac", content, "audio/flac")},
            )
            return response.status_code, response.json()

    return asyncio.run(run())


def test_zero_num_inference_steps_is_rejected_at_request_parsing_not_a_500_after_generation(tmp_path, monkeypatch):
    content = b"a fake but stable song file, take 4"
    session = _session(tmp_path)
    _seed_matching_row(session, content)
    monkeypatch.setattr(routes.settings, "outputs_dir", str(tmp_path))
    manager = _StubPipelineManager()
    app = _app(monkeypatch, manager, session)

    status_code, body = _post_multipart(app, content, num_inference_steps="0")

    assert status_code == 422, body
    assert manager.captured_params is None, "generate_aud_outpaint must never be reached for an invalid step count"
    session.close()


def test_negative_flow_guidance_scale_is_rejected_at_request_parsing_not_a_500_after_generation(tmp_path, monkeypatch):
    content = b"a fake but stable song file, take 5"
    session = _session(tmp_path)
    _seed_matching_row(session, content)
    monkeypatch.setattr(routes.settings, "outputs_dir", str(tmp_path))
    manager = _StubPipelineManager()
    app = _app(monkeypatch, manager, session)

    status_code, body = _post_multipart(app, content, flow_guidance_scale="-1.0")

    assert status_code == 422, body
    assert manager.captured_params is None, "generate_aud_outpaint must never be reached for an invalid guidance scale"
    session.close()
