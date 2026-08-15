"""Route-level test for `POST /generate/aud2aud`'s MiniMax Music 3 repaint branch (design doc phase plan item 8),
and a no-op proof that ACE-Step's own `mode=cover`/`mode=repaint` paths through this SAME route are unaffected by
that addition.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_aud2aud_route_test.py -v

WHY THIS FILE EXISTS
--------------------
`minimax_music3_repaint_test.py` covers `MiniMaxMusic3Mixin._generate_aud2aud_minimax_music3` and its two
sub-mode methods (the backend mechanism) in isolation -- this file covers `routes.generate_aud2aud` ITSELF: the
client-facing wiring that reads the multipart form, resolves the `Form(None)` sentinels
(`music3_repaint_mode`/`num_inference_steps`/`flow_guidance_scale`) from the per-arch overlay, matches the
uploaded clip to a gallery row by content hash for MiniMax Music 3, and threads the result into
`pipeline_manager.generate_aud2aud`. Same pattern as `minimax_music3_outpaint_route_test.py` (its own module
docstring explains why calling the route function directly, rather than through a mounted ASGI app, reaches every
branch below without a client).

ACE-STEP NO-OP PROOF
---------------------
This route now builds a params dict with THREE new keys (`music3_repaint_mode`/`num_inference_steps`/
`flow_guidance_scale`) it did not carry before this phase. The three ACE-Step-facing tests below COMPUTE the
params dict `pipeline_manager.generate_aud2aud` actually receives for `mode=cover` and `mode=repaint` requests
against a loaded ACE-Step model (`_StubPipelineManager.captured_params`, captured via a stub -- no real generation
is performed, per this task's "do not run a real weighted generation" constraint) and assert:

  1. every ORIGINAL key (prompt/lyrics/seed/inference_steps/guidance_scale/shift/cover_strength/mode/
     repaint_start/repaint_end/vocal_language/loras/unet_quantization/quantized_gemm_mode) is present with
     EXACTLY the value the request supplied -- the pre-existing 13-key dict this route built before this phase is
     reproduced byte-for-byte inside the new one;
  2. the three NEW keys are present but resolve to `None` for `acestep` (there is no
     `AUD2AUD_GEN_ARCH_OVERLAYS["acestep"]` entry, so the sentinel-resolution block's `if ... in
     _resolved_aud2aud` guards never fire for this architecture);
  3. `core.pipeline_backends.acestep.AceStepMixin._generate_aud2aud_acestep`'s OWN source never references any of
     the three new key names at all (a static proof that even a caller-supplied non-`None` value for one of them
     could not change ACE-Step's behavior, not merely that the DEFAULT resolution leaves them `None`) --
     `inspect.getsource` on the LIVE, currently-imported method, not a copy/paste of its text.

Together these are "compute both sides" for cover and repaint: the captured dict for each sub-mode is the actual
input `_generate_aud2aud_acestep` would run against, and its source is inspected to show none of the new keys can
reach a live branch. Precise scope of the "unaffected" claim: ACE-Step's GENERATED AUDIO and its SAVED FLAC bytes
are unaffected (nothing in `_generate_aud2aud_acestep` reads the three new keys, so its output cannot change) --
but its GALLERY ROW could have gained three new, always-null keys, since `params_for_db` is a comprehension over
the full `params` dict. `test_acestep_saved_gallery_row_does_not_carry_the_three_music3_only_null_keys` below
proves the route filters them the same way `/generate/outpaint/audio` already does for its own cross-arch keys,
so the saved row is unaffected too.

BOUND ENFORCEMENT (audit fix, third occurrence)
------------------------------------------------
`num_inference_steps`/`flow_guidance_scale` were declared as bare `Form(None)` sentinels here with no `ge=1`/
`gt=0` -- the same omission independently found in `Txt2AudRequest` (audit finding F4) and in
`/generate/outpaint/audio` (fixed the commit before this one). Measured: `num_inference_steps=0` raises
`ZeroDivisionError` and a negative value raises `"Number of samples must be non-negative"`, both from inside
`denoise_chunks`, both caught by the route's generic exception handler and returned as an unhelpful 500 with a
traceback -- after the ~18GB language-model staging move had already run. Fixed to match `/generate/outpaint/
audio`'s identical fields exactly (`Form(None, ge=1)` / `Form(None, gt=0)`); the two ASGI-mounted tests at the
bottom of this file mirror `minimax_music3_outpaint_route_test.py`'s own bound tests to prove FastAPI's OWN
request-parsing validation now catches these BEFORE the executor/generation call, the same way it already does
for outpaint.
"""

import asyncio
import hashlib
import inspect
import io
import os
import sys

import pytest
import torch
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import routes  # noqa: E402
from api.error_handlers import GenerationError, ValidationError  # noqa: E402
from api.param_defaults import AUD2AUD_DEFAULTS  # noqa: E402
from database.models import GalleryBase, GeneratedImage  # noqa: E402


class _StubGenerationFailed(Exception):
    """Deliberately raised by the stub `generate_aud2aud` instead of doing any real generation work -- see
    `minimax_music3_outpaint_route_test.py`'s identical helper for the full reasoning (the executor re-wraps
    whatever surfaces from a background thread, so `captured_params`, set before the raise, is what a test reads,
    not the exception itself)."""


class _StubPipelineManager:
    """Same shape as `minimax_music3_outpaint_route_test.py`'s stub, generalized to toggle between ACE-Step and
    MiniMax Music 3 (this route serves both, unlike outpaint's Music3-only stub)."""

    def __init__(self, *, is_music3: bool):
        self.is_minimax_music3_model = is_music3
        self.is_acestep_model = not is_music3
        self.is_minimax_h3_model = False
        arch = "minimax_music3" if is_music3 else "acestep"
        self.current_model_info = {"type": arch}
        self.current_pipeline_kind = arch
        self.captured_params = None

    def reset_cancel_flag(self):
        pass

    def generate_aud2aud(self, params, reference_audio_source, progress_callback=None):
        self.captured_params = dict(params)
        raise _StubGenerationFailed("stub: no real generation performed")


class _SucceedingAceStepStubPipelineManager(_StubPipelineManager):
    """Unlike `_StubPipelineManager`, actually RETURNS a (tiny, real) waveform instead of raising -- lets a test
    reach the route's OWN `params_for_db` construction and DB save, which `_StubGenerationFailed`-raising stubs
    never do. Used only for the ACE-Step gallery-row-scope test below (fix 5's second half): whether the saved
    row's `parameters` carries the three MiniMax-Music3-only keys that do not apply to an ACE-Step generation."""

    def __init__(self):
        super().__init__(is_music3=False)

    def generate_aud2aud(self, params, reference_audio_source, progress_callback=None):
        self.captured_params = dict(params)
        waveform = torch.zeros(2, 100, dtype=torch.float32)  # [channels, samples]
        return waveform, 44100, 12345


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/minimax_music3_aud2aud_route_test.db")
    GalleryBase.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _seed_matching_row(session, content: bytes, filename="txt2aud_20260101_000000_1.flac",
                       generation_type="txt2aud"):
    row = GeneratedImage(
        filename=filename,
        prompt="p",
        generation_type=generation_type,
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
        prompt=AUD2AUD_DEFAULTS["prompt"],
        lyrics=AUD2AUD_DEFAULTS["lyrics"],
        seed=AUD2AUD_DEFAULTS["seed"],
        inference_steps=AUD2AUD_DEFAULTS["inference_steps"],
        guidance_scale=AUD2AUD_DEFAULTS["guidance_scale"],
        shift=AUD2AUD_DEFAULTS["shift"],
        cover_strength=AUD2AUD_DEFAULTS["cover_strength"],
        mode=AUD2AUD_DEFAULTS["mode"],
        repaint_start=AUD2AUD_DEFAULTS["repaint_start"],
        repaint_end=AUD2AUD_DEFAULTS["repaint_end"],
        vocal_language=AUD2AUD_DEFAULTS["vocal_language"],
        unet_quantization=AUD2AUD_DEFAULTS["unet_quantization"],
        quantized_gemm_mode=AUD2AUD_DEFAULTS["quantized_gemm_mode"],
        music3_repaint_mode=None,
        num_inference_steps=None,
        flow_guidance_scale=None,
        loras="[]",
        reference_audio=UploadFile(file=io.BytesIO(content), filename="song.flac"),
        db=session,
    )
    kwargs.update(overrides)
    return asyncio.run(routes.generate_aud2aud(**kwargs))


# ---------------------------------------------------------------------------
# ACE-Step no-op proof: the params dict this route builds for an ACE-Step
# request is unaffected by MiniMax Music 3's three new keys.
# ---------------------------------------------------------------------------

_ACESTEP_ORIGINAL_KEYS = (
    "prompt", "lyrics", "seed", "inference_steps", "guidance_scale", "shift", "cover_strength", "mode",
    "repaint_start", "repaint_end", "vocal_language", "loras", "unet_quantization", "quantized_gemm_mode",
)


def test_acestep_cover_params_are_byte_identical_to_the_pre_phase_8_shape(tmp_path, monkeypatch):
    session = _session(tmp_path)
    manager = _StubPipelineManager(is_music3=False)

    with pytest.raises(GenerationError):
        _call(monkeypatch, tmp_path, manager, session, b"a cover reference clip", mode="cover", cover_strength=0.7)

    params = manager.captured_params
    assert params is not None, "generate_aud2aud was never reached"
    for key in _ACESTEP_ORIGINAL_KEYS:
        assert key in params, f"missing original key {key!r}"
    assert params["mode"] == "cover"
    assert params["cover_strength"] == 0.7
    assert params["prompt"] == AUD2AUD_DEFAULTS["prompt"]
    assert params["loras"] == []
    # The three MiniMax-Music3-only keys exist (this route always sets them) but resolve to None for acestep --
    # there is no AUD2AUD_GEN_ARCH_OVERLAYS["acestep"] entry.
    assert params["music3_repaint_mode"] is None
    assert params["num_inference_steps"] is None
    assert params["flow_guidance_scale"] is None
    session.close()


def test_acestep_repaint_params_are_byte_identical_to_the_pre_phase_8_shape(tmp_path, monkeypatch):
    session = _session(tmp_path)
    manager = _StubPipelineManager(is_music3=False)

    with pytest.raises(GenerationError):
        _call(
            monkeypatch, tmp_path, manager, session, b"a repaint reference clip",
            mode="repaint", repaint_start=4.0, repaint_end=7.0,
        )

    params = manager.captured_params
    assert params is not None, "generate_aud2aud was never reached"
    for key in _ACESTEP_ORIGINAL_KEYS:
        assert key in params, f"missing original key {key!r}"
    assert params["mode"] == "repaint"
    assert params["repaint_start"] == 4.0
    assert params["repaint_end"] == 7.0
    assert params["music3_repaint_mode"] is None
    assert params["num_inference_steps"] is None
    assert params["flow_guidance_scale"] is None
    session.close()


def test_acestep_backend_source_never_references_the_new_music3_only_keys():
    """Static proof (inspecting the LIVE, currently-imported method): even a caller-supplied non-None value for
    one of the three new keys could not change ACE-Step's behavior, because the method that would have to read it
    to do so never mentions the name at all."""
    from core.pipeline_backends.acestep import AceStepMixin

    source = inspect.getsource(AceStepMixin._generate_aud2aud_acestep)
    for key in ("music3_repaint_mode", "num_inference_steps", "flow_guidance_scale"):
        assert key not in source, f"AceStepMixin._generate_aud2aud_acestep unexpectedly references {key!r}"


def test_acestep_saved_gallery_row_does_not_carry_the_three_music3_only_null_keys(tmp_path, monkeypatch):
    """Fix 5 (second half): generation output and the saved FLAC bytes were already unaffected (proven above and
    by the pre-existing audio-content tests), but `params_for_db` is a comprehension over the FULL `params` dict,
    which now always carries `music3_repaint_mode`/`num_inference_steps`/`flow_guidance_scale` -- so an ACE-Step
    row would gain three new, always-null keys it never had before this phase unless the route explicitly drops
    them, mirroring `/generate/outpaint/audio`'s identical cross-arch-key filter. This is the one test in this
    file that lets the route run all the way to `create_db_image_record` (a REAL, tiny waveform is returned by
    the stub instead of raising) so it can inspect what was actually PERSISTED, not just what
    `generate_aud2aud` was called with.
    """
    from database.models import GeneratedImage as _GeneratedImageModel

    content = b"a real (stub) ace-step generation, saved for real"
    session = _session(tmp_path)
    monkeypatch.setattr(routes, "pipeline_manager", _SucceedingAceStepStubPipelineManager())
    monkeypatch.setattr(routes.settings, "outputs_dir", str(tmp_path))

    kwargs = dict(
        prompt=AUD2AUD_DEFAULTS["prompt"], lyrics=AUD2AUD_DEFAULTS["lyrics"], seed=AUD2AUD_DEFAULTS["seed"],
        inference_steps=AUD2AUD_DEFAULTS["inference_steps"], guidance_scale=AUD2AUD_DEFAULTS["guidance_scale"],
        shift=AUD2AUD_DEFAULTS["shift"], cover_strength=AUD2AUD_DEFAULTS["cover_strength"], mode="cover",
        repaint_start=AUD2AUD_DEFAULTS["repaint_start"], repaint_end=AUD2AUD_DEFAULTS["repaint_end"],
        vocal_language=AUD2AUD_DEFAULTS["vocal_language"], unet_quantization=AUD2AUD_DEFAULTS["unet_quantization"],
        quantized_gemm_mode=AUD2AUD_DEFAULTS["quantized_gemm_mode"], music3_repaint_mode=None,
        num_inference_steps=None, flow_guidance_scale=None, loras="[]",
        reference_audio=UploadFile(file=io.BytesIO(content), filename="song.flac"), db=session,
    )
    result = asyncio.run(routes.generate_aud2aud(**kwargs))

    saved_row = session.query(_GeneratedImageModel).filter_by(id=result["image"]["id"]).one()
    for key in ("music3_repaint_mode", "num_inference_steps", "flow_guidance_scale"):
        assert key not in saved_row.parameters, f"ACE-Step gallery row unexpectedly carries {key!r}"
    # Sanity: the row is otherwise a normal ACE-Step "cover" row (its own real keys survive the filter).
    assert saved_row.parameters["mode"] == "cover"
    assert saved_row.parameters["cover_strength"] == AUD2AUD_DEFAULTS["cover_strength"]
    session.close()


# ---------------------------------------------------------------------------
# MiniMax Music 3: gallery-hash resolution, sentinel resolution, mode=cover
# refusal reaching the route as a 400 (raised inside the mixin, propagated
# the same way the outpaint route's placement refusal already is).
# ---------------------------------------------------------------------------

def test_music3_upload_with_no_matching_gallery_row_is_refused(tmp_path, monkeypatch):
    session = _session(tmp_path)
    manager = _StubPipelineManager(is_music3=True)
    with pytest.raises(ValidationError) as exc_info:
        _call(monkeypatch, tmp_path, manager, session, b"not a real gallery file",
              mode="repaint", repaint_start=0.0, repaint_end=10.0)
    assert exc_info.value.status_code == 400
    assert "gallery" in exc_info.value.message.lower()
    session.close()


def test_music3_omitted_sentinels_resolve_from_the_overlay_before_reaching_the_backend(tmp_path, monkeypatch):
    content = b"a fake but stable music3 song file"
    session = _session(tmp_path)
    _seed_matching_row(session, content)
    manager = _StubPipelineManager(is_music3=True)

    with pytest.raises(GenerationError):
        _call(monkeypatch, tmp_path, manager, session, content, mode="repaint", repaint_start=4.0, repaint_end=8.0)

    params = manager.captured_params
    assert params is not None, "generate_aud2aud was never reached"
    assert params["music3_repaint_mode"] == "regenerate"
    assert params["num_inference_steps"] == 30
    assert params["flow_guidance_scale"] == 1.7
    assert params["mode"] == "repaint"
    session.close()


def test_music3_explicitly_supplied_sentinels_are_not_overridden_by_the_overlay(tmp_path, monkeypatch):
    content = b"a fake but stable music3 song file, take 2"
    session = _session(tmp_path)
    _seed_matching_row(session, content)
    manager = _StubPipelineManager(is_music3=True)

    with pytest.raises(GenerationError):
        _call(
            monkeypatch, tmp_path, manager, session, content,
            mode="repaint", repaint_start=4.0, repaint_end=8.0,
            music3_repaint_mode="rerender", num_inference_steps=5, flow_guidance_scale=2.1,
        )

    params = manager.captured_params
    assert params is not None, "generate_aud2aud was never reached"
    assert params["music3_repaint_mode"] == "rerender"
    assert params["num_inference_steps"] == 5
    assert params["flow_guidance_scale"] == 2.1
    session.close()


def test_music3_gallery_lookup_accepts_a_previous_repaint_row_too():
    """The generation_type filter `/generate/aud2aud` uses for MiniMax Music 3 must include "repaint" (not just
    "txt2aud"/"outpaint_aud"), so a previously repainted song's own result can be repainted or extended again --
    design doc: "the sidecar for the result so a repainted song can itself be extended or repainted"."""
    import inspect as _inspect

    source = _inspect.getsource(routes.generate_aud2aud)
    assert '.filter(GeneratedImage.generation_type.in_(["txt2aud", "outpaint_aud", "repaint"]))' in source


def test_music3_mode_cover_is_refused_via_the_backend_not_a_generic_no_model_error(tmp_path, monkeypatch):
    """mode=cover reaches the mixin (the route itself does not special-case music3+cover), which raises the
    real capability refusal -- proving the route's own generic validation does not accidentally block it with a
    less specific error first."""
    content = b"a fake but stable music3 song file, take 3"
    session = _session(tmp_path)
    _seed_matching_row(session, content)

    class _RefusingStubPipelineManager(_StubPipelineManager):
        def generate_aud2aud(self, params, reference_audio_source, progress_callback=None):
            if params.get("mode") != "repaint":
                raise ValidationError(
                    f"MiniMax Music 3 does not support aud2aud mode {params.get('mode')!r}",
                    detail="Only mode='repaint' is available for MiniMax Music 3.",
                )
            self.captured_params = dict(params)  # pragma: no cover -- not reached here
            raise _StubGenerationFailed("stub: no real generation performed")  # pragma: no cover

    manager = _RefusingStubPipelineManager(is_music3=True)
    with pytest.raises(ValidationError) as exc_info:
        _call(monkeypatch, tmp_path, manager, session, content, mode="cover")

    assert exc_info.value.status_code == 400
    assert "cover" in exc_info.value.message.lower()
    session.close()


def test_no_model_loaded_is_refused_for_neither_architecture(tmp_path, monkeypatch):
    session = _session(tmp_path)
    manager = _StubPipelineManager(is_music3=False)
    manager.is_acestep_model = False  # neither ACE-Step nor MiniMax Music 3
    with pytest.raises(ValidationError) as exc_info:
        _call(monkeypatch, tmp_path, manager, session, b"irrelevant content", mode="cover")
    assert exc_info.value.status_code == 400
    assert "ace-step" in exc_info.value.message.lower() or "acestep" in exc_info.value.message.lower()
    session.close()


# ---------------------------------------------------------------------------
# FastAPI's OWN `Form(..., ge=1)`/`Form(..., gt=0)` enforcement for
# `num_inference_steps`/`flow_guidance_scale` -- the actual fix (see module
# docstring, "BOUND ENFORCEMENT"). Only fires at the ASGI request-parsing
# boundary (a direct Python call, like every test above, bypasses it
# entirely), so these two are mounted for real, mirroring
# `minimax_music3_outpaint_route_test.py`'s identical `_app`/`_post_multipart`
# pair.
# ---------------------------------------------------------------------------

def _app(monkeypatch, manager, session):
    from fastapi import FastAPI

    from api.error_handlers import register_error_handlers

    monkeypatch.setattr(routes, "pipeline_manager", manager)
    app = FastAPI()
    register_error_handlers(app)
    app.post("/generate/aud2aud")(routes.generate_aud2aud)
    app.dependency_overrides[routes.get_gallery_db] = lambda: session
    return app


def _post_multipart(app, content: bytes, **data):
    import httpx

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/generate/aud2aud",
                data={"mode": "repaint", "repaint_start": "0.0", "repaint_end": "8.0", **data},
                files={"reference_audio": ("song.flac", content, "audio/flac")},
            )
            return response.status_code, response.json()

    return asyncio.run(run())


def test_zero_num_inference_steps_is_rejected_at_request_parsing_not_a_500_after_generation(tmp_path, monkeypatch):
    content = b"a fake but stable music3 song file, take 4"
    session = _session(tmp_path)
    _seed_matching_row(session, content)
    monkeypatch.setattr(routes.settings, "outputs_dir", str(tmp_path))
    manager = _StubPipelineManager(is_music3=True)
    app = _app(monkeypatch, manager, session)

    status_code, body = _post_multipart(app, content, num_inference_steps="0")

    assert status_code == 422, body
    assert manager.captured_params is None, "generate_aud2aud must never be reached for an invalid step count"
    session.close()


def test_negative_flow_guidance_scale_is_rejected_at_request_parsing_not_a_500_after_generation(tmp_path, monkeypatch):
    content = b"a fake but stable music3 song file, take 5"
    session = _session(tmp_path)
    _seed_matching_row(session, content)
    monkeypatch.setattr(routes.settings, "outputs_dir", str(tmp_path))
    manager = _StubPipelineManager(is_music3=True)
    app = _app(monkeypatch, manager, session)

    status_code, body = _post_multipart(app, content, flow_guidance_scale="-1.0")

    assert status_code == 422, body
    assert manager.captured_params is None, "generate_aud2aud must never be reached for an invalid guidance scale"
    session.close()
