"""Request/default tests; install CUDA import stubs before pytest collection."""
import pytest

from api.error_handlers import ValidationError
from api.generation_utils import resolve_audio_defaults, validate_yue2_request
from api.param_defaults import AUDIO_GEN_ARCH_OVERLAYS, TXT2AUD_DEFAULTS


def resolve(supplied):
    supplied = {"prompt": "Pop", **supplied}
    params = {**TXT2AUD_DEFAULTS, **dict.fromkeys(AUDIO_GEN_ARCH_OVERLAYS["yue2"]), **supplied}
    defaults = resolve_audio_defaults(params, set(supplied), "yue2")
    validate_yue2_request(params, supplied, defaults)
    return params


@pytest.mark.parametrize("cot, guidance", [("full", 1.0), ("melody", 1.0), ("off", 1.01)])
def test_guidance_follows_mode_only_when_omitted(cot, guidance):
    supplied = {"lyrics": "[Verse]\nHello", "yue2_cot": cot}
    assert resolve(supplied)["guidance_scale"] == guidance
    assert resolve({**supplied, "guidance_scale": 1.5})["guidance_scale"] == 1.5


def test_overlay_defaults_resolve():
    params = resolve({"lyrics": "[Verse]\nHello"})
    assert params["audio_duration"] == 360.0
    assert params["yue2_abc_max_tokens"] == 4096
    assert params["vae_tile_frames"] == 1024


@pytest.mark.parametrize("extra", [
    {"yue2_cot": "off", "yue2_abc": "X:1"}, {"lyrics": " "}, {"prompt": " "},
    {"yue2_cot": None}, {"negative_prompt": "noise"},
    {"reference_audio_enable": True}, {"unet_quantization": "int8"},
    {"blocks_to_swap": 2}, {"guidance_scale": float("nan")}, {"guidance_scale": 20.01},
    {"audio_duration": float("inf")}, {"num_inference_steps": 16},
    {"audio_duration": 0.001}, {"audio_duration": 1e308}, {"seed": -2}, {"seed": 2**63},
    {"attention_type": "flash"}, {"use_pinned_memory": True}, {"block_swap_ring_size": 3},
])
def test_invalid_or_unsupported_request_refused(extra):
    with pytest.raises(ValidationError):
        resolve({"lyrics": "[Verse]\nHello", **extra})


def test_yue2_accepts_stage_scoped_lora_request():
    params = resolve({"lyrics": "[Verse]\nHello", "loras": [{"path": "planner.safetensors"}]})
    assert params["loras"][0]["path"] == "planner.safetensors"


def test_ace_default_resolution_unchanged():
    params = dict(TXT2AUD_DEFAULTS)
    assert resolve_audio_defaults(params, set(), "acestep") == TXT2AUD_DEFAULTS


def test_whitespace_abc_is_omitted():
    assert resolve({"lyrics": "[Verse]\nHello", "yue2_cot": "off", "yue2_abc": " \n"})["yue2_abc"] == ""


@pytest.fixture
def routes(monkeypatch):
    import torch
    was_initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **k: None)
    monkeypatch.setattr(torch._C, "_cuda_init", lambda *a, **k: None)
    import api.routes as routes_module
    assert torch.cuda.is_initialized() == was_initialized
    return routes_module


@pytest.mark.parametrize("arch, extra, reaches_gpu_boundary", [
    ("acestep", {"ignored_legacy_key": True}, True),
    ("minimax_music3", {"ignored_legacy_key": True}, True),
    ("yue2", {"yue2_cot": "off"}, True),
    ("yue2", {"loras": [{"path": "planner.safetensors"}]}, True),
    ("yue2", {"negative_prompt": "noise"}, False),
    ("yue2", {"yue2_cot": "off", "yue2_abc": "X:1"}, False),
    ("yue2", {"attention_type": "flash"}, False),
])
def test_real_route_boundary(routes, monkeypatch, arch, extra, reaches_gpu_boundary):
    import asyncio
    import json
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from starlette.requests import Request
    from core.gpu_coordinator import gpu_coordinator

    @asynccontextmanager
    async def stop_before_gpu(**kwargs):
        raise ValidationError("verified-pre-gpu")
        yield  # pragma: no cover

    monkeypatch.setattr(gpu_coordinator, "generation_slot", stop_before_gpu)
    monkeypatch.setattr(routes, "pipeline_manager", SimpleNamespace(
        is_acestep_model=arch == "acestep", is_minimax_music3_model=arch == "minimax_music3",
        is_yue2_model=arch == "yue2", current_model_info={"type": arch},
        current_pipeline_kind=arch, reset_cancel_flag=lambda: None,
    ))
    body = {"prompt": "Pop", "lyrics": "[Verse]\nHello", **extra}

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/generate/txt2aud", "headers": []}, receive)
    with pytest.raises(ValidationError) as error:
        asyncio.run(routes.generate_txt2aud(routes.Txt2AudRequest(**body), request, db=None))
    assert ("verified-pre-gpu" in str(error.value)) is reaches_gpu_boundary


def test_sampling_request_bounds(routes):
    from pydantic import ValidationError as PydanticValidationError
    request = routes.Txt2AudRequest(temperature=0, top_k=0)
    assert request.temperature == 0 and request.top_k == 0
    for supplied in ({"temperature": 5.01}, {"temperature": -0.01}, {"top_k": -1}, {"top_k": 32769}):
        with pytest.raises(PydanticValidationError):
            routes.Txt2AudRequest(**supplied)


def test_yue2_peak_reservation_is_conservative(routes):
    assert routes._PEAK_VRAM_GB_BY_KIND["yue2"] >= 16.0


def test_sidecar_failure_keeps_saved_audio_successful(routes, monkeypatch):
    import asyncio
    import json
    import numpy as np
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from starlette.requests import Request
    from core.gpu_coordinator import gpu_coordinator
    from core.models.yue2 import artifacts
    from utils import audio_utils

    @asynccontextmanager
    async def gpu_stub(**kwargs):
        yield

    result = SimpleNamespace(
        waveform=np.zeros((2, 48), dtype=np.float32), sample_rate=48000, seed=3,
        abc_text="X:1", abc_ids=[1], truncated={"abc": False, "semantic": False},
        effective_config={}, timings={}, model_identity={},
    )

    async def generate_stub(*args, **kwargs):
        return result

    async def hash_stub(*args):
        return "audio-hash"

    def write_failure(*args, **kwargs):
        assert kwargs["media_sha256"] == "audio-hash"
        raise OSError("simulated sidecar write failure")

    monkeypatch.setattr(gpu_coordinator, "generation_slot", gpu_stub)
    monkeypatch.setattr(routes, "pipeline_manager", SimpleNamespace(
        is_yue2_model=True, current_model_info={"type": "yue2"},
        current_pipeline_kind="yue2", reset_cancel_flag=lambda: None,
    ))
    monkeypatch.setattr(routes, "_run_generation_in_executor", generate_stub)
    monkeypatch.setattr(routes, "extract_fp8_gemm_info", lambda *args: "")
    monkeypatch.setattr(routes, "extract_model_info", lambda *args: ("YuE2", "model-hash"))
    monkeypatch.setattr(routes, "_hash_saved_media", hash_stub)
    monkeypatch.setattr(audio_utils, "save_audio_with_metadata", lambda *args, **kwargs: "yue2_api_test.flac")
    monkeypatch.setattr(artifacts, "write_yue2_sidecar", write_failure)
    record = SimpleNamespace(id=1, to_dict=lambda: {"id": 1, "filename": "yue2_api_test.flac"})
    monkeypatch.setattr(routes, "create_db_image_record", lambda *args, **kwargs: record)
    db = SimpleNamespace(add=lambda *args: None, commit=lambda: None, refresh=lambda *args: None)
    body = {"prompt": "Pop", "lyrics": "[Verse]\nHello"}

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/generate/txt2aud", "headers": []}, receive)
    response = asyncio.run(routes.generate_txt2aud(routes.Txt2AudRequest(**body), request, db=db))
    assert response["success"] is True
    assert any(warning["code"] == "sidecar_write_failed" for warning in response["warnings"])
