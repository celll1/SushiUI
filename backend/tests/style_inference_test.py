"""The Phase 4 inference API loads a distributable style artifact without a base repo."""

from pathlib import Path
import base64
import asyncio
import io
import json
import math
import sys

from fastapi import FastAPI
import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style_training_fixture  # noqa: F401

from api.style_inference_routes import router
from api.error_handlers import register_error_handlers
from core.style_encoder.style_artifacts import ENCODER_SUFFIX, architecture_metadata, save_bundle
from core.style_encoder.style_model import StyleEncoder
from style_training_fixture import image_processor, tiny_vision


def _artifact(tmp_path):
    model = StyleEncoder(tiny_vision(), (1, -2), 8, 2, 12, 6, "full", None)
    processor = image_processor()
    metadata = {"format": "sushiui-style-encoder", "format_version": 1,
                "architecture": architecture_metadata(model),
                "preprocessing": json.loads(processor.to_json_string()), "style_max_num_patches": 16}
    return save_bundle(model, str(tmp_path), "best", metadata)


def _image_base64():
    image = Image.new("RGB", (16, 16), (64, 128, 192))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def test_sdxl_style_generation_route_validates_shape_and_passes_reference(monkeypatch):
    from core.style_encoder.sdxl_ip_inference import sdxl_style_generation_service

    calls = []
    def fake_generate(**kwargs):
        calls.append(kwargs)
        return "generated-png"
    monkeypatch.setattr(sdxl_style_generation_service, "generate", fake_generate)
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router, prefix="/api/v1")
    body = {"sdxl_model_path": "model.safetensors", "encoder_path": "encoder_style_encoder.safetensors",
            "adapter_path": "adapter.safetensors", "reference_image_base64": _image_base64(),
            "prompt": "portrait"}

    async def post(payload):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/v1/style/sdxl-generate", json=payload)

    result = asyncio.run(post(body))
    assert result.status_code == 200
    assert result.json() == {"image_base64": "generated-png", "seed": 0}
    assert calls[0]["reference"].size == (16, 16)
    assert calls[0]["width"] == 512
    result = asyncio.run(post({**body, "width": 513}))
    assert result.status_code == 400
    assert len(calls) == 1


def test_reference_selection_returns_original_candidate_index(tmp_path):
    encoder_path = _artifact(tmp_path)
    query = _image_base64()
    other = io.BytesIO()
    Image.new("RGB", (16, 16), (192, 32, 16)).save(other, format="PNG")
    body = {"encoder_path": encoder_path, "query_image_base64": query,
            "candidate_images_base64": [base64.b64encode(other.getvalue()).decode("ascii"), query]}
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router, prefix="/api/v1")

    async def post():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://test") as client:
            return await client.post("/api/v1/style/reference-select", json=body)

    result = asyncio.run(post())
    assert result.status_code == 200
    assert result.json()["selected_index"] == 1
    assert len(result.json()["cosine_similarities"]) == 2


def test_embed_and_similarity_use_the_distributable_artifact(tmp_path):
    path = _artifact(tmp_path)
    assert path.endswith(ENCODER_SUFFIX)
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router, prefix="/api/v1")
    def post(url, body):
        async def send():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                return await client.post(url, json=body)
        return asyncio.run(send())
    image = _image_base64()
    embedded = post("/api/v1/style/embed", {"encoder_path": path, "image_base64": image})
    assert embedded.status_code == 200
    vector = embedded.json()["embedding"]
    assert embedded.json()["dim"] == len(vector) == 6
    assert math.isclose(sum(x * x for x in vector), 1.0, abs_tol=1e-5)
    similarity = post("/api/v1/style/similarity", {
        "encoder_path": path, "image_a_base64": image, "image_b_base64": f"data:image/png;base64,{image}"})
    assert similarity.status_code == 200
    assert math.isclose(similarity.json()["cosine_similarity"], 1.0, abs_tol=1e-5)
    invalid = post("/api/v1/style/embed", {"encoder_path": path, "image_base64": "invalid"})
    assert invalid.status_code == 400
    missing = post("/api/v1/style/embed", {"encoder_path": str(tmp_path / "missing.safetensors"),
                                            "image_base64": image})
    assert missing.status_code == 404
