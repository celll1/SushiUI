"""CPU-only contract tests for SenseNova img2txt."""

import asyncio
import io
import json
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from api import routes
from api.arch_capabilities import TEXT_OUTPUT_MODES
from api.param_defaults import IMG2TXT_DEFAULTS
from core.models.sensenova.text_output import (
    build_effective_instruction,
    parse_structured_output,
)
from core.pipeline_backends.sensenova import SenseNovaMixin


def _png_upload(width=8, height=8):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "navy").save(buf, format="PNG")
    buf.seek(0)
    return UploadFile(file=buf, filename="source.png")


def _route_kwargs(upload):
    return {
        "images": [upload],
        "task": "caption_tags",
        "instruction": "",
        "hint_tags": json.dumps(["1girl", "outdoors"]),
        "max_new_tokens": 128,
        "do_sample": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": None,
        "repetition_penalty": None,
        "seed": 123,
        "prompt_template_version": 1,
        "loras": "[]",
    }


def test_defaults_and_capability_are_backend_owned():
    assert IMG2TXT_DEFAULTS["task"] == "caption"
    assert IMG2TXT_DEFAULTS["prompt_template_version"] == 1
    assert TEXT_OUTPUT_MODES == {"sensenova": ("img2txt",)}
    defaults = asyncio.run(routes.get_generation_defaults())
    capabilities = asyncio.run(routes.get_arch_capabilities())
    assert defaults["img2txt"] == IMG2TXT_DEFAULTS
    assert capabilities["text_output_modes"] == {"sensenova": ["img2txt"]}


def test_prompt_and_structured_result_preserve_generated_tag_order():
    prompt = build_effective_instruction(
        "caption_tags", "", ["second hint", "first hint"], 1)
    assert "context only" in prompt
    assert prompt.endswith('["second hint", "first hint"]')

    raw = '```json\n{"caption":"A scene","tags":["z","a"]}\n```'
    structured, warning = parse_structured_output("caption_tags", raw)
    assert structured == {"caption": "A scene", "tags": ["z", "a"]}
    assert warning is None


def test_structured_parse_failure_keeps_raw_result_out_of_structure():
    structured, warning = parse_structured_output("tags", "tag one, tag two")
    assert structured is None
    assert "raw_text is preserved" in warning


def test_wrong_model_is_refused_before_upload_read(monkeypatch):
    class NeverRead:
        async def read(self, _size=-1):
            raise AssertionError("wrong-model gate must run before upload decoding")

    monkeypatch.setattr(routes.pipeline_manager, "current_model_info", {"type": "sdxl"})
    monkeypatch.setattr(routes.pipeline_manager, "is_sensenova_model", False)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(routes.generate_img2txt(**_route_kwargs(NeverRead())))
    assert caught.value.status_code == 409


def test_route_returns_session_text_without_gallery_persistence(monkeypatch):
    monkeypatch.setattr(
        routes.pipeline_manager,
        "current_model_info",
        {"type": "sensenova", "source": "synthetic/model"},
    )
    monkeypatch.setattr(routes.pipeline_manager, "is_sensenova_model", True)
    monkeypatch.setattr(routes.pipeline_manager, "component_health", "healthy")
    monkeypatch.setattr(routes.pipeline_manager, "consume_load_warnings", lambda: [])
    monkeypatch.setattr(routes.pipeline_manager, "reset_cancel_flag", lambda: None)

    def fake_generate(params, image, progress_callback=None):
        assert image.mode == "RGB"
        assert params["hint_tags"] == ["1girl", "outdoors"]
        assert params["loras"] == []
        assert "context only" in params["instruction"]
        return '{"caption":"Synthetic","tags":["b","a"]}', 123, {
            "preprocess_seconds": 0.01,
            "generation_seconds": 0.02,
        }

    monkeypatch.setattr(routes.pipeline_manager, "generate_img2txt", fake_generate)

    from core.gpu_coordinator import gpu_coordinator

    @asynccontextmanager
    async def fake_slot(**_kwargs):
        yield

    monkeypatch.setattr(gpu_coordinator, "generation_slot", fake_slot)
    result = asyncio.run(routes.generate_img2txt(**_route_kwargs(_png_upload())))

    assert result["kind"] == "text"
    assert result["structured"] == {
        "caption": "Synthetic",
        "tags": ["b", "a"],
    }
    assert result["model"] == {"type": "sensenova", "source": "synthetic/model"}
    assert "image" not in result
    assert "filename" not in result


def test_pipeline_uses_understanding_pixels_and_never_moves_vae(monkeypatch):
    import torch
    from core.models.sensenova.vendor import utils as vendor_utils

    monkeypatch.setattr(
        vendor_utils,
        "load_image_native",
        lambda *_args, **_kwargs: (
            torch.zeros((4, 3 * 16 * 16), dtype=torch.float32),
            torch.tensor([[2, 2]]),
        ),
    )

    class FakeTransformer:
        patch_size = 16
        downsample_ratio = 0.5

        def to(self, _device):
            return self

        def chat(self, _tokenizer, pixels, instruction, config, **kwargs):
            assert pixels.dtype == torch.bfloat16
            assert instruction == "Describe it"
            assert kwargs["grid_hw"].tolist() == [[2, 2]]
            assert "temperature" not in config
            criterion = config["stopping_criteria"][0]
            assert criterion(None, None) is False
            return "A synthetic image"

    class Harness(SenseNovaMixin):
        device = "cpu"
        cancel_requested = False

        def __init__(self):
            self.moves = []
            self.sensenova_components = {
                "transformer": FakeTransformer(),
                "tokenizer": object(),
                "vae": object(),
            }

        def _unload_lora_sensenova(self):
            return 0

        def _load_lora_sensenova(self, configs, component_names=None):
            assert configs == []
            assert component_names == ["understanding"]
            return 0

        def _sensenova_move(self, component_name, target_device):
            self.moves.append((component_name, str(target_device)))
            return self.sensenova_components.get(component_name)

    harness = Harness()
    raw, seed, timing = harness._generate_img2txt_sensenova(
        {
            "instruction": "Describe it",
            "max_new_tokens": 16,
            "do_sample": False,
            "seed": 7,
        },
        Image.new("RGB", (8, 8)),
    )
    assert raw == "A synthetic image"
    assert seed == 7
    assert timing["generation_seconds"] >= 0
    assert harness.moves == [("transformer", "cpu"), ("transformer", "cpu")]
