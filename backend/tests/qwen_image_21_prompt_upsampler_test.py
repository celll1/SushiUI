import base64
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.extensions.minimax_h3_prompt_assistant import PromptAssistError
from core.extensions.qwen_image_21_prompt_upsampler import (
    _decode_image,
    _extract_result,
    resolve_official_artifact,
)


def test_extracts_answer_after_thinking_and_ratio():
    result = _extract_result(
        '<think>private reasoning</think>\n```json\n'
        '{"rewritten_prompt":"A detailed scene","wh_ratio":"16:9"}\n```',
        "t2i",
    )
    assert result == {
        "rewritten_prompt": "A detailed scene",
        "wh_ratio": "16:9",
        "ratio_follow": "",
    }


def test_i2i_ratio_follow_is_preserved():
    result = _extract_result(
        json.dumps({"rewritten_prompt": "Edit it", "wh_ratio": "", "ratio_follow": "<image1>"}),
        "i2i",
    )
    assert result["ratio_follow"] == "<image1>"


def test_rejects_invalid_ratio():
    with pytest.raises(PromptAssistError, match="unsupported ratio"):
        _extract_result('{"rewritten_prompt":"x","wh_ratio":"7:5"}', "t2i")


def test_decodes_image_data_url():
    buffer = io.BytesIO()
    Image.new("RGBA", (3, 2), (10, 20, 30, 128)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    image = _decode_image(f"data:image/png;base64,{encoded}")
    assert image.mode == "RGB"
    assert image.size == (3, 2)


def test_resolves_artifact_beside_variant(tmp_path):
    manifest = tmp_path / "prompt_enhancer" / "t2i" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    variant = tmp_path / "int8_convrot"
    variant.mkdir()
    assert resolve_official_artifact(str(variant), "t2i") == manifest
