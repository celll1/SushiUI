"""SigLIP2VisionEncoderWrapper must load both on-disk key layouts key-exactly."""

from pathlib import Path
import sys

import pytest
import torch
import transformers
from safetensors.torch import save_file
from transformers import Siglip2VisionConfig, Siglip2VisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.vision_encoder import SigLIP2VisionEncoderWrapper


@pytest.fixture(autouse=True)
def _no_processor_download(monkeypatch):
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", classmethod(lambda cls, *a, **k: object())
    )


def _source_model() -> Siglip2VisionModel:
    torch.manual_seed(0)
    # Shapes the wrapper's fallback config inference reproduces (heads = hidden/64, patch 16).
    config = Siglip2VisionConfig(
        hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=1,
        patch_size=16, num_patches=256,
    )
    return Siglip2VisionModel(config)


def _write(tmp_path, name, sd):
    path = tmp_path / name
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    return str(path)


@pytest.mark.parametrize("prefix", ["", "vision_model."])
def test_loads_bare_and_legacy_prefixed_keys(tmp_path, prefix):
    src = _source_model()
    sd = {f"{prefix}{k}": v for k, v in src.state_dict().items()}
    sd["header_token"] = torch.full((1, 1, 64), 0.5)
    wrapper = SigLIP2VisionEncoderWrapper(_write(tmp_path, "ve.safetensors", sd))

    loaded = wrapper.model.state_dict()
    assert loaded.keys() == src.state_dict().keys()
    for key, value in src.state_dict().items():
        assert torch.equal(loaded[key], value), key
    assert torch.equal(wrapper.header_token.data, sd["header_token"])


def test_save_roundtrip_is_loadable(tmp_path):
    src_path = _write(tmp_path, "src.safetensors", _source_model().state_dict())
    first = SigLIP2VisionEncoderWrapper(src_path)
    again = SigLIP2VisionEncoderWrapper(_write(tmp_path, "saved.safetensors", first.state_dict_for_save()))
    for key, value in first.model.state_dict().items():
        assert torch.equal(again.model.state_dict()[key], value), key


@pytest.mark.parametrize("mutation", ["drop", "rename"])
def test_refuses_mismatched_keys(tmp_path, mutation):
    sd = dict(_source_model().state_dict())
    victim = "encoder.layers.1.self_attn.q_proj.weight"
    tensor = sd.pop(victim)
    if mutation == "rename":
        sd["encoder.layers.1.self_attn.query.weight"] = tensor
    with pytest.raises(ValueError, match="does not match Siglip2VisionModel"):
        SigLIP2VisionEncoderWrapper(_write(tmp_path, "bad.safetensors", sd))
