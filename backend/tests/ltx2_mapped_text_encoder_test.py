import json

import pytest
import torch
from safetensors.torch import save_model, load_file, save_file

from core.models.ltx2.mapped_text_encoder import load_matching_mapped_encoder


def _package(tmp_path):
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    config = Gemma3Config(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, sliding_window=8),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, image_size=16, patch_size=8),
        mm_tokens_per_image=4,
    )
    model = Gemma3ForConditionalGeneration(config).to(torch.bfloat16).eval()
    model.save_pretrained(tmp_path)
    return model


def test_exact_native_weights_and_tied_embedding_forward(tmp_path):
    original = _package(tmp_path)
    loaded = load_matching_mapped_encoder(tmp_path, torch.bfloat16)
    assert loaded is not None
    assert loaded.lm_head.weight is loaded.model.language_model.embed_tokens.weight
    for key, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    tokens = torch.tensor([[1, 5, 7]])
    with torch.no_grad():
        assert torch.equal(original(input_ids=tokens).logits, loaded(input_ids=tokens).logits)


def test_dtype_conversion_keeps_standard_loader(tmp_path):
    _package(tmp_path)
    assert load_matching_mapped_encoder(tmp_path, torch.float32) is None


def test_missing_untied_weight_is_rejected(tmp_path):
    model = _package(tmp_path)
    model.model.language_model.layers[0].self_attn.q_proj.weight = None
    save_model(model, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Missing untied Gemma tensor"):
        load_matching_mapped_encoder(tmp_path, torch.bfloat16)


def test_shard_cannot_escape_component(tmp_path):
    _package(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"weight": "../model.safetensors"}}))
    with pytest.raises(ValueError, match="escapes"):
        load_matching_mapped_encoder(tmp_path, torch.bfloat16)


def test_legacy_official_keys_preserve_values(tmp_path):
    original = _package(tmp_path)
    state = load_file(str(tmp_path / "model.safetensors"))
    rules = (("model.language_model.", "language_model.model."),
             ("model.vision_tower.", "vision_tower.vision_model."),
             ("model.multi_modal_projector.", "multi_modal_projector."))
    legacy = {}
    for key, value in state.items():
        for new, old in rules:
            if key.startswith(new):
                key = old + key[len(new):]
                break
        legacy[key] = value
    save_file(legacy, str(tmp_path / "model.safetensors"))
    loaded = load_matching_mapped_encoder(tmp_path, torch.bfloat16)
    for key, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key


def test_shard_index_missing_tensor_is_rejected(tmp_path):
    _package(tmp_path)
    state = load_file(str(tmp_path / "model.safetensors"))
    mapping = dict.fromkeys(state, "model.safetensors")
    mapping["missing"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="index has missing tensors"):
        load_matching_mapped_encoder(tmp_path, torch.bfloat16)
