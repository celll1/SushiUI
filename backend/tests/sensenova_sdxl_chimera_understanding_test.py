from __future__ import annotations

import json
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from core.models.sensenova_sdxl_chimera.understanding import (
    _read_understanding_state_dict,
    is_understanding_tensor_key,
)
from core.models.sensenova.vendor.configuration_neo_chat import NEOLLMConfig
from core.models.sensenova.vendor.modeling_qwen3 import Qwen3Model


def test_understanding_key_filter_excludes_every_generation_component():
    accepted = (
        "transformer.vision_model.embeddings.patch_embedding.weight",
        "transformer.language_model.model.layers.0.self_attn.q_proj.weight",
        "transformer.language_model.model.layers.0.mlp.gate_proj.weight",
        "transformer.language_model.lm_head.weight",
    )
    refused = (
        "transformer.fm_modules.fm_head.conv.weight",
        "transformer.fm_modules.timestep_embedder.mlp.0.weight",
        "transformer.fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
        "transformer.language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
        "transformer.language_model.model.layers.0.mlp_mot_gen.gate_proj.weight",
        "transformer.language_model.model.norm_mot_gen.weight",
    )
    assert all(is_understanding_tensor_key(key) for key in accepted)
    assert not any(is_understanding_tensor_key(key) for key in refused)


def test_selective_reader_never_requests_generation_payload(tmp_path):
    source = tmp_path / "sensenova.safetensors"
    config = {"llm_config": {"hidden_size": 16, "num_hidden_layers": 1,
                              "num_attention_heads": 4, "num_key_value_heads": 2}}
    tensors = {
        "transformer.language_model.model.layers.0.self_attn.q_proj.weight": torch.ones(1),
        "transformer.vision_model.embeddings.patch_embedding.weight": torch.ones(1),
        "transformer.language_model.model.layers.0.self_attn.q_proj_mot_gen.weight": torch.ones(1),
        "transformer.fm_modules.fm_head.weight": torch.ones(1),
    }
    save_file(tensors, str(source), metadata={"sensenova_config": json.dumps(config)})
    requested = []

    from core.models.sensenova.loader import _LazySafetensorsSource

    original = _LazySafetensorsSource.get

    def audited(self, key):
        requested.append(key)
        assert is_understanding_tensor_key(key)
        return original(self, key)

    with patch.object(_LazySafetensorsSource, "get", audited):
        state, _metadata = _read_understanding_state_dict(str(source))
    assert requested
    assert set(state) == {
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "vision_model.embeddings.patch_embedding.weight",
    }


def test_qwen_prefix_capture_returns_only_selected_layer_kv():
    config = NEOLLMConfig(
        architectures=["Qwen3ForCausalLM"], vocab_size=32, hidden_size=32,
        intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=32,
        rope_theta_hw=1000.0, max_position_embeddings_hw=32,
    )
    config._attn_implementation = "eager"
    model = Qwen3Model(config).eval()
    embeds = torch.randn(1, 5, 32)
    indexes = torch.stack((torch.arange(5), torch.zeros(5, dtype=torch.long),
                           torch.zeros(5, dtype=torch.long)))
    outputs = model(
        inputs_embeds=embeds,
        indexes=indexes,
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        use_cache=False,
        capture_layers=(1, 3),
    )
    assert set(outputs.selected_kv) == {1, 3}
    assert len(outputs.hidden_states) == 2
    for key, value in outputs.selected_kv.values():
        assert key.shape == (1, 2, 5, 8)
        assert value.shape == (1, 2, 5, 8)
    assert outputs.past_key_values is None
