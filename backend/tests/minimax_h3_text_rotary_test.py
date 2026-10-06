"""H3 streamed text encoding matches the actual native Qwen3-VL text stack."""

from types import SimpleNamespace

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from core.models.minimax_h3.h3_pipeline_ops import encode_presentation


def test_streamed_text_matches_native_three_axis_rotary():
    torch.manual_seed(523)
    config = Qwen3VLTextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=64,
        rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True})
    model = Qwen3VLTextModel(config).eval()
    model.norm = torch.nn.Identity()
    encoder = SimpleNamespace(model=SimpleNamespace(language_model=model), config=config)
    ids = [4, 12, 21, 5, 38]
    with torch.no_grad():
        expected = model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state
        actual = encode_presentation(encoder, ids, device="cpu", dtype=torch.float32, layer=2)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual.shape == (1, len(ids), config.hidden_size)
    assert all(value.device.type == "cpu" for value in model.rotary_emb.buffers())
