"""Style encoder masking invariant and padding independence (design §4 I-1; T1, T2)."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from transformers import Siglip2VisionConfig, Siglip2VisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_model import LoRASpec, StyleEncoder, build_style_encoder, resolve_feature_layers
from core.style_encoder.style_views import apply_block_masks
from core.tagger.siglip2_tagger_model import LoRALinear

PATCH_DIM = 3 * 2 * 2


def _vision(attn: str) -> Siglip2VisionModel:
    torch.manual_seed(0)
    return Siglip2VisionModel(Siglip2VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                                  num_attention_heads=4, patch_size=2, num_patches=16,
                                                  attn_implementation=attn))


def _encoder(attn: str, mode: str = "full", feature_layers=(1, -1)) -> StyleEncoder:
    lora = LoRASpec(rank=2, alpha=2.0, dropout=0.0) if mode == "lora" else None
    model = StyleEncoder(_vision(attn), feature_layers, token_dim=16, pool_heads=4,
                         projector_hidden_dim=24, embed_dim=8, mode=mode, lora=lora)
    if lora is not None:
        for module in model.lora_modules.values():
            torch.nn.init.normal_(module.lora_B)
    return model.eval()


def _naflex(shapes, num_patches):
    mask = torch.zeros(len(shapes), num_patches, dtype=torch.int32)
    for b, (h, w) in enumerate(shapes):
        mask[b, :h * w] = 1
    return mask, torch.tensor(shapes, dtype=torch.int64)


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("fill", ["random", "nan", "inf", "1e30"])
@pytest.mark.parametrize("where", ["masked", "padding", "both"])
def test_outputs_ignore_pixels_at_masked_and_padding_positions(attn, mode, fill, where):
    model = _encoder(attn, mode)
    base_mask, shapes = _naflex([(3, 4), (4, 4)], num_patches=20)
    mask = apply_block_masks(base_mask, shapes, np.array([0.5, 0.4]), np.random.default_rng(3))
    invalid = mask == 0
    padding = base_mask == 0
    assert (invalid & ~padding).any() and padding.any()
    target = {"masked": invalid & ~padding, "padding": padding, "both": invalid}[where]
    torch.manual_seed(1)
    pixels = torch.randn(2, 20, PATCH_DIM)
    changed = pixels.clone()
    n = int(target.sum())
    changed[target] = {"random": 100.0 * torch.randn(n, PATCH_DIM), "nan": torch.full((n, PATCH_DIM), float("nan")),
                       "inf": torch.full((n, PATCH_DIM), float("inf")),
                       "1e30": torch.full((n, PATCH_DIM), 1e30)}[fill]

    with torch.no_grad():
        ref = model(pixels, mask, shapes)
        out = model(changed, mask, shapes)
    torch.testing.assert_close(out.embedding, ref.embedding)
    torch.testing.assert_close(out.features, ref.features)
    assert torch.equal(out.valid_mask, ~invalid) and torch.equal(ref.valid_mask, ~invalid)
    assert torch.all(out.features[invalid] == 0) and torch.all(ref.features[invalid] == 0)
    assert torch.allclose(ref.embedding.norm(dim=1), torch.ones(2))

    visible = changed.clone()
    visible[0, int(torch.nonzero(~invalid[0])[0])] += 1.0
    with torch.no_grad():
        moved = model(visible, mask, shapes)
    assert not torch.allclose(moved.embedding[0], ref.embedding[0])


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_padding_length_does_not_change_outputs(attn):
    model = _encoder(attn)
    torch.manual_seed(2)
    image = torch.randn(12, PATCH_DIM)
    short_mask, shapes = _naflex([(3, 4)], num_patches=12)
    long_mask, _ = _naflex([(3, 4)], num_patches=19)
    short = image.unsqueeze(0)
    long = torch.cat([image, 5.0 * torch.randn(7, PATCH_DIM)]).unsqueeze(0)
    with torch.no_grad():
        a = model(short, short_mask, shapes)
        b = model(long, long_mask, shapes)
    torch.testing.assert_close(b.embedding, a.embedding, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b.features[:, :12], a.features, atol=1e-5, rtol=1e-5)
    assert torch.all(b.features[:, 12:] == 0)


def test_trainable_parameters_cover_only_layers_that_run():
    full = _encoder("sdpa", "full", feature_layers=(2,))
    names = {name for name, _ in full.trainable_named_parameters()}
    assert any(n.startswith("vision_encoder.encoder.layers.1.") for n in names)
    assert any(n.startswith("vision_encoder.embeddings.") for n in names)
    assert not any(n.startswith(("vision_encoder.encoder.layers.2.", "vision_encoder.post_layernorm.",
                                 "vision_encoder.head.")) for n in names)

    lora = _encoder("sdpa", "lora", feature_layers=(2,))
    assert sorted(lora.lora_modules) == sorted(f"encoder.layers.{i}.self_attn.{p}" for i in (0, 1)
                                               for p in ("q_proj", "k_proj", "v_proj", "out_proj"))
    assert all(isinstance(m, LoRALinear) for m in lora.lora_modules.values())
    encoder_names = {n for n, _ in lora.trainable_named_parameters() if n.startswith("vision_encoder.")}
    assert encoder_names and all(n.endswith((".lora_A", ".lora_B")) for n in encoder_names)
    assert any(n.startswith("pooling.") for n, _ in lora.trainable_named_parameters())


def test_lora_that_adapts_no_layer_is_refused():
    with pytest.raises(ValueError, match="frozen"):
        _encoder("sdpa", "lora", feature_layers=(0,))


def test_unsupported_attention_implementation_is_refused():
    assert not Siglip2VisionModel._supports_flash_attn
    with pytest.raises(ValueError, match="flash_attention_2"):
        build_style_encoder("google/unused", "google/unused", "flash_attention_2", [-1], 16, 4, 24, 8, "full", None)
    vision = _vision("sdpa")
    vision.config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="flash_attention_2"):
        StyleEncoder(vision, [-1], 16, 4, 24, 8, "full", None)


def test_feature_layer_resolution():
    assert resolve_feature_layers([-1, 0, 1], 3) == (3, 0, 1)
    for bad in ([], [4], [-5], [1, -3]):
        with pytest.raises(ValueError):
            resolve_feature_layers(bad, 3)
