"""CPU contracts for SigLIP2 feature sharing and caption decoding."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Siglip2VisionConfig, Siglip2VisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.captioner.concepts import ConceptSelector
from core.captioner.features import CaptionFeatures, SigLIP2FeatureProvider
from core.captioner.model import CaptionConfig, CompactCaptioner
from core.tagger.siglip2_tagger_model import SigLIP2TaggerModel, SigLIP2TaggerLoRAModel


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_initialized()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    assert not torch.cuda.is_initialized()


def selector(budget=3):
    return ConceptSelector(["blue hair", "red hair", "1girl", "artist"],
                           ["General", "General", "General", "Artist"],
                           max_concepts=budget, allowed_categories=["General"])


def fixture(batch=2):
    torch.manual_seed(42)
    features = CaptionFeatures(torch.randn(batch, 5, 8),
                               torch.tensor([[1, 1, 1, 0, 0]] * batch, dtype=torch.bool),
                               torch.rand(batch, 5, 2), torch.randn(batch, 6), torch.randn(batch, 4))
    concepts = selector().select(features.tag_logits, mode="predicted")
    config = CaptionConfig(spatial_dim=8, global_dim=6, num_tags=4, vocab_size=12,
                           hidden_dim=24, heads=3, layers=2, ffn_dim=48, visual_queries=2,
                           max_length=8, dropout=0.0, num_categories=8, pad_id=0, bos_id=1, eos_id=2)
    return features, concepts, CompactCaptioner(config)


@pytest.mark.parametrize("kind", ["default", "custom", "lora"])
def test_real_naflex_features_preserve_logits_and_one_vision_call(kind):
    torch.manual_seed(1)
    encoder = Siglip2VisionModel(Siglip2VisionConfig(hidden_size=24, intermediate_size=48,
                                 num_hidden_layers=1, num_attention_heads=3, patch_size=4, num_patches=16))
    if kind == "lora":
        tagger = SigLIP2TaggerLoRAModel(4, encoder, hidden_size=24, lora_rank=2)
        for module in tagger._lora_modules.values():
            nn.init.normal_(module.lora_B, std=0.02)
    else:
        tagger = SigLIP2TaggerModel(4, encoder, hidden_size=24,
                                  cls_dim=16 if kind == "custom" else None)
    nn.init.normal_(tagger.head.weight)
    tagger.eval()
    pixels = torch.randn(2, 8, 48)
    pixels[1, :6] = pixels[0, :6]
    mask = torch.tensor([[1] * 6 + [0] * 2] * 2)
    shapes = torch.tensor([[2, 3], [2, 3]])
    with torch.no_grad():
        legacy = tagger(pixels, mask, shapes)
        out = encoder(pixel_values=pixels, pixel_attention_mask=mask, spatial_shapes=shapes)
        pooled = tagger.custom_pooler(out.last_hidden_state, mask) if kind == "custom" else out.pooler_output
        reference = tagger.head(pooled)
    calls = []
    handle = encoder.register_forward_hook(lambda *args: calls.append(1))
    try:
        features = SigLIP2FeatureProvider(tagger).extract(pixels, mask, shapes)
    finally:
        handle.remove()
    assert len(calls) == 1
    torch.testing.assert_close(legacy, reference)
    torch.testing.assert_close(features.tag_logits, reference)
    torch.testing.assert_close(features.tag_logits[0], features.tag_logits[1])
    torch.testing.assert_close(features.spatial_xy[0, 0], torch.tensor([1 / 6, 1 / 4]))
    torch.testing.assert_close(features.spatial_xy[0, 5], torch.tensor([5 / 6, 3 / 4]))
    assert features.global_features.shape[-1] == (16 if kind == "custom" else 24)
    assert not features.spatial_tokens.requires_grad
    tagger.train()
    with pytest.raises(ValueError, match="evaluation-mode"):
        SigLIP2FeatureProvider(tagger).extract(pixels, mask, shapes)


def test_fixed_resolution_geometry():
    class FixedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            from types import SimpleNamespace
            self.config = SimpleNamespace(patch_size=4)

        def forward(self, pixel_values):
            from types import SimpleNamespace
            batch, _, height, width = pixel_values.shape
            return SimpleNamespace(last_hidden_state=torch.ones(batch, height * width // 16, 8),
                                   pooler_output=torch.ones(batch, 8))
    tagger = SigLIP2TaggerModel(4, FixedEncoder(), hidden_size=8, is_naflex=False).eval()
    features = SigLIP2FeatureProvider(tagger).extract(torch.ones(1, 3, 8, 12))
    assert features.spatial_tokens.shape == (1, 6, 8)
    assert features.spatial_mask.all()
    torch.testing.assert_close(features.spatial_xy[0, -1], torch.tensor([5 / 6, 3 / 4]))


def test_concept_sources_order_filter_warnings_and_hash():
    select = selector(2)
    predicted = select.select(torch.zeros(1, 4), mode="predicted")
    assert predicted.ids.tolist() == [[2, 0]]  # canonical name breaks ties
    provided = select.select(torch.zeros(1, 4), mode="provided",
                             provided=[["blue_hair", "blue hair", "missing", "artist", "red hair", "1girl"]])
    assert provided.ids.tolist() == [[0, 1]]
    assert provided.sources.tolist() == [[1, 1]]
    assert provided.scores.eq(0).all()
    assert provided.warnings == (("unknown_tag:missing", "excluded_tag:artist", "provided_tags_truncated"),)
    merged = select.select(torch.tensor([[4.0, 0, 3, 100]]), mode="merged", provided=[["blue hair"]])
    assert merged.ids.tolist() == [[0, 2]]
    assert merged.sources.tolist() == [[1, 0]]
    torch.testing.assert_close(merged.scores[0, 1], torch.tensor(3.0).sigmoid())
    empty = select.select(torch.zeros(1, 4), mode="provided", provided=[[]])
    assert not empty.mask.any()
    assert selector().vocabulary_hash == select.vocabulary_hash
    other = ConceptSelector(["red hair", "blue hair", "1girl", "artist"],
                            select.categories, max_concepts=2, allowed_categories=["General"])
    assert other.vocabulary_hash != select.vocabulary_hash
    with pytest.raises(ValueError):
        select.select(torch.tensor([[float("nan")] * 4]), mode="predicted")
    with pytest.raises(ValueError):
        select.select(torch.zeros(1, 4), mode="provided")
    unknown_category = ConceptSelector(["known", "uncategorized"], ["General", "Unknown"],
                                       max_concepts=2, allowed_categories=["General"])
    filtered = unknown_category.select(torch.tensor([[0.0, 10.0]]), mode="predicted")
    assert filtered.ids[filtered.mask].tolist() == [0]


def test_spatial_and_concept_padding_cannot_affect_memory():
    features, concepts, model = fixture()
    model.eval()
    memory, mask = model.encode_memory(features, concepts)
    spatial = features.spatial_tokens.clone()
    spatial[:, 3:] = float("nan")
    xy = features.spatial_xy.clone()
    xy[:, 3:] = float("nan")
    changed = replace(features, spatial_tokens=spatial, spatial_xy=xy)
    padded, padded_mask = model.encode_memory(changed, concepts)
    torch.testing.assert_close(memory, padded)
    assert torch.equal(mask, padded_mask)
    empty = selector().select(features.tag_logits, mode="provided", provided=[[], []])
    empty_memory, empty_mask = model.encode_memory(features, empty)
    assert torch.isfinite(empty_memory).all()
    assert empty_mask[:, :3].all() and not empty_mask[:, 3:].any()
    with pytest.raises(ValueError):
        model.encode_memory(replace(features, spatial_mask=torch.zeros_like(features.spatial_mask)), concepts)


def test_cached_logits_equal_teacher_forcing_and_causal_future_independence():
    features, concepts, model = fixture()
    model.eval()
    memory, mask = model.encode_memory(features, concepts)
    tokens = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]])
    full = model.decode(tokens, memory, mask)
    caches = [None] * len(model.layers)
    cross = [layer.cross_attn.project_kv(memory) for layer in model.layers]
    for position in range(tokens.shape[1]):
        value = model._embed(tokens[:, position:position + 1], torch.tensor([position]))
        for index, layer in enumerate(model.layers):
            value, caches[index] = layer.step(value, caches[index], cross[index], mask[:, None, None])
        step_logits = F.linear(model.output_norm(value[:, 0]), model.word_embedding.weight)
        torch.testing.assert_close(step_logits, full[:, position], atol=2e-5, rtol=2e-5)
    alternative = tokens.clone()
    alternative[:, 2:] = 3
    torch.testing.assert_close(model.decode(alternative, memory, mask)[:, :2], full[:, :2])


def test_teacher_loss_backward_and_target_contract():
    features, concepts, model = fixture()
    sequences = torch.tensor([[1, 4, 5, 2, 0], [1, 7, 2, 0, 0]])
    loss = model.teacher_loss(features, concepts, sequences, label_smoothing=0.05)
    loss.backward()
    for name in ("spatial_projection", "global_projection", "tag_embedding", "word_embedding"):
        assert getattr(model, name).weight.grad.abs().sum() > 0
    for invalid in ([[1, 4, 0, 2]], [[1, 4, 5]], [[1, 2, 2]], [[1, 1, 2]]):
        with pytest.raises(ValueError):
            model.teacher_loss(features, concepts, torch.tensor(invalid * 2), label_smoothing=0.0)


def test_generation_eos_limit_and_cancellation():
    features, concepts, model = fixture()
    with pytest.raises(ValueError, match="eval"):
        model.generate(features, concepts, max_new_tokens=3)
    model.eval()
    with torch.no_grad():
        model.word_embedding.weight.zero_()
        model.output_norm.bias.fill_(1)
        model.word_embedding.weight[2].fill_(1)
    result = model.generate(features, concepts, max_new_tokens=8)
    assert result.token_ids.tolist() == [[2], [2]]
    assert result.lengths.tolist() == [1, 1] and result.reached_eos.all()
    with torch.no_grad():
        model.word_embedding.weight[2].zero_()
        model.word_embedding.weight[3].fill_(1)
    result = model.generate(features, concepts, max_new_tokens=3)
    assert result.token_ids.tolist() == [[3, 3, 3], [3, 3, 3]]
    assert not result.reached_eos.any()
    with pytest.raises(ValueError):
        model.generate(features, concepts, max_new_tokens=9)
    with pytest.raises(InterruptedError):
        model.generate(features, concepts, max_new_tokens=3, cancelled=lambda: True)


def test_tiny_caption_overfit_on_cpu():
    features, concepts, model = fixture(batch=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    sequence = torch.tensor([[1, 4, 5, 2]])
    initial = model.teacher_loss(features, concepts, sequence, label_smoothing=0.0).item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = model.teacher_loss(features, concepts, sequence, label_smoothing=0.0)
        loss.backward()
        optimizer.step()
    assert loss.item() < initial * 0.05
    result = model.eval().generate(features, concepts, max_new_tokens=4)
    assert result.token_ids.tolist() == [[4, 5, 2]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_memory_and_cache_generation_are_finite_in_supported_dtypes(dtype):
    features, concepts, model = fixture()
    model.to(dtype=dtype).eval()
    sequences = torch.tensor([[1, 4, 2], [1, 5, 2]])
    loss = model.teacher_loss(features, concepts, sequences, label_smoothing=0.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result = model.generate(features, concepts, max_new_tokens=3)
    assert 1 <= result.token_ids.shape[1] <= 3


def test_concept_permutation_and_variable_spatial_padding_preserve_decoding():
    features, concepts, model = fixture()
    model.eval()
    sequence = torch.tensor([[1, 4], [1, 5]])
    reference = model(features, concepts, sequence)
    permutation = torch.tensor([2, 0, 1])
    shuffled = replace(concepts, **{field: getattr(concepts, field)[:, permutation] for field in
                                    ("ids", "scores", "categories", "sources", "mask")})
    extended = replace(features,
                       spatial_tokens=torch.cat((features.spatial_tokens, torch.full((2, 7, 8), 50)), dim=1),
                       spatial_mask=torch.cat((features.spatial_mask, torch.zeros(2, 7, dtype=torch.bool)), dim=1),
                       spatial_xy=torch.cat((features.spatial_xy, torch.zeros(2, 7, 2)), dim=1))
    torch.testing.assert_close(model(extended, shuffled, sequence), reference, atol=2e-5, rtol=2e-5)
