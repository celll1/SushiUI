"""CPU tests for the SigLIP2-backed FoveaTag control model."""

from types import SimpleNamespace

import torch
from torch import nn

from core.tagger.foveatag.siglip2_control import (
    MaskedSlotPooler,
    SigLIP2ControlTagger,
    SigLIP2ViewInputs,
)


class FakeNaFlexEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_size)
        self.calls = []

    def forward(self, *, pixel_values, pixel_attention_mask, spatial_shapes):
        self.calls.append((pixel_values.shape, pixel_attention_mask, spatial_shapes))
        return SimpleNamespace(last_hidden_state=self.projection(pixel_values))


class FakeFixedEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 8, kernel_size=2, stride=2)

    def forward(self, *, pixel_values):
        features = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return SimpleNamespace(
            last_hidden_state=features,
            pooler_output=features.mean(dim=1),
        )


def test_control_model_flattens_views_and_returns_source_view_indices():
    encoder = FakeNaFlexEncoder(input_dim=6, hidden_size=8)
    model = SigLIP2ControlTagger(
        encoder,
        hidden_size=8,
        feature_dim=4,
        num_tags=3,
        evidence_slots_per_view=2,
        pool_heads=2,
    )
    overview = SigLIP2ViewInputs(
        pixel_values=torch.randn(2, 5, 6),
        pixel_attention_mask=torch.ones(2, 5, dtype=torch.bool),
        spatial_shapes=torch.tensor([[1, 5], [1, 5]]),
    )
    local = SigLIP2ViewInputs(
        pixel_values=torch.randn(2, 3, 4, 6),
        pixel_attention_mask=torch.ones(2, 3, 4, dtype=torch.bool),
        spatial_shapes=torch.ones(2, 3, 2, dtype=torch.long),
    )
    details = model(
        overview,
        local,
        local_view_mask=torch.tensor([[True, False, True], [True, True, True]]),
        return_details=True,
    )

    assert details.logits.shape == (2, 3)
    assert details.head_output.local_scores.shape == (2, 6, 3)
    assert details.evidence_view_indices.tolist() == [
        [0, 0, -1, -1, 2, 2],
        [0, 0, 1, 1, 2, 2],
    ]
    assert encoder.calls[1][0] == torch.Size([6, 4, 6])


def test_invalid_view_with_fully_masked_tokens_stays_finite():
    encoder = FakeNaFlexEncoder(input_dim=3, hidden_size=8)
    model = SigLIP2ControlTagger(
        encoder,
        hidden_size=8,
        feature_dim=4,
        num_tags=2,
        pool_heads=2,
    )
    overview = SigLIP2ViewInputs(torch.randn(1, 2, 3), torch.ones(1, 2, dtype=torch.bool))
    local = SigLIP2ViewInputs(torch.randn(1, 2, 2, 3), torch.zeros(1, 2, 2, dtype=torch.bool))

    output = model(overview, local, local_view_mask=torch.zeros(1, 2, dtype=torch.bool))

    assert torch.isfinite(output).all()


def test_zero_local_views_skip_second_encoder_call():
    encoder = FakeNaFlexEncoder(input_dim=3, hidden_size=8)
    model = SigLIP2ControlTagger(
        encoder,
        hidden_size=8,
        feature_dim=4,
        num_tags=2,
        pool_heads=2,
    )
    overview = SigLIP2ViewInputs(torch.randn(2, 3, 3), torch.ones(2, 3, dtype=torch.bool))
    local = SigLIP2ViewInputs(torch.empty(2, 0, 3, 3), torch.empty(2, 0, 3, dtype=torch.bool))

    details = model(overview, local, return_details=True)

    assert details.logits.shape == (2, 2)
    assert details.evidence_view_indices.shape == (2, 0)
    assert len(encoder.calls) == 1


def test_pooler_rejects_hidden_size_not_divisible_by_heads():
    try:
        MaskedSlotPooler(hidden_size=7, slot_count=2, num_heads=4)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected invalid attention dimensions to fail")


def test_fixed_resolution_encoder_path_uses_image_tensors():
    model = SigLIP2ControlTagger(
        FakeFixedEncoder(),
        hidden_size=8,
        feature_dim=4,
        num_tags=2,
        pool_heads=2,
        is_naflex=False,
    )
    overview = SigLIP2ViewInputs(torch.randn(1, 3, 8, 8))
    local = SigLIP2ViewInputs(torch.randn(1, 2, 3, 4, 4))

    logits = model(overview, local)

    assert logits.shape == (1, 2)


def test_baseline_preserving_control_is_exact_before_local_training():
    encoder = FakeFixedEncoder()
    baseline_head = nn.Linear(8, 2)
    model = SigLIP2ControlTagger(
        encoder,
        hidden_size=8,
        feature_dim=4,
        num_tags=2,
        pool_heads=2,
        is_naflex=False,
        baseline_head=baseline_head,
    )
    overview = SigLIP2ViewInputs(torch.randn(1, 3, 8, 8))
    local = SigLIP2ViewInputs(torch.randn(1, 4, 3, 4, 4))

    with torch.no_grad():
        expected = baseline_head(encoder(pixel_values=overview.pixel_values).pooler_output)
        actual = model(overview, local)

    torch.testing.assert_close(actual, expected)
    assert not any(parameter.requires_grad for parameter in model.baseline_head.parameters())
