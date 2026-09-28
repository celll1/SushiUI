"""CPU tests for FoveaTag evidence-slot aggregation."""

import torch

from core.tagger.foveatag.head import EvidenceTagHead


def _identity_mixer(head: EvidenceTagHead) -> None:
    for parameter in head.mixer.parameters():
        parameter.data.zero_()


def test_local_logsumexp_is_per_tag_and_respects_padding_mask():
    head = EvidenceTagHead(feature_dim=2, num_tags=2, temperature=0.5)
    _identity_mixer(head)
    with torch.no_grad():
        head.head.weight.copy_(torch.tensor([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))
        head.head.bias.zero_()

    evidence = torch.tensor([[[3.0, 0.0], [0.0, 2.0], [100.0, 100.0]]])
    output = head(
        torch.zeros((1, 2)),
        evidence,
        torch.tensor([[True, True, False]]),
    )

    expected = 0.5 * torch.logsumexp(evidence[0, :2] / 0.5, dim=0)
    torch.testing.assert_close(output.logits[0], expected)
    assert output.best_evidence_indices.tolist() == [[0, 1]]


def test_sample_with_no_valid_evidence_uses_only_global_term():
    head = EvidenceTagHead(feature_dim=2, num_tags=1)
    _identity_mixer(head)
    with torch.no_grad():
        head.head.weight.copy_(torch.tensor([[2.0, -1.0, 50.0, 50.0]]))
        head.head.bias.copy_(torch.tensor([0.25]))

    output = head(
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([[[100.0, 100.0]]]),
        torch.tensor([[False]]),
    )

    torch.testing.assert_close(output.logits, torch.tensor([[2.25]]))
    assert output.best_evidence_indices.tolist() == [[-1]]


def test_zero_slot_tensor_is_supported():
    head = EvidenceTagHead(feature_dim=4, num_tags=3)
    output = head(torch.randn(2, 4), torch.empty(2, 0, 4))

    assert output.logits.shape == (2, 3)
    assert output.local_scores.shape == (2, 0, 3)
    assert output.best_evidence_indices.tolist() == [[-1, -1, -1], [-1, -1, -1]]


def test_expand_head_preserves_existing_rows_and_appends_parameters():
    head = EvidenceTagHead(feature_dim=3, num_tags=2)
    old_weight = head.head.weight.detach().clone()
    old_bias = head.head.bias.detach().clone()

    new_weight, new_bias = head.expand_head(5)

    assert head.num_tags == 5
    assert new_weight.shape == (5, 6)
    assert new_bias is not None and new_bias.shape == (5,)
    torch.testing.assert_close(new_weight[:2], old_weight)
    torch.testing.assert_close(new_bias[:2], old_bias)


def test_overview_conditioning_changes_local_slots():
    torch.manual_seed(7)
    head = EvidenceTagHead(feature_dim=4, num_tags=2)
    evidence = torch.randn(1, 3, 4)

    first = head(torch.zeros(1, 4), evidence).mixed_evidence
    second = head(torch.tensor([[1.0, 0.0, -1.0, 0.5]]), evidence).mixed_evidence

    assert not torch.allclose(first, second)
