"""CPU tests for FoveaTag supervised and router loss composition."""

import torch
from torch import nn

from core.tagger.foveatag.training import FoveaTagTrainingLoss, prediction_consistency_loss


class MaskedMSE(nn.Module):
    def forward(self, logits, labels, loss_mask=None):
        loss = (logits - labels).pow(2)
        if loss_mask is not None:
            loss = loss * loss_mask
        return loss.mean()


def test_consistency_ignores_tag_when_either_view_lacks_density():
    first = torch.tensor([[5.0, 5.0]], requires_grad=True)
    second = torch.tensor([[-5.0, -5.0]], requires_grad=True)
    loss = prediction_consistency_loss(
        first,
        second,
        first_evidence_density=torch.tensor([[2.0, 0.5]]),
        second_evidence_density=torch.tensor([[2.0, 2.0]]),
        required_density=torch.tensor([1.0, 1.0]),
    )
    reference = prediction_consistency_loss(
        first[:, :1],
        second[:, :1],
        torch.tensor([[2.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([1.0]),
    )

    torch.testing.assert_close(loss, reference)


def test_consistency_with_no_resolved_tags_is_differentiable_zero():
    first = torch.randn(1, 2, requires_grad=True)
    second = torch.randn(1, 2, requires_grad=True)
    loss = prediction_consistency_loss(
        first,
        second,
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(2),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert first.grad is not None and second.grad is not None


def test_composite_loss_uses_tag_mask_and_optional_router_terms():
    objective = FoveaTagTrainingLoss(
        MaskedMSE(),
        consistency_weight=0.5,
        router_weight=2.0,
        diversity_weight=0.25,
    )
    logits = torch.tensor([[1.0, 10.0]], requires_grad=True)
    output = objective(
        logits,
        torch.zeros_like(logits),
        torch.tensor([[1.0, 0.0]]),
        second_logits=torch.zeros_like(logits),
        first_evidence_density=torch.ones_like(logits),
        second_evidence_density=torch.ones_like(logits),
        required_density=torch.zeros(2),
        router_scores=torch.tensor([[0.0, 1.0]], requires_grad=True),
        oracle_targets=torch.tensor([[0.0, 1.0]]),
        router_slot_scores=torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]),
    )
    expected = (
        output.classification
        + output.consistency * 0.5
        + output.router_ranking * 2.0
        + output.router_diversity * 0.25
    )

    torch.testing.assert_close(output.loss, expected)
    torch.testing.assert_close(output.classification, torch.tensor(0.5))
