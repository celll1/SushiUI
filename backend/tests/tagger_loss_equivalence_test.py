import pytest
import torch

from backend.core.tagger.tagger_loss import AsymmetricLossOptimized


def _legacy_asl(
    x,
    y,
    loss_mask,
    *,
    gamma_neg,
    gamma_pos,
    clip,
    eps,
    disable_grad_focal,
    reduction,
):
    targets = y
    anti_targets = 1.0 - y
    xs_pos = torch.sigmoid(x)
    xs_neg = 1.0 - xs_pos
    if clip is not None and clip > 0:
        xs_neg.add_(clip).clamp_(max=1.0)
    loss = targets * torch.log(xs_pos.clamp(min=eps))
    loss.add_(anti_targets * torch.log(xs_neg.clamp(min=eps)))
    if gamma_neg > 0 or gamma_pos > 0:
        if disable_grad_focal:
            with torch.no_grad():
                weight = torch.pow(
                    1.0 - xs_pos * targets - xs_neg * anti_targets,
                    gamma_pos * targets + gamma_neg * anti_targets,
                )
        else:
            weight = torch.pow(
                1.0 - xs_pos * targets - xs_neg * anti_targets,
                gamma_pos * targets + gamma_neg * anti_targets,
            )
        loss *= weight
    if loss_mask is not None:
        loss = loss * loss_mask
    if reduction == "mean":
        return -loss.mean()
    if reduction == "sum":
        return -loss.sum()
    return -loss


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("disable_grad_focal", [False, True])
@pytest.mark.parametrize("clip", [0.0, 0.05])
def test_asymmetric_loss_matches_legacy_forward_and_gradient(
    reduction, disable_grad_focal, clip
):
    logits = torch.tensor(
        [[-3.0, -0.25, 0.0, 2.5], [4.0, -1.5, 0.75, -0.01]],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    kwargs = dict(
        gamma_neg=4.0,
        gamma_pos=1.0,
        clip=clip,
        eps=1e-6,
        reduction=reduction,
    )
    actual_input = logits.clone().requires_grad_(True)
    expected_input = logits.clone().requires_grad_(True)

    actual = AsymmetricLossOptimized(
        **kwargs, disable_torch_grad_focal_loss=disable_grad_focal
    )(actual_input, targets, mask)
    expected = _legacy_asl(
        expected_input,
        targets,
        mask,
        **kwargs,
        disable_grad_focal=disable_grad_focal,
    )

    assert torch.equal(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    assert torch.equal(actual_input.grad, expected_input.grad)


def test_asymmetric_loss_does_not_retain_forward_tensors():
    criterion = AsymmetricLossOptimized()
    criterion(torch.zeros(1, 2, requires_grad=True), torch.zeros(1, 2))

    assert not any(isinstance(value, torch.Tensor) for value in vars(criterion).values())
