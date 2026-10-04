"""CPU tests for FoveaTag self-supervised objectives."""

import torch

from core.tagger.foveatag.objectives import (
    FoveaSelfSupervisedLoss,
    chroma_reconstruction_loss,
    feature_distribution_regularization,
    masked_patch_reconstruction_loss,
    overlap_correspondence_loss,
    rgb_to_chroma,
)


def _coordinates(offset=0.0):
    return torch.tensor([[[[0.0 + offset, 0.0], [1.0 + offset, 0.0]]]])


def test_overlap_correspondence_matches_cells_by_source_coordinates():
    first = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    second = torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]])
    second_coordinates = torch.flip(_coordinates(), dims=(2,))

    loss = overlap_correspondence_loss(
        first,
        _coordinates(),
        second,
        second_coordinates,
        max_coordinate_distance=0.01,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_non_overlapping_correspondence_is_differentiable_zero():
    first = torch.randn(1, 3, 1, 2, requires_grad=True)
    second = torch.randn(1, 3, 1, 2, requires_grad=True)
    loss = overlap_correspondence_loss(
        first,
        _coordinates(),
        second,
        _coordinates(offset=10.0),
        max_coordinate_distance=0.1,
    )

    loss.backward()
    assert loss.item() == 0.0
    assert first.grad is not None and second.grad is not None


def test_masked_reconstruction_ignores_unselected_pixels():
    target = torch.zeros(1, 3, 1, 2)
    prediction = target.clone()
    prediction[..., 1] = 100.0
    mask = torch.tensor([[[True, False]]])

    loss = masked_patch_reconstruction_loss(prediction, target, mask)

    torch.testing.assert_close(loss, torch.tensor(0.001))


def test_chroma_loss_detects_color_change_with_equal_luminance_sum():
    red = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
    green = torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]])
    mask = torch.ones(1, 1, 1, dtype=torch.bool)

    assert torch.isclose(red.sum(), green.sum())
    assert chroma_reconstruction_loss(red, green, mask) > 0
    assert rgb_to_chroma(red).shape == (1, 2, 1, 1)


def test_distribution_regularization_penalizes_constant_features():
    collapsed = torch.ones(1, 2, 2, 2)
    diverse = torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]]])

    collapsed_terms = feature_distribution_regularization(collapsed, collapsed)
    diverse_terms = feature_distribution_regularization(diverse, diverse)

    assert collapsed_terms[0] > diverse_terms[0]
    torch.testing.assert_close(diverse_terms[0], torch.tensor(0.0))
    torch.testing.assert_close(diverse_terms[1], torch.tensor(0.0))
    torch.testing.assert_close(collapsed_terms[2], torch.tensor(1.0))
    torch.testing.assert_close(diverse_terms[2], torch.tensor(0.0))


def test_distribution_regularization_has_finite_gradients():
    first = torch.randn(2, 4, 3, 3, requires_grad=True)
    second = torch.randn(2, 4, 3, 3, requires_grad=True)

    sum(feature_distribution_regularization(first, second)).backward()

    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()


def test_composite_loss_reports_weighted_components():
    objective = FoveaSelfSupervisedLoss(
        correspondence_weight=2.0,
        reconstruction_weight=3.0,
        chroma_weight=4.0,
        variance_weight=5.0,
        covariance_weight=6.0,
        centering_weight=7.0,
        max_coordinate_distance=0.1,
    )
    features = torch.tensor([[[[1.0]], [[0.0]]]])
    coordinates = torch.zeros(1, 1, 1, 2)
    target = torch.zeros(1, 3, 1, 1)
    prediction = torch.ones(1, 3, 1, 1)
    output = objective(
        features,
        coordinates,
        features,
        coordinates,
        prediction,
        target,
        torch.ones(1, 1, 1, dtype=torch.bool),
    )

    expected = (
        output.correspondence * 2
        + output.reconstruction * 3
        + output.chroma * 4
        + output.variance * 5
        + output.covariance * 6
        + output.centering * 7
    )
    torch.testing.assert_close(output.loss, expected)
