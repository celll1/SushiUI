"""CPU tests for FoveaTag evidence routing."""

import torch

from core.tagger.foveatag.router import (
    EvidenceRouter,
    build_scale_space_candidates,
    dense_oracle_targets,
    router_diversity_loss,
    router_ranking_loss,
    select_proposals,
    select_training_proposals,
    source_candidate_mask,
)


def test_candidate_builder_combines_every_cell_and_scale():
    coordinates = torch.tensor([[[[0.25, 0.5], [0.75, 0.5]]]])
    candidates = build_scale_space_candidates(coordinates, [(0.1, 0.2), (0.4, 0.5)])

    assert candidates.shape == (1, 4, 4)
    torch.testing.assert_close(candidates[0, 0], torch.tensor([0.25, 0.5, 0.1, 0.2]))
    torch.testing.assert_close(candidates[0, 3], torch.tensor([0.75, 0.5, 0.4, 0.5]))


def test_candidate_mask_rejects_letterbox_cells():
    boxes = torch.tensor([[[-0.1, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2], [1.1, 0.5, 0.2, 0.2]]])
    assert source_candidate_mask(boxes).tolist() == [[False, True, False]]


def test_router_scores_candidates_without_vocabulary_axis():
    router = EvidenceRouter(feature_dim=8, routing_dim=12, routing_slots=3)
    feature_map = torch.randn(2, 8, 2, 2)
    axis = torch.tensor([0.25, 0.75])
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack((x, y), dim=-1).unsqueeze(0).expand(2, -1, -1, -1)
    candidates = build_scale_space_candidates(coordinates, [(0.2, 0.2), (0.4, 0.4)])
    output = router(feature_map, coordinates, candidates)

    assert output.scores.shape == (2, 8)
    assert output.slot_scores.shape == (2, 3, 8)


def test_dense_oracle_penalizes_known_false_positives_and_is_detached():
    scores = torch.tensor([[[1.0, 10.0], [3.0, 0.0]]], requires_grad=True)
    overview = torch.tensor([[0.0, 0.0]], requires_grad=True)
    targets = dense_oracle_targets(scores, overview, torch.tensor([[True, False]]))

    torch.testing.assert_close(targets, torch.tensor([[0.0, 1.0]]))
    assert not targets.requires_grad
    loss = router_ranking_loss(torch.zeros_like(targets, requires_grad=True), targets)
    assert loss > 0


def test_dense_oracle_is_invariant_to_tag_identity():
    scores = torch.tensor([
        [[1.0, 8.0, 2.0], [4.0, 3.0, 0.5], [2.0, 7.0, 6.0]],
    ])
    overview = torch.tensor([[0.5, 1.0, 0.25]])
    positives = torch.tensor([[True, False, True]])
    permutation = torch.tensor([2, 0, 1])

    original = dense_oracle_targets(scores, overview, positives)
    renamed = dense_oracle_targets(
        scores[..., permutation],
        overview[..., permutation],
        positives[..., permutation],
    )

    torch.testing.assert_close(renamed, original)


def test_dense_oracle_scores_incremental_evidence_not_absolute_confidence():
    candidate_scores = torch.tensor([[[8.0, 0.0], [4.0, 0.0]]])
    overview_scores = torch.tensor([[7.5, 0.0]])
    positives = torch.tensor([[True, False]])

    targets = dense_oracle_targets(candidate_scores, overview_scores, positives)

    torch.testing.assert_close(targets, torch.tensor([[1.0, 0.0]]))


def test_dense_oracle_respects_unknown_label_mask():
    candidate_scores = torch.tensor([[[1.0, 10.0], [3.0, 0.0]]])
    overview_scores = torch.zeros(1, 2)
    positives = torch.tensor([[True, True]])
    loss_mask = torch.tensor([[1.0, 0.0]])

    targets = dense_oracle_targets(
        candidate_scores,
        overview_scores,
        positives,
        loss_mask=loss_mask,
    )

    from core.tagger.tagger_loss import AsymmetricLossOptimized
    loss = AsymmetricLossOptimized(reduction="none")
    baseline = loss(overview_scores, positives.float(), loss_mask).mean(-1)
    delta = torch.stack([baseline - loss(candidate_scores[:, i], positives.float(), loss_mask).mean(-1)
        for i in range(2)], dim=1).clamp_min(0)
    torch.testing.assert_close(targets, delta / delta.max(dim=1, keepdim=True).values)


def test_dense_oracle_can_reward_negative_suppression_without_positives():
    targets = dense_oracle_targets(torch.tensor([[[0.0], [5.0]]]),
        torch.tensor([[4.0]]), torch.zeros(1, 1))
    torch.testing.assert_close(targets, torch.tensor([[1.0, 0.0]]))


def test_dense_oracle_unknown_negatives_do_not_penalize_candidates():
    scores = torch.tensor([[[3.0, 10.0], [1.0, -10.0]]])
    targets = dense_oracle_targets(scores, torch.zeros(1, 2), torch.tensor([[1., 0.]]),
        loss_mask=torch.tensor([[.25, 0.]]))
    assert targets[0, 0] == 1 and 0 < targets[0, 1] < 1
    swapped_unknown = scores.clone()
    swapped_unknown[..., 1] *= -1
    torch.testing.assert_close(targets, dense_oracle_targets(swapped_unknown, torch.zeros(1, 2),
        torch.tensor([[1., 0.]]), loss_mask=torch.tensor([[.25, 0.]])))


def test_dense_oracle_all_unknown_or_worsening_candidates_get_no_reward():
    scores, overview = torch.tensor([[[0., 5.], [-2., 8.]]]), torch.tensor([[2., -3.]])
    labels = torch.tensor([[1., 0.]])
    assert not dense_oracle_targets(scores, overview, labels).any()
    assert not dense_oracle_targets(scores, overview, labels, loss_mask=torch.zeros_like(labels)).any()


def test_dense_oracle_uses_supplied_asl_parameters_and_soft_labels():
    from core.tagger.tagger_loss import AsymmetricLossOptimized
    loss = AsymmetricLossOptimized(gamma_pos=2, gamma_neg=1, clip=.1, reduction="none")
    labels, mask = torch.tensor([[.8, .1]]), torch.tensor([[.25, 1.]])
    scores, baseline = torch.tensor([[[2., -2.], [1., -1.]]]), torch.zeros(1, 2)
    improvements = torch.stack([loss(baseline, labels, mask).mean(-1) - loss(scores[:, i], labels, mask).mean(-1)
        for i in range(2)], dim=1).clamp_min(0)
    expected = improvements / improvements.max(dim=1, keepdim=True).values
    torch.testing.assert_close(dense_oracle_targets(scores, baseline, labels, mask, criterion=loss), expected)


def test_scale_space_nms_suppresses_overlapping_boxes():
    scores = torch.tensor([[3.0, 2.0, 1.0]])
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.51, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1]]])
    selected = select_proposals(scores, boxes, budget=2, nms_iou=0.5)

    assert selected.tolist() == [[0, 2]]


def test_training_selection_reserves_random_coverage():
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    boxes = torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]])
    generator = torch.Generator().manual_seed(4)
    selected = select_training_proposals(
        scores,
        boxes,
        budget=3,
        random_count=1,
        generator=generator,
    )

    assert selected[0, :2].tolist() == [0, 1]
    assert selected[0, 2].item() in {2, 3}


def test_diversity_penalty_is_lower_for_separate_slot_distributions():
    collapsed = torch.tensor([[[10.0, 0.0], [10.0, 0.0]]])
    separate = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])

    assert router_diversity_loss(separate) < router_diversity_loss(collapsed)


def test_ranking_loss_is_finite_differentiable_zero_when_all_candidates_are_masked():
    scores = torch.full((1, 2), -torch.inf, requires_grad=True)
    loss = router_ranking_loss(
        scores,
        torch.zeros_like(scores),
        candidate_mask=torch.zeros_like(scores, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(scores.grad).all()
