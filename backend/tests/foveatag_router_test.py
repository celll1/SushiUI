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


def test_dense_oracle_uses_only_positive_tags_and_is_detached():
    scores = torch.tensor([[[1.0, 10.0], [3.0, 0.0]]], requires_grad=True)
    overview = torch.tensor([[0.0, 0.0]], requires_grad=True)
    targets = dense_oracle_targets(scores, overview, torch.tensor([[True, False]]))

    torch.testing.assert_close(targets, torch.tensor([[1 / 3, 1.0]]))
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

    torch.testing.assert_close(targets, torch.tensor([[1 / 3, 1.0]]))


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
