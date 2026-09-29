"""CPU tests for bounded dense-oracle and router training."""

import copy

import torch

from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.model import FoveaTagger
from core.tagger.foveatag.router import EvidenceRouter
from core.tagger.foveatag.router_training import (
    evaluate_dense_candidates,
    train_router_batch,
)


def _components():
    encoder = FoveaEncoderConfig(
        widths=(8, 16, 32, 48),
        depths=(1, 1, 2, 1),
        window_size=4,
        attention_heads=(2, 3),
        mlp_ratio=2.0,
        scale_dropout=0.2,
    )
    model = FoveaTagger(
        3,
        encoder_config=encoder,
        feature_dim=16,
        evidence_slots_per_view=1,
        pool_heads=2,
    )
    router = EvidenceRouter(feature_dim=32, routing_dim=16, routing_slots=2)
    return model, router


def _inputs(candidates=5):
    overview_images = torch.randn(2, 3, 32, 32)
    overview_geometry = torch.zeros(2, 10)
    overview_geometry[:, 8:10] = 1
    candidate_images = torch.randn(2, candidates, 3, 16, 16)
    candidate_geometry = torch.zeros(2, candidates, 10)
    candidate_geometry[..., :4] = torch.rand(2, candidates, 4)
    candidate_geometry[..., 2:4] = candidate_geometry[..., 2:4] * 0.2 + 0.1
    candidate_geometry[..., 8:10] = 1
    boxes = candidate_geometry[..., :4].clone()
    return overview_images, overview_geometry, candidate_images, candidate_geometry, boxes


def test_dense_candidates_are_chunked_and_overview_is_encoded_once():
    model, _router = _components()
    inputs = _inputs()
    batch_sizes = []
    handle = model.encoder.register_forward_hook(
        lambda _module, values, _output: batch_sizes.append(values[0].shape[0])
    )
    try:
        oracle = evaluate_dense_candidates(
            model,
            *inputs[:4],
            chunk_size=2,
        )
    finally:
        handle.remove()

    assert oracle.overview_logits.shape == (2, 3)
    assert oracle.candidate_logits.shape == (2, 5, 3)
    assert batch_sizes == [2, 4, 4, 2]
    assert not oracle.overview_logits.requires_grad
    assert not oracle.candidate_logits.requires_grad


def test_dense_candidate_logits_do_not_change_with_chunk_size():
    model, _router = _components()
    inputs = _inputs()

    first = evaluate_dense_candidates(model, *inputs[:4], chunk_size=1)
    second = evaluate_dense_candidates(model, *inputs[:4], chunk_size=4)

    torch.testing.assert_close(first.overview_logits, second.overview_logits)
    torch.testing.assert_close(first.candidate_logits, second.candidate_logits)


def test_router_step_updates_only_router_from_generic_positive_utility():
    model, router = _components()
    inputs = _inputs(candidates=4)
    oracle = evaluate_dense_candidates(model, *inputs[:4], chunk_size=2)
    model_before = copy.deepcopy(model.state_dict())
    router_before = copy.deepcopy(router.state_dict())
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-2)
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    loss_mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])

    result = train_router_batch(
        router,
        optimizer,
        oracle,
        inputs[4],
        labels,
        loss_mask=loss_mask,
    )

    assert result.loss >= 0
    assert any(
        not torch.equal(router_before[name], value)
        for name, value in router.state_dict().items()
    )
    assert all(
        torch.equal(model_before[name], value)
        for name, value in model.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())
