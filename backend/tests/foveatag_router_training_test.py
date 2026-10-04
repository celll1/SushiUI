"""CPU tests for bounded dense-oracle and router training."""

import copy

import torch
from PIL import Image

from core.tagger.foveatag.batch import collate_view_batches
from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.model import FoveaTagger
from core.tagger.foveatag.router import EvidenceRouter
from core.tagger.foveatag.router_data import FoveaRouterBatch
from core.tagger.foveatag.router_training import (
    FoveaRouterTrainer,
    evaluate_dense_candidates,
    train_router_batch,
)
from core.tagger.foveatag.views import ViewConfig, build_views


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


def test_router_trainer_steps_and_reports_learned_and_random_fixed_k_recovery():
    model, router = _components()
    rendered = [
        build_views(
            Image.new("RGB", (40, 32), color),
            ViewConfig(
                overview_size=32,
                local_size=16,
                local_views=4,
                field_levels=(0.5, 1.0),
                scale_jitter=(1.0, 1.0),
            ),
            seed=index,
            training=False,
        )
        for index, color in enumerate(((20, 40, 60), (60, 40, 20)))
    ]
    view_batch = collate_view_batches(rendered)
    batch = FoveaRouterBatch(
        views=view_batch,
        candidate_boxes=view_batch.local_geometry[..., :4].clone(),
        labels=torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        loss_masks=torch.ones(2, 3),
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = FoveaRouterTrainer(
        model,
        router,
        optimizer,
        device="cpu",
        mixed_precision=None,
        oracle_chunk_size=2,
        scheduler=scheduler,
    )

    train_result = trainer.train_batch(batch)
    validation = trainer.validate([batch], budget=2, random_seed=11)
    repeated = trainer.validate([batch], budget=2, random_seed=11)

    assert train_result is not None and train_result.loss >= 0
    assert optimizer.param_groups[0]["lr"] == 5e-4
    assert validation.samples == 2
    assert validation.candidates_per_sample == 4
    assert validation.budget == 2
    assert 0 <= validation.mean_utility_recovery <= 1
    assert 0 <= validation.random_mean_utility_recovery <= 1
    assert validation == repeated
    assert all(parameter.grad is None for parameter in model.parameters())
