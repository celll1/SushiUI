"""CPU tests for routed FoveaTag joint fine-tuning."""

import torch
from torch import nn
from torch.nn import functional as F

from core.tagger.foveatag.batch import collate_view_batches
from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.joint_training import (
    FoveaJointTrainer,
    gather_selected_views,
)
from core.tagger.foveatag.model import FoveaTagger
from core.tagger.foveatag.router import EvidenceRouter
from core.tagger.foveatag.router_data import FoveaRouterBatch
from core.tagger.foveatag.views import ViewConfig, build_views
from PIL import Image


class _MaskedBCE(nn.Module):
    def forward(self, logits, labels, loss_mask):
        values = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        return (values * loss_mask).sum() / loss_mask.sum().clamp_min(1)


def _batch():
    config = ViewConfig(
        overview_size=16,
        local_size=16,
        local_views=4,
        field_levels=(0.5, 1.0),
        scale_jitter=(1.0, 1.0),
    )
    view_batch = collate_view_batches([
        build_views(Image.new("RGB", (48, 32), (80, 120, 160)), config, seed=3, training=False),
        build_views(Image.new("RGB", (32, 48), (160, 100, 40)), config, seed=5, training=False),
    ])
    return FoveaRouterBatch(
        views=view_batch,
        candidate_boxes=view_batch.local_geometry[..., :4].clone(),
        labels=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        loss_masks=torch.ones(2, 2),
    )


def _modules():
    encoder = FoveaEncoderConfig(
        widths=(8, 16, 32, 48),
        depths=(1, 1, 2, 1),
        window_size=4,
        attention_heads=(2, 3),
        mlp_ratio=2.0,
        scale_dropout=0.0,
    )
    model = FoveaTagger(
        2,
        encoder_config=encoder,
        feature_dim=16,
        evidence_slots_per_view=1,
        pool_heads=2,
    )
    router = EvidenceRouter(feature_dim=32, routing_dim=16, routing_slots=2)
    return model, router


def test_gather_selected_views_preserves_budget_and_invalid_slots():
    batch = _batch()
    selected = torch.tensor([[3, 1, -1], [0, 2, 1]])

    gathered = gather_selected_views(batch.views, selected)

    assert gathered.local_images.shape == (2, 3, 3, 16, 16)
    assert gathered.local_view_mask.tolist() == [[True, True, False], [True, True, True]]
    torch.testing.assert_close(gathered.local_geometry[0, 0], batch.views.local_geometry[0, 3])


def test_joint_training_updates_tagger_and_router_on_oracle_step():
    torch.manual_seed(7)
    model, router = _modules()
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *router.parameters()],
        lr=1e-3,
    )
    trainer = FoveaJointTrainer(
        model,
        router,
        _MaskedBCE(),
        optimizer,
        device="cpu",
        selection_budget=2,
        random_count=1,
        oracle_interval=2,
        oracle_chunk_size=2,
        mixed_precision=None,
    )
    model_before = model.head.weight.detach().clone()
    router_before = router.slots.detach().clone()

    result = trainer.train_batch(_batch(), global_step=2, random_seed=11)

    assert result is not None and result.oracle_used
    assert result.selected_views == 4
    assert not torch.equal(model.head.weight, model_before)
    assert not torch.equal(router.slots, router_before)


def test_joint_training_skips_dense_oracle_between_registered_intervals():
    torch.manual_seed(9)
    model, router = _modules()
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *router.parameters()],
        lr=1e-3,
    )
    trainer = FoveaJointTrainer(
        model,
        router,
        _MaskedBCE(),
        optimizer,
        device="cpu",
        selection_budget=2,
        random_count=1,
        oracle_interval=4,
        oracle_chunk_size=2,
        mixed_precision=None,
    )
    router_before = {name: value.detach().clone() for name, value in router.state_dict().items()}

    result = trainer.train_batch(_batch(), global_step=3, random_seed=13)

    assert result is not None and not result.oracle_used
    assert result.router_ranking == 0.0
    for name, value in router.state_dict().items():
        torch.testing.assert_close(value, router_before[name])


def test_joint_validation_uses_fixed_routed_budget_and_restores_modes():
    model, router = _modules()
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *router.parameters()],
        lr=1e-3,
    )
    trainer = FoveaJointTrainer(
        model,
        router,
        _MaskedBCE(),
        optimizer,
        device="cpu",
        selection_budget=2,
        random_count=1,
        oracle_interval=4,
        mixed_precision=None,
    )
    model.train()
    router.train()

    result = trainer.validate_epoch([_batch()], thresholds=(0.25, 0.5, 0.75))

    assert result.samples == 2
    assert result.known_labels == 4
    assert result.threshold in (0.25, 0.5, 0.75)
    assert model.training and router.training
