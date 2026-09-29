"""CPU tests for the Phase-3 branch runner contracts."""

import torch
from PIL import Image
from torch import nn

from core.tagger.foveatag.supervised_data import FoveaSupervisedSample
from core.tagger.foveatag.supervised_run import (
    FoveaRunControl,
    _encoder_config,
    _train_epoch,
    _validate_arguments,
)
from core.tagger.foveatag.views import ViewConfig, build_views


class _Samples:
    def __init__(self, count):
        views = build_views(
            Image.new("RGB", (16, 16), (1, 2, 3)),
            ViewConfig(overview_size=16, local_size=8, local_views=0),
            seed=0,
            training=False,
        )
        self.values = [
            FoveaSupervisedSample(
                views=views,
                label=torch.tensor([1.0, 0.0]),
                loss_mask=torch.ones(2),
            )
            for _ in range(count)
        ]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class _Trainer:
    def __init__(self):
        self.model = nn.Linear(2, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.5
        )

    def train_batch(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.model(batch.labels).square().mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return float(loss.detach())


def test_train_epoch_stops_at_exact_resumable_sample_offset(tmp_path):
    trainer = _Trainer()
    contract = {"input_mode": "source_native"}

    result, offset, step, added, stopped, accumulator = _train_epoch(
        trainer,
        _Samples(5),
        epoch=0,
        sample_offset=0,
        batch_size=2,
        workers=0,
        checkpoint_every_steps=1,
        log_every=10,
        global_step=0,
        output_dir=tmp_path,
        best_f1=0.0,
        best_threshold=0.5,
        run_contract=contract,
        history=[],
        initial_accumulator=None,
        remaining_step_budget=2,
    )

    assert stopped
    assert (offset, step, added) == (4, 2, 2)
    assert result["samples"] == 4
    assert accumulator["optimizer_steps"] == 2
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "progress.json").is_file()


def test_encoder_config_comes_only_from_scratch_metadata():
    metadata = {
        "initialization": "scratch",
        "external_weights": [],
        "run_contract": {
            "encoder": {
                "widths": [8, 16, 32, 48],
                "depths": [1, 1, 2, 1],
                "window_size": 4,
                "attention_heads": [2, 3],
                "mlp_ratio": 2.0,
                "geometry_dim": 10,
                "scale_dropout": 0.0,
            }
        },
    }

    config = _encoder_config(metadata)

    assert config.widths == (8, 16, 32, 48)
    metadata["external_weights"] = ["teacher"]
    try:
        _encoder_config(metadata)
    except ValueError as error:
        assert "pure-scratch" in str(error)
    else:
        raise AssertionError("expected external lineage to fail")


def test_branch_arguments_require_compute_matched_local_budget():
    values = dict(
        input_mode="source_native",
        train_counts={16: 10, 25: 10},
        validation_counts={16: 2, 25: 2},
        domain_weights={16: 0.15, 25: 0.85},
        epochs=1,
        epoch_samples=10,
        batch_size=2,
        workers=0,
        local_views=0,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_steps=0,
        checkpoint_every_steps=1,
        log_every=1,
        photo_loss_weight=0.25,
        max_source_pixels=40_000_000,
        stop_after_steps=None,
    )

    try:
        _validate_arguments(**values)
    except ValueError as error:
        assert "local_views" in str(error)
    else:
        raise AssertionError("expected zero local candidate budget to fail")


def test_train_epoch_honors_existing_lifecycle_stop_handle(tmp_path):
    trainer = _Trainer()
    control = FoveaRunControl()
    control.request_stop()

    result, offset, step, added, stopped, _accumulator = _train_epoch(
        trainer,
        _Samples(5),
        epoch=0,
        sample_offset=0,
        batch_size=2,
        workers=0,
        checkpoint_every_steps=10,
        log_every=10,
        global_step=0,
        output_dir=tmp_path,
        best_f1=0.0,
        best_threshold=0.5,
        run_contract={"input_mode": "overview_only"},
        history=[],
        initial_accumulator=None,
        remaining_step_budget=None,
        run_control=control,
    )

    assert stopped == "stopped_by_request"
    assert (offset, step, added) == (0, 0, 0)
    assert result["samples"] == 0
