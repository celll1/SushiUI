"""CPU tests for step-resumable FoveaTag router epochs."""

from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from core.tagger.foveatag.router_data import FoveaRouterSample
from core.tagger.foveatag.router_run import train_router_epoch
from core.tagger.foveatag.router_training import RouterTrainResult
from core.tagger.foveatag.views import ViewConfig, build_views


class _Samples:
    def __init__(self, count):
        self.values = []
        for index in range(count):
            views = build_views(
                Image.new("RGB", (16, 16), (index, 2, 3)),
                ViewConfig(
                    overview_size=16,
                    local_size=8,
                    local_views=2,
                    field_levels=(0.5, 1.0),
                    scale_jitter=(1.0, 1.0),
                ),
                seed=index,
                training=False,
            )
            boxes = torch.tensor([
                [
                    spec.source_box.center[0] / 16,
                    spec.source_box.center[1] / 16,
                    spec.source_box.width / 16,
                    spec.source_box.height / 16,
                ]
                for spec in views.local_specs
            ])
            self.values.append(FoveaRouterSample(
                views=views,
                candidate_boxes=boxes,
                label=torch.tensor([1.0, 0.0]),
                loss_mask=torch.ones(2),
            ))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class _Trainer:
    def __init__(self):
        self.router = nn.Linear(2, 2)
        self.optimizer = torch.optim.AdamW(self.router.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.5
        )

    def train_batch(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.router(batch.labels).square().mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return RouterTrainResult(
            loss=float(loss.detach()),
            ranking=0.2,
            diversity=0.3,
            mean_positive_utility=0.4,
        )


def test_router_epoch_stops_at_exact_checkpointed_sample_offset(tmp_path):
    trainer = _Trainer()
    events = []
    result, offset, step, added, stopped, accumulator = train_router_epoch(
        trainer,
        _Samples(5),
        epoch=0,
        sample_offset=0,
        batch_size=2,
        workers=0,
        checkpoint_every_steps=1,
        log_every=1,
        global_step=0,
        output_dir=tmp_path,
        best_utility_recovery=0.0,
        run_contract={"tagger_sha256": "abc"},
        history=[],
        initial_accumulator=None,
        remaining_step_budget=2,
        progress_callback=lambda event, payload: events.append((event, payload)),
    )

    assert stopped == "stopped_after_requested_steps"
    assert (offset, step, added) == (4, 2, 2)
    assert result["samples"] == 4
    assert result["optimizer_steps"] == 2
    assert accumulator["ranking_sum"] == 0.8
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "progress.json").is_file()
    checkpoint = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["sample_offset"] == 4
    assert checkpoint["global_step"] == 2
    assert checkpoint["epoch_accumulator"]["samples"] == 4
    assert len(events) == 2


def test_router_epoch_resume_keeps_prior_accumulator(tmp_path):
    trainer = _Trainer()
    prior = {
        "loss_sum": 2.0,
        "ranking_sum": 0.4,
        "diversity_sum": 0.6,
        "positive_utility_sum": 0.8,
        "samples": 2,
        "skipped": 0,
        "optimizer_steps": 1,
        "seconds": 1.0,
    }
    result, offset, step, added, stopped, _accumulator = train_router_epoch(
        trainer,
        _Samples(5),
        epoch=0,
        sample_offset=2,
        batch_size=2,
        workers=0,
        checkpoint_every_steps=10,
        log_every=10,
        global_step=1,
        output_dir=tmp_path,
        best_utility_recovery=0.0,
        run_contract={"tagger_sha256": "abc"},
        history=[],
        initial_accumulator=prior,
        remaining_step_budget=1,
    )

    assert stopped == "stopped_after_requested_steps"
    assert (offset, step, added) == (4, 2, 1)
    assert result["samples"] == 4
    assert result["optimizer_steps"] == 2
    assert (tmp_path / "latest.pt").is_file()


def test_router_epoch_honors_stop_before_decode(tmp_path):
    trainer = _Trainer()
    control = SimpleNamespace(should_stop=lambda: True)

    result, offset, step, added, stopped, _accumulator = train_router_epoch(
        trainer,
        _Samples(3),
        epoch=0,
        sample_offset=0,
        batch_size=2,
        workers=0,
        checkpoint_every_steps=10,
        log_every=10,
        global_step=0,
        output_dir=tmp_path,
        best_utility_recovery=0.0,
        run_contract={"tagger_sha256": "abc"},
        history=[],
        initial_accumulator=None,
        remaining_step_budget=None,
        run_control=control,
    )

    assert stopped == "stopped_by_request"
    assert (offset, step, added) == (0, 0, 0)
    assert result["samples"] == 0
