"""CPU tests for the Phase-3 branch runner contracts."""

import torch
from PIL import Image
from torch import nn

from core.tagger.foveatag.supervised_data import FoveaSupervisedSample
from core.tagger.foveatag.supervised_run import (
    FoveaRunControl,
    _bind_shared_data_contract,
    _bind_shared_training_contract,
    _encoder_config,
    _selected_sources_contract,
    _train_epoch,
    _validate_arguments,
    run_supervised_comparison,
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


class _RawSources:
    def __init__(self, paths):
        self.paths = paths

    def source_path_for_index(self, index):
        return str(self.paths[index])

    def dataset_id_for_index(self, index):
        return 29 if index < 2 else 16

    def tag_names_for_index(self, index):
        return (f"tag-{index}", "shared")


def test_selected_source_contract_binds_order_content_and_file_identity(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        path.write_bytes(bytes([index]))
        paths.append(path)
    dataset = _RawSources(paths)

    first = _selected_sources_contract(dataset, [0, 1, 2])
    reordered = _selected_sources_contract(dataset, [1, 0, 2])
    paths[1].write_bytes(b"changed")
    changed = _selected_sources_contract(dataset, [0, 1, 2])

    assert first["count"] == 3
    assert first["ordered_sources_and_supervision_sha256"] != reordered[
        "ordered_sources_and_supervision_sha256"
    ]
    assert first["ordered_sources_and_supervision_sha256"] != changed[
        "ordered_sources_and_supervision_sha256"
    ]


def test_shared_data_contract_is_immutable_across_branches(tmp_path):
    contract = {"format_version": 1, "kind": "foveatag_phase3_shared_data", "seed": 3}

    digest = _bind_shared_data_contract(tmp_path, contract)

    assert len(digest) == 64
    assert _bind_shared_data_contract(tmp_path, contract) == digest
    try:
        _bind_shared_data_contract(tmp_path, {**contract, "seed": 4})
    except ValueError as error:
        assert "shared data contract" in str(error)
    else:
        raise AssertionError("expected changed Phase-3 data to fail")


def test_shared_training_contract_is_immutable_across_branches(tmp_path):
    contract = {
        "format_version": 1,
        "kind": "foveatag_phase3_shared_training",
        "optimizer": {"learning_rate": 1e-4},
    }

    digest = _bind_shared_training_contract(tmp_path, contract)

    assert len(digest) == 64
    assert _bind_shared_training_contract(tmp_path, contract) == digest
    changed = {**contract, "optimizer": {"learning_rate": 2e-4}}
    try:
        _bind_shared_training_contract(tmp_path, changed)
    except ValueError as error:
        assert "shared training contract" in str(error)
    else:
        raise AssertionError("expected changed Phase-3 training settings to fail")


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


def test_comparison_runner_resumes_branches_and_stops_before_next_mode(tmp_path):
    first_resume = tmp_path / "overview_only" / "latest.pt"
    first_resume.parent.mkdir()
    first_resume.write_bytes(b"checkpoint")
    calls = []

    def branch_runner(**kwargs):
        calls.append(kwargs)
        mode = kwargs["input_mode"]
        return {
            "status": "complete" if mode == "overview_only" else "stopped_by_request",
            "input_mode": mode,
        }

    events = []
    result = run_supervised_comparison(
        comparison_dir=tmp_path,
        branch_runner=branch_runner,
        progress_callback=lambda event, payload: events.append((event, payload)),
        token="shared",
    )

    assert [call["input_mode"] for call in calls] == [
        "overview_only", "overview_placebo",
    ]
    assert calls[0]["resume"] == first_resume
    assert calls[1]["resume"] is None
    assert all(call["token"] == "shared" for call in calls)
    assert result["status"] == "stopped_by_request"
    assert result["completed_modes"] == ["overview_only"]
    assert result["active_mode"] == "overview_placebo"
    assert [payload["input_mode"] for event, payload in events if event == "branch_start"] == [
        "overview_only", "overview_placebo",
    ]
    assert (tmp_path / "comparison_summary.json").is_file()


def test_comparison_runner_honors_stop_before_loading_a_branch(tmp_path):
    control = FoveaRunControl()
    control.request_stop()

    result = run_supervised_comparison(
        comparison_dir=tmp_path,
        branch_runner=lambda **_kwargs: (_ for _ in ()).throw(AssertionError()),
        run_control=control,
    )

    assert result["status"] == "stopped_by_request"
    assert result["active_mode"] == "overview_only"
    assert result["completed_modes"] == []


def test_comparison_runner_completes_all_modes_in_registered_order(tmp_path):
    calls = []

    def branch_runner(**kwargs):
        calls.append(kwargs["input_mode"])
        return {"status": "complete", "input_mode": kwargs["input_mode"]}

    result = run_supervised_comparison(
        comparison_dir=tmp_path,
        branch_runner=branch_runner,
    )

    assert calls == ["overview_only", "overview_placebo", "source_native"]
    assert result["status"] == "complete"
    assert result["active_mode"] is None
    assert result["completed_modes"] == calls
    assert result["remaining_modes"] == []
