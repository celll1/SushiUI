"""CPU contracts for FoveaTag Phase-2 run bookkeeping."""

import pytest
import torch

from core.tagger.foveatag.pretraining_run import (
    _cosine_with_warmup,
    _emit_progress,
    _normalized_domain_weights,
    _parse_mapping,
    _restore_accumulator,
    _train_epoch,
    _training_summary,
)
from core.tagger.foveatag.run_control import FoveaRunControl


def test_epoch_accumulator_survives_segment_resume():
    first = _restore_accumulator(None)
    first["sums"]["loss"] = 8.0
    first["samples"] = 4
    first["optimizer_steps"] = 2
    first["seconds"] = 1.5

    resumed = _restore_accumulator(first)
    resumed["sums"]["loss"] += 6.0
    resumed["samples"] += 2
    resumed["optimizer_steps"] += 1
    resumed["seconds"] += 0.5
    summary = _training_summary(resumed)

    assert summary["samples"] == 6
    assert summary["optimizer_steps"] == 3
    assert summary["seconds"] == 2.0
    assert summary["loss"] == pytest.approx(14 / 6)


def test_domain_mapping_is_normalized_and_duplicate_safe():
    assert _normalized_domain_weights((16, 25), {16: 3.0, 25: 17.0}) == {
        16: 0.15,
        25: 0.85,
    }
    with pytest.raises(ValueError, match="duplicate"):
        _parse_mapping(("16=1", "16=2"), cast=int, name="validation-count")


def test_warmup_cosine_scheduler_is_step_resumable():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD((parameter,), lr=1e-3)
    scheduler = _cosine_with_warmup(
        optimizer,
        total_steps=20,
        warmup_steps=4,
        learning_rate=1e-3,
    )
    for _ in range(7):
        optimizer.step()
        scheduler.step()
    state = scheduler.state_dict()
    expected_lr = optimizer.param_groups[0]["lr"]

    restored_optimizer = torch.optim.SGD((torch.nn.Parameter(torch.ones(())),), lr=1e-3)
    restored = _cosine_with_warmup(
        restored_optimizer,
        total_steps=20,
        warmup_steps=4,
        learning_rate=1e-3,
    )
    restored.load_state_dict(state)
    restored_optimizer.param_groups[0]["lr"] = expected_lr

    assert restored.last_epoch == scheduler.last_epoch == 7
    assert restored_optimizer.param_groups[0]["lr"] == expected_lr


def test_shared_run_control_is_thread_safe_and_sticky():
    control = FoveaRunControl()

    assert not control.should_stop()
    control.request_stop()
    assert control.should_stop()
    control.request_stop()
    assert control.should_stop()


def test_pretraining_epoch_honors_stop_before_loading_data(tmp_path):
    control = FoveaRunControl()
    control.request_stop()
    events = []

    result, offset, step, added, stopped, accumulator = _train_epoch(
        None,
        None,
        None,
        [object()] * 10,
        epoch=2,
        sample_offset=4,
        batch_size=2,
        workers=0,
        amp_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_every_steps=5,
        log_every=1,
        global_step=17,
        output_dir=tmp_path,
        best_validation_loss=1.0,
        run_contract={},
        history=[],
        initial_accumulator=None,
        remaining_step_budget=None,
        run_control=control,
        progress_callback=lambda event, payload: events.append((event, dict(payload))),
    )

    assert stopped
    assert offset == 4
    assert step == 17
    assert added == 0
    assert result["optimizer_steps"] == 0
    assert accumulator["samples"] == 0
    assert events == [("train", {
        "status": "stopped",
        "epoch": 2,
        "sample_offset": 4,
        "global_step": 17,
    })]


def test_progress_reporter_failure_does_not_escape():
    def fail(_event, _payload):
        raise RuntimeError("reporter unavailable")

    _emit_progress(fail, "train", {"global_step": 3})
