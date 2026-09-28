"""Completion must retain optimizer state, including after a partial periodic save."""
import json
import sys
from pathlib import Path

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from core.training.base_trainer import BaseTrainer


class CompletionTrainer(BaseTrainer):
    def load_checkpoint(self, *args, **kwargs):
        pass

    def save_checkpoint(self, *args, **kwargs):
        pass

    def setup_trainable_parameters(self, *args, **kwargs):
        pass


def trainer(tmp_path):
    instance = object.__new__(CompletionTrainer)
    instance.output_dir = tmp_path
    instance.run_name = "completion"
    instance.run_id = None
    instance.log_prefix = "[test]"
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    instance.optimizer = torch.optim.AdamW([parameter], lr=.01)
    parameter.square().sum().backward()
    instance.optimizer.step()
    instance.save_checkpoint = lambda **kw: torch.save(parameter.detach(), tmp_path / "weights.pt")
    instance.save_training_state = lambda **kw: (tmp_path / "state.json").write_text(json.dumps(kw))
    instance._record_checkpoint_db_row = lambda **kw: None
    instance.save_ema_state = lambda **kw: None
    instance._save_ema_checkpoint = lambda **kw: None
    instance._periodic_save_with_space_guard = lambda **kw: instance._save_checkpoint_bundle(
        kw["step"], kw["epoch"], kw["batch_idx"], kw["multi_noise_timesteps"], kw["save_kind"])
    return instance


def complete(instance):
    instance._final_save_on_completion(step=10, epoch=0, batch_idx=10,
        multi_noise_timesteps=1, max_step_saves_to_keep=3,
        max_optimizer_saves_to_keep=1, save_every_n_steps=1000)


def test_completion_saves_real_optimizer_moments_and_resumes(tmp_path):
    instance = trainer(tmp_path)
    complete(instance)
    saved = torch.load(tmp_path / "completion_step_000010_optimizer.pt", weights_only=False)
    resumed = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    resumed.load_state_dict(saved)
    expected = instance.optimizer.state_dict()["state"][0]
    actual = resumed.state_dict()["state"][0]
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(actual[key], expected[key])
    assert json.loads((tmp_path / "state.json").read_text())["step"] == 10


def test_partial_periodic_optimizer_failure_is_retried_on_completion(tmp_path):
    instance = trainer(tmp_path)
    save_optimizer = instance.save_optimizer_state
    def fail(**kw):
        raise OSError("transient optimizer write failure")
    instance.save_optimizer_state = fail
    with pytest.raises(OSError):
        instance._save_checkpoint_bundle(10, 0, 10, 1)
    assert instance._last_periodic_checkpoint_step == 10
    assert instance._checkpoint_bundle_progress[1] == {"weights", "state"}
    instance.save_optimizer_state = save_optimizer
    complete(instance)
    assert (tmp_path / "completion_step_000010_optimizer.pt").exists()


def test_complete_periodic_bundle_is_not_saved_twice(tmp_path):
    instance = trainer(tmp_path)
    instance._save_checkpoint_bundle(10, 0, 10, 1)
    instance._periodic_save_with_space_guard = lambda **kw: pytest.fail("Duplicate final save")
    complete(instance)
