"""A resumed adapter must be loaded before step and optimizer state are restored."""

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from core.training.base_trainer import BaseTrainer, _require_loaded_resume_weights
from core.training.lora_trainer import LoRATrainer
from core.training.relora_trainer import ReLoRATrainer


def _owner(tmp_path: Path) -> LoRATrainer:
    owner = LoRATrainer.__new__(LoRATrainer)
    owner.output_dir = tmp_path
    owner.run_name = "run"
    owner.log_prefix = "[LoRA Trainer]"
    owner._loaded_checkpoint_path = None
    owner.resume_from_checkpoint = None
    owner.is_yue2 = False
    return owner


def test_train_loads_adapter_before_entering_base_train(monkeypatch, tmp_path):
    owner = _owner(tmp_path)
    calls = []
    monkeypatch.setattr(LoRATrainer, "_load_resume_adapter",
                        lambda self, requested: calls.append(("load", requested)))
    monkeypatch.setattr(BaseTrainer, "train",
                        lambda self, *args, **kwargs: calls.append(("train", kwargs["resume_from_checkpoint"])))

    owner.train(resume_from_checkpoint="latest")

    assert calls == [("load", "latest"), ("train", "latest")]


def test_constructor_loads_adapter_before_block_swap(monkeypatch, tmp_path):
    calls = []

    def base_init(self, **kwargs):
        calls.append("base")
        self.output_dir = tmp_path
        self.run_name = "run"
        self.resume_from_checkpoint = kwargs["resume_from_checkpoint"]
        self._loaded_checkpoint_path = None
        self.is_yue2 = False
        self.arch = SimpleNamespace(setup_block_swap=lambda trainer: calls.append("swap"))

    monkeypatch.setattr(BaseTrainer, "__init__", base_init)
    monkeypatch.setattr(LoRATrainer, "_create_adapter", lambda self: None)
    monkeypatch.setattr(LoRATrainer, "_apply_lora", lambda self: calls.append("inject"))
    monkeypatch.setattr(LoRATrainer, "_load_resume_adapter",
                        lambda self, requested: calls.append(("load", requested)))
    monkeypatch.setattr(LoRATrainer, "_setup_sensenova_phase_eviction", lambda self: None)

    LoRATrainer(train_config={"resume_from_checkpoint": "latest"})

    assert calls == ["base", "inject", ("load", "latest"), "swap"]


def test_latest_resume_loads_weights_and_records_exact_path(monkeypatch, tmp_path):
    owner = _owner(tmp_path)
    older = tmp_path / "run_step_000008.safetensors"
    newest = tmp_path / "run_step_000009.safetensors"
    older.touch()
    newest.touch()
    loaded = []
    monkeypatch.setattr(LoRATrainer, "load_checkpoint",
                        lambda self, path: loaded.append(path) or 9)

    owner._load_resume_adapter("latest")

    assert loaded == [str(newest)]
    assert owner._loaded_checkpoint_path == str(newest)
    _require_loaded_resume_weights(owner._loaded_checkpoint_path, newest)


def test_resume_refuses_orphan_state_and_missing_explicit_weights(tmp_path):
    owner = _owner(tmp_path)
    (tmp_path / "run_step_000009_state.json").touch()
    with pytest.raises(RuntimeError, match="no weight checkpoint"):
        owner._load_resume_adapter("latest")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        owner._load_resume_adapter("run_step_000009.safetensors")


def test_resume_refuses_checkpoint_without_matching_loaded_weights(tmp_path):
    selected = tmp_path / "run_step_000009.safetensors"
    with pytest.raises(RuntimeError, match="without loading"):
        _require_loaded_resume_weights(None, selected)
    with pytest.raises(RuntimeError, match="without loading"):
        _require_loaded_resume_weights(str(tmp_path / "run_step_000008.safetensors"), selected)


def test_relora_refuses_resume_when_merged_base_was_not_saved(monkeypatch, tmp_path):
    owner = ReLoRATrainer.__new__(ReLoRATrainer)
    checkpoint = tmp_path / "run_step_000009.safetensors"
    state_path = tmp_path / "run_step_000009_state.json"
    state_path.write_text(json.dumps({"merge_count": 1}), encoding="utf-8")
    monkeypatch.setattr(LoRATrainer, "_load_resume_adapter",
                        lambda self, requested: setattr(self, "_loaded_checkpoint_path", str(checkpoint)))

    with pytest.raises(RuntimeError, match="merged base weights"):
        owner._load_resume_adapter("latest")
