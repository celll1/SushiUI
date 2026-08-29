"""The caption-dropout conflict check at TRAINING START, not only on the route.

local/strategy/cfg_null_alignment/IMPLEMENTATION_STRATEGY.md section 4. A
hand-authored YAML never passes a request model, so the route's refusal cannot
see it; the run has to be refused where it actually starts, before the model
loads. Nothing here loads a checkpoint or opens a database.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/cfg_null_training_start_test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.cfg_null_resolver import (  # noqa: E402
    CFG_KEY, DATASET_CAPTION_CONFIGS_KEY, LEGACY_KEY,
)
from api.error_handlers import ValidationError  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402

_TRAIN_RUNNER_SOURCE = (BACKEND / "core" / "training"
                        / "train_runner.py").read_text(encoding="utf-8")


class _StubArch:
    name = "minit2i"
    cfg_null_stage = "collated"


class _StubTrainer:
    cfg_null_drop_rate = BaseTrainer.cfg_null_drop_rate
    log_prefix = "[cfg-null-test]"

    def __init__(self, **config):
        section = {CFG_KEY: None, LEGACY_KEY: None,
                   "danbooru_aug_enable": False,
                   "danbooru_aug_caption_dropout_rate": 0.0}
        section.update(config)
        self.config = section
        self.arch = _StubArch()


# ---------------------------------------------------------------------------
# What the trainer can see: its own train section
# ---------------------------------------------------------------------------

def test_the_trainer_refuses_an_explicit_rate_beside_danbooru_caption_dropout():
    trainer = _StubTrainer(**{CFG_KEY: 0.2, "danbooru_aug_enable": True,
                              "danbooru_aug_caption_dropout_rate": 0.1})
    with pytest.raises(ValidationError) as exc:
        trainer.cfg_null_drop_rate()
    assert "danbooru_aug_caption_dropout_rate" in exc.value.detail


def test_a_stale_rate_with_the_augmentation_off_is_not_a_conflict():
    """The rate is stored unconditionally and read only under the enable flag,
    so it drops no caption here -- same rule the route applies."""
    trainer = _StubTrainer(**{CFG_KEY: 0.2, "danbooru_aug_enable": False,
                              "danbooru_aug_caption_dropout_rate": 0.1})
    assert trainer.cfg_null_drop_rate() == 0.2


def test_a_legacy_minit2i_run_is_warned_not_refused(capsys):
    """Configs written before this feature keep working; the discrepancy is no
    longer silent."""
    trainer = _StubTrainer(**{"danbooru_aug_enable": True,
                              "danbooru_aug_caption_dropout_rate": 0.1})
    assert trainer.cfg_null_drop_rate() == 0.1
    assert "danbooru_aug_caption_dropout_rate" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# What the trainer cannot see on its own: the datasets
# ---------------------------------------------------------------------------

def test_the_trainer_refuses_on_the_dataset_configs_train_runner_parks_for_it():
    """Caption processing is a datasets-DB property the YAML deliberately does
    not carry, so train_runner reads it and parks it on the same config dict."""
    trainer = _StubTrainer(**{
        CFG_KEY: 0.2,
        DATASET_CAPTION_CONFIGS_KEY: [("portraits", {"caption_dropout_rate": 0.05})],
    })
    with pytest.raises(ValidationError) as exc:
        trainer.cfg_null_drop_rate()
    assert "portraits" in exc.value.detail
    assert "caption_dropout_rate" in exc.value.detail


def test_without_those_configs_the_dataset_half_simply_does_not_fire():
    """Stated rather than implied: a trainer constructed outside train_runner
    sees no dataset caption config and refuses only on what its config carries."""
    trainer = _StubTrainer(**{CFG_KEY: 0.2})
    assert trainer.cfg_null_drop_rate() == 0.2


# ---------------------------------------------------------------------------
# train_runner's pre-flight
# ---------------------------------------------------------------------------

class _StubQuery:
    def __init__(self, datasets):
        self._datasets = datasets
        self._id = None

    def filter(self, criterion):
        # The criterion is `Dataset.id == <value>`; read the value off it rather
        # than evaluating SQLAlchemy.
        self._id = criterion.right.value
        return self

    def first(self):
        return self._datasets.get(self._id)


class _StubDatasetsDb:
    def __init__(self, datasets):
        self._datasets = datasets

    def query(self, model):
        return _StubQuery(self._datasets)


class _StubDataset:
    def __init__(self, name, caption_processing):
        self.name = name
        self.path = f"/datasets/{name}"
        self.caption_processing = caption_processing


def _preflight(monkeypatch, train_config, caption_processing):
    from core.training import train_runner, training_config

    monkeypatch.setattr(training_config, "_detect_arch", lambda path: "minit2i")
    db = _StubDatasetsDb({7: _StubDataset("portraits", caption_processing)})
    train_runner._preflight_cfg_null_caption_conflict(
        train_config, "/models/minit2i.safetensors", [{"dataset_id": 7}], db)
    return train_config


def test_the_preflight_refuses_a_hand_authored_yaml(monkeypatch):
    with pytest.raises(ValueError) as exc:
        _preflight(monkeypatch, {CFG_KEY: 0.2, LEGACY_KEY: None},
                   {"caption_dropout_rate": 0.05})
    assert "portraits" in str(exc.value)
    assert CFG_KEY in str(exc.value)


def test_the_preflight_parks_what_it_read_for_the_trainer(monkeypatch):
    config = _preflight(monkeypatch, {CFG_KEY: 0.2, LEGACY_KEY: None},
                        {"caption_dropout_rate": 0.0})
    assert config[DATASET_CAPTION_CONFIGS_KEY] == [
        ("portraits", {"caption_dropout_rate": 0.0})]


def test_the_preflight_runs_before_the_scan_and_before_any_trainer(monkeypatch):
    """It has to refuse before the dataset scan (minutes) and before a model is
    loaded, which is the whole point of doing it here."""
    call = _TRAIN_RUNNER_SOURCE.index(
        "_preflight_cfg_null_caption_conflict(\n            train_config, "
        "run.base_model_path")
    assert call < _TRAIN_RUNNER_SOURCE.index('print(f"[TrainRunner] Loading {len(dataset_configs)}')
    assert call < _TRAIN_RUNNER_SOURCE.index("trainer = LoRATrainer(")
