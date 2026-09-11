"""`reconstruction_loss_weight` outside [0, 1] is refused before training starts.

Since `e742e4c6` every consuming architecture mixes the dual loss NORMALIZED,
``(1-w)*prediction + w*reconstruction``. A weight above 1 therefore makes the
PREDICTION term's coefficient negative -- the run would ascend the objective it
exists to minimize -- and a negative weight does the same to the reconstruction
term. Before this fix all three layers passed the number straight through:
the Pydantic field carried no bounds, `_build_train_section` did a bare
``p.get``, and `BaseTrainer.__init__` assigned the argument as given.

Refused, not clamped: a silently rounded weight starts an hours-long run the
operator did not configure. No stored run config carries a value outside the
range (surveyed over `training.db` before choosing), so nothing resumable
breaks.

Both entry paths are covered because they are independent: hand-written YAML
reaches `BaseTrainer` through `train_runner` without ever passing the request
model.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/reconstruction_loss_weight_bounds_test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

# The trainer stack builds a CUDA context on import; stub the trigger (AGENTS.md).
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from api.param_defaults import (  # noqa: E402
    RECONSTRUCTION_LOSS_WEIGHT_MAX,
    RECONSTRUCTION_LOSS_WEIGHT_MIN,
    TRAINING_DEFAULTS,
    validate_reconstruction_loss_weight,
)

KEY = "reconstruction_loss_weight"
IN_RANGE = [0.0, 0.3, 0.5, 1.0]
OUT_OF_RANGE = [1.0001, 1.5, 2.0, -0.0001, -0.3, -1.0]



@pytest.mark.parametrize("weight", IN_RANGE)
def test_the_closed_interval_endpoints_and_interior_pass(weight):
    assert validate_reconstruction_loss_weight(weight) == weight


@pytest.mark.parametrize("weight", OUT_OF_RANGE)
def test_out_of_range_raises_and_names_the_reason(weight):
    with pytest.raises(ValueError) as excinfo:
        validate_reconstruction_loss_weight(weight)
    message = str(excinfo.value)
    assert KEY in message and str(weight) in message
    assert "negative" in message


def test_nan_is_refused_rather_than_compared_away():
    with pytest.raises(ValueError):
        validate_reconstruction_loss_weight(float("nan"))


def test_none_means_unset_and_resolves_to_the_default():
    assert validate_reconstruction_loss_weight(None) == TRAINING_DEFAULTS[KEY]


def test_the_bounds_are_the_ones_the_mixing_formula_implies():
    assert (RECONSTRUCTION_LOSS_WEIGHT_MIN, RECONSTRUCTION_LOSS_WEIGHT_MAX) == (0.0, 1.0)



def _request(weight):
    from api.routes import TrainingRunCreateRequest
    return TrainingRunCreateRequest(
        training_method="lora",
        base_model_path="dummy.safetensors",
        **{KEY: weight},
    )


@pytest.mark.parametrize("weight", IN_RANGE)
def test_pydantic_accepts_the_range(weight):
    assert getattr(_request(weight), KEY) == weight


@pytest.mark.parametrize("weight", OUT_OF_RANGE)
def test_pydantic_rejects_out_of_range(weight):
    from pydantic import ValidationError
    with pytest.raises(ValidationError) as excinfo:
        _request(weight)
    assert KEY in str(excinfo.value)


def test_the_pydantic_default_comes_from_param_defaults():
    from api.routes import TrainingRunCreateRequest
    request = TrainingRunCreateRequest(
        training_method="lora", base_model_path="dummy.safetensors")
    assert getattr(request, KEY) == TRAINING_DEFAULTS[KEY]



def _train_section(weight):
    from core.training.training_config import _build_train_section
    return _build_train_section(
        {KEY: weight}, total_steps=10, epochs=None,
        train_unet=True, train_text_encoder=False)


@pytest.mark.parametrize("weight", IN_RANGE)
def test_config_generation_emits_the_range_unchanged(weight):
    assert _train_section(weight)[KEY] == weight


@pytest.mark.parametrize("weight", OUT_OF_RANGE)
def test_config_generation_refuses_out_of_range(weight):
    with pytest.raises(ValueError):
        _train_section(weight)


def test_an_omitted_key_still_emits_the_default():
    from core.training.training_config import _build_train_section
    section = _build_train_section(
        {}, total_steps=10, epochs=None, train_unet=True, train_text_encoder=False)
    assert section[KEY] == TRAINING_DEFAULTS[KEY]



def _trainer_class():
    from core.training.base_trainer import BaseTrainer

    class _Probe(BaseTrainer):
        loaded = False

        def _load_model_components(self):
            type(self).loaded = True

        def setup_trainable_parameters(self):
            return []

        def save_checkpoint(self, step, epoch):
            pass

        def load_checkpoint(self, checkpoint_path):
            return 0

    return _Probe


def _construct(weight, tmp_path):
    cls = _trainer_class()
    trainer = cls(model_path="dummy.safetensors", output_dir=str(tmp_path),
                  device="cpu", **{KEY: weight})
    return trainer, cls


@pytest.mark.parametrize("weight", IN_RANGE)
def test_the_trainer_stores_an_in_range_weight_exactly(weight, tmp_path):
    trainer, cls = _construct(weight, tmp_path)
    assert getattr(trainer, KEY) == weight
    assert cls.loaded, "construction should have reached model loading"


@pytest.mark.parametrize("weight", OUT_OF_RANGE)
def test_the_trainer_refuses_out_of_range_before_the_model_loads(weight, tmp_path):
    cls = _trainer_class()
    with pytest.raises(ValueError) as excinfo:
        cls(model_path="dummy.safetensors", output_dir=str(tmp_path),
            device="cpu", **{KEY: weight})
    assert KEY in str(excinfo.value)
    assert not cls.loaded, "refused only after paying for the model load"


def test_a_yaml_config_reaches_the_trainer_without_the_request_model(tmp_path):
    """Why layer 3 is needed: train_runner reads YAML and calls the trainer directly."""
    runner = (BACKEND / "core" / "training" / "train_runner.py").read_text(encoding="utf-8")
    assert f"train_config.get('{KEY}'" in runner


# ---------------------------------------------------------------------------
# The spec states the same interval
# ---------------------------------------------------------------------------

def test_openapi_declares_the_same_bounds_exactly_once():
    text = (REPO / "openapi.yaml").read_text(encoding="utf-8")
    assert text.count(f"{KEY}:") == 1, "duplicate blocks are last-key-wins in YAML"

    spec = yaml.safe_load(text)
    prop = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"][KEY]
    assert prop["minimum"] == RECONSTRUCTION_LOSS_WEIGHT_MIN
    assert prop["maximum"] == RECONSTRUCTION_LOSS_WEIGHT_MAX
    assert prop["default"] == TRAINING_DEFAULTS[KEY]



@pytest.mark.parametrize("layer,source,marker", [
    ("routes", "backend/api/routes.py",
     f"    {KEY}: float = 0.0"),
    ("training_config", "backend/core/training/training_config.py",
     f'train["{KEY}"] = p.get("{KEY}", 0.0)'),
    ("base_trainer", "backend/core/training/base_trainer.py",
     f"        self.{KEY} = {KEY}\n"),
])
def test_negative_control_the_shipped_layer_was_a_pass_through(layer, source, marker):
    """What each layer looked like at `62080e47`, so the fix is provably a change."""
    import subprocess
    before = subprocess.run(
        ["git", "show", f"62080e47:{source}"], cwd=REPO,
        capture_output=True, text=True, encoding="utf-8", check=True).stdout
    assert marker in before, f"{layer}: expected the pre-fix pass-through"

    now = (REPO / source).read_text(encoding="utf-8")
    assert marker not in now, f"{layer}: the pass-through is still there"
