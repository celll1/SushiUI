"""CPU-only contracts for activation-dispatch configuration reachability."""

import ast
import inspect
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.param_defaults import TRAINING_DEFAULTS
from core.training.base_trainer import (
    BaseTrainer,
    _resolve_activation_dispatch_settings,
)


def _fallbacks():
    return {
        "enable": False,
        "margin_gb": 1.0,
        "seed_coef": 24.0e-6,
        "residual_frac": 0.85,
        "threshold_mb": 4,
    }


def test_run_config_overrides_constructor_fallbacks_for_adapter_trainers():
    resolved = _resolve_activation_dispatch_settings({
        "activation_dispatch_enable": True,
        "activation_dispatch_margin_gb": 2.5,
        "activation_dispatch_seed_coef": 0.125,
        "activation_dispatch_residual_frac": 0.6,
        "activation_dispatch_threshold_mb": 9,
    }, **_fallbacks())

    assert resolved == {
        "enable": True,
        "margin_gb": 2.5,
        "seed_coef": 0.125,
        "residual_frac": 0.6,
        "threshold_mb": 9,
    }


def test_direct_callers_retain_explicit_constructor_values():
    assert _resolve_activation_dispatch_settings(None, **{
        "enable": True,
        "margin_gb": 3.0,
        "seed_coef": 0.25,
        "residual_frac": 0.5,
        "threshold_mb": 7,
    }) == {
        "enable": True,
        "margin_gb": 3.0,
        "seed_coef": 0.25,
        "residual_frac": 0.5,
        "threshold_mb": 7,
    }


def test_base_trainer_signature_uses_api_default_ssot():
    params = inspect.signature(BaseTrainer.__init__).parameters
    for key in (
        "activation_dispatch_enable",
        "activation_dispatch_margin_gb",
        "activation_dispatch_seed_coef",
        "activation_dispatch_residual_frac",
        "activation_dispatch_threshold_mb",
    ):
        assert params[key].default == TRAINING_DEFAULTS[key]


def test_every_runner_created_base_trainer_receives_train_config():
    runner = Path(__file__).parents[1] / "core" / "training" / "train_runner.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    trainer_names = {
        "LoRATrainer", "ReLoRATrainer", "FullParameterTrainer", "ControlNetTrainer",
    }
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id in trainer_names]

    assert {call.func.id for call in calls} == trainer_names
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "train_config" in keywords, call.func.id
