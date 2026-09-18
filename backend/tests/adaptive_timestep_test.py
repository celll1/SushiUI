import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.adaptive_timestep import (
    AdaptiveTimestepSampler,
    validate_adaptive_timestep_config,
)
from core.training.timestep_sampler import (
    LogitNormalTimestepSampler,
    MorphingTimestepSampler,
    UniformTimestepSampler,
)


def _config(mode="bounded", **overrides):
    config = {
        "mode": mode,
        "warmup_updates": 0,
        "control_interval": 1,
        "bins": 2,
        "morph_updates": 3,
        "cooldown_updates": 0,
        "min_observations": 2,
    }
    config.update(overrides)
    return config


def test_validation_refuses_unsafe_ranges():
    with pytest.raises(ValueError, match="coverage_floor"):
        validate_adaptive_timestep_config({"mode": "bounded", "coverage_floor": 0})
    with pytest.raises(ValueError, match="bins"):
        validate_adaptive_timestep_config({"mode": "bounded", "bins": 1})
    with pytest.raises(ValueError, match="auto_observe_controls"):
        validate_adaptive_timestep_config({"mode": "auto", "auto_observe_controls": 0})


def test_x0_equivalent_flow_loss_uses_noise_fraction():
    t0 = AdaptiveTimestepSampler(
        UniformTimestepSampler(), _config("observe"), convention="t0")
    t1 = AdaptiveTimestepSampler(
        UniformTimestepSampler(), _config("observe"), convention="t1")
    _, loss_t0 = t0._bin_and_x0_loss(0.75, 4.0)
    _, loss_t1 = t1._bin_and_x0_loss(0.25, 4.0)
    assert loss_t0 == pytest.approx(2.25)
    assert loss_t1 == pytest.approx(2.25)


def test_observe_mode_never_changes_sampler():
    base = UniformTimestepSampler()
    sampler = AdaptiveTimestepSampler(base, _config("observe"), convention="t1")
    sampler.observe(torch.tensor([0.1]), 1.0)
    sampler.observe(torch.tensor([0.9]), 1.0)
    sampler.set_optimizer_update_step(1)
    assert sampler.current is base
    assert sampler.status()["control_count"] == 1
    assert sampler.status()["action"] == "observed"


def test_bounded_mode_builds_quantile_morph_and_obeys_density_limits():
    sampler = AdaptiveTimestepSampler(
        UniformTimestepSampler(), _config(), convention="t1")
    # Seed both bins, then make the noisy bin's fast EMA worsen relative to slow.
    sampler.observe(torch.tensor([0.9]), 1.0)
    sampler.observe(torch.tensor([0.1]), 1.0)
    for _ in range(20):
        sampler.observe(torch.tensor([0.1]), 4.0)
    sampler.set_optimizer_update_step(1)
    assert isinstance(sampler.current, MorphingTimestepSampler)
    ratios = sampler.status()["density_ratio"]
    assert min(ratios) >= sampler.config["coverage_floor"]
    assert max(ratios) <= sampler.config["max_density_ratio"]


def test_auto_observes_then_promotes_when_controls_and_bins_are_ready():
    base = UniformTimestepSampler()
    sampler = AdaptiveTimestepSampler(
        base,
        _config(
            "auto", auto_observe_controls=2, auto_min_bin_observations=1,
            auto_min_bin_probability=0.01,
        ),
        convention="t1",
    )
    sampler.observe(torch.tensor([0.1]), 1.0)
    sampler.observe(torch.tensor([0.9]), 1.0)
    sampler.set_optimizer_update_step(1)
    assert sampler.current is base
    assert sampler.status()["effective_mode"] == "observe"
    assert sampler.status()["action"] == "auto_observing"

    sampler.observe(torch.tensor([0.1]), 2.0)
    sampler.observe(torch.tensor([0.9]), 2.0)
    sampler.set_optimizer_update_step(2)
    status = sampler.status()
    assert isinstance(sampler.current, MorphingTimestepSampler)
    assert status["auto_promoted"] is True
    assert status["effective_mode"] == "bounded"
    assert status["auto_promotion_update"] == 2
    restored = AdaptiveTimestepSampler(
        UniformTimestepSampler(), sampler.config, convention="t1",
        resume_state=json.loads(json.dumps(sampler.state(), allow_nan=False)),
    )
    assert restored.status()["auto_promoted"] is True
    assert restored.status()["effective_mode"] == "bounded"


def test_state_is_strict_json_and_round_trips_active_morph():
    sampler = AdaptiveTimestepSampler(
        UniformTimestepSampler(), _config(), convention="t1")
    sampler.observe(torch.tensor([0.1]), 1.0)
    sampler.observe(torch.tensor([0.9]), 2.0)
    sampler.set_optimizer_update_step(1)
    state = sampler.state()
    payload = json.dumps(state, allow_nan=False)
    restored = AdaptiveTimestepSampler(
        UniformTimestepSampler(), _config(), convention="t1",
        resume_state=json.loads(payload))
    assert restored.status() == sampler.status()
    assert isinstance(restored.current, MorphingTimestepSampler)

    with pytest.raises(ValueError, match="base distribution changed"):
        AdaptiveTimestepSampler(
            LogitNormalTimestepSampler(mean=0.0, std=1.0), _config(),
            convention="t1", resume_state=json.loads(payload))


def test_batch_greater_than_one_is_refused_not_silently_misbinned():
    sampler = AdaptiveTimestepSampler(
        UniformTimestepSampler(), _config("observe"), convention="t1")
    with pytest.raises(ValueError, match="batch_size=1"):
        sampler.observe(torch.tensor([0.1, 0.9]), 1.0)


def test_api_validation_and_openapi_are_wired(monkeypatch):
    from api.routes import _check_timestep_sampling
    from api.param_defaults import TRAINING_DEFAULTS

    request = SimpleNamespace(
        batch_size=2,
        timestep_sampling={"distribution": "uniform", "adaptive": {"mode": "observe"}},
    )
    with pytest.raises(Exception, match="batch_size=1"):
        _check_timestep_sampling(request)
    from core.model_loader import ModelLoader
    monkeypatch.setattr(
        ModelLoader, "detect_model_type",
        staticmethod(lambda _path: "sensenova_sdxl_chimera"),
    )
    request.batch_size = 1
    request.base_model_path = "dummy"
    _check_timestep_sampling(request)
    default = TRAINING_DEFAULTS["timestep_sampling"]["adaptive"]
    assert default["mode"] == "off"
    spec = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "openapi.yaml").read_text(encoding="utf-8"))
    schema = spec["components"]["schemas"]["AdaptiveTimestepSampling"]
    assert schema["properties"]["mode"]["enum"] == [
        "off", "observe", "auto", "bounded"]
