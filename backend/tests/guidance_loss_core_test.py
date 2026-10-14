"""Architecture-independent guidance-target loss core."""
from types import SimpleNamespace

import pytest
import torch

from core.training import guidance_loss as guidance

NAMES = ("weight", "mix_mode", "scale", "schedule", "extrapolation_max_rms", "weight_schedule",
         "high_noise_weight", "ramp_start", "ramp_end")
VALUES = {"weight": 0.25, "mix_mode": "blend", "scale": 4.0, "schedule": "constant",
          "extrapolation_max_rms": 0.5, "weight_schedule": "high_noise_smoothstep",
          "high_noise_weight": 0.75, "ramp_start": 0.4, "ramp_end": 0.9}


def test_canonical_and_legacy_keys_resolve_identically_and_canonical_wins():
    legacy = {guidance.LEGACY_PREFIX + name: value for name, value in VALUES.items()}
    canonical = {guidance.PREFIX + name: value for name, value in VALUES.items()}
    for name in NAMES:
        assert guidance.setting(legacy, name) == guidance.setting(canonical, name) == VALUES[name]
    both = dict(legacy, **{guidance.PREFIX + "scale": 2.0})
    assert guidance.setting(both, "scale") == 2.0


def test_guided_target_is_exact_and_null_rows_are_ordinary():
    torch.manual_seed(0)
    prediction, target, uncond = (torch.randn(3, 4, 5) for _ in range(3))
    sigma = torch.tensor([0.2, 0.5, 0.9])
    eligible = torch.tensor([True, True, False])
    config = {guidance.PREFIX + "weight": 1.0, guidance.PREFIX + "scale": 3.0,
              guidance.PREFIX + "schedule": "sigma"}
    condition = guidance.condition(config, eligible)
    loss, _, _ = guidance.objective(prediction, target, uncond, sigma, condition)
    scale = (1 + 2 * sigma).view(-1, 1, 1)
    guided = uncond + scale * (target - uncond)
    expected = torch.stack([
        (prediction[0] - guided[0]).square().mean(),
        (prediction[1] - guided[1]).square().mean(),
        (prediction[2] - target[2]).square().mean(),
    ]).mean()
    torch.testing.assert_close(loss, expected)


def test_inactive_and_invalid_settings():
    eligible = torch.ones(2, dtype=torch.bool)
    assert guidance.condition({}, eligible) is None
    assert not guidance.active({})
    assert guidance.active({guidance.PREFIX + "weight_schedule": "high_noise_smoothstep",
                            guidance.PREFIX + "high_noise_weight": 0.5})
    with pytest.raises(ValueError):
        guidance.weight({guidance.PREFIX + "weight": 1.5})
    with pytest.raises(ValueError):
        guidance.condition({guidance.PREFIX + "weight": 0.5, guidance.PREFIX + "scale": 20.0}, eligible)


def test_missing_mix_mode_key_keeps_blend_for_saved_runs():
    trainer = SimpleNamespace(config={guidance.PREFIX + "weight": 0.5})
    condition = guidance.condition(trainer.config, torch.ones(2, dtype=torch.bool))
    mix, selected = guidance.selection(trainer, torch.tensor([0.3, 0.7]), condition)
    torch.testing.assert_close(mix, torch.tensor([0.5, 0.5]))
    assert selected.all()


def test_capability_table_gates_architectures():
    from api.arch_capabilities import TRAINING_DECLARED_ARCHS, training_feature_unsupported_reason
    supported = {"qwen_image_21", "ming_image_01", "anima", "zimage", "krea2", "flux2", "sensenova_sdxl_chimera", "ltx2", "lens", "minit2i", "sd15", "sdxl", "ideogram4"}
    for arch in supported:
        assert training_feature_unsupported_reason(arch, "guidance_loss") is None
    for arch in TRAINING_DECLARED_ARCHS - supported:
        assert training_feature_unsupported_reason(arch, "guidance_loss")
    assert "distilled" in training_feature_unsupported_reason("minimax_h3", "guidance_loss")


def test_legacy_yaml_round_trips_to_canonical_request():
    from api.routes import TrainingRunCreateRequest, _extract_request_params_from_yaml
    legacy = {"qwen_guidance_loss_weight": 0.25, "qwen_guidance_loss_scale": 4.0}
    params = _extract_request_params_from_yaml({"train": legacy}, "lora")
    assert params["guidance_loss_weight"] == 0.25 and params["guidance_loss_scale"] == 4.0
    assert params["guidance_loss_mix_mode"] == "blend"
    assert all(params[guidance.LEGACY_PREFIX + name] is None for name in guidance.NAMES)
    request = TrainingRunCreateRequest(**{**params, "training_method": "lora", "base_model_path": "offline"})
    assert request.guidance_loss_weight == 0.25
