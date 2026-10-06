"""Public VAE swap gates agree with Chimera's saved target-grid contract."""
from unittest.mock import patch

import pytest


def test_public_swap_capability_defaults_and_preflight():
    from api.arch_capabilities import training_feature_advisories, training_feature_unsupported_reason
    from api.param_defaults import BUNDLE_VAE_DEFAULTS_BY_ARCH
    from core.training.vae_swap import initialization_mode, preflight_vae_swap
    arch = "sensenova_sdxl_chimera"
    assert training_feature_unsupported_reason(arch, "vae_swap", "full_finetune") is None
    assert training_feature_unsupported_reason(arch, "vae_swap", "lora")
    assert training_feature_advisories(arch, "full_finetune")["vae_swap"]["level"] == "experimental"
    assert BUNDLE_VAE_DEFAULTS_BY_ARCH[arch] is True
    config = dict(vae_swap_source="registry:flux1", chimera_training_stage="unet")
    assert initialization_mode(config, arch) == "paired_spatial"
    for stage in ("unet", "joint"):
        assert preflight_vae_swap({**config, "chimera_training_stage": stage}, arch=arch,
            method="full_finetune", bundle_vae_explicit_false=False) == config["vae_swap_source"]
    for updates, method, bundle, message in (
        ({}, "lora", False, "full_finetune"),
        ({"vae_swap_new_channel_init": "zero"}, "full_finetune", False, "paired_spatial"),
        ({"chimera_training_stage": "bridge_align"}, "full_finetune", False, "unet or joint"),
        ({}, "full_finetune", True, "bundled target codec"),
    ):
        with pytest.raises(ValueError, match=message):
            preflight_vae_swap({**config, **updates}, arch=arch, method=method,
                bundle_vae_explicit_false=bundle)


@pytest.mark.parametrize("stage", ["unet", "joint", "bridge_align"])
def test_runner_no_longer_refuses_supported_swap_before_components_load(stage):
    from core.training.train_runner import _apply_chimera_training_contract
    config = dict(vae_swap_source="registry:flux1", chimera_training_stage=stage,
        chimera_clip_hidden_weight=.5, chimera_clip_pooled_weight=.5)
    manifest = dict(model_type="sensenova_sdxl_chimera", format_version=2,
        prediction=dict(type="flow_velocity"))
    with patch("core.model_loader.ModelLoader.detect_model_type", return_value="sensenova_sdxl_chimera"), \
         patch("core.models.sensenova_sdxl_chimera.artifact.read_artifact_documents", return_value=(manifest, {})):
        if stage == "bridge_align":
            with pytest.raises(ValueError, match="unet or joint"):
                _apply_chimera_training_contract("artifact", "full_finetune", config)
        else:
            assert _apply_chimera_training_contract("artifact", "full_finetune", config)
            assert config["vae_swap_source"] == "registry:flux1"


def test_v4_artifact_rejects_missing_or_changed_drift_contract_before_tensor_load(tmp_path):
    from copy import deepcopy
    from core.models.sensenova_sdxl_chimera.vae_artifact import validate_documents, POLAR_DRIFT_CONTRACT
    from core.models.sensenova_sdxl_chimera.artifact import prediction_contract
    from core.models.sensenova_sdxl_chimera.flow import FLOW_V4_PREDICTION
    declaration = dict(version=1, method="paired_spatial", token_policy="target_native",
        old_channels=4, new_channels=2, old_scale=8, new_scale=16, source="synthetic",
        ndim=4, norm="identity", norm_pack=1, input_channels=3)
    target = prediction_contract(FLOW_V4_PREDICTION, latent_channels=2,
        latent_mean=[.1, -.2], latent_centered_second_moment=.7)
    source = prediction_contract(FLOW_V4_PREDICTION,
        latent_mean=[.1]*4, latent_centered_second_moment=.8)
    record = dict(version=1, declaration=declaration, source_prediction=source, target_prediction=target,
        polar_boundary=dict(POLAR_DRIFT_CONTRACT), calibration=dict(images=2, positions_per_channel=12,
            latent_mean=target["latent_mean"], latent_centered_second_moment=.7))
    manifest = dict(vae_alignment=record, prediction=target, vae=dict(latent_channels=2, scale_factor=16))
    runtime = dict(vae_alignment=deepcopy(record), unet=dict(in_channels=4, out_channels=4))
    assert validate_documents(manifest, runtime) == declaration
    for value in (None, {**POLAR_DRIFT_CONTRACT, "version": 2}):
        manifest["vae_alignment"]["polar_boundary"] = value
        runtime["vae_alignment"] = deepcopy(manifest["vae_alignment"])
        with pytest.raises(ValueError, match="drift contract"):
            validate_documents(manifest, runtime)


@pytest.mark.parametrize("kind", ["endpoint_observable_velocity", "endpoint_observable_residual"])
def test_runner_preserves_aligned_v2_statistics_without_native_four_channel_revalidation(kind):
    from core.training.train_runner import _apply_chimera_training_contract
    from core.models.sensenova_sdxl_chimera.artifact import prediction_contract
    declaration = dict(version=1, method="paired_spatial", token_policy="target_native",
        old_channels=4, new_channels=2, old_scale=8, new_scale=16, source="synthetic",
        ndim=4, norm="identity", norm_pack=1, input_channels=3)
    manifest = dict(model_type="sensenova_sdxl_chimera", format_version=3,
        prediction=prediction_contract(kind, latent_channels=2, latent_mean=[.1, -.2],
            latent_centered_second_moment=.7), vae_alignment=dict(declaration=declaration))
    config = dict(chimera_training_stage="unet", chimera_flow_version="auto",
        chimera_v2_latent_mean=[99.]*4, chimera_v2_latent_centered_second_moment=99.)
    with patch("core.model_loader.ModelLoader.detect_model_type", return_value="sensenova_sdxl_chimera"), \
         patch("core.models.sensenova_sdxl_chimera.artifact.read_artifact_documents", return_value=(manifest, {})):
        assert _apply_chimera_training_contract("artifact", "full_finetune", config)
        assert config["chimera_v2_latent_mean"] is None
        assert config["chimera_v2_latent_centered_second_moment"] is None
        assert config["chimera_v2_parameterization"] == (
            "direct_velocity" if kind == "endpoint_observable_velocity" else "analytic_residual")
        config["chimera_flow_version"] = "v1"
        with pytest.raises(ValueError, match="saved prediction contract"):
            _apply_chimera_training_contract("artifact", "full_finetune", config)
