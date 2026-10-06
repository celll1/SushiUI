"""Public audio swap contracts against real tiny codec/artifact headers."""

import pytest

from audio_vae_source_test import small_codec
from acestep_audio_artifact_test import artifact
from core.training.vae_swap import preflight_vae_swap, initialization_mode


def test_audio_preflight_accepts_auto_and_temporal_with_native_grid(tmp_path):
    codec = small_codec()
    codec.save_pretrained(tmp_path / "target")
    source = f"file:{tmp_path / 'target'}"
    kwargs = dict(arch="acestep", method="full_finetune", bundle_vae_explicit_false=False)
    for mode in (None, "auto", "paired_temporal"):
        config = dict(vae_swap_source=source)
        if mode is not None:
            config["vae_swap_new_channel_init"] = mode
        assert initialization_mode(config, "acestep") == "paired_temporal"
        assert preflight_vae_swap(config, **kwargs) == source
    from core.models.common.vae_source import describe_vae_source
    assert describe_vae_source(source, arch="acestep")["compatible"] is True


@pytest.mark.parametrize("patch,match", [
    ({"vae_swap_new_channel_init": "zero"}, "paired_temporal"),
    ({"vae_swap_new_channel_init": "paired_spatial"}, "paired_temporal"),
    ({"train_unet": False}, "train_unet"),
    ({"vae_swap_calibration_samples": True}, "sample count"),
    ({"vae_swap_calibration_samples": 257}, "sample count"),
])
def test_audio_preflight_refuses_invalid_training_contract_before_weight_load(patch, match):
    with pytest.raises(ValueError, match=match):
        preflight_vae_swap(dict(vae_swap_source="file:missing", **patch),
                           arch="acestep", method="full_finetune", bundle_vae_explicit_false=False)


def test_audio_preflight_requires_full_training_and_codec_bundle(tmp_path):
    _, _, _, path = artifact(tmp_path, 4, True)
    config = dict(vae_swap_source=f"model:{path}")
    with pytest.raises(ValueError, match="full_finetune"):
        preflight_vae_swap(config, arch="acestep", method="lora", bundle_vae_explicit_false=False)
    with pytest.raises(ValueError, match="bundl"):
        preflight_vae_swap(config, arch="acestep", method="full_finetune", bundle_vae_explicit_false=True)
    with pytest.raises(ValueError, match="bundle_vae=false"):
        preflight_vae_swap({}, arch="acestep", method="full_finetune", bundle_vae_explicit_false=True,
                           base_model_path=str(path))
    with pytest.raises(ValueError, match="Unknown"):
        initialization_mode({"vae_swap_new_channel_init": "paired_temporal"}, "sdxl")


def test_public_selector_lists_audio_codec_and_bundled_checkpoint(tmp_path, monkeypatch):
    import api.routes as routes
    small_codec().save_pretrained(tmp_path / "codec")
    _, _, _, path = artifact(tmp_path, 4, True)
    monkeypatch.setattr(routes, "_override_scan_dirs", lambda db: [str(tmp_path)])
    monkeypatch.setattr(routes, "_training_vae_export_dirs", lambda: [])
    result = routes.list_vae_sources(arch="acestep", db=None)["sources"]
    standalone = result["standalone"]
    assert len(standalone) == 1 and standalone[0]["source"] == f"file:{tmp_path / 'codec'}"
    bundled = result["extract_from_model"]
    assert len(bundled) == 1 and bundled[0]["source"] == f"model:{path}"
    assert standalone[0]["compatible"] and bundled[0]["compatible"]
    assert all(not candidate["compatible"] for candidate in result["registry"])
    payload = routes.TrainingRunCreateRequest(training_method="full_finetune", base_model_path="synthetic", total_steps=1,
        vae_swap_source=standalone[0]["source"], vae_swap_new_channel_init="paired_temporal")
    assert payload.model_dump()["vae_swap_new_channel_init"] == "paired_temporal"
    import yaml
    from core.training.training_config import TrainingConfigGenerator
    text = TrainingConfigGenerator.generate_full_finetune_config(payload.model_dump(), run_name="audio",
        base_model_path="synthetic", output_dir=str(tmp_path), dataset_path=str(tmp_path))
    process = yaml.safe_load(text)["config"]["process"][0]
    assert process["train"]["vae_swap_new_channel_init"] == "paired_temporal"
    restored = routes._extract_request_params_from_yaml(process, "full_finetune")
    assert restored["vae_swap_new_channel_init"] == "paired_temporal"
    assert restored["vae_swap_source"] == standalone[0]["source"]


def test_explicit_temporal_calibrates_through_training_owner(tmp_path):
    from acestep_vae_training_test import setup, audio_files
    trainer, resolved, _ = setup(tmp_path)
    trainer.config["vae_swap_new_channel_init"] = "paired_temporal"
    trainer.arch.apply_vae_swap(trainer, resolved)
    trainer.arch.calibrate_before_training(trainer, audio_files(tmp_path))
    assert trainer.transformer.decoder._audio_latent_alignment.ready
