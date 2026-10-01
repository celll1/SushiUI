"""Offline public request/YAML/runner contract into native full fine-tuning."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from PIL import Image
from api.routes import TrainingRunCreateRequest, _check_ming_conditioner_params
from api.arch_capabilities import TRAINING_UNSUPPORTED, training_feature_unsupported_reason
from core.training.training_config import TrainingConfigGenerator
from core.training.train_runner import _prepare_training_process_config, _is_bf16_native_base_model, _validate_latent_io
from core.training.full_parameter_trainer import FullParameterTrainer
from core.models.ming_image_01.training_options import preflight_training
from ming_image_01_conditioning_test import write_tiny_package


@pytest.mark.parametrize("profile,train_dit,groups", [("generation_edit", True, []),
    ("layer_decompose", False, ["projections"])])
def test_public_full_request_to_runner_to_native_training(tmp_path, profile, train_dit, groups):
    root = tmp_path / "package"
    root.mkdir()
    write_tiny_package(root, profile=profile)
    target = tmp_path / "target.png"
    Image.new("RGBA", (64, 64), (100, 80, 60, 255)).save(target)
    if profile == "layer_decompose":
        layer = tmp_path / "layer.png"
        Image.new("RGBA", (64, 64), (120, 50, 30, 120)).save(layer)
        target.with_name(target.name + ".ming.json").write_text(json.dumps({"schema_version": 1,
            "profile": profile, "targets": [
                {"path": target.name, "role": "auxiliary_canvas", "weight": 0},
                {"path": layer.name, "role": "layer_1", "weight": 1}], "references": [target.name]}))
    request = TrainingRunCreateRequest(training_method="full_finetune", base_model_path=str(root),
        total_steps=1, train_unet=train_dit, train_text_encoder=False, ming_conditioner_groups=groups,
        use_reference_images=profile == "layer_decompose", base_resolutions=[64],
        text_encoding_mode="swap_onthefly", latent_encoding_mode="swap_onthefly")
    _check_ming_conditioner_params(request)
    payload = request.model_dump()
    payload["_explicit_fields"] = sorted(request.model_fields_set)
    config = yaml.safe_load(TrainingConfigGenerator.generate_full_finetune_config(payload,
        run_name="run", base_model_path=str(root), output_dir=str(tmp_path / "run")))
    process, train, network, method = _prepare_training_process_config(config, str(root))
    assert method == "full_finetune" and train["train_text_encoder"] == bool(groups)
    assert _is_bf16_native_base_model(str(root)), "Config detection must recognize renamed Ming packages"
    assert "full_finetune" not in TRAINING_UNSUPPORTED.get("ming_image_01", {})
    assert training_feature_unsupported_reason("ming_image_01", "reference_images", method) is None
    owner = FullParameterTrainer(model_path=str(root), output_dir=process["training_folder"],
        device="cpu", weight_dtype="fp32", training_dtype="fp32", vae_dtype="fp32",
        mixed_precision=False, attention_backend="native", train_unet=train["train_unet"],
        train_text_encoder=train["train_text_encoder"], train_config=train)
    _validate_latent_io(owner, train)
    before_dit = owner.transformer.layers[0].attention.to_q.weight.detach().clone()
    before_groups = {name: value.detach().clone() for name, value in owner.text_encoder.named_parameters()
                     if value.requires_grad}
    dataset = SimpleNamespace(unique_id="public-ming", dataset_root=tmp_path,
        items=[{"image_path": str(target), "caption": "word5", "width": 64, "height": 64}],
        reload_for_epoch=lambda **kwargs: None)
    owner.train([dataset], num_epochs=1, total_steps=1, batch_size=1,
        save_every_n_steps=1, sample_every_n_steps=0, enable_bucketing=False, base_resolutions=[64],
        text_encoding_mode=train["text_encoding_mode"], latent_encoding_mode=train["latent_encoding_mode"])
    assert torch.equal(before_dit, owner.transformer.layers[0].attention.to_q.weight) == (not train_dit)
    if groups:
        values = dict(owner.text_encoder.named_parameters())
        assert any(not torch.equal(before, values[name]) for name, before in before_groups.items())
    assert (owner.output_dir / "run_step_000001" / "manifest.json").is_file()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("method", ["relora", "controlnet", "vae_decoder"])
def test_non_native_methods_remain_refused_before_load(method):
    with pytest.raises(ValueError, match="pending|does not support|ControlNet"):
        preflight_training({}, method=method)


@pytest.mark.parametrize("patch,message", [
    ({"train_unet": False, "train_text_encoder": True, "ming_conditioner_groups": []}, "requires DiT"),
    ({"ming_conditioner_groups": ["projections"], "text_encoding_mode": "cpu_prefetch"}, "frozen"),
    ({"train_image_encoder": True}, "native vision"),
    ({"noise_process": "add_noise"}, "flow"),
    ({"prediction_target": "epsilon"}, "velocity"),
])
def test_native_contract_refuses_incompatible_run_before_model_load(patch, message):
    with pytest.raises(ValueError, match=message):
        preflight_training(patch, method="full_finetune")


def test_minimal_native_full_yaml_inherits_auto_prediction(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    write_tiny_package(root)
    config = yaml.safe_load(TrainingConfigGenerator.generate_full_finetune_config({"total_steps": 1,
        "train_text_encoder": False}, run_name="minimal", base_model_path=str(root), output_dir=str(tmp_path / "run")))
    _, train, _, _ = _prepare_training_process_config(config, str(root))
    assert train["noise_process"] == "auto"
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("config,message", [
    ({"train_text_encoder": True}, "conditioner LoRA"),
    ({"ming_conditioner_groups": ["language"]}, "conditioner LoRA"),
    ({"train_unet": False}, "requires DiT"),
])
def test_native_lora_preflight_rejects_unconnected_or_empty_scope(config, message):
    with pytest.raises(ValueError, match=message):
        preflight_training(config, method="lora")
