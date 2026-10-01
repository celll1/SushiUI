"""CPU full-training caption scope, rates and native bundle roundtrip."""
import json
from types import SimpleNamespace
import pytest
import torch
import yaml
from PIL import Image
from api.routes import TrainingRunCreateRequest, _check_ming_conditioner_params, _check_adapter_params
from core.training.training_config import TrainingConfigGenerator
from core.training.train_runner import _prepare_training_process_config
from core.training.full_parameter_trainer import FullParameterTrainer
from core.models.ming_image_01.loader import load_components
from ming_image_01_conditioning_test import write_tiny_package


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
@pytest.mark.parametrize("train_dit,choice,rate", [(False, True, .0002), (True, False, .0002),
    (True, None, 0.), (False, True, 0.)])
def test_public_full_caption_scope_rates_and_export(tmp_path, profile, train_dit, choice, rate):
    root = tmp_path/"package"
    root.mkdir()
    write_tiny_package(root, profile=profile)
    request = TrainingRunCreateRequest(training_method="full_finetune", base_model_path=str(root),
        total_steps=1, train_unet=train_dit, train_text_encoder=False, ming_conditioner_groups=[],
        train_adapter=choice, adapter_lr=rate, base_resolutions=[64],
        text_encoding_mode="swap_onthefly", latent_encoding_mode="swap_onthefly")
    _check_ming_conditioner_params(request)
    _check_adapter_params(request)
    payload = request.model_dump()
    payload["_explicit_fields"] = sorted(request.model_fields_set)
    serialized = TrainingConfigGenerator.generate_full_finetune_config(payload,
        run_name="run", base_model_path=str(root), output_dir=str(tmp_path/"run"))
    process, train, _, method = _prepare_training_process_config(yaml.safe_load(serialized), str(root))
    assert method == "full_finetune" and train["train_adapter"] == choice and train["adapter_lr"] == rate
    owner = FullParameterTrainer(model_path=str(root), output_dir=process["training_folder"],
        device="cpu", weight_dtype="fp32", training_dtype="fp32", vae_dtype="fp32",
        mixed_precision=False, attention_backend="native", train_unet=train_dit,
        train_text_encoder=False, train_config=train)
    caption_on = train_dit if choice is None else choice
    assert all(parameter.requires_grad == caption_on for parameter in owner.transformer.cap_embedder.parameters())
    groups = owner.setup_trainable_parameters()
    caption = [group for group in groups if group["name"] == "caption"]
    assert bool(caption) == caption_on
    if caption:
        assert caption[0]["lr"] == rate and caption[0]["component"] == "adapter"
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(ids) == len(set(ids))
    before = {name: value.detach().clone() for name, value in owner.transformer.named_parameters()}
    image = tmp_path/"target.png"
    Image.new("RGBA", (64, 64), (100, 70, 60, 220)).save(image)
    if profile == "layer_decompose":
        layer = tmp_path/"layer.png"
        Image.new("RGBA", (64, 64), (40, 80, 100, 100)).save(layer)
        image.with_name(image.name + ".ming.json").write_text(json.dumps({"schema_version": 1,
            "profile": profile, "targets": [{"path": image.name, "role": "auxiliary_canvas", "weight": 0},
                {"path": layer.name, "role": "layer_1", "weight": 1}], "references": [image.name]}))
    dataset = SimpleNamespace(unique_id="caption-full", dataset_root=tmp_path,
        items=[{"image_path": str(image), "caption": "word5", "width": 64, "height": 64}],
        reload_for_epoch=lambda **kwargs: None)
    owner.train([dataset], num_epochs=1, total_steps=1, batch_size=1, save_every_n_steps=1,
        sample_every_n_steps=0, enable_bucketing=False, base_resolutions=[64],
        text_encoding_mode="swap_onthefly", latent_encoding_mode="swap_onthefly")
    current = dict(owner.transformer.named_parameters())
    assert all(torch.isfinite(value).all() for value in current.values())
    changed_caption = any(not torch.equal(value, current[name]) for name, value in before.items() if name.startswith("cap_embedder."))
    changed_dit = any(not torch.equal(value, current[name]) for name, value in before.items() if not name.startswith("cap_embedder."))
    assert changed_caption == (caption_on and rate > 0)
    assert changed_dit == train_dit
    restored = load_components(owner.output_dir/"run_step_000001", dtype=torch.float32, conditioner_attention="eager")
    actual = restored.transformer.state_dict()
    assert all(torch.equal(value, actual[name]) for name, value in owner.transformer.state_dict().items())
    assert not torch.cuda.is_initialized()
