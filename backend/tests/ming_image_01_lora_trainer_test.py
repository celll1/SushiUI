"""CPU native LoRA trainer, shared checkpoint codec and generation roundtrip."""
import json
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from core.training.lora_trainer import LoRATrainer
from core.models.ming_image_01.loader import load_components
from core.models.ming_image_01.generation_adapters import generation_adapters
from api.arch_capabilities import TRAINING_UNSUPPORTED
from ming_image_01_conditioning_test import write_tiny_package


def make_owner(root, output, *, projection_only=False, resume=None):
    return LoRATrainer(model_path=str(root), output_dir=str(output), lora_rank=2, lora_alpha=2,
        device="cpu", weight_dtype="fp32", training_dtype="fp32", vae_dtype="fp32",
        mixed_precision=False, attention_backend="native", train_unet=not projection_only,
        train_text_encoder=False,
        resume_from_checkpoint=resume,
        train_config={"ming_conditioner_groups": [], "train_unet": not projection_only,
                      "train_adapter": projection_only, "gradient_checkpointing": True, "save_optimizer_state": True})


@pytest.mark.parametrize("projection_only", [False, True])
@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_native_lora_shared_trainer_save_resume_and_generation(tmp_path, monkeypatch, projection_only, profile):
    monkeypatch.setitem(TRAINING_UNSUPPORTED, "ming_image_01", {})
    root = tmp_path/"package"
    root.mkdir()
    write_tiny_package(root, profile=profile)
    owner = make_owner(root, tmp_path/"run", projection_only=projection_only)
    if projection_only:
        owner.config["adapter_lr"] = 0.
        assert owner.setup_trainable_parameters()[0]["lr"] == 0.
        owner.config["adapter_lr"] = .0002
        assert owner.setup_trainable_parameters()[0]["lr"] == .0002
    base_states = {name: value.detach().clone() for name, value in owner.transformer.named_parameters()
                   if not value.requires_grad}
    initial = {name: value.detach().clone() for name, value in owner.transformer.named_parameters()
               if value.requires_grad}
    assert initial
    assert all(torch.isfinite(value).all() for value in owner.transformer.parameters())
    assert all(not parameter.requires_grad for parameter in owner.text_encoder.parameters())
    image = tmp_path/"target.png"
    Image.new("RGBA", (64, 64), (120, 80, 30, 200)).save(image)
    if profile == "layer_decompose":
        layer = tmp_path/"layer.png"
        Image.new("RGBA", (64, 64), (20, 60, 120, 100)).save(layer)
        image.with_name(image.name + ".ming.json").write_text(json.dumps({"schema_version": 1,
            "profile": profile, "targets": [{"path": image.name, "role": "auxiliary_canvas", "weight": 0},
                {"path": layer.name, "role": "layer_1", "weight": 1}], "references": [image.name]}))
    dataset = SimpleNamespace(unique_id="ming-lora", dataset_root=tmp_path,
        items=[{"image_path": str(image), "caption": "word5", "width": 64, "height": 64}],
        reload_for_epoch=lambda **kwargs: None)
    owner.train([dataset], num_epochs=1, total_steps=1, batch_size=1, save_every_n_steps=1,
        sample_every_n_steps=0, enable_bucketing=False, base_resolutions=[64],
        latent_encoding_mode="swap_onthefly", text_encoding_mode="swap_onthefly")
    current = dict(owner.transformer.named_parameters())
    assert all(torch.isfinite(value).all() for value in current.values())
    assert any(not torch.equal(before, current[name]) for name, before in initial.items())
    assert all(torch.equal(before, current[name]) for name, before in base_states.items())
    checkpoint = owner.output_dir/"run_step_000001.safetensors"
    with safe_open(checkpoint, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    assert metadata["model_type"] == "ming_image_01"
    assert metadata["ming_image_01_profile"] == profile
    resumed = make_owner(root, tmp_path/"resumed", projection_only=projection_only, resume=str(checkpoint))
    for name, layer in owner.lora_layers.items():
        for key, value in layer.branch_tensors().items():
            torch.testing.assert_close(value, resumed.lora_layers[name].branch_tensors()[key], rtol=0, atol=0)
    generated = load_components(root, dtype=torch.float32, conditioner_attention="eager")
    selected = "cap_embedder.1" if projection_only else "layers.0.attention.to_q"
    trained = owner.transformer.get_submodule(selected)
    data = torch.randn(3, trained.original_module.in_features)
    base = generated.transformer.get_submodule(selected)
    with generation_adapters(generated, [{"path": str(checkpoint)}], resolve_path=str):
        torch.testing.assert_close(generated.transformer.get_submodule(selected)(data), trained(data), rtol=1e-6, atol=1e-7)
    assert generated.transformer.get_submodule(selected) is base
    bad = tmp_path/"wrong-profile.safetensors"
    save_file(load_file(str(checkpoint)), str(bad), metadata={**metadata, "ming_image_01_profile": "layer_decompose" if profile == "generation_edit" else "generation_edit"})
    snapshot = {name: value.detach().clone() for name, value in resumed.transformer.named_parameters()}
    with pytest.raises(ValueError, match="profile"):
        resumed.load_checkpoint(str(bad))
    assert all(torch.equal(value, dict(resumed.transformer.named_parameters())[name]) for name, value in snapshot.items())
    bad_scope = tmp_path/"wrong-scope.safetensors"
    tensors = load_file(str(checkpoint))
    tensors["extra.lora_down.weight"] = torch.ones(2, 2)
    save_file(tensors, str(bad_scope), metadata=metadata)
    with pytest.raises(ValueError, match="scope"):
        resumed.load_checkpoint(str(bad_scope))
    assert all(torch.equal(value, dict(resumed.transformer.named_parameters())[name]) for name, value in snapshot.items())
    assert not torch.cuda.is_initialized()
