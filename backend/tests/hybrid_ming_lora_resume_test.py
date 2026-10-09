"""Real shared trainer restores structural weights and named optimizer state."""
import shutil
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from core.training.lora_trainer import LoRATrainer
from core.models.ming_image_01.loader import load_components
from ming_image_01_conditioning_test import write_tiny_package
from hybrid_ming_test import hybrid_for


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_shared_hybrid_lora_resume_from_native_package_matches_continuation(tmp_path, monkeypatch, profile):
    import json
    from core.training import latent_cache
    monkeypatch.setattr(latent_cache, "get_cache_base_dir", lambda: str(tmp_path / "cache"))
    root = tmp_path / "native"
    root.mkdir()
    write_tiny_package(root, profile=profile)
    components = load_components(root, dtype=torch.float32, conditioner_attention="eager")
    from core.models.ming_image_01.residual_unet import restore_topology
    restore_topology(components.transformer, None)
    save_file(components.transformer.state_dict(), str(root / "transformer" / "diffusion_pytorch_model.safetensors"))
    spec = hybrid_for(components.transformer)
    del components
    items = []
    for index in range(2):
        image = tmp_path / f"target{index}.png"
        Image.new("RGBA", (64, 64), (120, 80 + index, 30, 200)).save(image)
        if profile == "layer_decompose":
            layer = tmp_path / f"layer{index}.png"
            Image.new("RGBA", (64, 64), (20, 60, 120, 100)).save(layer)
            image.with_name(image.name + ".ming.json").write_text(json.dumps({"schema_version": 1,
                "profile": profile, "targets": [{"path": image.name, "role": "auxiliary_canvas", "weight": 0},
                    {"path": layer.name, "role": "layer_1", "weight": 1}], "references": [image.name]}))
        items.append(dict(image_path=str(image), caption="word5", width=64, height=64))
    dataset = SimpleNamespace(unique_id="hybrid-ming-resume", dataset_root=tmp_path,
        items=items, reload_for_epoch=lambda **kwargs: None)
    def trainer(folder, resume=None):
        config = dict(ming_conditioner_groups=[], seed=17, gradient_checkpointing=True,
            dit_rewiring_mode="inherit" if resume else "configure",
            dit_rewiring_spec=None if resume else spec)
        return LoRATrainer(model_path=str(root), output_dir=str(folder), lora_rank=2, lora_alpha=2,
            train_unet=True, train_text_encoder=False, device="cpu", weight_dtype="fp32",
            training_dtype="fp32", vae_dtype="fp32", mixed_precision=False,
            attention_backend="native", train_config=config, resume_from_checkpoint=resume)
    def train(owner, resume=None):
        owner.train([dataset], num_epochs=1, total_steps=2, batch_size=1,
            save_every_n_steps=1, sample_every_n_steps=0, enable_bucketing=False,
            max_step_saves_to_keep=3, max_optimizer_saves_to_keep=3, base_resolutions=[64],
            latent_encoding_mode="swap_onthefly", text_encoding_mode="swap_onthefly",
            resume_from_checkpoint=resume)
    baseline = trainer(tmp_path / "baseline" / "run")
    train(baseline)
    checkpoint = baseline.output_dir / "run_step_000001.safetensors"
    saved = torch.load(checkpoint.with_name("run_step_000001_optimizer.pt"), map_location="cpu", weights_only=True)
    names = [name for group in saved["_sushi_param_names"] for name in group]
    assert sum("rewiring_modules" in name for name in names) == len(list(baseline.transformer.rewiring_modules.parameters()))
    destination = tmp_path / "resumed" / "run"
    destination.mkdir(parents=True)
    for path in baseline.output_dir.glob("run_step_000001*"): shutil.copy2(path, destination / path.name)
    selected = str(destination / checkpoint.name)
    resumed = trainer(destination, selected)
    assert resumed.transformer._residual_unet_spec == spec
    train(resumed, selected)
    for name, value in baseline.transformer.state_dict().items():
        torch.testing.assert_close(value, resumed.transformer.state_dict()[name], rtol=2e-5, atol=2e-6)
    for parameter in resumed.transformer.rewiring_modules.parameters():
        assert resumed.optimizer.state[parameter]["step"].item() == 2
    assert not torch.cuda.is_initialized()
