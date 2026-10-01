"""CPU native package through the real shared full-parameter trainer."""
from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def forbidden_cuda(*args, **kwargs):
    raise AssertionError("Ming full trainer test attempted CUDA initialization")
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)
from ming_image_01_conditioning_test import write_tiny_package
from core.training.full_parameter_trainer import FullParameterTrainer
from api.arch_capabilities import TRAINING_UNSUPPORTED

@pytest.mark.parametrize("latent_mode,text_mode,groups", [
    ("onthefly_gpu", "onthefly_gpu", []),
    ("swap_onthefly", "swap_onthefly", []),
    ("pre_encoded_cache", "pre_encoded_cache", []),
    ("swap_onthefly", "cpu_prefetch", []),
    ("swap_onthefly", "swap_onthefly", ["projections"]),
])
def test_real_full_trainer_step(tmp_path, monkeypatch, latent_mode, text_mode, groups):
    from core.training import latent_cache
    monkeypatch.setattr(latent_cache, "get_cache_base_dir", lambda: str(tmp_path / "cache"))
    root = tmp_path / "package"
    root.mkdir()
    _, _, topology = write_tiny_package(root)
    # The public entry point stays staged until the complete lifecycle passes.
    monkeypatch.setitem(TRAINING_UNSUPPORTED, "ming_image_01", {})
    owner = FullParameterTrainer(model_path=str(root), output_dir=str(tmp_path / "run"),
        device="cpu", weight_dtype="fp32", training_dtype="fp32", vae_dtype="fp32",
        mixed_precision=False, attention_backend="native",
        train_config={"ming_conditioner_groups": groups, "seed": 17,
            "latent_encoding_mode": latent_mode, "text_encoding_mode": text_mode})
    assert owner.is_ming_image_01 and owner.arch.name == "ming_image_01"
    assert owner.transformer._residual_unet_spec == topology
    assert owner.transformer.training and owner.text_encoder.cacheable == (not groups)
    assert not owner.vae.training
    assert all(p.device.type == "cpu" for p in owner.transformer.parameters())
    from types import SimpleNamespace
    from PIL import Image
    image_path = tmp_path / "target.png"
    Image.new("RGBA", (64, 64), (100, 80, 60, 255)).save(image_path)
    dataset = SimpleNamespace(unique_id="tiny-ming", dataset_root=tmp_path,
        items=[{"image_path": str(image_path), "caption": "word5", "width": 64, "height": 64}],
        reload_for_epoch=lambda **kwargs: None)
    before = owner.transformer.layers[0].attention.to_q.weight.detach().clone()
    projection_before = {name: p.detach().clone() for name, p in owner.text_encoder.named_parameters()
                         if p.requires_grad}
    owner.train([dataset], num_epochs=1, total_steps=1, batch_size=1,
        save_every_n_steps=1, sample_every_n_steps=0,
        enable_bucketing=False, base_resolutions=[64],
        latent_encoding_mode=latent_mode, text_encoding_mode=text_mode)
    assert not torch.equal(before, owner.transformer.layers[0].attention.to_q.weight)
    if groups:
        current = dict(owner.text_encoder.named_parameters())
        assert any(not torch.equal(value, current[name]) for name, value in projection_before.items())
    from core.training.base_trainer import _list_checkpoint_entries
    checkpoint = owner.output_dir / "run_step_000001"
    assert checkpoint in _list_checkpoint_entries(owner.output_dir)
    assert (owner.output_dir / "run_step_000001_optimizer.pt").is_file()
    assert (owner.output_dir / "run_step_000001_state.json").is_file()
    if latent_mode == text_mode == "onthefly_gpu":
        trained = {k: v.detach().clone() for k, v in owner.transformer.state_dict().items()}
        resumed = FullParameterTrainer(model_path=str(root), output_dir=str(owner.output_dir),
            device="cpu", weight_dtype="fp32", training_dtype="fp32", vae_dtype="fp32",
            mixed_precision=False, attention_backend="native", resume_from_checkpoint="latest",
            train_config={"ming_conditioner_groups": [], "seed": 17,
                "latent_encoding_mode": latent_mode, "text_encoding_mode": text_mode})
        assert Path(resumed._loaded_checkpoint_path) == checkpoint
        assert resumed.transformer._residual_unet_spec == topology
        assert all(torch.equal(v, resumed.transformer.state_dict()[k]) for k, v in trained.items())
        resumed.train([dataset], num_epochs=2, total_steps=2, batch_size=1,
            save_every_n_steps=1, sample_every_n_steps=0, enable_bucketing=False,
            base_resolutions=[64], latent_encoding_mode=latent_mode, text_encoding_mode=text_mode,
            resume_from_checkpoint="latest")
        assert (owner.output_dir / "run_step_000002").is_dir()
        assert all(int(state["step"]) == 2 for state in resumed.optimizer.state.values())
    assert not torch.cuda.is_initialized()


def test_checkpoint_discovery_ignores_unrelated_manifest(tmp_path):
    import json
    from core.training.base_trainer import _list_checkpoint_entries
    for name, payload in [("other_step_000002", {"model_type": "other"}),
                          ("broken_step_000003", []), ("partial_step_000004", None)]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "manifest.json").write_text("{" if payload is None else json.dumps(payload))
    assert _list_checkpoint_entries(tmp_path) == []


@pytest.mark.parametrize("groups", [[], ["vision", "projections"]])
def test_real_layer_training_routes_ordered_supervision(tmp_path, monkeypatch, groups):
    import json
    from types import SimpleNamespace
    from PIL import Image
    from core.training.ops import ming_image_01_ops
    root = tmp_path / "package"
    root.mkdir()
    _, _, topology = write_tiny_package(root, profile="layer_decompose")
    monkeypatch.setitem(TRAINING_UNSUPPORTED, "ming_image_01", {})
    owner = FullParameterTrainer(model_path=str(root), output_dir=str(tmp_path / "run"),
        device="cpu", weight_dtype="fp32", training_dtype="fp32", vae_dtype="fp32",
        mixed_precision=False, attention_backend="native",
        train_config={"ming_conditioner_groups": groups, "seed": 17})
    data = tmp_path / "data"
    data.mkdir()
    paths = ["canvas.png", "layer1.png", "layer2.png"]
    for index, name in enumerate(paths):
        Image.new("RGBA", (64, 64), (40 + 50 * index, 80, 100, index * 80)).save(data / name)
    (data / "canvas.png.ming.json").write_text(json.dumps({"schema_version": 1,
        "profile": "layer_decompose", "targets": [
            {"path": name, "role": "auxiliary_canvas" if i == 0 else f"layer_{i}", "weight": [0, 1, .5][i]}
            for i, name in enumerate(paths)], "references": ["canvas.png"]}))
    dataset = SimpleNamespace(unique_id="tiny-layer", dataset_root=data,
        items=[{"image_path": str(data / "canvas.png"), "caption": "word5", "width": 64, "height": 64}],
        reload_for_epoch=lambda **kwargs: None)
    calls = []
    actual = ming_image_01_ops.train_step
    def record_batch(trainer, batch, **kwargs):
        calls.append((tuple(batch.clean.shape), batch.frame_weights.detach().clone(),
                      [tuple(ref.shape) for ref in batch.reference_latents]))
        return actual(trainer, batch, **kwargs)
    monkeypatch.setattr(ming_image_01_ops, "train_step", record_batch)
    trained_before = {name: p.detach().clone() for name, p in owner.text_encoder.named_parameters() if p.requires_grad}
    owner.train([dataset], total_steps=2, batch_size=1, save_every_n_steps=2,
        multi_noise_timesteps=2, gradient_accumulation_steps=2,
        sample_every_n_steps=0, enable_bucketing=False, base_resolutions=[64],
        latent_encoding_mode="swap_onthefly", text_encoding_mode="swap_onthefly")
    assert len(calls) == 2
    assert all(int(state["step"]) == 1 for state in owner.optimizer.state.values())
    for shape, weights, refs in calls:
        assert shape == (1, 16, 3, 8, 8)
        torch.testing.assert_close(weights, torch.tensor([[0., 1., .5]]))
        assert refs == [(16, 1, 8, 8)]
    if groups:
        current = dict(owner.text_encoder.named_parameters())
        assert any("vision" in name and not torch.equal(value, current[name]) for name, value in trained_before.items())
    from core.models.ming_image_01.loader import load_components
    restored = load_components(owner.output_dir / "run_step_000002", dtype=torch.float32, conditioner_attention="eager")
    assert restored.artifact.profile == "layer_decompose"
    assert restored.transformer._residual_unet_spec == topology
    assert not torch.cuda.is_initialized()
