"""Exact native shared-loop continuation from a pending-gradient batch seam."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from ming_image_01_conditioning_test import write_tiny_package
from core.training.lora_trainer import LoRATrainer
from api.arch_capabilities import TRAINING_UNSUPPORTED
from core.models.ming_image_01.training_resume import RUNTIME_KEY


@pytest.mark.parametrize("text_mode,groups,profile,projection_only", [
    ("cpu_prefetch", [], "generation_edit", False),
    ("cpu_prefetch", [], "layer_decompose", False),
    ("swap_onthefly", [], "generation_edit", True),
    ("swap_onthefly", [], "layer_decompose", True),
])
def test_lora_mid_epoch_mnt_accumulation_resume_matches_uninterrupted(tmp_path, monkeypatch, text_mode, groups, profile, projection_only):
    from core.training import latent_cache
    from core.training.ops import ming_image_01_ops
    monkeypatch.setattr(latent_cache, "get_cache_base_dir", lambda: str(tmp_path / "cache"))
    root = tmp_path / "package"
    root.mkdir()
    write_tiny_package(root, profile=profile)
    items = []
    for index in range(4):
        path = tmp_path / f"target_{index}.png"
        Image.new("RGBA", (64, 64), (80 + index * 20, 60, 40, 255)).save(path)
        if profile == "layer_decompose":
            layers = [f"layer_{index}_{number}.png" for number in (1, 2)]
            for number, name in enumerate(layers):
                Image.new("RGBA", (64, 64), (60, 90 + number * 20, 40, 120)).save(tmp_path / name)
            path.with_name(path.name + ".ming.json").write_text(json.dumps({"schema_version": 1,
                "profile": profile, "targets": [
                    {"path": path.name, "role": "auxiliary_canvas", "weight": 0},
                    {"path": layers[0], "role": "layer_1", "weight": 1},
                    {"path": layers[1], "role": "layer_2", "weight": .5}], "references": [path.name]}))
        items.append({"image_path": str(path), "caption": f"word{index + 5}", "width": 64, "height": 64})
    dataset = SimpleNamespace(unique_id="resume-ming", dataset_root=tmp_path, items=items,
        reload_for_epoch=lambda **kwargs: None)
    traces = []
    original_loss = ming_image_01_ops.flow_loss
    def recorded_loss(transformer, **kwargs):
        traces.append(tuple(kwargs[key].detach().clone() for key in ("clean", "noise", "sigma")))
        return original_loss(transformer, **kwargs)
    monkeypatch.setattr(ming_image_01_ops, "flow_loss", recorded_loss)
    config = {"ming_conditioner_groups": groups, "seed": 17,
        "latent_encoding_mode": "swap_onthefly", "text_encoding_mode": text_mode,
        "train_adapter": projection_only, "train_unet": not projection_only, "gradient_checkpointing": True}
    def trainer(folder, resume=None):
        return LoRATrainer(model_path=str(root), output_dir=str(folder), lora_rank=2, lora_alpha=2,
            train_unet=not projection_only, train_text_encoder=False,
            device="cpu", weight_dtype="fp32", training_dtype="fp32", vae_dtype="fp32",
            mixed_precision=False, attention_backend="native", train_config=config.copy(),
            resume_from_checkpoint=resume)
    def train(owner, resume=None):
        owner.train([dataset], num_epochs=1, total_steps=8, batch_size=1,
            multi_noise_timesteps=2, gradient_accumulation_steps=4,
            save_every_n_steps=2, sample_every_n_steps=0, enable_bucketing=False,
            max_step_saves_to_keep=4, max_optimizer_saves_to_keep=4,
            base_resolutions=[64], latent_encoding_mode="swap_onthefly", text_encoding_mode=text_mode,
            resume_from_checkpoint=resume)
    baseline = trainer(tmp_path / "baseline" / "run")
    train(baseline)
    baseline_traces = traces.copy()
    assert len(baseline_traces) == 8
    checkpoint = baseline.output_dir / "run_step_000002.safetensors"
    assert checkpoint.is_file()
    state = json.loads((baseline.output_dir / "run_step_000002_state.json").read_text())
    assert state["epoch"] == 0 and state["batch_idx"] == 1
    payload = torch.load(baseline.output_dir / "run_step_000002_optimizer.pt", map_location="cpu", weights_only=True)
    assert payload[RUNTIME_KEY]["gradients"], "The checkpoint should retain two of four accumulated backwards"
    destination = tmp_path / "resumed" / "run"
    destination.mkdir(parents=True)
    for path in baseline.output_dir.glob("run_step_000002*"):
        if path.is_dir():
            shutil.copytree(path, destination / path.name)
        else:
            shutil.copy2(path, destination / path.name)
    traces.clear()
    resumed = trainer(destination, "latest")
    train(resumed, "latest")
    assert len(traces) == 6
    for expected, actual in zip(baseline_traces[2:], traces):
        assert all(torch.equal(a, b) for a, b in zip(expected, actual)), "Resumed data/noise/timesteps differ"
    for name in ("transformer", "text_encoder"):
        expected = getattr(baseline, name).state_dict()
        actual = getattr(resumed, name).state_dict()
        assert all(torch.equal(value, actual[key]) for key, value in expected.items()), f"{name} continuation differs"
    expected = baseline.optimizer.state_dict()["state"]
    actual = resumed.optimizer.state_dict()["state"]
    assert expected.keys() == actual.keys()
    for index, values in expected.items():
        assert all(torch.equal(value, actual[index][key]) for key, value in values.items())
    assert baseline._optimizer_update_step == resumed._optimizer_update_step == 2
    assert baseline.lr_scheduler.last_epoch == resumed.lr_scheduler.last_epoch
    assert resumed._ming_pending_runtime is None
    assert not torch.cuda.is_initialized()


