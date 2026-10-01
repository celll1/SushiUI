"""Exact native shared-loop continuation from a pending-gradient batch seam."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from ming_image_01_conditioning_test import write_tiny_package
from core.training.full_parameter_trainer import FullParameterTrainer
from api.arch_capabilities import TRAINING_UNSUPPORTED
from core.models.ming_image_01.training_resume import RUNTIME_KEY


@pytest.mark.parametrize("text_mode,groups,profile,caption_only", [("cpu_prefetch", [], "generation_edit", False),
    ("onthefly_gpu", ["projections"], "generation_edit", False),
    ("cpu_prefetch", [], "layer_decompose", False),
    pytest.param("cpu_prefetch", [], "generation_edit", True, id="caption_design"),
    pytest.param("cpu_prefetch", [], "layer_decompose", True, id="caption_layer")])
def test_mid_epoch_mnt_accumulation_resume_matches_uninterrupted(tmp_path, monkeypatch, text_mode, groups, profile, caption_only):
    from core.training import latent_cache
    from core.training.ops import ming_image_01_ops
    monkeypatch.setattr(latent_cache, "get_cache_base_dir", lambda: str(tmp_path / "cache"))
    monkeypatch.setitem(TRAINING_UNSUPPORTED, "ming_image_01", {})
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
        "train_unet": not caption_only, "train_adapter": True if caption_only else None,
        "adapter_lr": .0002 if caption_only else None}
    def trainer(folder, resume=None):
        return FullParameterTrainer(model_path=str(root), output_dir=str(folder), train_unet=not caption_only,
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
    checkpoint = baseline.output_dir / "run_step_000002"
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



def runtime_owner():
    parameter = torch.nn.Parameter(torch.tensor([2.]))
    optimizer = torch.optim.AdamW([parameter])
    scaler = torch.amp.GradScaler("cpu", init_scale=8.)
    scaler.scale((parameter * 3).sum()).backward()
    owner = SimpleNamespace(optimizer=optimizer, fused_optimizer_groups=None,
        device=torch.device("cpu"), run_seed=17, run_seed_drawn=False, _grad_accum_steps=4,
        grad_scaler=scaler, use_grad_scaler=True,
        _build_ema_param_name_map=lambda: {id(parameter): "transformer.weight"})
    return owner, parameter


def test_runtime_scaler_rng_and_pending_gradient_roundtrip():
    import random
    import numpy as np
    from core.models.ming_image_01.training_resume import capture_runtime, queue_runtime, restore_runtime
    owner, parameter = runtime_owner()
    runtime = capture_runtime(owner)
    expected = (random.random(), np.random.random(), torch.rand(4))
    gradient = parameter.grad.clone()
    parameter.grad = None
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    owner.grad_scaler.update(new_scale=32.)
    owner._ming_pending_runtime = runtime
    restore_runtime(owner)
    assert owner.grad_scaler.get_scale() == 8.
    assert torch.equal(parameter.grad, gradient)
    assert random.random() == expected[0] and np.random.random() == expected[1]
    assert torch.equal(torch.rand(4), expected[2])
    owner.run_seed_drawn = True
    queue_runtime(owner, {RUNTIME_KEY: runtime})
    assert owner.run_seed == 17
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("field,value,message", [
    ("gradient_accumulation_steps", 2, "gradient accumulation"),
    ("torch_device", "cuda", "device type"),
    ("gradients", {"unknown.weight": torch.ones(1)}, "ownership"),
    ("gradients", {"transformer.weight": torch.ones(2)}, "shape/precision"),
    ("grad_scaler", None, "scaler mode"),
])
def test_runtime_invalid_contract_refused_before_gradient_mutation(field, value, message):
    from core.models.ming_image_01.training_resume import capture_runtime, restore_runtime
    owner, parameter = runtime_owner()
    runtime = capture_runtime(owner)
    runtime[field] = value
    owner._ming_pending_runtime = runtime
    original = parameter.grad
    with pytest.raises(ValueError, match=message):
        restore_runtime(owner)
    assert parameter.grad is original
    assert not torch.cuda.is_initialized()
