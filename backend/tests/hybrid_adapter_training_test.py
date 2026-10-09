"""Hybrid full-module weights share adapter training, save and resume ownership."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from hybrid_adapter_payload_test import case, native
from core.training.lora_trainer import LoRATrainer
from core.training.base_trainer import BaseTrainer
from core.rewiring.training_adapters import prepare_hybrid_resume


def owner_for(model, spec, algorithm="lora"):
    owner = SimpleNamespace(transformer=model, config={"training_method": "lora", "ming_conditioner_groups": []},
        arch=SimpleNamespace(name=spec["architecture"]), train_unet=True, train_text_encoder=False,
        learning_rate=.001, unet_lr=.001, lora_layers={}, lora_rank=2, lora_alpha=2,
        device="cpu", training_dtype=torch.float32, blocks_to_swap=0, gradient_checkpointing=False,
        log_prefix="[HybridAdapterTest]", adapter_algorithm=algorithm, weight_decompose=False, adapter_config={})
    model.requires_grad_(False)
    if spec["architecture"] == "anima":
        from core.training.adapters.anima_adapter import AnimaLoRAAdapter as Adapter
        owner.is_anima = True
    elif spec["architecture"] == "qwen_image_21":
        from core.training.adapters.qwen_image_21_adapter import QwenImage21LoRAAdapter as Adapter
        owner.is_qwen_image_21 = True
    else:
        from core.training.adapters.ming_image_01_adapter import MingImage01LoRAAdapter as Adapter
        class Conditioner(nn.Identity):
            def set_trainable(self, _): self.requires_grad_(False)
        owner.is_ming_image_01 = True
        owner.ming_conditioner_trainable_groups = set()
        owner.ming_image_01_components = SimpleNamespace(transformer=model, conditioner=Conditioner(),
            vae=nn.Identity(), artifact=SimpleNamespace(profile="generation_edit"))
    owner.adapter = Adapter(owner, 2, 2, torch.float32)
    return owner


@contextmanager
def generation(model, spec, path):
    configs = [{"path": str(path), "strength": 1}]
    if spec["architecture"] == "anima":
        from core.pipeline_backends.anima import AnimaMixin
        class Owner(AnimaMixin): _anima_resolve_lora_path = staticmethod(str)
        owner = Owner()
        owner.anima_components = {"transformer": model}
        load, unload = owner._load_lora_anima, owner._unload_lora_anima
    elif spec["architecture"] == "qwen_image_21":
        from core.pipeline_backends.qwen_image_21 import QwenImage21Mixin
        class Owner(QwenImage21Mixin): _qwen21_resolve_lora_path = staticmethod(str)
        owner = Owner()
        owner.qwen_image_21_components = {"transformer": model}
        load, unload = owner._load_lora_qwen21, owner._unload_lora_qwen21
    else:
        from core.models.ming_image_01.generation_adapters import generation_adapters
        components = SimpleNamespace(transformer=model, artifact=SimpleNamespace(profile="generation_edit"))
        with generation_adapters(components, configs, resolve_path=str): yield
        return
    load(configs)
    try: yield
    finally: unload()


@pytest.mark.parametrize("native_resume", [False, True])
@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
def test_train_save_resume_and_generation_keep_all_added_weights(case, tmp_path, native_resume, algorithm):
    model, spec, restore, forward = case
    from core.adapters.capability import trainable_pairs
    if (algorithm, False) not in trainable_pairs(spec["architecture"]):
        pytest.skip("Architecture capability does not enable this algebra")
    base = deepcopy(model)
    owner = owner_for(model, spec, algorithm)
    LoRATrainer._apply_lora(owner)
    groups = owner.adapter.setup_trainable_parameters(owner.lora_layers)
    params = [parameter for group in groups for parameter in group["params"]]
    assert len(params) == len(set(map(id, params)))
    structural = list(model.rewiring_modules.parameters())
    assert set(map(id, structural)) <= set(map(id, params))
    frozen = {name: value.clone() for name, value in model.named_parameters() if not value.requires_grad}
    before = {name: value.clone() for name, value in model.rewiring_modules.named_parameters()}
    optimizer = torch.optim.AdamW(groups)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        forward(model).square().mean().backward()
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in structural)
        optimizer.step()
    assert any(not torch.equal(value, dict(model.rewiring_modules.named_parameters())[name]) for name, value in before.items())
    assert all(torch.equal(value, dict(model.named_parameters())[name]) for name, value in frozen.items())
    names = BaseTrainer._build_ema_param_name_map(owner)
    assert len({names[id(value)] for value in params}) == len(params)
    assert all("rewiring_modules" in names[id(value)] for value in structural)
    output = tmp_path / "hybrid_step_000002.safetensors"
    owner.adapter.save_checkpoint(owner.lora_layers, 2, 1, output)
    with safe_open(str(output), framework="pt", device="cpu") as handle:
        assert handle.metadata()["sushi.rewiring.full_module"] == "1"
        assert "rewiring_modules.alpha" not in handle.keys()
    resumed_model = deepcopy(base)
    if native_resume: restore(resumed_model, None)
    resumed = owner_for(resumed_model, spec, algorithm)
    prepare_hybrid_resume(resumed, output)
    LoRATrainer._apply_lora(resumed)
    identities = list(map(id, resumed_model.rewiring_modules.parameters()))
    assert LoRATrainer.load_checkpoint(resumed, str(output)) == 2
    assert identities == list(map(id, resumed_model.rewiring_modules.parameters()))
    restored_names = BaseTrainer._build_ema_param_name_map(resumed)
    assert set(names.values()) == set(restored_names.values())
    with torch.no_grad(): torch.testing.assert_close(forward(resumed_model), forward(model), rtol=0, atol=0)
    inference = deepcopy(base)
    original = inference.rewiring_modules
    with generation(inference, spec, output):
        with torch.no_grad(): torch.testing.assert_close(forward(inference), forward(model), rtol=1e-5, atol=1e-6)
    assert inference.rewiring_modules is original
    tensors = load_file(str(output))
    with safe_open(str(output), framework="pt", device="cpu") as handle: metadata = handle.metadata()
    tensors[next(key for key in tensors if key.startswith("rewiring_modules."))].fill_(float("nan"))
    broken = tmp_path / "broken.safetensors"
    save_file(tensors, str(broken), metadata=metadata)
    previous = deepcopy(resumed_model.state_dict())
    with pytest.raises(ValueError): LoRATrainer.load_checkpoint(resumed, str(broken))
    assert all(torch.equal(value, resumed_model.state_dict()[key]) for key, value in previous.items())


def test_api_to_lora_yaml_preserves_hybrid_declaration(case, monkeypatch):
    import yaml
    from api.routes import TrainingRunCreateRequest
    from core.training.training_config import TrainingConfigGenerator
    from core.training.train_runner import _prepare_training_process_config
    _, spec, _, _ = case
    from core.model_loader import ModelLoader
    monkeypatch.setattr(ModelLoader, "detect_model_type", staticmethod(lambda _: spec["architecture"]))
    request = TrainingRunCreateRequest(training_method="lora", base_model_path="native",
        total_steps=2, dit_rewiring_mode="configure", dit_rewiring_spec=spec,
        train_unet=True, train_text_encoder=False, ming_conditioner_groups=[])
    payload = request.model_dump()
    payload["_explicit_fields"] = sorted(request.model_fields_set)
    document = TrainingConfigGenerator.generate_lora_config(payload, run_name="probe",
        base_model_path="native", output_dir="probe")
    _, config, _, method = _prepare_training_process_config(yaml.safe_load(document), "native")
    assert method == "lora"
    assert config["dit_rewiring_mode"] == "configure"
    assert config["dit_rewiring_spec"] == spec
