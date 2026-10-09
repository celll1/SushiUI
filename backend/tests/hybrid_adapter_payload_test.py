"""Structural adapter replacements share the session's component lifetime."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from core.adapters import AdapterComponent, AdapterSession
from core.rewiring.adapters import FULL_MODULE_METADATA, HybridPayloadTransaction
from core.rewiring.spec import checkpoint_metadata
from hybrid_anima_test import native


@pytest.fixture(params=["anima", "qwen", "ming"])
def case(request, native):
    if request.param == "anima":
        from core.models.anima.residual_unet import restore_topology
        from hybrid_anima_test import forward
        model, spec = native
    elif request.param == "qwen":
        from core.models.qwen_image_21.residual_unet import restore_topology
        from hybrid_qwen_test import model_and_spec
        from qwen_partition_full_checkpoint_test import _forward
        model, spec, _ = model_and_spec()
        restore_topology(model, spec)
        forward = lambda module: _forward(module, torch.ones(1, 8, 12, 8))
    else:
        from core.models.ming_image_01.residual_unet import restore_topology
        from core.models.ming_image_01.model_definition import create_transformer
        from hybrid_ming_test import config, hybrid_for
        model = create_transformer(config("zero_masked"), device="cpu", dtype=torch.float32)
        spec = hybrid_for(model)
        restore_topology(model, spec)
        forward = lambda module: module([torch.ones(16, 1, 16, 24)], torch.tensor([.5]),
            [torch.ones(5, 8)], cap_feats_2=[torch.ones(3, 24)]).sample[0]
    with torch.no_grad():
        for parameter in model.rewiring_modules.parameters(): parameter.fill_(.01)
    yield model, spec, restore_topology, forward
    assert not torch.cuda.is_initialized()


def file_for(model, spec):
    from core.rewiring.provenance import model_metadata
    return SimpleNamespace(metadata=dict(checkpoint_metadata(spec), **model_metadata(model),
        **{FULL_MODULE_METADATA: "1"}),
        tensors={"rewiring_modules." + key: value.detach().clone()
                 for key, value in model.rewiring_modules.state_dict().items()},
        strength=1, apply_to_unet=True, step_range=None)


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("ending", ["unload", "release", "reload"])
def test_payload_matches_fitted_forward_and_restores_original(case, inherited, ending):
    model, spec, restore, forward = case
    from hybrid_provenance_test import record
    original_record = dict(record(), origin_spec=spec)
    model._hybrid_initialization = original_record
    reference = deepcopy(model)
    reference._hybrid_initialization["seed"] = 719
    file = file_for(reference, spec)
    if not inherited: restore(model, None)
    original = getattr(model, "rewiring_modules", None)
    native_weights = {key: value.clone() for key, value in model.state_dict().items()
                      if not key.startswith("rewiring_modules.")}
    with torch.no_grad(): before = forward(model)
    session = AdapterSession(resolve_path=lambda value: value)
    component = AdapterComponent(name="transformer", module=model,
        iter_targets=lambda _: (), build_branch=lambda _: None)
    session.bind(component)
    transaction = HybridPayloadTransaction(model, [file], spec, restore)
    assert getattr(model, "rewiring_modules", None) is original
    transaction.install()
    assert model._hybrid_initialization["seed"] == 719
    session.own_cleanup(component, transaction.restore)
    with torch.no_grad(): torch.testing.assert_close(forward(model), forward(reference), rtol=1e-5, atol=1e-6)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in native_weights.items())
    if ending == "release": session.release()
    elif ending == "reload":
        replacement = deepcopy(reference)
        session.bind(AdapterComponent(name="transformer", module=replacement,
            iter_targets=lambda _: (), build_branch=lambda _: None))
        assert replacement.rewiring_modules is not original
    else: session.unload([component])
    assert getattr(model, "rewiring_modules", None) is original
    assert getattr(model, "_hybrid_initialization", None) == (original_record if inherited else None)
    with torch.no_grad(): torch.testing.assert_close(forward(model), before, rtol=0, atol=0)
    transaction.restore()


@pytest.mark.parametrize("invalid", ["missing", "nan", "strength", "range", "two_owners", "undeclared"])
def test_payload_refusal_does_not_change_base(case, invalid):
    model, spec, restore, _ = case
    file = file_for(model, spec)
    original = model.rewiring_modules
    before = deepcopy(model.state_dict())
    files = [file]
    if invalid == "missing": file.tensors.pop(next(iter(file.tensors)))
    elif invalid == "nan": file.tensors[next(iter(file.tensors))].fill_(float("nan"))
    elif invalid == "strength": file.strength = .5
    elif invalid == "range": file.step_range = [0, 500]
    elif invalid == "two_owners": files.append(deepcopy(file))
    else: file.metadata.pop(FULL_MODULE_METADATA)
    with pytest.raises(ValueError): HybridPayloadTransaction(model, files, spec, restore)
    assert model.rewiring_modules is original
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_real_generation_loader_applies_and_unloads_structural_payload(case, tmp_path):
    from safetensors.torch import save_file
    model, spec, restore, forward = case
    reference = deepcopy(model)
    file = file_for(reference, spec)
    architecture = spec["architecture"]
    if architecture == "anima":
        from core.pipeline_backends.anima import AnimaMixin
        class Owner(AnimaMixin):
            _anima_resolve_lora_path = staticmethod(str)
        owner = Owner()
        owner.anima_components = {"transformer": model}
        stem = "lora_unet_blocks_0_self_attn_q_proj"
        target = model.blocks[0].self_attn.q_proj
        load, unload = owner._load_lora_anima, owner._unload_lora_anima
    elif architecture == "qwen_image_21":
        from core.pipeline_backends.qwen_image_21 import QwenImage21Mixin
        class Owner(QwenImage21Mixin):
            _qwen21_resolve_lora_path = staticmethod(str)
        owner = Owner()
        owner.qwen_image_21_components = {"transformer": model}
        stem = "lora_unet_transformer_blocks__0__attn__to_q"
        target = model.transformer_blocks[0].attn.to_q
        load, unload = owner._load_lora_qwen21, owner._unload_lora_qwen21
    else:
        from core.models.ming_image_01.lora import iter_lora_slots, flatten_to_key
        parent, attr, path = next(iter_lora_slots(model))
        target = parent[attr] if isinstance(attr, int) else getattr(parent, attr)
        stem = flatten_to_key(path)
        file.metadata["ming_image_01_profile"] = "generation_edit"
    file.tensors.update({stem + ".lora_down.weight": torch.ones(2, target.in_features),
                         stem + ".lora_up.weight": torch.zeros(target.out_features, 2)})
    file.metadata["model_type"] = architecture
    output = tmp_path / "hybrid.safetensors"
    save_file(file.tensors, str(output), metadata=file.metadata)
    restore(model, None)
    configs = [{"path": str(output), "strength": 1}]
    if architecture == "ming_image_01":
        from core.models.ming_image_01.generation_adapters import generation_adapters
        components = SimpleNamespace(transformer=model, artifact=SimpleNamespace(profile="generation_edit"))
        with generation_adapters(components, configs, resolve_path=str):
            assert model._residual_unet_spec == spec
            with torch.no_grad(): torch.testing.assert_close(forward(model), forward(reference), rtol=1e-5, atol=1e-6)
    else:
        assert load(configs) == 1
        assert model._residual_unet_spec == spec
        with torch.no_grad(): torch.testing.assert_close(forward(model), forward(reference), rtol=1e-5, atol=1e-6)
        unload()
    assert not hasattr(model, "rewiring_modules")
    assert not hasattr(model, "_residual_unet_spec")
