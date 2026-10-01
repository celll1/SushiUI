"""CPU common-session lifecycle and native generation adapter wiring."""
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
from core.models.ming_image_01.generation_adapters import generation_adapters
from core.models.ming_image_01.lora import flatten_to_key
from ming_image_01_pipeline_test import manager, params


def write_adapter(path, model, *, profile="generation_edit", metadata=None):
    stem = flatten_to_key("layers.0.attention.to_q")
    base = model.layers[0].attention.to_q
    down, up = torch.ones(2, base.in_features) * .03, torch.ones(base.out_features, 2) * .07
    save_file({stem + ".lora_down.weight": down, stem + ".lora_up.weight": up,
               stem + ".alpha": torch.tensor(2.)}, str(path),
              metadata={"model_type": "ming_image_01", "ming_image_01_profile": profile, **(metadata or {})})
    return down, up


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_session_layering_step_gate_and_exception_cleanup(tmp_path, profile):
    _, c = manager(profile)
    base = c.transformer.layers[0].attention.to_q
    path = tmp_path/"native.safetensors"
    down, up = write_adapter(path, c.transformer, profile=profile)
    configs = [{"path": str(path), "strength": .5}, {"path": str(path), "strength": .25}]
    data = torch.randn(2, base.in_features)
    with pytest.raises(RuntimeError, match="cancel"):
        with generation_adapters(c, configs, resolve_path=str) as session:
            session.set_step(0, 2)
            changed = c.transformer.layers[0].attention.to_q
            expected = base(data) + .75 * torch.nn.functional.linear(torch.nn.functional.linear(data, down), up)
            torch.testing.assert_close(changed(data), expected)
            raise RuntimeError("cancel")
    assert c.transformer.layers[0].attention.to_q is base
    with generation_adapters(c, [{"path": str(path), "step_range": [500, 1000]}], resolve_path=str) as session:
        session.set_step(0, 2)
        torch.testing.assert_close(c.transformer.layers[0].attention.to_q(data), base(data))
        session.set_step(1, 2)
        assert not torch.equal(c.transformer.layers[0].attention.to_q(data), base(data))
    assert c.transformer.layers[0].attention.to_q is base


def test_wrong_profile_refuses_without_wrapping(tmp_path):
    _, c = manager("generation_edit")
    base = c.transformer.layers[0].attention.to_q
    path = tmp_path/"layer.safetensors"
    write_adapter(path, c.transformer, profile="layer_decompose")
    with pytest.raises(Exception, match="profile"):
        with generation_adapters(c, [{"path": str(path)}], resolve_path=str):
            pytest.fail("wrong profile accepted")
    assert c.transformer.layers[0].attention.to_q is base


def test_real_generation_with_adapter_restores_slot_and_reuses_model(tmp_path, monkeypatch):
    value, c = manager("generation_edit")
    base = c.transformer.layers[0].attention.to_q
    path = tmp_path/"native.safetensors"
    write_adapter(path, c.transformer)
    from core.extensions.lora_manager import lora_manager
    monkeypatch.setattr(lora_manager, "_resolve_lora_path", str)
    events = []
    def callback(index, timestep, latents):
        events.append(c.transformer.layers[0].attention.to_q is not base)
    value._generate_txt2img_ming_image_01(params(loras=[{"path": str(path)}]), step_callback=callback)
    assert events == [True]
    assert c.transformer.layers[0].attention.to_q is base
    result, _, _ = value._generate_txt2img_ming_image_01(params())
    assert result.groups[0].values[0].mode == "RGBA"
    assert not torch.cuda.is_initialized()


def test_adapter_topology_restored_after_context_and_conflicts_refused(tmp_path):
    from core.models.ming_image_01.model_definition import create_transformer
    from core.models.ming_image_01.residual_unet import restore_topology, model_topology_metadata
    from ming_image_01_native_test import tiny_config
    from ming_image_01_rewiring_test import spec_for
    config = tiny_config("zero_masked")
    config["n_layers"] = 6
    model = create_transformer(config, device="cpu", dtype=torch.float32)
    c = SimpleNamespace(transformer=model, artifact=SimpleNamespace(profile="generation_edit"))
    topology = spec_for(model)
    restore_topology(model, topology)
    metadata = model_topology_metadata(model)
    restore_topology(model, None)
    path = tmp_path/"unet.safetensors"
    write_adapter(path, model, metadata=metadata)
    base = model.layers[0].attention.to_q
    with generation_adapters(c, [{"path": str(path)}], resolve_path=str):
        assert model._residual_unet_spec == topology
    assert getattr(model, "_residual_unet_spec", None) is None
    assert model.layers[0].attention.to_q is base
    dense = tmp_path/"dense.safetensors"
    write_adapter(dense, model)
    with pytest.raises(ValueError, match="different residual topologies"):
        with generation_adapters(c, [{"path": str(path)}, {"path": str(dense)}], resolve_path=str):
            pytest.fail("mixed topologies accepted")
    assert getattr(model, "_residual_unet_spec", None) is None
    assert model.layers[0].attention.to_q is base


def test_foreign_latent_refused_before_install(tmp_path):
    from core.adapters.base_identity import BaseLatentIdentity
    _, c = manager("generation_edit")
    base = c.transformer.layers[0].attention.to_q
    path = tmp_path/"foreign.safetensors"
    write_adapter(path, c.transformer, metadata=BaseLatentIdentity(latent_channels=4).to_metadata())
    with pytest.raises(Exception, match="latent"):
        with generation_adapters(c, [{"path": str(path)}], resolve_path=str):
            pytest.fail("foreign latent accepted")
    assert c.transformer.layers[0].attention.to_q is base


def test_generation_and_training_capability_axes_remain_separate():
    from api.arch_capabilities import adapter_families_payload
    entry = adapter_families_payload()["ming_image_01"]
    assert entry["supported"] == ["lora"]
    assert entry["trainable"] == []
