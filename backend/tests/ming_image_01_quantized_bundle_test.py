"""Native CPU quantized package preservation, strict preflight and generation."""
import pytest
import torch
from safetensors.torch import load_file, save_file
from ming_image_01_conditioning_test import write_tiny_package
from core.models.ming_image_01.loader import load_components
from core.models.ming_image_01.checkpoint import save_bundle
from core.models.common.int8_runtime_quantize import quantize_linears_in_place, already_weight_only_quantized


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_quantized_bundle_shards_exact_state_and_native_generation(tmp_path, profile):
    source = tmp_path / "source"
    source.mkdir()
    write_tiny_package(source, profile=profile)
    c = load_components(source, dtype=torch.float32, conditioner_attention="eager")
    quantize_linears_in_place(c.transformer, arch="ming_image_01", compute_dtype=torch.float32)
    count = already_weight_only_quantized(c.transformer)
    assert count > 0
    from ming_image_01_rewiring_test import spec_for
    from core.models.ming_image_01.residual_unet import restore_topology
    topology = spec_for(c.transformer)
    restore_topology(c.transformer, topology)
    exported = save_bundle(c, tmp_path / "export", max_shard_bytes=4096)
    assert (tmp_path / "export/transformer/diffusion_pytorch_model.safetensors.index.json").exists()
    loaded = load_components(exported, dtype=torch.float32, conditioner_attention="eager")
    assert already_weight_only_quantized(loaded.transformer) == count
    assert loaded.transformer._residual_unet_spec == topology
    old, new = c.transformer.state_dict(), loaded.transformer.state_dict()
    assert old.keys() == new.keys()
    for key in old:
        assert old[key].dtype == new[key].dtype
        assert torch.equal(old[key].view(torch.uint8), new[key].view(torch.uint8)), key
    from ming_image_01_pipeline_test import manager, params, cpu_only
    from PIL import Image
    value, _ = manager(profile)
    value._install_ming_image_01({"ming_native": loaded, "transformer": loaded.transformer,
        "vae": loaded.vae, "text_encoder": loaded.conditioner},
        model_id="quantized", source_type="local", source=str(exported))
    images = [Image.new("RGBA", (64,64), (80,60,40,200))] if profile == "layer_decompose" else []
    result, _, _ = value._ming_image_run(params(width=64, height=64, ming_image_01_layers=2), images=images)
    assert result.groups[0].values[0].mode == "RGBA"
    cpu_only(loaded)


@pytest.mark.parametrize("damage", ["missing_scale", "wrong_scale", "unknown_marker", "orphan_scale"])
def test_malformed_quantization_refused_before_conditioner_loading(tmp_path, monkeypatch, damage):
    write_tiny_package(tmp_path)
    weight_file = tmp_path / "transformer/diffusion_pytorch_model.safetensors"
    state = load_file(str(weight_file))
    path = "layers.0.attention.to_q"
    state[path + ".weight"] = state[path + ".weight"].to(torch.int8)
    state[path + ".weight_scale"] = torch.ones(state[path + ".weight"].shape[0])
    if damage == "missing_scale":
        del state[path + ".weight_scale"]
    elif damage == "wrong_scale":
        state[path + ".weight_scale"] = torch.ones(1)
    elif damage == "unknown_marker":
        state[path + ".comfy_quant"] = torch.ones(1, dtype=torch.uint8)
    else:
        state["layers.0.attention.to_k.weight_scale"] = torch.ones(24)
    save_file(state, str(weight_file))
    from core.models.ming_image_01.vendor.modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
    def forbidden(*args, **kwargs):
        raise AssertionError("Malformed package reached conditioner loading")
    monkeypatch.setattr(BailingMM2NativeForConditionalGeneration, "from_pretrained", forbidden)
    with pytest.raises(ValueError, match="Ming"):
        load_components(tmp_path, dtype=torch.float32, conditioner_attention="eager")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixed_int8_fp8_codes_and_scales_survive_compute_dtype_selection(tmp_path, dtype):
    write_tiny_package(tmp_path)
    path = tmp_path / "transformer/diffusion_pytorch_model.safetensors"
    state = load_file(str(path))
    for leaf, format_ in (("to_q", torch.int8), ("to_k", torch.float8_e4m3fn)):
        key = "layers.0.attention." + leaf
        state[key + ".weight"] = state[key + ".weight"].mul(64).to(format_)
        state[key + ".weight_scale"] = torch.full((24,), 1/64.)
    save_file(state, str(path))
    c = load_components(tmp_path, dtype=dtype, conditioner_attention="eager")
    assert c.transformer.layers[0].attention.to_q.__class__.__name__ == "Int8Linear"
    assert c.transformer.layers[0].attention.to_k.__class__.__name__ == "Fp8Linear"
    actual = c.transformer.state_dict()
    for key, value in state.items():
        if key.endswith(".weight_scale") or value.dtype in (torch.int8, torch.float8_e4m3fn):
            assert actual[key].dtype == value.dtype
            assert torch.equal(actual[key].view(torch.uint8), value.view(torch.uint8)), key
        else:
            assert actual[key].dtype == dtype
    x = torch.randn(2,24, dtype=dtype)
    assert torch.isfinite(c.transformer.layers[0].attention.to_q(x)).all()
    assert torch.isfinite(c.transformer.layers[0].attention.to_k(x)).all()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("scale", [0., float("nan")])
def test_invalid_scale_payload_refused(tmp_path, scale):
    write_tiny_package(tmp_path)
    path = tmp_path / "transformer/diffusion_pytorch_model.safetensors"
    state = load_file(str(path))
    name = "layers.0.attention.to_q"
    state[name + ".weight"] = state[name + ".weight"].to(torch.int8)
    state[name + ".weight_scale"] = torch.full((24,), scale)
    save_file(state, str(path))
    from core.models.ming_image_01.artifact import inspect_artifact
    from core.models.ming_image_01.quantization import prepare_quantized_transformer, load_quantized_transformer
    artifact = inspect_artifact(tmp_path)
    model = prepare_quantized_transformer(artifact, torch.float32)
    with pytest.raises(ValueError, match="finite and positive"):
        load_quantized_transformer(model, artifact, torch.float32)
    assert not torch.cuda.is_initialized()
