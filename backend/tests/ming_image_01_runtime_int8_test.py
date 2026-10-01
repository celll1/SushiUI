"""CPU real runtime converter, native generation and adapter ordering."""
import pytest
import torch
from PIL import Image
from core.models.common.int8_runtime_quantize import already_weight_only_quantized, RUNTIME_INT8_ARCHS
from ming_image_01_pipeline_test import manager, params, cpu_only
from ming_image_01_generation_lora_test import write_adapter


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_runtime_conversion_native_generation_persistence_and_census(tmp_path, monkeypatch, profile):
    value, c = manager(profile)
    if profile == "layer_decompose":
        with torch.no_grad():
            c.transformer.x_pad_token.fill_(.02)
            c.transformer.cap_pad_token.fill_(.03)
    protected = {name: id(module) for name, module in c.transformer.named_modules()
                 if isinstance(module, torch.nn.Linear) and (name.startswith(("all_x_embedder.",
                     "all_final_layer.", "t_embedder.", "cap_embedder.")) or ".adaLN_modulation." in name)}
    frozen = {name: parameter.detach().clone() for name, parameter in c.conditioner.named_parameters()}
    warnings = []
    monkeypatch.setattr("core.vram_optimization._add_generation_warning", lambda message, code: warnings.append(code))
    images = [Image.new("RGBA", (32, 32), (80, 60, 40, 200))] if profile == "layer_decompose" else []
    result, _, _ = value._ming_image_run(params(unet_quantization="int8", ming_image_01_layers=2), images=images)
    count = already_weight_only_quantized(c.transformer)
    assert count > 0 and value._runtime_int8_converted
    assert "ming_image_01" in RUNTIME_INT8_ARCHS
    assert all(id(c.transformer.get_submodule(name)) == identity for name, identity in protected.items())
    assert all(torch.equal(parameter, dict(c.conditioner.named_parameters())[name]) for name, parameter in frozen.items())
    assert result.groups[0].values[0].mode == "RGBA"
    assert value._runtime_int8_audit["converted"]["int8"] + value._runtime_int8_audit["converted"]["e4m3"] == count
    assert value.current_model_info["parameter_accounting"].endswith("excludes_buffers_and_quantization_payloads")
    value._ming_image_run(params(unet_quantization=None, ming_image_01_layers=2), images=images)
    assert already_weight_only_quantized(c.transformer) == count
    assert "runtime_quantization_persistent" in warnings
    value._ming_image_run(params(unet_quantization="none", ming_image_01_layers=2), images=images)
    assert already_weight_only_quantized(c.transformer) == count
    cpu_only(c)


def test_int8_before_lora_keeps_quantized_base_after_request(tmp_path, monkeypatch):
    value, c = manager("generation_edit")
    path = tmp_path/"native.safetensors"
    write_adapter(path, c.transformer)
    from core.extensions.lora_manager import lora_manager
    monkeypatch.setattr(lora_manager, "_resolve_lora_path", str)
    seen = []
    def callback(index, timestep, latents):
        layer = c.transformer.layers[0].attention.to_q
        seen.append(hasattr(layer, "original_module") or hasattr(layer, "base"))
    result, _, _ = value._generate_txt2img_ming_image_01(params(unet_quantization="int8",
        loras=[{"path": str(path)}]), step_callback=callback)
    assert seen == [True]
    assert already_weight_only_quantized(c.transformer) > 0
    assert result.groups[0].values[0].mode == "RGBA"
    assert c.transformer.layers[0].attention.to_q.__class__.__name__ in {"Int8Linear", "Fp8Linear"}
    cpu_only(c)


def test_other_formats_refuse_before_any_weight_conversion():
    value, c = manager("generation_edit")
    with pytest.raises(Exception, match="INT8 only"):
        value._generate_txt2img_ming_image_01(params(unet_quantization="fp8_e4m3fn"))
    assert already_weight_only_quantized(c.transformer) == 0
    cpu_only(c)


@pytest.mark.parametrize("profile,padding", [("generation_edit", "zero_masked"), ("layer_decompose", "learned")])
def test_nested_unet_runtime_int8_generation_preserves_topology(profile, padding):
    from core.models.ming_image_01.model_definition import create_transformer
    from core.models.ming_image_01.residual_unet import restore_topology
    from ming_image_01_rewiring_test import config, spec_for
    value, c = manager(profile)
    c.transformer = create_transformer(config(padding), device="cpu", dtype=torch.float32).eval()
    c.transformer.requires_grad_(False)
    if padding == "learned":
        with torch.no_grad():
            c.transformer.x_pad_token.fill_(.02)
            c.transformer.cap_pad_token.fill_(.03)
    topology = spec_for(c.transformer)
    restore_topology(c.transformer, topology)
    value.ming_image_01_components["transformer"] = c.transformer
    request = params(width=64, height=64, unet_quantization="int8", ming_image_01_layers=2)
    images = [Image.new("RGBA", (64, 64), (50, 90, 100, 220))] if profile == "layer_decompose" else []
    result, _, _ = value._ming_image_run(request, images=images)
    assert c.transformer._residual_unet_spec == topology
    assert request["dit_rewiring_actual"]["spec"] == topology
    assert already_weight_only_quantized(c.transformer) > 0
    assert result.groups[0].values[0].mode == "RGBA"
    cpu_only(c)
