"""Native ConvRot marker, CPU eager execution and package restoration."""
import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from ming_image_01_conditioning_test import write_tiny_package
from core.models.ming_image_01.loader import load_components
from core.models.ming_image_01.checkpoint import save_bundle
from core.models.ming_image_01.quantization import prepare_quantized_transformer
from core.models.common.quantized_checkpoint_guard import encode_comfy_quant_marker
from core.models.common.convrot_int8_linear import ConvRotInt8Linear


def marker():
    return encode_comfy_quant_marker({"format":"int8_tensorwise", "convrot":True,"convrot_groupsize":256})


def make_package(tmp_path, profile):
    from core.models.ming_image_01.model_definition import create_transformer
    from core.models.ming_image_01.residual_unet import restore_topology
    from ming_image_01_rewiring_test import spec_for
    from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight
    source = tmp_path/"source"
    source.mkdir()
    write_tiny_package(source, profile=profile)
    c = load_components(source,dtype=torch.float32,conditioner_attention="eager")
    config = dict(c.transformer.config)
    config.update(dim=256,n_heads=8,n_kv_heads=8,axes_dims=[8,12,12])
    c.transformer = create_transformer(config,device="cpu",dtype=torch.float32).eval()
    if profile == "layer_decompose":
        with torch.no_grad():
            c.transformer.x_pad_token.fill_(.02)
            c.transformer.cap_pad_token.fill_(.03)
    c.artifact.configs["mlp"]["diffusion_inner_dim"] = 256
    c.conditioner.native.diffusion_inner_dim = 256
    c.conditioner.native.proj_directvlm[1] = torch.nn.Linear(48,256)
    c.conditioner.set_trainable(set())
    restore_topology(c.transformer,spec_for(c.transformer))
    old = c.transformer.layers[0].attention.to_q
    replacement = ConvRotInt8Linear(256,256,False,torch.float32,
        convrot_groupsize=256,marker_numel=marker().numel(),device="cpu")
    q,scale = quantize_int8_convrot_weight(old.weight.detach(),256)
    replacement.weight=q.contiguous()
    replacement.weight_scale=scale.reshape(-1).float().contiguous()
    replacement.comfy_quant=marker()
    c.transformer.layers[0].attention.to_q=replacement
    c.transformer.requires_grad_(False)
    output = tmp_path/"convrot"
    manifest=save_bundle(c,output)
    return c,manifest


@pytest.mark.parametrize("profile", ["generation_edit","layer_decompose"])
def test_real_convrot_native_bundle_cpu_generation_and_gradient(tmp_path, profile):
    c,manifest=make_package(tmp_path,profile)
    loaded=load_components(manifest,dtype=torch.float32,conditioner_attention="eager")
    layer=loaded.transformer.layers[0].attention.to_q
    assert isinstance(layer,ConvRotInt8Linear)
    assert loaded.transformer._residual_unet_spec == c.transformer._residual_unet_spec
    for key,value in c.transformer.state_dict().items():
        assert torch.equal(value.view(torch.uint8), loaded.transformer.state_dict()[key].view(torch.uint8)),key
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
    x=torch.randn(2,256,requires_grad=True)
    actual=layer(x)
    expected=F.linear(x,dequantize_int8_convrot_weight(layer.weight,layer.weight_scale.reshape(-1,1),256))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is None for parameter in loaded.transformer.parameters())
    from ming_image_01_pipeline_test import manager,params,cpu_only
    from PIL import Image
    value,_=manager(profile)
    value._install_ming_image_01({"ming_native":loaded,"transformer":loaded.transformer,
        "text_encoder":loaded.conditioner,"vae":loaded.vae},model_id="convrot",source_type="local",source=str(manifest))
    images=[Image.new("RGBA",(64,64),(80,60,40,200))] if profile=="layer_decompose" else []
    result,_,_=value._ming_image_run(params(width=64,height=64,ming_image_01_layers=2),images=images)
    assert result.groups[0].values[0].mode=="RGBA"
    cpu_only(loaded)
    from api.generation_utils import extract_fp8_gemm_info
    assert "convrot_int8(comfy-kitchen)" in extract_fp8_gemm_info(value)
    from api.quantized_export_job import _export_ming_package, export_status
    assert export_status(value)["exportable"]
    exported=_export_ming_package(value,str(tmp_path/"reexport"),None,"loaded",lambda *a:None)
    restored=load_components(exported["output_path"],dtype=torch.float32,conditioner_attention="eager")
    assert isinstance(restored.transformer.layers[0].attention.to_q,ConvRotInt8Linear)
    assert restored.transformer._residual_unet_spec == loaded.transformer._residual_unet_spec
    assert torch.equal(restored.transformer.layers[0].attention.to_q.comfy_quant,marker())


@pytest.mark.parametrize("damage", ["unknown","missing_scale","bad_groupsize","bad_width","wrong_weight","marker_dtype"])
def test_convrot_bad_contract_refused_before_conditioner_loading(tmp_path,monkeypatch,damage):
    write_tiny_package(tmp_path)
    path=tmp_path/"transformer/diffusion_pytorch_model.safetensors"
    state=load_file(str(path))
    name="layers.0.attention.to_q"
    state[name+".weight"]=state[name+".weight"].to(torch.int8)
    state[name+".weight_scale"]=torch.ones(24)
    state[name+".comfy_quant"]=marker()
    if damage=="unknown":
        state[name+".comfy_quant"]=encode_comfy_quant_marker({"format":"unknown"})
    elif damage=="bad_groupsize":
        state[name+".comfy_quant"]=encode_comfy_quant_marker({"format":"int8_tensorwise","convrot":True,"convrot_groupsize":128})
    elif damage=="missing_scale":
        del state[name+".weight_scale"]
    elif damage=="wrong_weight":
        state[name+".weight"]=state[name+".weight"].float()
    elif damage=="marker_dtype":
        state[name+".comfy_quant"]=state[name+".comfy_quant"].float()
    save_file(state,str(path))
    from core.models.ming_image_01.vendor.modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
    def forbidden(*a,**k): raise AssertionError("Malformed ConvRot reached conditioner loading")
    monkeypatch.setattr(BailingMM2NativeForConditionalGeneration,"from_pretrained",forbidden)
    with pytest.raises(ValueError,match="ConvRot"):
        load_components(tmp_path,dtype=torch.float32,conditioner_attention="eager")
    assert not torch.cuda.is_initialized()


def test_column_scales_normalize_only_declared_convrot_and_keep_topology(tmp_path):
    c,manifest=make_package(tmp_path,"generation_edit")
    path=manifest.parent/"transformer/diffusion_pytorch_model.safetensors"
    state=load_file(str(path))
    name="layers.0.attention.to_q"
    state[name+".weight_scale"]=state[name+".weight_scale"].reshape(-1,1)
    from safetensors import safe_open
    with safe_open(str(path),framework="pt",device="cpu") as handle:
        metadata=handle.metadata()
    save_file(state,str(path),metadata=metadata)
    loaded=load_components(manifest,dtype=torch.float32,conditioner_attention="eager")
    layer=loaded.transformer.layers[0].attention.to_q
    assert layer.weight_scale.shape==(256,)
    assert torch.equal(layer.weight_scale,c.transformer.layers[0].attention.to_q.weight_scale)
    assert loaded.transformer._residual_unet_spec==c.transformer._residual_unet_spec
    assert not torch.cuda.is_initialized()


def test_missing_runtime_refuses_before_conditioner_payloads(tmp_path,monkeypatch):
    _,manifest=make_package(tmp_path,"generation_edit")
    from core.models.common import convrot_int8_linear
    from core.models.ming_image_01.vendor.modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
    def unavailable(): raise RuntimeError("ConvRot runtime unavailable")
    def forbidden(*a,**k): raise AssertionError("Missing runtime reached conditioner loading")
    monkeypatch.setattr(convrot_int8_linear,"require_convrot_int8_runtime",unavailable)
    monkeypatch.setattr(BailingMM2NativeForConditionalGeneration,"from_pretrained",forbidden)
    with pytest.raises(RuntimeError,match="runtime unavailable"):
        load_components(manifest,dtype=torch.float32,conditioner_attention="eager")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("profile", ["generation_edit","layer_decompose"])
def test_convrot_lora_generation_keeps_packed_base_and_nested_unet(tmp_path,monkeypatch,profile):
    _,manifest=make_package(tmp_path,profile)
    c=load_components(manifest,dtype=torch.float32,conditioner_attention="eager")
    from ming_image_01_pipeline_test import manager,params,cpu_only
    from ming_image_01_generation_lora_test import write_adapter
    from core.extensions.lora_manager import lora_manager
    from PIL import Image
    value,_=manager(profile)
    value._install_ming_image_01({"ming_native":c,"transformer":c.transformer,
        "text_encoder":c.conditioner,"vae":c.vae},model_id="convrot",source_type="local",source=str(manifest))
    base=c.transformer.layers[0].attention.to_q
    codes=base.weight.clone()
    topology=c.transformer._residual_unet_spec
    path=tmp_path/"lora.safetensors"
    from core.models.ming_image_01.residual_unet import model_topology_metadata
    write_adapter(path,c.transformer,profile=profile,metadata=model_topology_metadata(c.transformer))
    monkeypatch.setattr(lora_manager,"_resolve_lora_path",str)
    events=[]
    def callback(index,timestep,latents):
        events.append((c.transformer.layers[0].attention.to_q is not base, bool(torch.isfinite(latents).all())))
    images=[Image.new("RGBA",(64,64),(80,60,40,200))] if profile=="layer_decompose" else []
    result,_,_=value._ming_image_run(params(width=64,height=64,ming_image_01_layers=2,
        loras=[{"path":str(path)}]),images=images,step_callback=callback)
    assert events==[(True,True)]
    assert result.groups[0].values[0].mode=="RGBA"
    assert c.transformer.layers[0].attention.to_q is base
    assert torch.equal(base.weight,codes) and torch.equal(base.comfy_quant,marker())
    assert c.transformer._residual_unet_spec==topology
    cpu_only(c)
