"""Public quantized native LoRA training, frozen payloads and checkpoint reuse."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from ming_image_01_conversion_test import make_source
from core.models.ming_image_01.conversion import convert_package
from ming_image_01_lora_trainer_test import make_owner


def same_bits(a,b):
    return a.dtype==b.dtype and torch.equal(a.view(torch.uint8),b.view(torch.uint8))


@pytest.mark.parametrize("profile",["generation_edit","layer_decompose"])
@pytest.mark.parametrize("format_",["int8","int8_convrot"])
def test_public_quantized_native_lora_updates_only_branches_and_restores(tmp_path,monkeypatch,profile,format_):
    source=tmp_path/"source"
    original,_=make_source(source,profile)
    if format_=="int8":
        from safetensors.torch import save_file
        from core.models.ming_image_01.residual_unet import model_topology_metadata
        with torch.no_grad(): original.layers[0].attention.to_q.weight.zero_()
        save_file(original.state_dict(),str(source/"transformer/diffusion_pytorch_model.safetensors"),
                  metadata=model_topology_metadata(original))
    converted=convert_package(source,tmp_path/"quantized",format_=format_)
    root=Path(converted["manifest"]).parent
    from api.routes import TrainingRunCreateRequest,_check_ming_conditioner_params,_check_adapter_params
    from core.training.training_config import TrainingConfigGenerator
    from core.training.train_runner import _prepare_training_process_config
    from core.training.lora_trainer import LoRATrainer
    import yaml
    request=TrainingRunCreateRequest(training_method="lora",base_model_path=str(root),total_steps=1,
        train_unet=True,train_text_encoder=False,ming_conditioner_groups=[],lora_rank=2,lora_alpha=2,
        base_resolutions=[64],text_encoding_mode="swap_onthefly",latent_encoding_mode="swap_onthefly")
    _check_ming_conditioner_params(request)
    _check_adapter_params(request)
    payload=request.model_dump()
    payload["_explicit_fields"]=sorted(request.model_fields_set)
    serialized=TrainingConfigGenerator.generate_lora_config(payload,run_name="run",base_model_path=str(root),output_dir=str(tmp_path/"run"))
    process,train,network,method=_prepare_training_process_config(yaml.safe_load(serialized),str(root))
    assert method=="lora"
    from core.models.ideogram4.vendor import int8_linear,fp8_linear
    monkeypatch.setattr(int8_linear,"_INT8_MM_ENABLED",True)
    monkeypatch.setattr(fp8_linear,"_SCALED_MM_ENABLED",True)
    owner=LoRATrainer(model_path=str(root),output_dir=process["training_folder"],lora_rank=2,lora_alpha=2,
        device="cpu",weight_dtype="fp32",training_dtype="fp32",vae_dtype="fp32",mixed_precision=False,
        attention_backend="native",train_unet=True,train_text_encoder=False,train_config=train)
    assert owner.ming_quantized_base_training["execution"]=="dequant_autograd"
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    for module in owner.transformer.modules():
        if isinstance(module,int8_linear.Int8Linear): assert not module._allow_int8_mm
        if isinstance(module,fp8_linear.Fp8Linear): assert not module._allow_scaled_mm
        if isinstance(module,ConvRotInt8Linear): assert module._force_dequant
    if format_=="int8": assert owner.ming_quantized_base_training["formats"]["e4m3"]>0
    buffers={name:value.clone() for name,value in owner.transformer.named_buffers()}
    frozen={name:value.clone() for name,value in owner.transformer.named_parameters() if not value.requires_grad}
    branches={name:value.clone() for name,value in owner.transformer.named_parameters() if value.requires_grad}
    assert branches
    path=tmp_path/"target.png"
    Image.new("RGBA",(64,64),(120,80,30,200)).save(path)
    if profile=="layer_decompose":
        layer=tmp_path/"layer.png"
        Image.new("RGBA",(64,64),(20,60,120,100)).save(layer)
        path.with_name(path.name+".ming.json").write_text(json.dumps({"schema_version":1,"profile":profile,
            "targets":[{"path":path.name,"role":"auxiliary_canvas","weight":0},
                       {"path":layer.name,"role":"layer_1","weight":1}],"references":[path.name]}))
    dataset=SimpleNamespace(unique_id="quantized-ming",dataset_root=tmp_path,
        items=[{"image_path":str(path),"caption":"word5","width":64,"height":64}],reload_for_epoch=lambda **k:None)
    owner.train([dataset],num_epochs=1,total_steps=1,batch_size=1,save_every_n_steps=1,sample_every_n_steps=0,
        enable_bucketing=False,base_resolutions=[64],latent_encoding_mode="swap_onthefly",text_encoding_mode="swap_onthefly")
    current=dict(owner.transformer.named_parameters())
    assert any(not torch.equal(value,current[name]) for name,value in branches.items())
    assert all(torch.isfinite(value).all() for value in current.values())
    assert all(torch.equal(value,current[name]) for name,value in frozen.items())
    assert all(same_bits(value,dict(owner.transformer.named_buffers())[name]) for name,value in buffers.items())
    checkpoint=owner.output_dir/"run_step_000001.safetensors"
    with safe_open(str(checkpoint),framework="pt",device="cpu") as handle:
        recorded=json.loads(handle.metadata()["ming_quantized_base_training"])
    assert recorded==owner.ming_quantized_base_training
    resumed=make_owner(root,tmp_path/"resumed",resume=str(checkpoint))
    for name,branch in owner.lora_layers.items():
        for key,value in branch.branch_tensors().items():
            assert same_bits(value,resumed.lora_layers[name].branch_tensors()[key])
    from core.models.ming_image_01.loader import load_components
    from core.models.ming_image_01.generation_adapters import generation_adapters
    c=load_components(root,dtype=torch.float32,conditioner_attention="eager")
    base=c.transformer.layers[0].attention.to_q
    topology=c.transformer._residual_unet_spec
    with generation_adapters(c,[{"path":str(checkpoint)}],resolve_path=str):
        assert c.transformer.layers[0].attention.to_q is not base
    assert c.transformer.layers[0].attention.to_q is base
    assert c.transformer._residual_unet_spec==topology
    assert not torch.cuda.is_initialized()
