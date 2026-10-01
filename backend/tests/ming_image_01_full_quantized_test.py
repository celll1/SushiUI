"""Explicit native packed-base full training, floating exports and provenance."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from core.models.ming_image_01.quantization import materialize_training_linears, saved_full_initialization
from core.models.ming_image_01.conversion import convert_package
from ming_image_01_conversion_test import make_source
from core.models.ming_image_01.loader import load_components
from core.training.full_parameter_trainer import FullParameterTrainer


def package(tmp_path,profile,format_):
    source=tmp_path/'source';model,_=make_source(source,profile)
    if format_=='int8':
        from safetensors.torch import save_file
        from core.models.ming_image_01.residual_unet import model_topology_metadata
        with torch.no_grad():model.layers[0].attention.to_q.weight.zero_()
        save_file(model.state_dict(),str(source/'transformer/diffusion_pytorch_model.safetensors'),metadata=model_topology_metadata(model))
    value=convert_package(source,tmp_path/'quantized',format_=format_)
    return Path(value['manifest']).parent


@pytest.mark.parametrize("dtype",[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize("format_",['int8','int8_convrot'])
def test_materialized_weights_match_native_dequant_forward_and_receive_gradients(tmp_path,dtype,format_):
    root=package(tmp_path,'generation_edit',format_)
    c=load_components(root,dtype=dtype,conditioner_attention='eager')
    from core.models.ideogram4.vendor.int8_linear import Int8Linear
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    selected={name:m for name,m in c.transformer.named_modules() if isinstance(m,(Int8Linear,Fp8Linear,ConvRotInt8Linear))}
    assert selected
    expected={}
    inputs={}
    for name,child in selected.items():
        x=torch.randn(2,child.in_features,dtype=dtype,requires_grad=True)
        inputs[name]=x
        expected[name]=child._dequant_forward(x).detach()
    report=materialize_training_linears(c.transformer,dtype)
    assert sum(report['formats'].values())==len(selected)
    if format_=='int8':assert report['formats']['e4m3']>0
    assert report['lossy_source'] and report['weight_dtype']==str(dtype).removeprefix('torch.')
    assert not any(key.endswith(('.weight_scale','.comfy_quant')) for key in c.transformer.state_dict())
    for name,x in inputs.items():
        child=c.transformer.get_submodule(name)
        assert type(child) is torch.nn.Linear and child.weight.dtype==dtype and child.weight.requires_grad
        actual=child(x)
        torch.testing.assert_close(actual,expected[name],rtol=1e-3 if dtype!=torch.float32 else 1e-6,atol=1e-3 if dtype!=torch.float32 else 1e-6)
        actual.float().square().mean().backward()
        assert child.weight.grad is not None and torch.isfinite(child.weight.grad).all()
    assert materialize_training_linears(c.transformer,dtype) is None
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("profile",['generation_edit','layer_decompose'])
@pytest.mark.parametrize("format_",['int8','int8_convrot'])
def test_public_full_optin_trains_exports_floating_and_preserves_resume_origin(tmp_path,profile,format_):
    root=package(tmp_path,profile,format_)
    from api.routes import TrainingRunCreateRequest,_check_ming_conditioner_params,_check_qwen_full_finetune_params
    from core.training.training_config import TrainingConfigGenerator
    from core.training.train_runner import _prepare_training_process_config
    from core.training.adapters.base_adapter import reject_quantized_base
    import yaml
    request=TrainingRunCreateRequest(training_method='full_finetune',base_model_path=str(root),total_steps=2,
        full_finetune_dequantize_int8_base=True,train_unet=True,train_text_encoder=False,ming_conditioner_groups=[])
    _check_ming_conditioner_params(request);_check_qwen_full_finetune_params(request)
    payload=request.model_dump();payload['_explicit_fields']=sorted(request.model_fields_set)
    serialized=TrainingConfigGenerator.generate_full_finetune_config(payload,run_name='run',base_model_path=str(root),output_dir=str(tmp_path/'run'))
    process,train,_,method=_prepare_training_process_config(yaml.safe_load(serialized),str(root))
    assert method=='full_finetune' and train['full_finetune_dequantize_int8_base'] is True
    owner=FullParameterTrainer(model_path=str(root),output_dir=process['training_folder'],device='cpu',weight_dtype='fp32',training_dtype='fp32',vae_dtype='fp32',mixed_precision=False,attention_backend='native',train_unet=True,train_text_encoder=False,train_config=train)
    reject_quantized_base(owner.transformer,model_label='Ming test')
    groups=owner.setup_trainable_parameters()
    expected={id(p) for p in owner.transformer.parameters() if p.requires_grad}
    actual=[id(p) for g in groups for p in g['params']]
    assert expected==set(actual) and len(actual)==len(set(actual))
    before={name:p.clone() for name,p in owner.transformer.named_parameters()}
    image=tmp_path/'target.png';Image.new('RGBA',(64,64),(100,70,50,180)).save(image)
    if profile=='layer_decompose':
        layer=tmp_path/'layer.png';Image.new('RGBA',(64,64),(30,80,120,120)).save(layer)
        image.with_name(image.name+'.ming.json').write_text(json.dumps({'schema_version':1,'profile':profile,'targets':[{'path':image.name,'role':'auxiliary_canvas','weight':0},{'path':layer.name,'role':'layer_1','weight':1}],'references':[image.name]}))
    dataset=SimpleNamespace(unique_id='materialized-full',dataset_root=tmp_path,items=[{'image_path':str(image),'caption':'word5','width':64,'height':64}],reload_for_epoch=lambda **k:None)
    owner.train([dataset],num_epochs=2,total_steps=2,batch_size=1,save_every_n_steps=1,sample_every_n_steps=0,enable_bucketing=False,base_resolutions=[64],text_encoding_mode='swap_onthefly',latent_encoding_mode='swap_onthefly')
    assert any(not torch.equal(p,before[name]) for name,p in owner.transformer.named_parameters() if 'attention.to_' in name)
    assert all(torch.isfinite(p).all() for p in owner.transformer.parameters())
    checkpoint=owner.output_dir/'run_step_000002'
    metadata=json.loads((checkpoint/'training.json').read_text())
    assert metadata['ming_full_initialization']==owner.ming_full_initialization
    restored=load_components(checkpoint,dtype=torch.float32,conditioner_attention='eager')
    assert restored.transformer._residual_unet_spec==owner.transformer._residual_unet_spec
    assert all(torch.equal(t,restored.transformer.state_dict()[key]) for key,t in owner.transformer.state_dict().items())
    assert not any(key.endswith(('.weight_scale','.comfy_quant')) for key in restored.transformer.state_dict())
    assert saved_full_initialization(restored.artifact)==owner.ming_full_initialization
    fresh=FullParameterTrainer(model_path=str(checkpoint),output_dir=str(tmp_path/'fresh'),device='cpu',weight_dtype='fp32',training_dtype='fp32',vae_dtype='fp32',mixed_precision=False,attention_backend='native',train_text_encoder=False,train_config={'ming_conditioner_groups':[]})
    assert fresh.ming_full_initialization==owner.ming_full_initialization
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('format_',['int8','int8_convrot'])
def test_full_packed_base_without_explicit_optin_is_refused(tmp_path,format_):
    root=package(tmp_path,'generation_edit',format_)
    with pytest.raises(NotImplementedError,match='quantiz'):
        FullParameterTrainer(model_path=str(root),output_dir=str(tmp_path/'run'),device='cpu',weight_dtype='fp32',training_dtype='fp32',vae_dtype='fp32',mixed_precision=False,attention_backend='native',train_text_encoder=False,train_config={'ming_conditioner_groups':[]})


@pytest.mark.parametrize('method',['lora','relora','controlnet'])
def test_public_api_refuses_materialization_for_non_full_methods(tmp_path,method):
    root=package(tmp_path,'generation_edit','int8')
    from api.routes import TrainingRunCreateRequest,_check_qwen_full_finetune_params
    from fastapi import HTTPException
    request=TrainingRunCreateRequest(training_method=method,base_model_path=str(root),full_finetune_dequantize_int8_base=True)
    with pytest.raises(HTTPException):_check_qwen_full_finetune_params(request)


@pytest.mark.parametrize('value',['true',1,None])
def test_direct_ming_preflight_refuses_non_boolean_optin(value):
    from core.models.ming_image_01.training_options import preflight_training
    with pytest.raises(ValueError,match='boolean'):preflight_training({'full_finetune_dequantize_int8_base':value,'ming_conditioner_groups':[]},method='full_finetune')


@pytest.mark.parametrize('field,value',[
    ('schema_version',True),('execution','packed'),('weight_dtype','int8'),
    ('lossy_source',False),('formats',{'int8':True,'e4m3':0,'convrot_int8':0}),
    ('formats',{'int8':0,'e4m3':0,'convrot_int8':0}),
    ('formats',{'int8':-1,'e4m3':1,'convrot_int8':0}),
])
def test_saved_initialization_refuses_invalid_provenance(tmp_path,field,value):
    metadata={'schema_version':1,'execution':'materialized_floating',
        'formats':{'int8':1,'e4m3':0,'convrot_int8':0},'weight_dtype':'float32','lossy_source':True}
    metadata[field]=value
    (tmp_path/'training.json').write_text(json.dumps({'ming_full_initialization':metadata}))
    with pytest.raises(ValueError,match='initialization metadata'):
        saved_full_initialization(SimpleNamespace(root=tmp_path))


def test_full_materialization_refuses_non_cpu_payload_before_replacing_slots():
    from core.models.ideogram4.vendor.int8_linear import Int8Linear
    with torch.device('meta'):
        child=Int8Linear(4,4,False,torch.float32)
    model=torch.nn.Sequential(child)
    with pytest.raises(ValueError,match='CPU-resident'):
        materialize_training_linears(model,torch.float32)
    assert model[0] is child


@pytest.mark.parametrize('arch,method,export,allowed',[
    ('qwen_image_21','full_finetune','none',True),
    ('qwen_image_21','lora','none',False),
    ('sdxl','full_finetune','none',False),
    ('ming_image_01','full_finetune','int8_convrot',False),
])
def test_shared_optin_gate_preserves_other_architecture_boundaries(monkeypatch,arch,method,export,allowed):
    from api.routes import TrainingRunCreateRequest,_check_qwen_full_finetune_params
    from core.training import training_config
    from fastapi import HTTPException
    monkeypatch.setattr(training_config,'_detect_arch',lambda _:arch)
    request=TrainingRunCreateRequest(training_method=method,base_model_path='unused',
        full_finetune_dequantize_int8_base=True,training_export_format=export)
    if allowed:
        _check_qwen_full_finetune_params(request)
    else:
        with pytest.raises(HTTPException):_check_qwen_full_finetune_params(request)
