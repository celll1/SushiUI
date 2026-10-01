"""Prepared Ming projection precision, native inference and full-training ownership."""
import json
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file,save_file
from core.models.ming_image_01.loader import load_components
from core.models.ming_image_01.checkpoint import save_bundle
from ming_image_01_conditioner_quantized_test import source


def pack_projections(root,convrot=False):
    path=root/'mlp/model.safetensors';state=load_file(str(path))
    from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight
    from core.models.common.quantized_checkpoint_guard import encode_comfy_quant_marker
    for index,key in enumerate(('proj_in.weight','proj_out.weight','proj_directvlm.1.weight')):
        stem=key.removesuffix('.weight')
        if convrot:
            code,scale=quantize_int8_convrot_weight(state[key],256)
            state[key]=code.contiguous();state[stem+'.weight_scale']=scale.float().contiguous()
            state[stem+'.comfy_quant']=encode_comfy_quant_marker({'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256})
        else:
            state[key]=(state[key]*64).to(torch.int8 if index%2==0 else torch.float8_e4m3fn)
            state[stem+'.weight_scale']=torch.full((state[key].shape[0],),1/64.)
    save_file(state,str(path))
    return {k:v for k,v in state.items() if k.endswith(('.weight_scale','.comfy_quant')) or v.dtype in (torch.int8,torch.float8_e4m3fn)}


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('convrot',[False,True])
def test_packed_projection_actual_rgba_inference_and_native_roundtrip(tmp_path,profile,dtype,convrot):
    root=source(tmp_path,profile,wide=convrot);packed=pack_projections(root,convrot)
    c=load_components(root,dtype=dtype,conditioner_attention='eager')
    state=c.conditioner.projection_state_dict()
    for key,value in packed.items():
        actual=state[key]
        assert actual.dtype==value.dtype and torch.equal(actual.view(torch.uint8),value.reshape(actual.shape).view(torch.uint8))
    assert c.conditioner.native.query_tokens_dict['2x2'].dtype==dtype
    assert c.conditioner.native.proj_directvlm[0].weight.dtype==dtype
    from ming_image_01_pipeline_test import manager,params
    value,_=manager(profile)
    value._install_ming_image_01({'ming_native':c,'transformer':c.transformer,'vae':c.vae,'text_encoder':c.conditioner},model_id='packed-projections',source_type='local',source=str(root))
    images=[Image.new('RGBA',(64,64),(70,90,120,180))] if profile=='layer_decompose' else []
    finite=[]
    result,_,_=value._ming_image_run(params(width=64,height=64,ming_image_01_layers=1),images=images,
        step_callback=lambda i,t,latents:finite.append(bool(torch.isfinite(latents).all())))
    assert finite==[True] and result.groups[0].values[0].mode=='RGBA'
    restored=load_components(save_bundle(c,tmp_path/'saved'),dtype=dtype,conditioner_attention='eager')
    assert all(torch.equal(v.view(torch.uint8),restored.conditioner.projection_state_dict()[k].view(torch.uint8)) for k,v in state.items())
    assert restored.transformer._residual_unet_spec==c.transformer._residual_unet_spec
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage',['missing_scale','wrong_scale','query_codes','norm_codes','unknown_slot'])
def test_packed_projection_malformed_preflight_precedes_weight_payload(tmp_path,monkeypatch,damage):
    root=source(tmp_path,'generation_edit');pack_projections(root)
    path=root/'mlp/model.safetensors';state=load_file(str(path))
    if damage=='missing_scale':del state['proj_in.weight_scale']
    elif damage=='wrong_scale':state['proj_in.weight_scale']=torch.ones(1)
    elif damage=='query_codes':state['query_tokens_dict.2x2']=state['query_tokens_dict.2x2'].to(torch.int8)
    elif damage=='norm_codes':state['proj_directvlm.0.weight']=state['proj_directvlm.0.weight'].to(torch.int8)
    else:state['unknown.weight']=torch.ones(4,4,dtype=torch.int8)
    save_file(state,str(path))
    def forbidden(*a,**k):raise AssertionError('Payload loaded before projection preflight')
    monkeypatch.setattr('safetensors.torch.load_file',forbidden)
    with pytest.raises(ValueError,match='Ming quantized|Unsupported Ming'):
        load_components(root,dtype=torch.float32,conditioner_attention='eager')


@pytest.mark.parametrize('convrot',[False,True])
def test_packed_projection_full_optin_updates_and_floating_restart(tmp_path,convrot):
    root=source(tmp_path,'generation_edit',wide=convrot);pack_projections(root,convrot)
    from ming_image_01_conditioner_full_quantized_test import owner,train,dataset
    with pytest.raises(NotImplementedError,match='quantiz'):
        owner(root,tmp_path/'refused',['projections'])
    value=owner(root,tmp_path/'run',['projections'],optin=True)
    before=value.text_encoder.native.proj_in.weight.detach().clone()
    train(value,dataset(tmp_path,'generation_edit'))
    assert not torch.equal(before,value.text_encoder.native.proj_in.weight)
    assert isinstance(value.text_encoder.native.proj_in,torch.nn.Linear)
    report=value.ming_conditioner_full_initialization
    assert sum(report['formats'].values())==3 and report['lossy_source']
    checkpoint=value.output_dir/'run_step_000002'
    fresh=owner(checkpoint,tmp_path/'fresh',['projections'])
    assert fresh.ming_conditioner_full_initialization==report
    assert torch.equal(fresh.text_encoder.native.proj_in.weight,value.text_encoder.native.proj_in.weight)
    assert not torch.cuda.is_initialized()


def test_dit_convrot_training_export_preserves_packed_projection_bytes(tmp_path):
    from ming_image_01_conversion_test import make_source
    from core.training.ming_full_export import export_ming_full_checkpoint
    run=tmp_path/'run';run.mkdir();root=run/'run_step_000001'
    make_source(root,'generation_edit');pack_projections(root)
    (root/'training.json').write_text(json.dumps({'schema_version':1,'kind':'full_parameter_weights','profile':'generation_edit'}))
    before=(root/'mlp/model.safetensors').read_bytes()
    exported=export_ming_full_checkpoint(root,run)
    assert (exported/'mlp/model.safetensors').read_bytes()==before
    assert (root/'mlp/model.safetensors').read_bytes()==before
    c=load_components(exported,dtype=torch.float32,conditioner_attention='eager')
    from core.models.ideogram4.vendor.int8_linear import Int8Linear
    assert isinstance(c.conditioner.native.proj_in,Int8Linear)
    assert c.transformer._residual_unet_spec is not None
    assert not torch.cuda.is_initialized()



@pytest.mark.parametrize('convrot',[False,True])
def test_queries_train_through_frozen_packed_projections(tmp_path,convrot):
    root=source(tmp_path,'generation_edit',wide=convrot);packed=pack_projections(root,convrot)
    from ming_image_01_conditioner_full_quantized_test import owner,train,dataset
    value=owner(root,tmp_path/'run',['queries'])
    before=value.text_encoder.native.query_tokens_dict['2x2'].detach().clone()
    initial={k:v.detach().clone() for k,v in value.text_encoder.projection_state_dict().items() if k in packed}
    train(value,dataset(tmp_path,'generation_edit'))
    assert not torch.equal(before,value.text_encoder.native.query_tokens_dict['2x2'])
    assert all(torch.equal(v.view(torch.uint8),value.text_encoder.projection_state_dict()[k].view(torch.uint8)) for k,v in initial.items())
    assert value.ming_conditioner_full_initialization is None
    assert sum(value.ming_conditioner_quantized_base_training['formats'].values())==3
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('convrot',[False,True])
def test_all_packed_conditioner_components_query_training_and_checkpoint(tmp_path,convrot):
    from ming_image_01_conditioner_quantized_test import pack
    from ming_image_01_conditioner_full_quantized_test import owner,train,dataset
    root=source(tmp_path,'layer_decompose',wide=convrot)
    pack(root,'mllm',convrot=convrot);pack(root,'connector',convrot=convrot);pack_projections(root,convrot)
    value=owner(root,tmp_path/'run',['queries'])
    state=value.text_encoder.state_dict()
    before={k:t.detach().clone() for k,t in state.items() if k.endswith(('.weight_scale','.comfy_quant')) or t.dtype in (torch.int8,torch.float8_e4m3fn)}
    initial=value.text_encoder.native.query_tokens_dict['2x2'].detach().clone()
    assert sum(value.ming_conditioner_quantized_base_training['formats'].values())==7
    train(value,dataset(tmp_path,'layer_decompose'))
    assert not torch.equal(initial,value.text_encoder.native.query_tokens_dict['2x2'])
    assert all(torch.equal(t.view(torch.uint8),value.text_encoder.state_dict()[k].view(torch.uint8)) for k,t in before.items())
    restored=load_components(value.output_dir/'run_step_000002',dtype=torch.float32,conditioner_attention='eager')
    assert all(torch.equal(t.view(torch.uint8),restored.conditioner.state_dict()[k].view(torch.uint8)) for k,t in before.items())
    assert torch.equal(restored.conditioner.native.query_tokens_dict['2x2'],value.text_encoder.native.query_tokens_dict['2x2'])
    assert not torch.cuda.is_initialized()
