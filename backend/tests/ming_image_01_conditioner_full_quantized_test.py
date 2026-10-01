"""Packed conditioner full ownership, lossy materialization and floating restart."""
import json
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from core.training.full_parameter_trainer import FullParameterTrainer
from core.models.ming_image_01.loader import load_components
from core.models.ming_image_01.quantization import saved_full_initialization
from ming_image_01_conditioner_quantized_test import source,pack


def package(tmp_path,profile,format_):
    root=source(tmp_path,profile,wide=format_=='int8_convrot')
    pack(root,'mllm',convrot=format_=='int8_convrot');pack(root,'connector',convrot=format_=='int8_convrot')
    return root


def owner(root,output,groups,*,optin=False,config=None):
    return FullParameterTrainer(model_path=str(root),output_dir=str(output),device='cpu',weight_dtype='fp32',
        training_dtype='fp32',vae_dtype='fp32',mixed_precision=False,attention_backend='native',
        train_unet=False,train_text_encoder=True,train_config={
            'ming_conditioner_groups':groups,'train_unet':False,'train_adapter':False,
            'full_finetune_dequantize_int8_base':optin,'text_encoding_mode':'swap_onthefly',
            'latent_encoding_mode':'swap_onthefly','seed':17,**(config or {})})


def dataset(tmp_path,profile):
    image=tmp_path/'target.png';Image.new('RGBA',(64,64),(100,70,40,180)).save(image)
    if profile=='layer_decompose':
        layer=tmp_path/'layer.png';Image.new('RGBA',(64,64),(40,80,110,120)).save(layer)
        image.with_name(image.name+'.ming.json').write_text(json.dumps({'schema_version':1,'profile':profile,
            'targets':[{'path':image.name,'role':'auxiliary_canvas','weight':0},
                       {'path':layer.name,'role':'layer_1','weight':1}], 'references':[image.name]}))
    return SimpleNamespace(unique_id='conditioner-full',dataset_root=tmp_path,
        items=[{'image_path':str(image),'caption':'word5','width':64,'height':64}],reload_for_epoch=lambda **k:None)


def train(value,data):
    value.train([data],num_epochs=2,total_steps=2,batch_size=1,save_every_n_steps=1,sample_every_n_steps=0,
        enable_bucketing=False,base_resolutions=[64],text_encoding_mode='swap_onthefly',latent_encoding_mode='swap_onthefly')


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('format_',['int8','int8_convrot'])
def test_public_conditioner_full_optin_trains_packed_paths_saves_and_restarts_floating(tmp_path,profile,format_):
    root=package(tmp_path,profile,format_)
    from api.routes import TrainingRunCreateRequest,_check_ming_conditioner_params,_check_qwen_full_finetune_params
    from core.training.training_config import TrainingConfigGenerator
    from core.training.train_runner import _prepare_training_process_config
    import yaml
    request=TrainingRunCreateRequest(training_method='full_finetune',base_model_path=str(root),total_steps=2,
        train_unet=False,train_adapter=False,train_text_encoder=True,ming_conditioner_groups=['language','connector'],
        full_finetune_dequantize_int8_base=True,text_encoding_mode='swap_onthefly',latent_encoding_mode='swap_onthefly')
    _check_ming_conditioner_params(request);_check_qwen_full_finetune_params(request)
    serialized=TrainingConfigGenerator.generate_full_finetune_config(request.model_dump(),run_name='run',base_model_path=str(root),output_dir=str(tmp_path/'run'))
    _,config,_,method=_prepare_training_process_config(yaml.safe_load(serialized),str(root))
    assert method=='full_finetune' and config['full_finetune_dequantize_int8_base'] is True
    value=owner(root,tmp_path/'run',['language','connector'],optin=True,config=config)
    report=value.ming_conditioner_full_initialization
    assert report['lossy_source'] and sum(report['formats'].values())==4
    assert report['formats']['convrot_int8']==4 if format_=='int8_convrot' else report['formats']['e4m3']==2
    assert not any(key.endswith(('.weight_scale','.comfy_quant')) for key in value.text_encoder.state_dict())
    groups=value.setup_trainable_parameters()
    actual=[id(p) for group in groups for p in group['params']]
    expected={id(p) for p in value.text_encoder.parameters() if p.requires_grad}
    assert set(actual)==expected and len(actual)==len(set(actual))
    before={name:p.detach().clone() for name,p in value.text_encoder.named_parameters()}
    dit={key:p.clone() for key,p in value.transformer.state_dict().items()}
    train(value,dataset(tmp_path,profile))
    current=dict(value.text_encoder.named_parameters())
    for selector in ('model.model.layers.0.attention.query_key_value.weight','connector.model.layers.0.self_attn.q_proj.weight'):
        name='native.'+selector
        assert not torch.equal(current[name],before[name]),name
    assert all(torch.isfinite(p).all() for p in current.values())
    assert all(torch.equal(p,value.transformer.state_dict()[key]) for key,p in dit.items())
    checkpoint=value.output_dir/'run_step_000002'
    metadata=json.loads((checkpoint/'training.json').read_text())
    assert metadata['ming_conditioner_full_initialization']==report
    restored=load_components(checkpoint,dtype=torch.float32,conditioner_attention='eager')
    assert saved_full_initialization(restored.artifact,component='conditioner')==report
    assert all(torch.equal(t,restored.conditioner.state_dict()[key]) for key,t in value.text_encoder.state_dict().items())
    assert restored.transformer._residual_unet_spec==value.transformer._residual_unet_spec
    fresh=owner(checkpoint,tmp_path/'fresh',['language','connector'])
    assert fresh.ming_conditioner_full_initialization==report
    fresh_checkpoint=fresh.adapter.write_checkpoint(0,0,tmp_path/'fresh-float')
    assert json.loads((fresh_checkpoint.parent/'training.json').read_text())['ming_conditioner_full_initialization']==report
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('group',['queries','projections'])
def test_unpacked_group_can_train_through_frozen_convrot_conditioner_without_materialization(tmp_path,group):
    root=package(tmp_path,'generation_edit','int8_convrot')
    value=owner(root,tmp_path/'run',[group])
    assert value.ming_conditioner_full_initialization is None
    assert value.ming_conditioner_quantized_base_training['formats']['convrot_int8']==4
    before={key:t.clone() for key,t in value.text_encoder.state_dict().items() if key.endswith(('.weight_scale','.comfy_quant')) or t.dtype==torch.int8}
    initial={name:p.detach().clone() for name,p in value.text_encoder.named_parameters() if p.requires_grad}
    train(value,dataset(tmp_path,'generation_edit'))
    current=dict(value.text_encoder.named_parameters())
    assert any(not torch.equal(p,current[name]) for name,p in initial.items())
    assert all(torch.equal(t,value.text_encoder.state_dict()[key]) for key,t in before.items())
    metadata=json.loads((value.output_dir/'run_step_000002/training.json').read_text())
    assert metadata['ming_conditioner_quantized_base_training']['execution']=='dequant_autograd'
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('group',['language','connector'])
def test_selected_packed_group_without_optin_refused_before_optimizer(tmp_path,group):
    root=package(tmp_path,'generation_edit','int8')
    with pytest.raises(NotImplementedError,match='quantiz'):
        owner(root,tmp_path/'run',[group])


@pytest.mark.parametrize('field,value',[('schema_version',True),('lossy_source',False)])
def test_conditioner_initialization_provenance_is_strict(tmp_path,field,value):
    report={'schema_version':1,'execution':'materialized_floating','formats':{'int8':1,'e4m3':0,'convrot_int8':0},'weight_dtype':'float32','lossy_source':True}
    report[field]=value
    (tmp_path/'training.json').write_text(json.dumps({'ming_conditioner_full_initialization':report}))
    with pytest.raises(ValueError,match='initialization metadata'):
        saved_full_initialization(SimpleNamespace(root=tmp_path),component='conditioner')


def test_materialized_packed_vision_updates_with_reference_supervision(tmp_path):
    from safetensors.torch import load_file,save_file
    root=package(tmp_path,'layer_decompose','int8')
    path=root/'mllm/model.safetensors';state=load_file(str(path))
    key='vision.blocks.0.mlp.down_proj.weight'
    state[key]=(state[key]*64).to(torch.float8_e4m3fn)
    state[key.removesuffix('.weight')+'.weight_scale']=torch.full((state[key].shape[0],),1/64.)
    save_file(state,str(path))
    value=owner(root,tmp_path/'run',['vision'],optin=True)
    before=value.text_encoder.native.vision.blocks[0].mlp.down_proj.weight.detach().clone()
    assert value.ming_conditioner_full_initialization['formats']['e4m3']==3
    train(value,dataset(tmp_path,'layer_decompose'))
    after=value.text_encoder.native.vision.blocks[0].mlp.down_proj.weight
    assert not torch.equal(before,after) and torch.isfinite(after).all()
    assert not torch.cuda.is_initialized()


def test_joint_dit_conditioner_materialization_preserves_both_origin_records(tmp_path):
    from safetensors.torch import load_file,save_file
    from core.models.ming_image_01.residual_unet import model_topology_metadata
    root=package(tmp_path,'generation_edit','int8')
    original=load_components(root,dtype=torch.float32,conditioner_attention='eager')
    path=root/'transformer/diffusion_pytorch_model.safetensors';state=load_file(str(path))
    key='layers.0.attention.to_q.weight'
    state[key]=(state[key]*64).to(torch.int8)
    state[key.removesuffix('.weight')+'.weight_scale']=torch.full((state[key].shape[0],),1/64.)
    save_file(state,str(path),metadata=model_topology_metadata(original.transformer))
    value=owner(root,tmp_path/'run',['language','connector'],optin=True)
    assert value.ming_full_initialization['formats']['int8']==1
    assert sum(value.ming_conditioner_full_initialization['formats'].values())==4
    manifest=value.adapter.write_checkpoint(0,0,tmp_path/'joint-float')
    metadata=json.loads((manifest.parent/'training.json').read_text())
    assert metadata['ming_full_initialization']==value.ming_full_initialization
    assert metadata['ming_conditioner_full_initialization']==value.ming_conditioner_full_initialization
    fresh=owner(manifest,tmp_path/'fresh',['language','connector'])
    assert fresh.ming_full_initialization==value.ming_full_initialization
    assert fresh.ming_conditioner_full_initialization==value.ming_conditioner_full_initialization
    assert fresh.transformer._residual_unet_spec==value.transformer._residual_unet_spec
    assert not torch.cuda.is_initialized()
