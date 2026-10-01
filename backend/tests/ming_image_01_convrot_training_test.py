"""Offline Ming ConvRot policy/metadata dispatch; CUDA kernels remain owner acceptance."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from core.models.ming_image_01 import convrot_training as policy
from core.models.common.convrot_int8_linear import ConvRotInt8Linear


def model(packed=True):
    result=nn.Module()
    result.layers=nn.ModuleList([nn.Module()])
    if packed:
        result.layers[0].q=ConvRotInt8Linear(256,8,False,torch.bfloat16,convrot_groupsize=256,marker_numel=1,device='cpu')
    else:result.layers[0].q=nn.Linear(256,8)
    result.noise_refiner=nn.ModuleList([]);result.context_refiner=nn.ModuleList([])
    return result


def owner(mode='auto', **kwargs):
    return SimpleNamespace(config={'qwen_convrot_training_forward':mode},device='cuda',training_dtype=torch.bfloat16,blocks_to_swap=0,**kwargs)


@pytest.mark.parametrize('mode', ['auto','dequant','transient_bf16','cached_bf16','prefetch_bf16'])
def test_resolved_compatible_modes(mode):
    effective,blocks,depth=policy.resolve_policy(owner(mode),model())
    assert effective==('prefetch_bf16' if mode=='auto' else mode)
    assert blocks==2 and depth==1


@pytest.mark.parametrize('field,value', [('qwen_convrot_training_forward','bad'),('qwen_convrot_backward_cache_blocks',True),('qwen_convrot_backward_cache_blocks',0),('qwen_convrot_backward_cache_blocks',9),('qwen_convrot_backward_prefetch_depth',True),('qwen_convrot_backward_prefetch_depth',-1),('qwen_convrot_backward_prefetch_depth',8),('qwen_convrot_backward_prefetch_depth',2)])
def test_preflight_invalid_settings_before_device_access(field,value):
    with pytest.raises(ValueError):policy.settings({field:value},method='lora')


@pytest.mark.parametrize('damage', ['dense','cpu','fp32','swap'])
def test_incompatible_optimized_policy_is_refused_before_move(damage):
    value=owner('prefetch_bf16');native=model(damage!='dense')
    if damage=='cpu':value.device='cpu'
    if damage=='fp32':value.training_dtype=torch.float32
    if damage=='swap':value.blocks_to_swap=1
    native.to=lambda **k:pytest.fail('Incompatible policy attempted model move')
    with pytest.raises(ValueError):policy.configure_training(value,native)


@pytest.mark.parametrize('mode', ['transient_bf16','cached_bf16','prefetch_bf16'])
def test_full_training_refuses_optimized_frozen_mode(mode):
    with pytest.raises(ValueError,match='LoRA'):policy.settings({'qwen_convrot_training_forward':mode},method='full_finetune')


@pytest.mark.parametrize('mode', ['transient_bf16','cached_bf16','prefetch_bf16'])
def test_configure_dispatches_exact_shared_policy_and_ming_groups(monkeypatch,mode):
    from core.models.common import quantized_frozen_training as shared
    native=model();value=owner(mode);calls=[];cache=object()
    native.to=lambda **kw:calls.append(('move',kw))
    def enable(module,**kw):
        assert module is native
        calls.append((mode,kw))
        module.layers[0].q._frozen_training_fused=True
        if mode=='transient_bf16':return 1
        if mode=='cached_bf16':return 1,4096
        return 1,2048,cache
    target={'transient_bf16':'enable_frozen_training_fused','cached_bf16':'enable_frozen_training_cached_backward','prefetch_bf16':'enable_frozen_training_prefetch_backward'}[mode]
    monkeypatch.setattr(shared,target,enable)
    result=policy.configure_training(value,native)
    assert calls[0]==('move',{'device':'cuda'})
    assert result['execution']==policy.BASE_FORWARD and result['policy']==mode
    assert not native.layers[0].q._force_dequant
    assert native.layers[0].q._frozen_training_fused
    if mode=='prefetch_bf16':
        assert calls[1][1]['block_groups']==policy.MING_BLOCK_GROUPS
        assert value.ming_convrot_backward_prefetch_cache is cache
    else:assert value.ming_convrot_backward_prefetch_cache is None
    assert policy.checkpoint_metadata(value)=={'ming_base_forward':policy.BASE_FORWARD}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('current,saved,valid', [('dequant',{},True),('dequant',{'ming_base_forward':policy.BASE_FORWARD},False),('cached_bf16',{},False),('cached_bf16',{'ming_base_forward':policy.BASE_FORWARD},True),('prefetch_bf16',{'ming_base_forward':policy.BASE_FORWARD},True),('transient_bf16',{'ming_base_forward':'unknown'},False)])
def test_resume_base_function_contract(current,saved,valid):
    value=SimpleNamespace(ming_convrot_training_forward=current)
    if valid:policy.validate_resume_metadata(value,saved)
    else:
        with pytest.raises(ValueError,match='base-forward'):policy.validate_resume_metadata(value,saved)


def test_cpu_auto_retains_actual_differentiable_dequant_path():
    value=owner();value.device='cpu';native=model()
    native.to=lambda **kw:pytest.fail('CPU auto must not request optimized placement')
    report=policy.configure_training(value,native)
    assert report['execution']=='dequant_autograd'
    assert value.ming_convrot_training_forward=='dequant' and native.layers[0].q._force_dequant
    assert policy.checkpoint_metadata(value)=={}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
def test_real_lora_checkpoint_contract_refused_before_branch_mutation(tmp_path,profile):
    from safetensors.torch import load_file,save_file
    from safetensors import safe_open
    from ming_image_01_conditioning_test import write_tiny_package
    from ming_image_01_lora_trainer_test import make_owner
    root=tmp_path/'package';root.mkdir();write_tiny_package(root,profile=profile)
    value=make_owner(root,tmp_path/'run')
    value.save_checkpoint(0,0)
    checkpoint=value.output_dir/'run_step_000000.safetensors'
    with safe_open(str(checkpoint),framework='pt',device='cpu') as handle:metadata=handle.metadata()
    assert 'ming_base_forward' not in metadata
    altered=tmp_path/'optimized.safetensors'
    save_file(load_file(str(checkpoint)),str(altered),metadata={**metadata,'ming_base_forward':policy.BASE_FORWARD})
    snapshot={name:param.clone() for name,param in value.transformer.named_parameters()}
    with pytest.raises(ValueError,match='base-forward'):value._load_resume_adapter(str(altered))
    assert all(torch.equal(before,dict(value.transformer.named_parameters())[name]) for name,before in snapshot.items())
    value._load_resume_adapter(str(checkpoint))
    assert not torch.cuda.is_initialized()


def test_public_config_preserves_convrot_fields_and_full_auto_ignores_ring_geometry(tmp_path):
    import yaml
    from api.routes import TrainingRunCreateRequest
    from core.training.training_config import TrainingConfigGenerator
    from core.training.train_runner import _prepare_training_process_config
    request=TrainingRunCreateRequest(training_method='lora',total_steps=1,base_model_path=str(tmp_path),qwen_convrot_training_forward='prefetch_bf16',qwen_convrot_backward_cache_blocks=4,qwen_convrot_backward_prefetch_depth=2)
    payload=request.model_dump();payload['_explicit_fields']=sorted(request.model_fields_set)
    text=TrainingConfigGenerator.generate_lora_config(payload,run_name='run',base_model_path=str(tmp_path),output_dir=str(tmp_path/'run'))
    _,config,_,_= _prepare_training_process_config(yaml.safe_load(text),str(tmp_path))
    assert policy.settings(config,method='lora')==('prefetch_bf16',4,2)
    assert policy.settings({'qwen_convrot_backward_cache_blocks':1,'qwen_convrot_backward_prefetch_depth':1},method='full_finetune')==('auto',1,1)


@pytest.mark.parametrize('packed,dtype,metadata,valid',[(True,torch.bfloat16,{'ming_base_forward':policy.BASE_FORWARD},True),(False,torch.bfloat16,{'ming_base_forward':policy.BASE_FORWARD},False),(True,torch.float32,{'ming_base_forward':policy.BASE_FORWARD},False),(True,torch.bfloat16,{'ming_base_forward':'unknown'},False),(False,torch.float32,{},True)])
def test_generation_requires_saved_convrot_base_function(packed,dtype,metadata,valid):
    native=model(packed)
    if packed:native.layers[0].q.compute_dtype=dtype
    if valid:policy.validate_generation_metadata(native,metadata)
    else:
        with pytest.raises(ValueError):policy.validate_generation_metadata(native,metadata)

@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
def test_generation_session_refuses_contract_before_wrapping_and_can_retry(tmp_path,profile):
    from ming_image_01_pipeline_test import manager
    from ming_image_01_generation_lora_test import write_adapter
    from core.models.ming_image_01.generation_adapters import generation_adapters
    _,components=manager(profile)
    model=components.transformer;base=model.layers[0].attention.to_q
    bad=tmp_path/'bad.safetensors';good=tmp_path/'good.safetensors'
    write_adapter(bad,model,profile=profile,metadata={'ming_base_forward':policy.BASE_FORWARD})
    write_adapter(good,model,profile=profile)
    with pytest.raises(Exception,match='ConvRot base'):
        with generation_adapters(components,[{'path':str(bad)}],resolve_path=str):
            pytest.fail('Incompatible base was accepted')
    assert model.layers[0].attention.to_q is base
    with generation_adapters(components,[{'path':str(good)}],resolve_path=str):
        assert model.layers[0].attention.to_q is not base
    assert model.layers[0].attention.to_q is base
    assert not torch.cuda.is_initialized()
