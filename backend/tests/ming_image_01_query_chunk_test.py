"""Full-KV Ming query chunk equivalence, masks, packed reuse and public training."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from core.models.ming_image_01.attention import joint_attention,install_attention
from core.attention.registry import BACKENDS
from core.attention.backends import _native_varlen


@pytest.mark.parametrize('chunk',[1,4])
@pytest.mark.parametrize('kind',['key_bool','query_bool','additive'])
def test_chunked_joint_attention_dense_oracle_and_qkv_gradients(chunk,kind):
    torch.manual_seed(39)
    values=[torch.randn(2,11 if i==0 else 13,3,8) for i in range(3)]
    if kind=='key_bool':
        mask=torch.ones(2,13,dtype=torch.bool);mask[0,-3:]=False;mask[1]=False
        oracle_mask=mask[:,None,None,:]
    else:
        mask=torch.rand(2,1,11,13)>.3;mask[:,:,2]=False
        oracle_mask=mask if kind=='query_bool' else torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,-torch.inf)
        mask=oracle_mask
    first=[v.clone().requires_grad_() for v in values];second=[v.clone().requires_grad_() for v in values]
    expected=torch.nn.functional.scaled_dot_product_attention(first[0].transpose(1,2),first[1].transpose(1,2),
        first[2].transpose(1,2),attn_mask=oracle_mask).transpose(1,2)
    actual=joint_attention(*second,mask,backend='native',query_chunk_tokens=chunk)
    expected.square().sum().backward();actual.square().sum().backward()
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    for a,b in zip(first,second):torch.testing.assert_close(a.grad,b.grad,rtol=3e-5,atol=3e-6)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('chunk',[2,4,16])
def test_packed_chunk_queries_keep_all_rows_and_reuse_one_kv_pack(monkeypatch,chunk):
    torch.manual_seed(5);calls=[]
    mask=torch.tensor([[True,False,True,True,False],[False,True,True,True,True]])
    def kernel(q,k,v,cu_q,cu_k,max_q,max_k,**kwargs):
        calls.append((k.data_ptr(),v.data_ptr(),cu_q.tolist(),cu_k.tolist(),max_q,max_k))
        return _native_varlen(q,k,v,cu_q,cu_k,max_q,max_k,**kwargs)
    monkeypatch.setitem(BACKENDS,'flash',replace(BACKENDS['flash'],needs_half_dtype=False,varlen_fn=kernel))
    values=[torch.randn(2,5,3,8) for _ in range(3)]
    first=[v.clone().requires_grad_() for v in values];second=[v.clone().requires_grad_() for v in values]
    expected=torch.nn.functional.scaled_dot_product_attention(first[0].transpose(1,2),first[1].transpose(1,2),
        first[2].transpose(1,2),attn_mask=mask[:,None,None,:]).transpose(1,2)
    actual=joint_attention(*second,mask,backend='flash',query_chunk_tokens=chunk)
    expected.square().sum().backward();actual.square().sum().backward()
    assert len({(call[0],call[1]) for call in calls})==1
    assert all(call[3]==[0,3,7] and call[5]==4 and call[4]<=chunk for call in calls)
    assert sum(call[4] for call in calls)==5 and actual.shape==second[0].shape
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    for a,b in zip(first,second):torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=2e-6)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('padding',['zero_masked','learned'])
@pytest.mark.parametrize('rewired',[False,True])
def test_native_dit_chunking_preserves_reference_layer_rope_and_gradients(padding,rewired):
    from ming_image_01_rewiring_test import config,spec_for
    from core.models.ming_image_01.model_definition import create_transformer
    from core.models.ming_image_01.residual_unet import restore_topology
    torch.manual_seed(23);original=create_transformer(config(padding),device='cpu',dtype=torch.float32)
    if padding=='learned':
        torch.nn.init.constant_(original.x_pad_token,.02);torch.nn.init.constant_(original.cap_pad_token,.03)
    if rewired:restore_topology(original,spec_for(original))
    chunked=deepcopy(original);install_attention(chunked,'native',query_chunk_tokens=33)
    frames=3 if padding=='learned' else 1
    images=[torch.randn(16,frames,16,16) for _ in range(2)]
    references=[torch.randn(16,1,16,16),torch.randn(16,2,16,16)]
    queries=[torch.randn(5,8) for _ in range(2)];captions=[torch.randn(3,24),torch.randn(35,24)]
    outputs=[];leaves=[]
    for model in (original,chunked):
        x,q,c=[[v.clone().requires_grad_() for v in group] for group in (images,queries,captions)]
        result=model(x,torch.tensor([.2,.6]),q,cap_feats_2=c,ref_x=references).sample
        sum(v.square().mean() for v in result).backward();outputs.append(result);leaves.append(x+q+c)
    for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    for a,b in zip(*leaves):torch.testing.assert_close(a.grad,b.grad,rtol=5e-4,atol=3e-6)
    for name,p in original.named_parameters():
        other=dict(chunked.named_parameters())[name]
        assert (p.grad is None)==(other.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,other.grad,rtol=5e-4,atol=3e-6)
    assert original.state_dict().keys()==chunked.state_dict().keys()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('method',['full_finetune','lora'])
def test_public_query_chunk_reaches_all_processors_and_guided_training(tmp_path,profile,method):
    from ming_image_01_conditioner_quantized_test import source
    from ming_image_01_conditioner_full_quantized_test import dataset
    from api.routes import TrainingRunCreateRequest,_check_ming_conditioner_params
    from core.training.training_config import TrainingConfigGenerator
    from core.training.train_runner import _prepare_training_process_config
    from core.training.full_parameter_trainer import FullParameterTrainer
    from core.training.lora_trainer import LoRATrainer
    import yaml
    root=source(tmp_path,profile)
    request=TrainingRunCreateRequest(training_method=method,base_model_path=str(root),train_text_encoder=False,
        ming_conditioner_groups=[],qwen_full_kv_query_chunk_tokens=17,qwen_guidance_loss_weight=1.,total_steps=2,seed=17)
    _check_ming_conditioner_params(request)
    factory=TrainingConfigGenerator.generate_full_finetune_config if method=='full_finetune' else TrainingConfigGenerator.generate_lora_config
    serialized=factory(request.model_dump(),run_name='run',base_model_path=str(root),output_dir=str(tmp_path/'run'))
    _,config,_,resolved=_prepare_training_process_config(yaml.safe_load(serialized),str(root))
    assert resolved==method and config['qwen_full_kv_query_chunk_tokens']==17
    klass=FullParameterTrainer if method=='full_finetune' else LoRATrainer
    value=klass(model_path=str(root),output_dir=str(tmp_path/'run'),device='cpu',weight_dtype='fp32',training_dtype='fp32',
        vae_dtype='fp32',mixed_precision=False,attention_backend='native',train_text_encoder=False,train_config=config,
        **({} if method=='full_finetune' else {'lora_rank':2,'lora_alpha':2}))
    before={k:p.detach().clone() for k,p in value.transformer.named_parameters() if p.requires_grad}
    value.train([dataset(tmp_path,profile)],num_epochs=1,total_steps=2,batch_size=1,multi_noise_timesteps=2,
        save_every_n_steps=2,sample_every_n_steps=0,enable_bucketing=False,base_resolutions=[64],
        latent_encoding_mode='swap_onthefly',text_encoding_mode='swap_onthefly')
    assert value.transformer._ming_full_kv_query_chunk_tokens==17
    assert all(block.attention.processor._query_chunk_tokens==17 for group in ('layers','noise_refiner','context_refiner')
               for block in getattr(value.transformer,group))
    assert any(not torch.equal(p,dict(value.transformer.named_parameters())[k]) for k,p in before.items())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('chunk',[True,-1,1.5])
def test_invalid_query_chunk_refused_before_weight_payload(chunk):
    from core.models.ming_image_01.training_options import preflight_training
    with pytest.raises(ValueError,match='nonnegative integer'):
        preflight_training({'ming_conditioner_groups':[],'qwen_full_kv_query_chunk_tokens':chunk},method='full_finetune')
