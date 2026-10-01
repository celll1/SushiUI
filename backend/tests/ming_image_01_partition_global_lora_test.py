"""Public ordinary LoRA plus native global summary training/restore."""
import pytest
import torch

@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])

def test_public_global_lora_training_and_checkpoint_restore(tmp_path,profile):
    method="lora"
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
        ming_conditioner_groups=[],dit_partition_global_adapter_enabled=True,dit_partition_global_rank=8,dit_partition_global_tokens=3,
        dit_gradient_checkpointing_blocks=2,dit_partition_gradient_checkpointing_blocks=1,
        dit_partition_training_enabled=True,dit_partition_mode="fixed",
        dit_partition_fixed_count=4,dit_partition_split_ratio_min=.5,dit_partition_split_ratio_max=.5,
        qwen_full_kv_query_chunk_tokens=17,qwen_guidance_loss_weight=1.,total_steps=2,seed=17)
    _check_ming_conditioner_params(request)
    factory=TrainingConfigGenerator.generate_full_finetune_config if method=='full_finetune' else TrainingConfigGenerator.generate_lora_config
    serialized=factory(request.model_dump(),run_name='run',base_model_path=str(root),output_dir=str(tmp_path/'run'))
    _,config,_,resolved=_prepare_training_process_config(yaml.safe_load(serialized),str(root))
    assert resolved==method and config['qwen_full_kv_query_chunk_tokens']==17
    klass=FullParameterTrainer if method=='full_finetune' else LoRATrainer
    value=klass(model_path=str(root),output_dir=str(tmp_path/'run'),device='cpu',weight_dtype='fp32',training_dtype='fp32',
        vae_dtype='fp32',mixed_precision=False,attention_backend='native',train_text_encoder=False,train_config=config,
        **({} if method=='full_finetune' else {'lora_rank':2,'lora_alpha':2}))
    metrics={}
    original_metric=value.log_extra_metric
    def record(k,v):metrics[k]=v;original_metric(k,v)
    value.log_extra_metric=record
    before={k:p.detach().clone() for k,p in value.transformer.named_parameters() if p.requires_grad}
    value.train([dataset(tmp_path,profile)],num_epochs=1,total_steps=2,batch_size=1,multi_noise_timesteps=2,gradient_accumulation_steps=2,
        save_every_n_steps=2,sample_every_n_steps=0,enable_bucketing=False,base_resolutions=[128],
        latent_encoding_mode='swap_onthefly',text_encoding_mode='swap_onthefly')
    assert value.transformer._ming_full_kv_query_chunk_tokens==17
    assert all(block.attention.processor._query_chunk_tokens==17 for group in ('layers','noise_refiner','context_refiner')
               for block in getattr(value.transformer,group))
    assert any(not torch.equal(p,dict(value.transformer.named_parameters())[k]) for k,p in before.items())
    assert value.transformer._training_gradient_checkpointing_blocks==2
    assert metrics["ming_partition_count"]==4.
    from core.models.ming_image_01.partition_global import GLOBAL_NAME
    branch=value.lora_layers[GLOBAL_NAME]
    assert branch is value.transformer.ming_partition_global_adapter
    assert any(p is branch.out.weight for g in value.optimizer.param_groups for p in g['params'])
    assert branch.out.weight.detach().abs().sum()>0
    checkpoint=value.output_dir/'run_step_000002.safetensors'
    from safetensors import safe_open
    import json
    with safe_open(checkpoint,framework='pt',device='cpu') as f:metadata=f.metadata()
    assert json.loads(metadata[GLOBAL_NAME])==branch.metadata()
    restored=LoRATrainer(model_path=str(root),output_dir=str(tmp_path/'restored'),device='cpu',weight_dtype='fp32',
        training_dtype='fp32',vae_dtype='fp32',mixed_precision=False,attention_backend='native',train_text_encoder=False,
        train_config=config.copy(),lora_rank=2,lora_alpha=2,resume_from_checkpoint=str(checkpoint))
    for name,layer in value.lora_layers.items():
        assert all(torch.equal(t,restored.lora_layers[name].branch_tensors()[k]) for k,t in layer.branch_tensors().items())
    from safetensors.torch import load_file,save_file
    for damage in ('version','rank','extra','nan'):
        state=load_file(str(checkpoint));bad_metadata=dict(metadata)
        info=json.loads(metadata[GLOBAL_NAME])
        if damage=='version':info['schema_version']=True
        elif damage=='rank':info['rank']=9
        elif damage=='extra':info['unexpected']=1
        else:state[GLOBAL_NAME+'.out.weight'][0,0]=float('nan')
        bad_metadata[GLOBAL_NAME]=json.dumps(info)
        path=tmp_path/(damage+'.safetensors');save_file(state,str(path),metadata=bad_metadata)
        snapshot={k:p.detach().clone() for k,p in restored.transformer.named_parameters()}
        with pytest.raises(ValueError):restored.load_checkpoint(str(path))
        assert all(torch.equal(p,dict(restored.transformer.named_parameters())[k]) for k,p in snapshot.items())
    assert not torch.cuda.is_initialized()
