"""Offline public request/YAML/runner and real partition training loop."""
import pytest
import torch

@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('method',['full_finetune','lora'])
def test_public_partition_trains_all_profiles_and_methods(tmp_path,profile,method):
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
        ming_conditioner_groups=[],dit_gradient_checkpointing_blocks=2,dit_partition_gradient_checkpointing_blocks=1,
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
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('frames,limit,expected',[(1,64,1),(3,100,2),(3,40,4)])
def test_threshold_counts_native_layer_target_tokens(frames,limit,expected):
    from types import SimpleNamespace
    from core.models.ming_image_01.partition_training import region_count
    owner=SimpleNamespace(config={'dit_partition_training_enabled':True,'dit_partition_mode':'token_threshold',
        'dit_partition_full_frame_max_tokens':limit})
    assert region_count(owner,torch.zeros(1,16,frames,16,16))==expected


def test_rewiring_partition_exception_only_applies_to_ming():
    from core.rewiring.training import validate_training_features
    config={'dit_partition_training_enabled':True}
    validate_training_features(config,'ming_image_01')
    with pytest.raises(ValueError,match='incompatible'):
        validate_training_features(config,'qwen_image_21')
    validate_training_features({'dit_partition_global_adapter_enabled':True},'ming_image_01')
    with pytest.raises(ValueError,match='repa'):
        validate_training_features({'repa_enable':True},'ming_image_01')
