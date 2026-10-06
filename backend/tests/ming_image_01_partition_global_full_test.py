"""Actual native full training and independent global bundle preservation."""
import pytest
import torch

@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('swapped',[False,True])

def test_public_global_full_training_updates_and_restores_bundle(tmp_path,profile,swapped):
    method="full_finetune"
    from ming_image_01_conditioner_quantized_test import source
    from ming_image_01_conditioner_full_quantized_test import dataset
    from api.routes import TrainingRunCreateRequest,_check_ming_conditioner_params
    from core.training.training_config import TrainingConfigGenerator
    from core.training.train_runner import _prepare_training_process_config
    from core.training.full_parameter_trainer import FullParameterTrainer
    from core.training.lora_trainer import LoRATrainer
    import yaml
    root=source(tmp_path,profile)
    swap_options={}
    if swapped:
        from diffusers import AutoencoderKL
        target=AutoencoderKL(in_channels=4,out_channels=4,latent_channels=24,block_out_channels=(8,)*5,
            down_block_types=("DownEncoderBlock2D",)*5,up_block_types=("UpDecoderBlock2D",)*5,
            norm_num_groups=4,layers_per_block=1,scaling_factor=.7,shift_factor=.2)
        target.save_pretrained(tmp_path/"target_codec")
        swap_options=dict(vae_swap_source=f"file:{tmp_path/'target_codec'}",vae_swap_calibration_samples=2,
            vae_swap_calibration_resolution=128)
    request=TrainingRunCreateRequest(training_method=method,base_model_path=str(root),train_text_encoder=False,
        ming_conditioner_groups=[],dit_partition_global_adapter_enabled=True,dit_partition_global_rank=8,dit_partition_global_tokens=3,
        dit_gradient_checkpointing_blocks=2,dit_partition_gradient_checkpointing_blocks=1,
        dit_partition_training_enabled=True,dit_partition_mode="fixed",
        dit_partition_fixed_count=4,dit_partition_split_ratio_min=.5,dit_partition_split_ratio_max=.5,
        qwen_full_kv_query_chunk_tokens=17,qwen_guidance_loss_weight=1.,total_steps=2,seed=17,**swap_options)
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
    branch=value.transformer.ming_partition_global_adapter
    if swapped:
        assert value.arch.pixel_align==32 and value.arch.wiring.latent_channels==24
    assert all(p.requires_grad for p in branch.parameters())
    before={k:p.detach().clone() for k,p in value.transformer.named_parameters() if p.requires_grad}
    data=dataset(tmp_path,profile)
    if swapped:
        from PIL import Image
        import numpy as np
        for index,name in enumerate(("target.png","layer.png") if profile=="layer_decompose" else ("target.png",)):
            Image.fromarray(np.random.default_rng(index).integers(0,256,(128,128,4),dtype=np.uint8)).save(tmp_path/name)
        data.items[0].update(width=128,height=128)
    value.train([data],num_epochs=1,total_steps=2,batch_size=1,multi_noise_timesteps=2,gradient_accumulation_steps=2,
        save_every_n_steps=2,sample_every_n_steps=0,enable_bucketing=False,base_resolutions=[128],
        latent_encoding_mode='swap_onthefly',text_encoding_mode='swap_onthefly')
    assert value.transformer._ming_full_kv_query_chunk_tokens==17
    assert all(block.attention.processor._query_chunk_tokens==17 for group in ('layers','noise_refiner','context_refiner')
               for block in getattr(value.transformer,group))
    assert any(not torch.equal(p,dict(value.transformer.named_parameters())[k]) for k,p in before.items())
    assert value.transformer._training_gradient_checkpointing_blocks==2
    assert metrics["ming_partition_count"]==4.
    assert any(p is branch.out.weight for group in value.optimizer.param_groups for p in group["params"])
    assert branch.out.weight.detach().abs().sum()>0
    from core.models.ming_image_01.loader import load_components
    from core.models.ming_image_01.checkpoint import save_bundle
    saved=value.output_dir/"run_step_000002"
    loaded=load_components(saved,dtype=torch.float32,conditioner_attention="eager")
    if swapped:
        pair=value.transformer.vae_latent_bridge
        assert pair.ready and loaded.transformer.vae_latent_bridge.ready
        assert loaded.vae.spatial_scale_factor==16 and loaded.vae.config.z_dim==24
        assert pair.declaration["noise_policy"]=="gaussian_row_orthogonal_v1"
        for key,buffer in pair.noise_alignment.state_dict().items():
            assert torch.equal(buffer,loaded.transformer.vae_latent_bridge.noise_alignment.state_dict()[key])
    payload=loaded.transformer._ming_partition_global_payload
    assert payload["metadata"]==branch.metadata()
    assert all(torch.equal(v,payload["state"][k]) for k,v in branch.export_tensors().items())
    assert not hasattr(loaded.transformer,"ming_partition_global_adapter")
    copied=save_bundle(loaded,tmp_path/"copied")
    reloaded=load_components(copied,dtype=torch.float32,conditioner_attention="eager")
    assert all(torch.equal(v,reloaded.transformer._ming_partition_global_payload["state"][k]) for k,v in payload["state"].items())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_sidecar_preserves_stored_precision_and_tensor_identity(tmp_path,dtype):
    from types import SimpleNamespace
    from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter,write_payload,payload_from_directory
    branch=MingPartitionGlobalAdapter(4,24,rank=8,summary_tokens=3,dtype=dtype)
    write_payload(SimpleNamespace(ming_partition_global_adapter=branch),tmp_path)
    payload=payload_from_directory(tmp_path,latent_channels=4,hidden_dim=24)
    assert all(v.dtype==dtype and torch.equal(v,branch.export_tensors()[k]) for k,v in payload['state'].items())


@pytest.mark.parametrize('config,message',[
    ({'dit_partition_global_adapter_enabled':False},'disabled'),
    ({'dit_partition_global_adapter_enabled':True,'dit_partition_training_enabled':False},'spatial'),
    ({'dit_partition_global_adapter_enabled':True,'dit_partition_training_enabled':True,'dit_partition_global_rank':4},'dimensions'),
])
def test_saved_global_payload_requires_matching_training_options(config,message):
    from types import SimpleNamespace
    from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter,prepare_full_training
    branch=MingPartitionGlobalAdapter(4,24,rank=8,summary_tokens=3)
    model=torch.nn.Linear(4,24);model.in_channels=4;model.dim=24
    model._ming_partition_global_payload={'metadata':branch.metadata(),'state':branch.export_tensors()}
    owner=SimpleNamespace(config=config,transformer=model,train_unet=True,weight_dtype=torch.float32)
    with pytest.raises(ValueError,match=message):prepare_full_training(owner)
    assert hasattr(model,'_ming_partition_global_payload') and not hasattr(model,'ming_partition_global_adapter')
