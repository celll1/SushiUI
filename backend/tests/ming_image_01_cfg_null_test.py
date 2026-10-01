"""Ming native zero CFG labels through public configuration and real shared loops."""
import pytest
import torch
from core.models.ming_image_01.conditioning import MingConditioning,apply_cfg_null_conditioning
from core.training.arch.ming_image_01 import MingImage01ArchHandler


def test_dual_stream_null_is_out_of_place_and_has_selected_row_gradients():
    q=torch.randn(2,4,8,requires_grad=True);d=torch.randn(2,3,24,requires_grad=True)
    original=MingConditioning(q,d);before=(q.detach().clone(),d.detach().clone())
    handler=MingImage01ArchHandler()
    null,aux=handler.apply_cfg_null_collated(None,original,'sentinel',torch.tensor([True,False]))
    assert aux=='sentinel' and torch.equal(q,before[0]) and torch.equal(d,before[1])
    assert null.queries.shape==q.shape and null.direct.shape==d.shape
    assert torch.equal(null.queries[0],original.zero().queries[0]) and torch.equal(null.direct[0],original.zero().direct[0])
    (null.queries.sum()+null.direct.sum()).backward()
    assert q.grad[0].count_nonzero()==0 and d.grad[0].count_nonzero()==0
    assert torch.equal(q.grad[1],torch.ones_like(q[1])) and torch.equal(d.grad[1],torch.ones_like(d[1]))
    later=apply_cfg_null_conditioning(original,torch.tensor([False,True]))
    assert torch.equal(later.queries[0],q[0]) and later.queries[1].count_nonzero()==0


@pytest.mark.parametrize('mask',[torch.ones(2),torch.ones(1,dtype=torch.bool)])
def test_invalid_null_label_refused(mask):
    with pytest.raises(ValueError,match='boolean'):
        apply_cfg_null_conditioning(MingConditioning(torch.ones(2,4,8),torch.ones(2,3,24)),mask)


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('method',['full_finetune','lora'])
def test_public_cfg_null_configuration_reaches_actual_mnt_training(tmp_path,monkeypatch,profile,method):
    from ming_image_01_conditioner_quantized_test import source
    from ming_image_01_conditioner_full_quantized_test import dataset
    from api.routes import TrainingRunCreateRequest,_check_cfg_null_params
    from api.arch_capabilities import CFG_NULL_STAGE_BY_ARCH
    from api.param_defaults import CFG_UNCOND_DROP_DEFAULTS_BY_ARCH
    from core.training.training_config import TrainingConfigGenerator
    from core.training.train_runner import _prepare_training_process_config
    from core.training.full_parameter_trainer import FullParameterTrainer
    from core.training.lora_trainer import LoRATrainer
    from core.training.ops import ming_image_01_ops
    import yaml
    root=source(tmp_path,profile)
    request=TrainingRunCreateRequest(training_method=method,base_model_path=str(root),cfg_uncond_drop_rate=1.,
        cfg_uncond_drop_per_mnt=True,ming_conditioner_groups=[],train_text_encoder=False,total_steps=2)
    _check_cfg_null_params(request,[])
    assert CFG_NULL_STAGE_BY_ARCH['ming_image_01']==MingImage01ArchHandler.cfg_null_stage=='collated'
    assert CFG_UNCOND_DROP_DEFAULTS_BY_ARCH['ming_image_01']==0.
    factory=TrainingConfigGenerator.generate_full_finetune_config if method=='full_finetune' else TrainingConfigGenerator.generate_lora_config
    serialized=factory(request.model_dump(),run_name='run',base_model_path=str(root),output_dir=str(tmp_path/'run'))
    _,config,_,resolved=_prepare_training_process_config(yaml.safe_load(serialized),str(root))
    assert resolved==method and config['cfg_uncond_drop_rate']==1.
    klass=FullParameterTrainer if method=='full_finetune' else LoRATrainer
    extra={} if method=='full_finetune' else {'lora_rank':2,'lora_alpha':2}
    value=klass(model_path=str(root),output_dir=str(tmp_path/'run'),device='cpu',weight_dtype='fp32',
        training_dtype='fp32',vae_dtype='fp32',mixed_precision=False,attention_backend='native',
        train_text_encoder=False,train_config=config,**extra)
    calls=[];original=ming_image_01_ops.flow_loss
    def record(transformer,**kwargs):
        condition=kwargs['conditioning']
        assert condition.queries.count_nonzero()==0 and condition.direct.count_nonzero()==0
        assert kwargs['cfg_drop_mask'] is None
        if profile=='layer_decompose':
            assert kwargs['reference_latents'][0] is not None
            assert kwargs['frame_weights'][0,0]==0 and kwargs['frame_weights'][0,1]>0
        calls.append(condition.queries.shape)
        return original(transformer,**kwargs)
    monkeypatch.setattr(ming_image_01_ops,'flow_loss',record)
    metrics=[];original_step=ming_image_01_ops.train_step
    def record_step(*args,**kwargs):
        result=original_step(*args,**kwargs)
        assert value._last_loss_per_sample is not None and torch.isfinite(value._last_loss_per_sample).all()
        metrics.append(value._last_loss_per_sample.clone())
        return result
    monkeypatch.setattr(ming_image_01_ops,'train_step',record_step)
    before={k:p.detach().clone() for k,p in value.transformer.named_parameters() if p.requires_grad}
    value.train([dataset(tmp_path,profile)],num_epochs=1,total_steps=2,batch_size=1,multi_noise_timesteps=2,
        save_every_n_steps=2,sample_every_n_steps=0,enable_bucketing=False,base_resolutions=[64],
        latent_encoding_mode='swap_onthefly',text_encoding_mode='swap_onthefly')
    assert len(calls)==2
    assert any(not torch.equal(p,dict(value.transformer.named_parameters())[k]) for k,p in before.items())
    assert len(metrics)==2
    assert not torch.cuda.is_initialized()
