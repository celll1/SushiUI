"""Native Ming guidance oracles, frame supervision and real public training."""
from types import SimpleNamespace
import pytest
import torch
from core.models.ming_image_01.guidance import prepare_guidance,guidance_loss


def test_bf16_null_forward_does_not_poison_trainable_weight_cast_cache(monkeypatch):
    from core.training.ops import ming_image_01_ops
    from core.models.ming_image_01.conditioning import MingConditioning
    model=torch.nn.Linear(4,4)
    calls=[]
    def objective(transformer,*,clean,**kwargs):
        prediction=transformer(clean.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        calls.append(prediction.requires_grad)
        target=torch.ones_like(prediction)
        per_item=(prediction.float()-target.float()).square().flatten(1).mean(1)
        return SimpleNamespace(prediction=prediction,target=target,per_item_loss=per_item,loss=per_item.mean())
    monkeypatch.setattr(ming_image_01_ops,'flow_loss',objective)
    trainer=SimpleNamespace(device='cpu',training_dtype=torch.bfloat16,mixed_precision=True,
        transformer=model,reconstruction_loss_weight=0.,log_extra_metric=lambda *args:None,
        config={'qwen_guidance_loss_weight':1.,'qwen_guidance_loss_mix_mode':'blend',
                'qwen_guidance_loss_scale':2.,'qwen_guidance_loss_schedule':'constant'})
    batch=SimpleNamespace(clean=torch.randn(1,4,1,4,4),reference_latents=None,
        conditioning=MingConditioning(torch.ones(1,2,4),torch.ones(1,2,4)),frame_weights=torch.ones(1,1))
    loss,_,_=ming_image_01_ops.train_step(trainer,batch,timesteps=torch.tensor([.5]))
    assert calls==[False,True] and loss.requires_grad
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>0 for p in model.parameters())


@pytest.mark.parametrize('schedule',['constant','sigma'])
@pytest.mark.parametrize('cap',[0.,.25])
def test_frame_weighted_guidance_matches_independent_dense_oracle(schedule,cap):
    sigma=torch.tensor([.2,.8]);weights=torch.tensor([[0.,1.,2.],[0.,1.,1.]])
    prediction=torch.randn(2,16,3,4,4,requires_grad=True);target=torch.randn_like(prediction)
    uncond=torch.randn_like(prediction);uncond[:,:,0]=10000
    normal_frames=(prediction-target).square().mean(dim=(1,3,4))
    normal=(normal_frames*weights).sum(1)/weights.sum(1)
    result=SimpleNamespace(prediction=prediction,target=target,per_item_loss=normal,loss=normal.mean())
    metrics={};trainer=SimpleNamespace(config={'qwen_guidance_loss_weight':.4,'qwen_guidance_loss_mix_mode':'blend',
        'qwen_guidance_loss_scale':3.,'qwen_guidance_loss_schedule':schedule,
        'qwen_guidance_loss_extrapolation_max_rms':cap},log_extra_metric=lambda k,v:metrics.update({k:v}))
    condition,selection,selected=prepare_guidance(trainer,sigma,None)
    loss,per_item=guidance_loss(trainer,result,uncond,sigma,weights,condition,selection,selected)
    scale=torch.full_like(sigma,3.) if schedule=='constant' else 1+2*sigma
    if cap:
        squared=(sigma[:,None,None,None,None]*(target-uncond)).square().mean(dim=(1,3,4))
        rms=((squared*weights).sum(1)/weights.sum(1)).sqrt()
        scale=1+torch.minimum(scale-1,cap/rms)
    guided=uncond+scale[:,None,None,None,None]*(target-uncond)
    expected_guided=((prediction-guided).square().mean(dim=(1,3,4))*weights).sum(1)/weights.sum(1)
    expected=.6*normal+.4*expected_guided
    torch.testing.assert_close(per_item,expected);torch.testing.assert_close(loss,expected.mean())
    loss.backward()
    assert prediction.grad[:,:,0].count_nonzero()==0 and prediction.grad[:,:,1:].abs().sum()>0
    assert metrics['ming_guidance_loss_selected_fraction']==1.


def test_stochastic_guidance_excludes_native_null_labels_and_shares_draw():
    trainer=SimpleNamespace(config={'qwen_guidance_loss_weight':.5,'qwen_guidance_loss_mix_mode':'stochastic'},
        _qwen_guidance_draw=torch.tensor([.1,.1,.9]))
    _,selection,selected=prepare_guidance(trainer,torch.tensor([.2,.5,.8]),torch.tensor([False,True,False]))
    assert torch.equal(selected,torch.tensor([True,False,False]))
    assert torch.equal(selection,selected.float())


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('method',['full_finetune','lora'])
@pytest.mark.parametrize('mode',['blend','stochastic'])
def test_public_guidance_objective_trains_native_profiles(tmp_path,monkeypatch,profile,method,mode):
    from ming_image_01_conditioner_quantized_test import source
    from ming_image_01_conditioner_full_quantized_test import dataset
    from api.routes import TrainingRunCreateRequest
    from core.training.training_config import TrainingConfigGenerator
    from core.training.train_runner import _prepare_training_process_config
    from core.training.full_parameter_trainer import FullParameterTrainer
    from core.training.lora_trainer import LoRATrainer
    from core.training.ops import ming_image_01_ops
    import yaml
    root=source(tmp_path,profile)
    request=TrainingRunCreateRequest(training_method=method,base_model_path=str(root),train_text_encoder=False,
        ming_conditioner_groups=[],qwen_guidance_loss_weight=1.,qwen_guidance_loss_mix_mode=mode,
        qwen_guidance_loss_schedule='sigma',qwen_guidance_loss_extrapolation_max_rms=.5,total_steps=2,seed=17)
    factory=TrainingConfigGenerator.generate_full_finetune_config if method=='full_finetune' else TrainingConfigGenerator.generate_lora_config
    serialized=factory(request.model_dump(),run_name='run',base_model_path=str(root),output_dir=str(tmp_path/'run'))
    _,config,_,resolved=_prepare_training_process_config(yaml.safe_load(serialized),str(root))
    assert resolved==method and config['qwen_guidance_loss_weight']==1.
    klass=FullParameterTrainer if method=='full_finetune' else LoRATrainer
    extra={} if method=='full_finetune' else {'lora_rank':2,'lora_alpha':2}
    value=klass(model_path=str(root),output_dir=str(tmp_path/'run'),device='cpu',weight_dtype='fp32',
        training_dtype='fp32',vae_dtype='fp32',mixed_precision=False,attention_backend='native',
        train_text_encoder=False,train_config=config,**extra)
    calls=[];original=ming_image_01_ops.flow_loss;metrics={}
    old_metric=value.log_extra_metric
    def metric(k,v):metrics[k]=v;old_metric(k,v)
    monkeypatch.setattr(value,'log_extra_metric',metric)
    def record(transformer,**kwargs):
        calls.append((torch.is_grad_enabled(),kwargs['noise'].clone(),kwargs['sigma'].clone()))
        if not torch.is_grad_enabled():
            assert kwargs['conditioning'].queries.count_nonzero()==0 and kwargs['conditioning'].direct.count_nonzero()==0
        if profile=='layer_decompose':assert kwargs['reference_latents'][0] is not None and kwargs['frame_weights'][0,0]==0
        return original(transformer,**kwargs)
    monkeypatch.setattr(ming_image_01_ops,'flow_loss',record)
    before={k:p.detach().clone() for k,p in value.transformer.named_parameters() if p.requires_grad}
    value.train([dataset(tmp_path,profile)],num_epochs=1,total_steps=2,batch_size=1,multi_noise_timesteps=2,
        save_every_n_steps=2,sample_every_n_steps=0,enable_bucketing=False,base_resolutions=[64],
        latent_encoding_mode='swap_onthefly',text_encoding_mode='swap_onthefly')
    assert [item[0] for item in calls]==[False,True,False,True]
    for a,b in zip(calls[::2],calls[1::2]):assert torch.equal(a[1],b[1]) and torch.equal(a[2],b[2])
    assert any(not torch.equal(p,dict(value.transformer.named_parameters())[k]) for k,p in before.items())
    assert metrics['ming_guidance_loss_selected_fraction']==1. and all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
def test_sigma_null_shared_draw_selects_guided_null_and_ordinary_paths(tmp_path,monkeypatch,profile):
    from ming_image_01_conditioner_quantized_test import source
    from ming_image_01_conditioner_full_quantized_test import dataset
    from core.training.full_parameter_trainer import FullParameterTrainer
    from core.training.ops import qwen_image_21_ops,ming_image_01_ops
    root=source(tmp_path,profile)
    value=FullParameterTrainer(model_path=str(root),output_dir=str(tmp_path/'run'),device='cpu',weight_dtype='fp32',
        training_dtype='fp32',vae_dtype='fp32',mixed_precision=False,attention_backend='native',train_text_encoder=False,
        train_config={'ming_conditioner_groups':[],'seed':17,'qwen_guidance_loss_weight':.5,
            'qwen_guidance_loss_mix_mode':'stochastic','qwen_cfg_null_sigma_schedule':True,
            'cfg_uncond_drop_rate':.5,'cfg_uncond_drop_per_mnt':True})
    draws=iter([.1,.6,.95,.2]);old_sigma=qwen_image_21_ops.sigma_null_drop_mask
    def sigma_draw(*args):
        with monkeypatch.context() as scope:
            scope.setattr(torch,'rand_like',lambda x:torch.full_like(x,next(draws)))
            return old_sigma(*args)
    monkeypatch.setattr(qwen_image_21_ops,'sigma_null_drop_mask',sigma_draw)
    labels=[];old_step=ming_image_01_ops.train_step
    def step(*args,**kwargs):
        labels.append(kwargs['guidance_drop_mask'].clone())
        return old_step(*args,**kwargs)
    monkeypatch.setattr(ming_image_01_ops,'train_step',step)
    forwards=[];old_loss=ming_image_01_ops.flow_loss
    def loss(*args,**kwargs):
        is_grad=torch.is_grad_enabled();forwards.append(is_grad)
        if not is_grad or bool(labels[-1].any()):
            assert kwargs['conditioning'].queries.count_nonzero()==0 and kwargs['conditioning'].direct.count_nonzero()==0
        return old_loss(*args,**kwargs)
    monkeypatch.setattr(ming_image_01_ops,'flow_loss',loss)
    value.train([dataset(tmp_path,profile)],num_epochs=2,total_steps=4,batch_size=1,multi_noise_timesteps=2,
        save_every_n_steps=4,sample_every_n_steps=0,enable_bucketing=False,base_resolutions=[64],
        latent_encoding_mode='swap_onthefly',text_encoding_mode='swap_onthefly')
    assert [bool(mask[0]) for mask in labels]==[False,True,False,False]
    assert forwards==[False,True,True,True,False,True]
    assert not torch.cuda.is_initialized()


def test_guidance_bf16_native_loop_preserves_prediction_precision(tmp_path,monkeypatch):
    from ming_image_01_conditioner_quantized_test import source
    from ming_image_01_conditioner_full_quantized_test import dataset
    from core.training.full_parameter_trainer import FullParameterTrainer
    from core.training.ops import ming_image_01_ops
    root=source(tmp_path,'generation_edit')
    value=FullParameterTrainer(model_path=str(root),output_dir=str(tmp_path/'run'),device='cpu',weight_dtype='bf16',
        training_dtype='bf16',vae_dtype='fp32',mixed_precision=False,attention_backend='native',train_text_encoder=False,
        train_config={'ming_conditioner_groups':[],'seed':17,'qwen_guidance_loss_weight':1.,
            'qwen_guidance_loss_mix_mode':'blend','qwen_guidance_loss_extrapolation_max_rms':.5,'unet_lr':.01})
    outputs=[];old=ming_image_01_ops.flow_loss
    def record(*args,**kwargs):
        result=old(*args,**kwargs);outputs.append(result.prediction.dtype)
        assert torch.isfinite(result.prediction).all()
        return result
    monkeypatch.setattr(ming_image_01_ops,'flow_loss',record)
    before=value.transformer.layers[0].attention.to_q.weight.detach().clone()
    value.train([dataset(tmp_path,'generation_edit')],num_epochs=1,total_steps=2,batch_size=1,multi_noise_timesteps=2,
        save_every_n_steps=2,sample_every_n_steps=0,enable_bucketing=False,base_resolutions=[64],
        latent_encoding_mode='swap_onthefly',text_encoding_mode='swap_onthefly')
    assert len(outputs)==4 and outputs[::2]==outputs[1::2]
    assert not torch.equal(before,value.transformer.layers[0].attention.to_q.weight)
    assert not torch.cuda.is_initialized()


def test_guidance_refuses_trainable_conditioner_before_any_forward(tmp_path,monkeypatch):
    from ming_image_01_conditioner_quantized_test import source
    from ming_image_01_conditioner_full_quantized_test import dataset,owner
    from core.training.ops import ming_image_01_ops
    root=source(tmp_path,'generation_edit')
    value=owner(root,tmp_path/'run',['queries'],config={'qwen_guidance_loss_weight':1.})
    def forbidden(*a,**k):raise AssertionError('Guidance reached a forward with trainable conditioner')
    monkeypatch.setattr(ming_image_01_ops,'flow_loss',forbidden)
    with pytest.raises(ValueError,match='frozen text encoder'):
        value.train([dataset(tmp_path,'generation_edit')],num_epochs=1,total_steps=1,batch_size=1,
            save_every_n_steps=1,sample_every_n_steps=0,enable_bucketing=False,base_resolutions=[64],
            latent_encoding_mode='swap_onthefly',text_encoding_mode='swap_onthefly')
    assert not torch.cuda.is_initialized()


def test_zero_low_weight_high_noise_ramp_still_selects_guidance():
    trainer=SimpleNamespace(config={'qwen_guidance_loss_weight':0.,'qwen_guidance_loss_mix_mode':'stochastic',
        'qwen_guidance_loss_weight_schedule':'high_noise_smoothstep','qwen_guidance_loss_high_noise_weight':1.,
        'qwen_guidance_loss_ramp_start':.2,'qwen_guidance_loss_ramp_end':.8},_qwen_guidance_draw=torch.full((3,),.1))
    condition,_,selected=prepare_guidance(trainer,torch.tensor([0.,.5,1.]),None)
    assert condition is not None and torch.equal(selected,torch.tensor([False,True,True]))


@pytest.mark.parametrize('weight',[0.,1.])
def test_public_guidance_rejects_trainable_conditioner_before_payload(tmp_path,monkeypatch,weight):
    from ming_image_01_conditioner_quantized_test import source
    from api.routes import TrainingRunCreateRequest,_check_ming_conditioner_params
    from fastapi import HTTPException
    root=source(tmp_path,'generation_edit')
    request=TrainingRunCreateRequest(training_method='full_finetune',base_model_path=str(root),
        ming_conditioner_groups=['queries'],qwen_guidance_loss_weight=weight,
        qwen_guidance_loss_weight_schedule='high_noise_smoothstep',qwen_guidance_loss_high_noise_weight=1.)
    def forbidden(*a,**k):raise AssertionError('Guidance ownership gate loaded weights')
    monkeypatch.setattr('safetensors.torch.load_file',forbidden)
    with pytest.raises(HTTPException,match='frozen text encoder'):_check_ming_conditioner_params(request)
