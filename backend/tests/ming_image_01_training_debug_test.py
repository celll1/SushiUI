"""Native debug math/frames, actual backward invariance and deferred VAE ownership."""
from copy import deepcopy
from types import SimpleNamespace
import inspect
from pathlib import Path
import pytest
import torch
from PIL import Image
from core.models.ming_image_01.training_debug import save_debug,flush_pending_debug_previews
from core.models.ming_image_01.training_data import MingTrainingBatch
from core.models.ming_image_01.conditioning import MingConditioning
from core.models.ming_image_01.partition import MingPartitionAdapter
from core.models.ming_image_01.partition_training import backward_regions
from core.models.ming_image_01.model_definition import create_transformer
from core.training.ops.ming_image_01_ops import train_step
from ming_image_01_rewiring_test import config,spec_for,restore_topology
from ming_image_01_sampling_test import make_sampler


def fixture(profile,guided):
    torch.manual_seed(19)
    model=create_transformer(config('learned' if profile=='layer_decompose' else 'zero_masked'),device='cpu',dtype=torch.float32)
    if profile=='layer_decompose':
        torch.nn.init.constant_(model.x_pad_token,.02);torch.nn.init.constant_(model.cap_pad_token,.03)
    restore_topology(model,spec_for(model))
    frames=3 if profile=='layer_decompose' else 1
    clean=torch.randn(2,16,frames,16,16);noise=torch.randn_like(clean)
    weights=torch.tensor([[0.,1.,2.],[0.,2.,1.]]) if frames==3 else torch.ones(2,1)
    batch=MingTrainingBatch(clean,MingConditioning(torch.randn(2,5,8),torch.randn(2,3,24)),weights,
        [torch.randn(16,1,16,16),torch.randn(16,2,16,16)])
    metrics={};settings={'qwen_debug_latent_view':'pixel','qwen_guidance_loss_weight':.4 if guided else 0.,
        'qwen_guidance_loss_mix_mode':'blend','qwen_guidance_loss_scale':4.,'qwen_guidance_loss_schedule':'sigma',
        'qwen_guidance_loss_extrapolation_max_rms':.25}
    owner=SimpleNamespace(transformer=model,device='cpu',training_dtype=torch.float32,mixed_precision=False,
        vae=make_sampler(profile).components.vae,vae_dtype=torch.float32,config=settings,
        reconstruction_loss_weight=.3,_active_mnt_noise=noise,log_extra_metric=lambda k,v:metrics.__setitem__(k,v))
    plan=MingPartitionAdapter().build_fixed_plan(8,8,count=4,halo=0,split_ratio_min=.5,split_ratio_max=.5)
    return owner,batch,noise,torch.tensor([.2,.7]),plan


def filmstrip(x):
    return torch.cat([frame[:1].float().cpu() for frame in x.unbind(2)],dim=-1)


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('partition',[False,True])
@pytest.mark.parametrize('guided',[False,True])
def test_actual_native_debug_snapshot_matches_core_oracle_without_changing_backward(tmp_path,profile,partition,guided):
    owner,batch,noise,sigma,plan=fixture(profile,guided);baseline=deepcopy(owner);trace=[]
    def observe(module,args,kwargs,output):trace.append(torch.stack([value[:,:batch.clean.shape[2]] for value in output[0]]).detach().clone())
    handle=owner.transformer.register_forward_hook(observe,with_kwargs=True)
    before=torch.get_rng_state().clone()
    if partition:
        actual=backward_regions(owner,batch,sigma=sigma,noise=noise,plan=plan,backward_scale=.125,
            debug_save_path=tmp_path,debug_captions=['word5','word6'],debug_reference_image_paths=['ref.png',None])
        expected=backward_regions(baseline,batch,sigma=sigma,noise=noise,plan=plan,backward_scale=.125)
    else:
        values=train_step(owner,batch,timesteps=sigma,debug_save_path=tmp_path,
            debug_captions=['word5','word6'],debug_reference_image_paths=['ref.png',None])
        (.125*values[0]).backward();actual=tuple(float(x) for x in values)
        other=train_step(baseline,batch,timesteps=sigma);(.125*other[0]).backward();expected=tuple(float(x) for x in other)
    handle.remove()
    assert torch.equal(before,torch.get_rng_state()) and actual==expected
    for (name,a),(_,b) in zip(owner.transformer.named_parameters(),baseline.transformer.named_parameters()):
        if a.grad is None:assert b.grad is None
        else:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    data=torch.load(tmp_path/'latents_t0.2000.pt',weights_only=True)
    noisy=(1-sigma[:,None,None,None,None])*batch.clean+sigma[:,None,None,None,None]*noise
    predicted=trace[-1][:1]
    null=trace[0][:1] if guided else None
    if partition:
        def assemble(outputs):
            canvas=torch.zeros_like(batch.clean[:1])
            for region,value in zip(plan.regions,outputs):
                core,local=region.core,region.core_in_input
                canvas[...,2*core.top:2*core.bottom,2*core.left:2*core.right]=value[:1,...,2*local.top:2*local.bottom,2*local.left:2*local.right]
            return canvas
        predicted=assemble(trace[-4:]);null=assemble(trace[:4]) if guided else None
    torch.testing.assert_close(data['predicted_velocity'],filmstrip(predicted),rtol=0,atol=0)
    torch.testing.assert_close(data['actual_velocity'],filmstrip(batch.clean-noise),rtol=0,atol=0)
    torch.testing.assert_close(data['predicted_latent'],filmstrip(noisy[:1]+float(sigma[0])*predicted),rtol=0,atol=0)
    if guided:
        assert data['uncond_prediction_status']=='computed'
        torch.testing.assert_close(data['uncond_predicted_latent'],filmstrip(noisy[:1]+float(sigma[0])*null),rtol=0,atol=0)
    assert data['velocity_sign']=='clean_minus_noise' and data['timestep_convention']=='sigma'
    assert data['caption']=='word5' and data['reference_image_path']=='ref.png'
    assert data['frame_weights']==batch.frame_weights[0].tolist()
    assert data['frame_roles']==(['image'] if profile=='generation_edit' else ['auxiliary_canvas','layer_1','layer_2'])
    if partition:
        assert data['partition_count']==4 and data['partition_grid_size']==[16*batch.clean.shape[2],16]
        assert len(data['partition_regions'])==4*batch.clean.shape[2]
    assert all(not p.exists() for p in tmp_path.glob('*.webp'))
    original=next(owner.vae.parameters());device,dtype,training=original.device,original.dtype,owner.vae.training
    flush_pending_debug_previews(owner)
    assert not hasattr(owner,'_pending_ming_debug_previews')
    assert next(owner.vae.parameters()).device==device and next(owner.vae.parameters()).dtype==dtype and owner.vae.training==training
    with Image.open(tmp_path/'decode_t0.2000_target.webp') as image:
        assert image.mode=='RGBA' and image.size==(128*batch.clean.shape[2],128)
    assert len(list(tmp_path.glob('*.webp')))==(4 if guided else 3)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('failure',['decode','save'])
def test_pixel_preview_failure_releases_pending_state_and_restores_vae(tmp_path,monkeypatch,failure):
    owner,batch,noise,sigma,plan=fixture('generation_edit',False)
    save_debug(owner,tmp_path,clean=batch.clean,noisy=noise,target=batch.clean-noise,prediction=noise,
        sigma=sigma,frame_weights=batch.frame_weights,loss=1.,recon_loss=2.)
    owner.vae.train();parameter=next(owner.vae.parameters());dtype=parameter.dtype;device=parameter.device
    def refuse(*a,**k):raise RuntimeError('forced debug failure')
    if failure=='decode':monkeypatch.setattr(owner.vae,'decode',refuse)
    else:monkeypatch.setattr(Image.Image,'save',refuse)
    with pytest.raises(RuntimeError,match='forced debug'):flush_pending_debug_previews(owner)
    assert not hasattr(owner,'_pending_ming_debug_previews') and owner.vae.training
    assert next(owner.vae.parameters()).dtype==dtype and next(owner.vae.parameters()).device==device
    assert not torch.cuda.is_initialized()


def test_no_debug_path_does_not_inspect_or_allocate_tensors():
    save_debug(None,None,clean=None,noisy=None,target=None,prediction=None,sigma=None,
               frame_weights=None,loss=None,recon_loss=None)
    flush_pending_debug_previews(SimpleNamespace())


def test_shared_dispatch_and_public_metadata_static_contract():
    from core.training.base_trainer import BaseTrainer
    from core.training.arch.ming_image_01 import MingImage01ArchHandler
    source=inspect.getsource(BaseTrainer._execute_forward_backward)
    assert 'debug_save_path=debug_save_path' in source and 'ming_image_01.training_debug import flush_pending_debug_previews' in source
    source=inspect.getsource(BaseTrainer._execute_ming_partitioned_forward_backward)
    assert 'debug_captions=debug_captions' in source and 'flush_pending_debug_previews(self)' in source
    assert 'debug_save_path=ctx.debug_save_path' in inspect.getsource(MingImage01ArchHandler.train_step)
    root=Path(__file__).resolve().parents[2]
    routes=(root/'backend/api/routes.py').read_text(encoding='utf-8')
    assert '"RGBA" if data.get("model_type") == "ming_image_01"' in routes
    assert '"native_frame_count", "frame_roles"' in routes
    frontend=(root/'frontend/src/components/training/TrainingConfig.tsx').read_text(encoding='utf-8')
    assert 'debugLatents && (isQwenImage21Model || isMingImage01Model)' in frontend


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('partition',[False,True])
def test_shared_base_dispatch_decodes_only_after_backward(tmp_path,monkeypatch,profile,partition):
    from types import MethodType
    from core.training.base_trainer import BaseTrainer
    from ming_image_01_shared_step_test import shared_owner,prepared,step_args
    from ming_image_01_training_batch_test import components
    value=components(profile);owner=shared_owner(value)
    owner.config={'qwen_debug_latent_view':'pixel','dit_partition_training_enabled':partition,
                  'dit_partition_mode':'fixed','dit_partition_fixed_count':2,'dit_partition_halo_tokens':0,
                  'dit_partition_split_ratio_min':.5,'dit_partition_split_ratio_max':.5}
    owner.vae=value.vae;owner.vae_dtype=torch.float32;owner.log_prefix='[MingTest]'
    owner.log_extra_metric=lambda *a,**k:None
    owner._execute_ming_partitioned_forward_backward=MethodType(BaseTrainer._execute_ming_partitioned_forward_backward,owner)
    frames=3 if profile=='layer_decompose' else 1
    batch=prepared(value,torch.zeros(3,4,frames,64,64));noise=torch.randn_like(batch.clean)
    args=step_args(batch,noise);args['debug_save_path']=tmp_path
    original=value.vae.decode;observed=[]
    def after_backward(*a,**k):
        assert any(p.grad is not None for p in value.transformer.parameters())
        assert not torch.is_grad_enabled()
        observed.append(True);return original(*a,**k)
    monkeypatch.setattr(value.vae,'decode',after_backward)
    owner._execute_forward_backward(**args)
    assert observed and (tmp_path/'decode_t0.2000_target.webp').exists()
    assert not hasattr(owner,'_pending_ming_debug_previews')
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mode',['RGB','RGBA'])
def test_bounded_media_preview_preserves_native_alpha(tmp_path,mode):
    from api.training_media import cached_image_preview
    source=tmp_path/'source.webp'
    Image.new(mode,(128,128),(70,90,110,120) if mode=='RGBA' else (70,90,110)).save(source,'WEBP',lossless=True)
    path,_=cached_image_preview(source,tmp_path/'cache',256)
    with Image.open(path) as image:
        assert image.mode==mode
        if mode=='RGBA':assert image.getpixel((0,0))[3]==120
