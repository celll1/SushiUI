"""Native scheduler previews, primary Layer frame, bounded RGBA display and callback demand."""
from types import SimpleNamespace
import base64
import io
import pytest
import torch
from PIL import Image
from core.models.ming_image_01.generation_preview import scheduler_preview,primary_latent,decode_preview
from core.models.ming_image_01.style_generation import normalize_styles,prepare_styles
from core.models.ming_image_01.partition_generation import resolve_partition_plan
from api.generation_utils import preview_arch_kwargs,create_progress_callback_factory
from ming_image_01_style_generation_test import sampler,entries
from ming_image_01_sampling_test import condition,make_sampler


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('mode',['plain','style','partition'])
def test_native_scheduler_preview_matches_actual_guided_velocity_and_preserves_trajectory(profile,mode):
    torch.manual_seed(17);value=sampler(profile,rewired=True)
    frames=3 if profile=='layer_decompose' else 1
    args=dict(task='layer-decompose' if profile=='layer_decompose' else 'text-to-image',conditioning=condition(2),
        width=128,height=64,steps=2,cfg=2.,seed=71,requested_layers=2 if frames>1 else 1,output_type='latent')
    if frames>1:args['reference_pixels']=torch.randn(2,4,64,128)
    if mode=='style':
        args['styles']=prepare_styles(value.components.vae,normalize_styles(entries()),width=128,height=64,batch=2,seed=71,check_cancel=lambda:None)
        args['style_combine_mode']='stack'
    elif mode=='partition':args['partition_plan']=resolve_partition_plan({'mode':'auto','max_target_tokens':16*frames,'halo_tokens':0},4,8,frames)
    baseline=value.run(**args);native_steps=[];expected=[];previews=[];calls=[]
    handle=value.components.transformer.register_forward_pre_hook(lambda *a:calls.append(True))
    scheduler=value.pipe.scheduler;original=scheduler.step
    def oracle(model_output,timestep,sample,*a,**kw):
        sigma=scheduler.sigmas[len(expected)].to(sample)
        expected.append((sample-sigma*model_output).clone())
        return original(model_output,timestep,sample,*a,**kw)
    scheduler.step=oracle
    actual=value.run(**args,preview_callback=lambda i,n,t,x,p:previews.append((i,x.clone(),p.clone())),
        step_callback=lambda i,n,t,x:native_steps.append(x.clone()))
    handle.remove()
    assert scheduler.__dict__['step'] is oracle
    del scheduler.step
    for i,(_index,latent,predicted) in enumerate(previews):
        assert _index==i and torch.equal(latent,native_steps[i])
        torch.testing.assert_close(predicted,expected[i],rtol=0,atol=0)
        selected=primary_latent(predicted,profile)
        assert selected.shape==(2,16,8,16)
        if frames>1:assert torch.equal(selected,predicted[:,:,1])
    assert all(torch.equal(a,b) for group,other in zip(actual,baseline) for a,b in zip(group.values,other.values))
    count=len(calls);calls=[];handle=value.components.transformer.register_forward_pre_hook(lambda *a:calls.append(True))
    value.run(**args);handle.remove();assert len(calls)==count
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('failure',['callback','cancel'])
def test_native_preview_failure_restores_scheduler_and_next_request_recovers(failure):
    value=make_sampler('generation_edit');stop=[False];scheduler=value.pipe.scheduler
    original=scheduler.step.__func__;args=dict(task='text-to-image',conditioning=condition(),width=32,height=32,
        steps=2,cfg=2.,seed=7,requested_layers=1,output_type='latent')
    def preview(*a):
        if failure=='cancel':stop[0]=True
        else:raise RuntimeError('preview callback failed')
    with pytest.raises(RuntimeError,match='cancelled|preview callback failed'):
        value.run(**args,preview_callback=preview,cancelled=lambda:stop[0])
    assert 'step' not in scheduler.__dict__ and scheduler.step.__func__ is original
    assert not value.components.transformer._forward_pre_hooks
    stop[0]=False;assert value.run(**args)


def test_scheduler_prediction_owns_storage_and_restores_nested_instance_methods():
    class Scheduler:
        sigmas=torch.tensor([.75])
        def step(self,prediction,timestep,sample):return sample+.1*prediction
    scheduler=Scheduler();custom=lambda p,t,s:s+.1*p;scheduler.step=custom
    first={'index':0,'clean':None};second={'index':0,'clean':None}
    sample=torch.ones(2,16,4,4);prediction=torch.ones_like(sample)*2
    with scheduler_preview(scheduler,first):
        outer=scheduler.step
        with scheduler_preview(scheduler,second):scheduler.step(prediction,0,sample)
        assert scheduler.step is outer and second['clean'] is None
        torch.testing.assert_close(first['clean'],torch.full_like(sample,-.5),rtol=0,atol=0)
        first['clean'].add_(99);assert sample.eq(1).all() and prediction.eq(2).all()
    assert scheduler.step is custom and first['clean'] is None


def test_declined_predicted_clean_steps_keep_native_output_without_snapshot():
    value=make_sampler('generation_edit');seen=[]
    def preview(i,n,t,x,p):seen.append((i,p))
    preview.wants_predicted_x0=lambda i,n:i==n-1
    value.run(task='text-to-image',conditioning=condition(),width=32,height=32,steps=3,cfg=1.,seed=7,
        requested_layers=1,output_type='latent',preview_callback=preview)
    assert [item[1] is None for item in seen]==[True,True,False]


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
def test_real_native_preview_is_bounded_rgb_checkerboard_without_weight_or_rng_changes(profile):
    vae=make_sampler(profile).components.vae
    before={k:v.clone() for k,v in vae.state_dict().items()};rng=torch.get_rng_state().clone()
    latent=torch.zeros(2,16,48,64)
    image=decode_preview(vae,latent)
    assert image.mode=='RGB' and image.size==(256,192)
    assert all(torch.equal(v,vae.state_dict()[k]) for k,v in before.items())
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()


def test_transparent_native_preview_uses_display_checkerboard():
    def decode(value,return_dict):return (torch.cat([torch.ones(1,3,1,16,16),-torch.ones(1,1,1,16,16)],dim=1),)
    vae=SimpleNamespace(device='cpu',dtype=torch.float32,config=SimpleNamespace(shift_factor=0.,scaling_factor=1.),decode=decode)
    image=decode_preview(vae,torch.zeros(1,16,2,2))
    assert image.getpixel((0,0))==(235,235,235) and image.getpixel((8,0))==(190,190,190)


@pytest.mark.parametrize('shape,profile',[( (1,4,4,4),'generation_edit'),((1,16,1,4,4),'layer_decompose'),((1,16,4),'generation_edit')])
def test_primary_frame_refuses_incompatible_layout(shape,profile):
    with pytest.raises(ValueError):primary_latent(torch.zeros(shape),profile)


def test_public_factory_uses_native_predicted_clean_and_respects_interval_without_transport(monkeypatch):
    from ming_image_01_pipeline_test import manager,params
    owner,components=manager('generation_edit')
    kwargs=preview_arch_kwargs(owner,None)
    assert kwargs['preview_predicted_x0'] and callable(kwargs['native_preview_decoder'])
    class WS:
        def __init__(self):self.messages=[]
        def send_progress_sync(self,*a,**k):self.messages.append((a,k))
    class TAE:
        def decode_latent(self,*a,**k):raise AssertionError('Ming used another architecture preview decoder')
    import api.generation_status as status
    monkeypatch.setattr(status,'update_progress',lambda *a,**k:None)
    ws=WS();callback=create_progress_callback_factory(TAE(),ws,**kwargs,preview_interval=2)
    calls=[];native=kwargs['native_preview_decoder']
    def observe(latent):calls.append(latent.clone());return native(latent)
    kwargs['native_preview_decoder']=observe
    callback=create_progress_callback_factory(TAE(),ws,**kwargs,preview_interval=2)
    output,_,_=owner.generate_txt2img(params(steps=4,cfg_scale=2.),progress_callback=callback)
    assert len(ws.messages)==4 and len(calls)==3
    assert [bool(item[1]['preview_image']) for item in ws.messages]==[True,False,True,True]
    for _args,message in ws.messages:
        if message['preview_image']:
            with Image.open(io.BytesIO(base64.b64decode(message['preview_image']))) as image:assert image.mode=='RGB'
    assert output.primary.mode=='RGBA' and not torch.cuda.is_initialized()
