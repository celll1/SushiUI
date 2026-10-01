"""Native CPU scheduler/CFG/style lifecycle; public API routing remains separate."""
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from core.inference.reference_style import StyleContext,StyleTransferConfig
from core.models.ming_image_01.style_generation import PreparedStyle,MingStyleExecutor,normalize_styles,prepare_styles
from core.models.ming_image_01.style_transfer import style_scope
from core.models.ming_image_01.sampling import MingSampler,MingGenerationCancelled
from core.models.ming_image_01.model_definition import create_transformer
from ming_image_01_rewiring_test import config,spec_for,restore_topology
from ming_image_01_sampling_test import make_sampler,condition


def sampler(profile,rewired=False):
    value=make_sampler(profile)
    if rewired:
        cfg=config('learned' if profile=='layer_decompose' else 'zero_masked')
        model=create_transformer(cfg,device='cpu',dtype=torch.float32).eval()
        if profile=='layer_decompose':
            torch.nn.init.constant_(model.x_pad_token,.02);torch.nn.init.constant_(model.cap_pad_token,.03)
        restore_topology(model,spec_for(model))
        value.components.transformer=model;value.components.artifact.configs['transformer']=cfg
        value=MingSampler(value.components)
    return value


def entries():
    return [{'image':Image.new('RGBA',(64,64),(120,40,60,180)),'ref_k_strength':.4,'adain_strength':.3,'start_step':0,'end_step':500},
            {'image':Image.new('RGBA',(64,64),(20,90,110,140)),'ref_k_strength':.3,'adain_strength':.2,'start_step':500,'end_step':1000}]


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('cfg',[1.,2.])
@pytest.mark.parametrize('combine',['stack','common_concept'])
def test_native_sampler_multistyle_cfg_step_gates_nested_unet_and_seed_reuse(profile,cfg,combine):
    torch.manual_seed(17);value=sampler(profile,rewired=True)
    styles=prepare_styles(value.components.vae,normalize_styles(entries()),width=64,height=64,batch=2,seed=71,check_cancel=lambda:None)
    args=dict(task='layer-decompose' if profile=='layer_decompose' else 'text-to-image',conditioning=condition(2),width=64,height=64,steps=3,cfg=cfg,seed=71,requested_layers=2 if profile=='layer_decompose' else 1,output_type='latent')
    if profile=='layer_decompose':args['reference_pixels']=torch.randn(2,4,64,64)
    processors=[block.attention.processor for block in value.components.transformer.layers]
    snapshot={key:tensor.clone() for key,tensor in value.components.transformer.state_dict().items()}
    modes=[]
    def observe(module,inputs):
        info=getattr(module.layers[0].attention,'_ming_style_info',None)
        modes.append(None if info is None else info['context'].mode)
    handle=value.components.transformer.register_forward_pre_hook(observe)
    actual=value.run(**args,styles=styles,style_combine_mode=combine)
    handle.remove()
    expected_modes=['capture','inject']+([None] if cfg>1 else [])+['capture','capture','inject']+([None] if cfg>1 else [])+['capture','inject']+([None] if cfg>1 else [])
    assert modes==expected_modes
    repeated=value.run(**args,styles=styles,style_combine_mode=combine)
    baseline=value.run(**args)
    assert all(torch.equal(a,b) for group,other in zip(actual,repeated) for a,b in zip(group.values,other.values))
    assert any(not torch.equal(a,b) for group,other in zip(actual,baseline) for a,b in zip(group.values,other.values))
    assert all(torch.isfinite(item).all() for group in actual for item in group.values)
    assert value.pipe.transformer.style_executor is None
    assert all(block.attention.processor is processor for block,processor in zip(value.components.transformer.layers,processors))
    assert all(torch.equal(tensor,value.components.transformer.state_dict()[key]) for key,tensor in snapshot.items())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
def test_executor_conditional_only_prediction_matches_independent_native_oracle(profile):
    torch.manual_seed(23);value=sampler(profile,rewired=True);model=value.components.transformer
    frames=3 if profile=='layer_decompose' else 1;batch=2
    x=[torch.randn(16,frames,8,8) for _ in range(4)]
    x[2:]=[item.clone() for item in x[:2]]
    caps=[torch.randn(5,8) for _ in range(batch)];direct=[torch.randn(3,24),torch.randn(7,24)]
    caps+= [torch.zeros_like(item) for item in caps];direct += [torch.zeros_like(item) for item in direct]
    refs=[torch.randn(16,1,8,8),torch.randn(16,2,8,8)];refs+= [item.clone() for item in refs]
    t=torch.full((4,),.4);cfg=StyleTransferConfig(block_range=(0,5),ref_k_strength=.4,adain_strength=.3)
    style=PreparedStyle(torch.randn(2,16,8,8),torch.randn(2,16,8,8),cfg)
    scheduler=SimpleNamespace(sigmas=torch.tensor([.6,0.]))
    executor=MingStyleExecutor(model,scheduler,[style],batch=2,total=1,combine_mode='stack',cancelled=lambda:False)
    cond=dict(t=t[:2],cap_feats=caps[:2],cap_feats_2=direct[:2],ref_x=refs[:2],return_dict=False)
    with torch.no_grad():
        sigma=scheduler.sigmas[0]
        noisy=((1-sigma)*style.clean+sigma*style.noise).unsqueeze(2).expand(-1,-1,frames,-1,-1)
        captured=StyleContext('capture',cfg)
        with style_scope(model,captured,list(noisy.unbind())):model(x=list(noisy.unbind()),**cond)
        injected=StyleContext('inject',cfg,refs=[(captured.store,cfg)],combine_mode='stack')
        with style_scope(model,injected,x[:2]):positive=model(x=x[:2],**cond)[0]
        negative=model(x=x[2:],t=t[2:],cap_feats=caps[2:],cap_feats_2=direct[2:],ref_x=refs[2:],return_dict=False)[0]
        actual=executor.predict(x,t,caps,encoder_hidden_states_2=direct,ref_hidden_states=refs)[0]
    for a,b in zip(actual,positive+negative):torch.testing.assert_close(a,b,rtol=0,atol=0)
    executor.close();assert executor.closed and executor.styles==()
    with pytest.raises(RuntimeError,match='closed'):executor.predict(x,t,caps,encoder_hidden_states_2=direct)


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('during',['capture','inject'])
def test_style_cancellation_restores_sampler_processors_and_allows_next_request(profile,during):
    value=sampler(profile);stop=[False]
    styles=prepare_styles(value.components.vae,normalize_styles(entries()[:1]),width=64,height=64,batch=1,seed=17,check_cancel=lambda:None)
    original=[block.attention.processor for block in value.components.transformer.layers]
    def trigger(module,inputs):
        info=getattr(module.layers[0].attention,'_ming_style_info',None)
        if info and info['context'].mode==during:stop[0]=True
    handle=value.components.transformer.register_forward_pre_hook(trigger)
    args=dict(task='layer-decompose' if profile=='layer_decompose' else 'text-to-image',conditioning=condition(),width=64,height=64,steps=2,cfg=2.,seed=17,requested_layers=1)
    if profile=='layer_decompose':args['reference_pixels']=torch.randn(1,4,64,64)
    with pytest.raises(MingGenerationCancelled):value.run(**args,styles=styles,style_combine_mode='stack',cancelled=lambda:stop[0])
    handle.remove();stop[0]=False
    assert value.pipe.transformer.style_executor is None
    assert all(block.attention.processor is processor for block,processor in zip(value.components.transformer.layers,original))
    assert all(not hasattr(block.attention,'_ming_style_info') for block in value.components.transformer.layers)
    output=value.run(**args)
    assert output[0].values[0].mode=='RGBA' and not torch.cuda.is_initialized()


def test_prepare_style_noise_is_independent_reproducible_and_uses_native_vae():
    value=sampler('generation_edit');normalized=normalize_styles(entries())
    before=torch.get_rng_state().clone()
    prepared=prepare_styles(value.components.vae,normalized,width=64,height=64,batch=2,seed=71,check_cancel=lambda:None)
    assert torch.equal(before,torch.get_rng_state())
    again=prepare_styles(value.components.vae,normalized,width=64,height=64,batch=2,seed=71,check_cancel=lambda:None)
    assert all(torch.equal(a.clean,b.clean) and torch.equal(a.noise,b.noise) for a,b in zip(prepared,again))
    expected=torch.randn(prepared[0].noise.shape,generator=torch.Generator().manual_seed(71+29023))
    assert torch.equal(prepared[0].noise,expected)
    assert not torch.equal(prepared[0].noise,prepared[1].noise)
    assert all(item.clean.shape==(2,16,8,8) for item in prepared)


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
def test_noop_and_inactive_styles_preserve_exact_native_trajectory(profile):
    value=sampler(profile);cfg=StyleTransferConfig(start_step=1000,end_step=1000)
    style=PreparedStyle(torch.randn(1,16,8,8),torch.randn(1,16,8,8),cfg)
    args=dict(task='layer-decompose' if profile=='layer_decompose' else 'text-to-image',conditioning=condition(),width=64,height=64,steps=1,cfg=2.,seed=17,requested_layers=1,output_type='latent')
    if profile=='layer_decompose':args['reference_pixels']=torch.randn(1,4,64,64)
    baseline=value.run(**args);inactive=value.run(**args,styles=[style],style_combine_mode='stack')
    cfg.start_step=0;cfg.ref_k_strength=cfg.adain_strength=0
    noop=value.run(**args,styles=[style],style_combine_mode='stack')
    assert all(torch.equal(a,b) and torch.equal(a,c) for group,other,third in zip(baseline,inactive,noop) for a,b,c in zip(group.values,other.values,third.values))


@pytest.mark.parametrize('damage',['image','nan','adain','steps','mode','blocks'])
def test_invalid_style_options_are_refused(damage):
    entry=entries()[0]
    updates={'image':{'image':None},'nan':{'ref_k_strength':float('nan')},'adain':{'adain_strength':2},'steps':{'start_step':800,'end_step':100},'mode':{'value_mode':'bad'},'blocks':{'block_range':[4,1]}}
    entry.update(updates[damage])
    with pytest.raises(ValueError):normalize_styles([entry])


@pytest.mark.parametrize('target_tokens,reference_tokens',[(1,1),(1,5),(5,1),(5,7)])
def test_adain_single_token_bottleneck_has_finite_population_variance(target_tokens,reference_tokens):
    from core.inference.reference_style import adain
    generator=torch.Generator().manual_seed(47)
    target=torch.randn(2,target_tokens,3,8,generator=generator,requires_grad=True)
    reference=torch.randn(2,reference_tokens,3,8,generator=generator)
    def stats(value):
        mean=value.mean(1,keepdim=True)
        variance=((value-mean)**2).sum(1,keepdim=True)/max(value.shape[1]-1,1)
        return mean,variance.sqrt()+1e-6
    mean,std=stats(target);ref_mean,ref_std=stats(reference)
    expected=(target-mean)/std*ref_std+ref_mean
    actual=adain(target,reference)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    assert torch.isfinite(actual).all()
    actual.square().mean().backward()
    assert torch.isfinite(target.grad).all()
