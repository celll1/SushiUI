"""CPU native Reference Guide endpoint/window/ownership and manager integration."""
import base64
import io
import json
import pytest
import torch
from PIL import Image
from core.models.ming_image_01.reference_guide import GuideConfig, PreparedGuide, normalize_guides, prepare_guides, blend_guides
from ming_image_01_sampling_test import make_sampler, condition, options
from ming_image_01_pipeline_test import manager, params, cpu_only
from api.error_handlers import ValidationError


def guide(**overrides):
    return {"is_reference_guide":True,"image":Image.new("RGBA",(32,32),(120,30,50,90)),"strength":1.,"start_step":0,"end_step":1000,**overrides}


@pytest.mark.parametrize("dtype",[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize("profile",["generation_edit","layer_decompose"])
def test_blend_endpoints_order_windows_and_protected_auxiliary(dtype,profile):
    clean=torch.full((2,16,4,4),2.,dtype=dtype)
    noise=torch.full_like(clean,6.)
    first=PreparedGuide(clean,noise,GuideConfig(Image.new("RGBA",(1,1)),1.,0,1000))
    latents=torch.full_like(clean,-1.) if profile=="generation_edit" else torch.full((2,16,3,4,4),-1.,dtype=dtype)
    zero=blend_guides(latents,[first],index=1,total=2,sigma=0.,profile=profile)
    selected=zero if profile=="generation_edit" else zero[:,:,1:]
    assert torch.equal(selected,torch.full_like(selected,2.)) and zero.dtype==dtype
    one=blend_guides(latents,[first],index=0,total=2,sigma=1.,profile=profile)
    selected=one if profile=="generation_edit" else one[:,:,1:]
    assert torch.equal(selected,torch.full_like(selected,6.))
    if profile=="layer_decompose":assert torch.equal(zero[:,:,0],latents[:,:,0]) and torch.equal(one[:,:,0],latents[:,:,0])
    window=PreparedGuide(clean,noise,GuideConfig(first.config.image,.5,500,500))
    assert blend_guides(latents,[window],index=0,total=3,sigma=.5,profile=profile) is latents
    middle=blend_guides(latents,[window],index=1,total=3,sigma=.5,profile=profile)
    selected=middle if profile=="generation_edit" else middle[:,:,1:]
    assert torch.equal(selected,torch.full_like(selected,1.5))
    extrapolate=PreparedGuide(clean,noise,GuideConfig(first.config.image,2.,0,1000))
    extended=blend_guides(latents,[extrapolate],index=1,total=2,sigma=0.,profile=profile)
    selected=extended if profile=="generation_edit" else extended[:,:,1:]
    assert torch.equal(selected,torch.full_like(selected,5.))
    second=PreparedGuide(clean*2,noise,first.config)
    ordered=blend_guides(latents,[first,second],index=1,total=2,sigma=0.,profile=profile)
    selected=ordered if profile=="generation_edit" else ordered[:,:,1:]
    assert torch.equal(selected,torch.full_like(selected,4.))


@pytest.mark.parametrize("change",[{"strength":float('nan')},{"strength":2.1},{"strength":-1},{"start_step":900,"end_step":800},{"end_step":1001},{"start_step":True},{"image":None},{"is_reference_guide":False}])
def test_config_refuses_malformed_guide(change):
    with pytest.raises(ValueError):normalize_guides([guide(**change)])


def test_native_vae_normalization_alpha_and_independent_noise_do_not_advance_main_rng():
    from core.models.ming_image_01.training_data import encode_frames
    import numpy as np
    sampler=make_sampler("generation_edit")
    vae=sampler.components.vae
    config=normalize_guides([guide()])
    before=torch.get_rng_state().clone()
    prepared=prepare_guides(vae,config,width=32,height=32,batch=2,seed=7,check_cancel=lambda:None)
    assert torch.equal(before,torch.get_rng_state())
    pixels=torch.from_numpy(np.asarray(config[0].image).copy()).permute(2,0,1).float()/127.5-1
    expected=encode_frames(vae,pixels[None,:,None].expand(2,-1,-1,-1,-1)).squeeze(2)
    assert torch.equal(prepared[0].clean,expected)
    again=prepare_guides(vae,config,width=32,height=32,batch=2,seed=7,check_cancel=lambda:None)
    assert torch.equal(prepared[0].noise,again[0].noise)
    assert not torch.equal(prepared[0].noise[0],prepared[0].noise[1])
    altered=normalize_guides([guide(image=Image.new("RGBA",(32,32),(120,30,50,255)))])
    different=prepare_guides(vae,altered,width=32,height=32,batch=2,seed=7,check_cancel=lambda:None)
    assert not torch.equal(prepared[0].clean,different[0].clean)
    assert prepare_guides(vae,normalize_guides([guide(strength=0)]),width=32,height=32,batch=2,seed=7,check_cancel=lambda:None)==[]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("profile",["generation_edit","layer_decompose"])
def test_native_sampler_next_sigma_and_frame_exclusions(profile):
    sampler=make_sampler(profile)
    c=condition(batch=2)
    kw={**options(),"steps":1,"requested_layers":2 if profile=='layer_decompose' else 1}
    if profile=='layer_decompose':kw['reference_pixels']=torch.zeros(2,4,32,32)
    task='layer-decompose' if profile=='layer_decompose' else 'text-to-image'
    baseline=[]
    sampler.run(task=task,conditioning=c,step_callback=lambda i,n,t,z:baseline.append(z.clone()),**kw)
    prepared=prepare_guides(sampler.components.vae,normalize_guides([guide()]),width=32,height=32,batch=2,seed=kw['seed'],check_cancel=lambda:None)
    guided=[]
    result=sampler.run(task=task,conditioning=c,reference_guides=prepared,step_callback=lambda i,n,t,z:guided.append(z.clone()),**kw)
    assert torch.equal(sampler.components.scheduler.sigmas[-1],torch.tensor(0.))
    if profile=='generation_edit':assert torch.equal(guided[0],prepared[0].clean)
    else:
        assert torch.equal(guided[0][:,:,0],baseline[0][:,:,0])
        assert torch.equal(guided[0][:,:,1],prepared[0].clean) and torch.equal(guided[0][:,:,2],prepared[0].clean)
    assert len(result)==2 and all(v.mode=='RGBA' for g in result for v in g.values)


@pytest.mark.parametrize("operation",['txt2img','img2img','inpaint','outpaint'])
def test_public_transport_manager_operations_protected_pixels_and_cancel(operation):
    from api.generation_utils import process_controlnet_configs
    source=Image.new("RGBA",(32,32),(30,60,90,100))
    buffer=io.BytesIO();source.save(buffer,format='PNG')
    raw={"model_path":"__reference_guide__","is_reference_guide":True,"image_base64":base64.b64encode(buffer.getvalue()).decode()}
    decoded,*_=process_controlnet_configs([raw])
    value,components=manager('generation_edit')
    call=params(controlnets=[raw],controlnet_images=decoded,steps=2)
    if operation=='txt2img':invoke=lambda callback=None:value.generate_txt2img(call,step_callback=callback)
    elif operation=='img2img':invoke=lambda callback=None:value.generate_img2img(call,source,step_callback=callback)
    elif operation=='inpaint':
        mask=Image.new('L',(32,32),0);mask.paste(255,(16,0,32,32))
        invoke=lambda callback=None:value.generate_inpaint(call,source,mask,step_callback=callback)
    else:
        source=source.resize((16,16));call.update(canvas_width=32,canvas_height=32,place_x=8,place_y=8,place_width=16,place_height=16)
        invoke=lambda callback=None:value.generate_outpaint(call,source,step_callback=callback)
    first,*_=invoke()
    repeated,*_=invoke()
    assert first.primary.tobytes()==repeated.primary.tobytes()
    if operation=='inpaint':assert first.primary.crop((0,0,16,32)).tobytes()==source.crop((0,0,16,32)).tobytes()
    if operation=='outpaint':assert first.primary.crop((8,8,24,24)).tobytes()==source.tobytes()
    def cancel(*a):value.cancel_requested=True
    with pytest.raises(RuntimeError,match='cancelled'):invoke(cancel)
    value.cancel_requested=False
    recovered,*_=invoke()
    assert recovered.primary.tobytes()==first.primary.tobytes()
    cpu_only(components)


@pytest.mark.parametrize("changes",[{"controlnets":[{"model_path":"__reference_guide__","is_reference_guide":True}]},{"controlnet_images":[guide(strength=2.1)]},{"controlnets":[{"model_path":"actual-controlnet"}]},{"style_transfers":[{"image":Image.new('RGBA',(32,32))}]}])
def test_manager_early_refusal_before_residency(monkeypatch,changes):
    value,components=manager('generation_edit')
    monkeypatch.setattr(components.conditioner,'to',lambda *a,**k:pytest.fail('Invalid guide changed residency'))
    with pytest.raises(ValidationError):value.generate_txt2img(params(**changes))


def test_generation_schema_reference_guide_defaults_and_spec():
    import asyncio
    from api.schema_routes import get_generation_defaults
    from api.param_defaults import REFERENCE_GUIDE_DEFAULTS
    import yaml
    from pathlib import Path
    assert asyncio.run(get_generation_defaults())['reference_guide']==REFERENCE_GUIDE_DEFAULTS
    spec=yaml.safe_load((Path(__file__).parents[2]/'openapi.yaml').read_text(encoding='utf-8'))
    schema=spec['components']['schemas']['MingReferenceGuide']
    for key in ('strength','start_step','end_step'):assert schema['properties'][key]['default']==REFERENCE_GUIDE_DEFAULTS[key]


@pytest.mark.parametrize("profile,format_",[("generation_edit","int8"),("layer_decompose","int8_convrot")])
def test_native_guide_with_quantized_nested_unet_and_lora(tmp_path,monkeypatch,profile,format_):
    from ming_image_01_conversion_test import make_source
    from core.models.ming_image_01.conversion import convert_package
    from core.model_loader import ModelLoader
    from ming_image_01_generation_lora_test import write_adapter
    from core.models.ming_image_01.residual_unet import model_topology_metadata
    from core.extensions.lora_manager import lora_manager
    source=tmp_path/'source';make_source(source,profile)
    converted=convert_package(source,tmp_path/'quantized',format_=format_)
    loaded=ModelLoader.load_ming_image_01_from_path(converted['manifest'],torch.float32)
    c=loaded['ming_native'];value,_=manager(profile)
    value._install_ming_image_01(loaded,model_id='native',source_type='local',source=str(converted['manifest']))
    topology=c.transformer._residual_unet_spec
    path=tmp_path/'guide-lora.safetensors'
    write_adapter(path,c.transformer,profile=profile,metadata=model_topology_metadata(c.transformer))
    monkeypatch.setattr(lora_manager,'_resolve_lora_path',str)
    base=c.transformer.layers[0].attention.to_q
    packed={key:t.clone() for key,t in c.transformer.named_buffers()}
    config=guide(image=Image.new('RGBA',(64,64),(100,50,70,180)))
    prepared=prepare_guides(c.vae,normalize_guides([config]),width=64,height=64,batch=1,seed=7,check_cancel=lambda:None)
    captures=[]
    images=[Image.new('RGBA',(64,64),(80,40,60,200))] if profile=='layer_decompose' else []
    output,*_=value._ming_image_run(params(width=64,height=64,ming_image_01_layers=2,
        controlnet_images=[config],loras=[{'path':str(path)}]),images=images,
        step_callback=lambda i,t,z:captures.append(z.clone()))
    observed=captures[0] if profile=='generation_edit' else captures[0][:,:,1]
    assert torch.equal(observed,prepared[0].clean)
    assert c.transformer.layers[0].attention.to_q is base and c.transformer._residual_unet_spec==topology
    assert all(torch.equal(t.view(torch.uint8),dict(c.transformer.named_buffers())[key].view(torch.uint8)) for key,t in packed.items())
    assert all(image.mode=='RGBA' for image in output.groups[0].values)
    cpu_only(c)


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
def test_real_half_precision_native_vae_and_dit_reference_guide(dtype):
    value,c=manager('generation_edit')
    c.vae.to(dtype=dtype);c.transformer.to(dtype=dtype)
    captured=[]
    output,*_=value.generate_txt2img(params(controlnet_images=[guide()]),
        step_callback=lambda i,t,z:captured.append(z.clone()))
    assert len(captured)==1 and captured[0].dtype==torch.float32 and torch.isfinite(captured[0]).all()
    assert output.primary.mode=='RGBA'
    cpu_only(c)



def test_native_dynamic_shift_uses_actual_next_sigma_at_each_step():
    sampler=make_sampler('generation_edit')
    sampler.components.scheduler.register_to_config(use_dynamic_shifting=True)
    prepared=prepare_guides(sampler.components.vae,normalize_guides([guide()]),width=32,height=32,batch=1,seed=53,check_cancel=lambda:None)
    captures=[]
    sampler.run(task='text-to-image',conditioning=condition(),reference_guides=prepared,
        step_callback=lambda i,n,t,z:captures.append(z.clone()),**{**options(),"steps":3})
    assert len(captures)==3
    assert 0<sampler.components.scheduler.sigmas[1]<1
    for index,actual in enumerate(captures):
        sigma=sampler.components.scheduler.sigmas[index+1]
        expected=(1-sigma)*prepared[0].clean+sigma*prepared[0].noise
        assert torch.equal(actual,expected)
