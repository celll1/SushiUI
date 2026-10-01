"""Offline public transport and real native manager style dispatch; no HTTP/CUDA."""
import asyncio
import base64
import inspect
import io
import json
from pathlib import Path
import pytest
import torch
from PIL import Image
from api.generation_utils import process_controlnet_configs
from api.error_handlers import ValidationError
from api.param_defaults import STYLE_TRANSFER_UI_DEFAULTS
from ming_image_01_pipeline_test import manager,params,cpu_only


def transport(alpha=113):
    buffer=io.BytesIO();Image.new('RGBA',(64,64),(170,50,90,alpha)).save(buffer,format='PNG')
    return {**STYLE_TRANSFER_UI_DEFAULTS,'image_base64':base64.b64encode(buffer.getvalue()).decode(),
            'strength':.4,'style_adain_strength':.3,'style_blocks':'0-1'}


def decode(entries):
    guides,single,styles,combine=process_controlnet_configs(entries,preserve_style_alpha=True)
    return dict(controlnets=entries,controlnet_images=guides,style_transfer=single,
                style_transfers=styles,style_combine_mode=combine)


@pytest.mark.parametrize('task',['txt2img','img2img','inpaint','outpaint'])
def test_design_public_style_and_guide_all_image_paths_preserve_native_rgba(task):
    torch.manual_seed(67);value,components=manager('generation_edit')
    entries=[transport(),{**transport(190),'style_combine_mode':'common_concept'}]
    entries[0]['style_combine_mode']='common_concept'
    guide={**transport(90),'model_path':'__reference_guide__','is_style_transfer':False,
           'is_reference_guide':True,'strength':.2,'start_step':0,'end_step':1000}
    inputs=decode([*entries,guide]);traces=[];request=params(width=64,height=64,steps=2,cfg_scale=2.,**inputs)
    image=Image.new('RGBA',(64,64),(60,80,100,140));mask=Image.new('L',(64,64),0);mask.paste(255,(32,0,64,64))
    def run(request,trace):
        callback=lambda i,t,x:trace.append(x.clone())
        if task=='txt2img':return value.generate_txt2img(request,step_callback=callback)[0]
        if task=='img2img':return value.generate_img2img(request,image,step_callback=callback)[0]
        if task=='inpaint':return value.generate_inpaint(request,image,mask,step_callback=callback)[0]
        return value.generate_outpaint({**request,'canvas_width':64,'canvas_height':64,
            'place_x':16,'place_y':16,'place_width':32,'place_height':32},image,step_callback=callback)[0]
    actual=run(request,traces);repeat=[];repeated=run(request,repeat)
    baseline=[];run(params(width=64,height=64,steps=2,cfg_scale=2.),baseline)
    assert len(traces)==len(repeat)==2 and all(torch.equal(a,b) for a,b in zip(traces,repeat))
    assert any(not torch.equal(a,b) for a,b in zip(traces,baseline))
    assert all(torch.isfinite(x).all() for x in traces)
    assert actual.primary.mode=='RGBA' and actual.primary.tobytes()==repeated.primary.tobytes()
    if task=='inpaint':assert actual.primary.crop((0,0,32,64)).tobytes()==image.crop((0,0,32,64)).tobytes()
    if task=='outpaint':assert actual.primary.crop((16,16,48,48)).tobytes()==image.resize((32,32)).tobytes()
    assert all(not hasattr(block.attention,'_ming_style_info') for block in components.transformer.layers)
    assert not components.transformer._forward_pre_hooks
    cpu_only(components)


@pytest.mark.parametrize('combine',['stack','common_concept'])
def test_layer_public_multi_style_keeps_auxiliary_and_ordered_output_roles(combine):
    value,components=manager('layer_decompose')
    entries=[{**transport(),'style_combine_mode':combine},transport(200)]
    request=params(width=64,height=64,steps=2,cfg_scale=2.,ming_image_01_layers=2,**decode(entries))
    traces=[];result,_,_=value.generate_img2img(request,Image.new('RGBA',(64,64),(70,90,40,150)),step_callback=lambda i,t,x:traces.append(x.clone()))
    assert result.groups[0].roles==['auxiliary_canvas','layer_1','layer_2']
    assert all(image.mode=='RGBA' for image in result.groups[0].values)
    assert all(torch.isfinite(x).all() for x in traces)
    cpu_only(components)


@pytest.mark.parametrize('fault',['missing','bad_image','controlnet','ambiguous','combine','partition','attention','guidance','offset','release','nan'])
def test_public_style_faults_refuse_before_residency_and_next_request_recovers(monkeypatch,fault):
    value,components=manager('generation_edit');inputs=decode([transport()])
    if fault=='missing':inputs['style_transfers']=[];inputs['style_transfer']=None
    elif fault=='bad_image':inputs['style_transfers'][0]['image']=None
    elif fault=='controlnet':inputs['controlnets']=[{'model_path':'actual-controlnet'}]
    elif fault=='ambiguous':inputs['controlnets'][0]['is_reference_guide']=True
    elif fault=='combine':inputs['style_combine_mode']='bad'
    elif fault=='partition':inputs['ming_image_01_partition']={'mode':'auto','max_target_tokens':8,'halo_tokens':0}
    elif fault=='attention':inputs['attention_type']='flash'
    elif fault=='guidance':inputs['style_transfers'][0]['style_guidance_scale']=1
    elif fault=='offset':inputs['style_transfers'][0]['rope_offset']=2
    elif fault=='release':inputs['style_transfers'][0]['late_release']=.5
    else:inputs['style_transfers'][0]['ref_k_strength']=float('nan')
    from core.models.ming_image_01.residency import MingPhaseResidency
    with monkeypatch.context() as local:
        def forbidden(*a,**k):raise AssertionError('Invalid style reached residency')
        local.setattr(MingPhaseResidency,'__call__',forbidden)
        with pytest.raises(ValidationError):value.generate_txt2img(params(width=64,height=64,**inputs))
    assert value.generate_txt2img(params())[0].primary.mode=='RGBA'
    cpu_only(components)


def test_shared_transport_alpha_is_preserved_only_for_native_request():
    raw=[transport()]
    native=process_controlnet_configs(raw,preserve_style_alpha=True)[2][0]['image']
    legacy=process_controlnet_configs(raw)[2][0]['image']
    assert native.mode=='RGBA' and native.getpixel((0,0))[3]==113 and legacy.mode=='RGB'
    assert decode([{**transport(),'image_base64':'invalid'}])['style_transfers']==[]


def test_defaults_routes_spec_and_frontend_static_contract():
    from api.schema_routes import get_generation_defaults
    from api import routes
    payload=asyncio.run(get_generation_defaults())
    assert payload['style_transfer']==STYLE_TRANSFER_UI_DEFAULTS
    for name in ('generate_txt2img','generate_img2img','generate_inpaint','generate_outpaint'):
        source=inspect.getsource(getattr(routes,name))
        assert 'preserve_style_alpha=bool(pipeline_manager.is_ming_image_01_model)' in source
        for key in ('style_transfer','style_transfers','style_combine_mode'):assert f'"{key}"' in source
    root=Path(__file__).resolve().parents[2]
    import yaml
    spec=yaml.safe_load((root/'openapi.yaml').read_text(encoding='utf-8'))
    assert 'style_transfer' in spec['paths']['/schema/generation-defaults']['get']['description']
    selector=(root/'frontend/src/components/common/ControlNetSelector.tsx').read_text(encoding='utf-8')
    assert 'const defaults = generationDefaults?.style_transfer' in selector
    assert 'value={cn.style_blocks ?? ""}' in selector
    assert 'disabled={disabled || !generationDefaults?.style_transfer}' in selector
    assert 'disabled={disabled || mingNativeConditioning}' in selector
    assert 'style_transfer?: Pick<ControlNetConfig' in (root/'frontend/src/utils/api.ts').read_text(encoding='utf-8')
