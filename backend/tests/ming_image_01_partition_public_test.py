"""Offline request/schema contracts and native manager partition dispatch."""
import inspect
import json
from pathlib import Path
import pytest
import torch
from PIL import Image
from api.param_defaults import GENERATION_DEFAULTS
from core.models.ming_image_01.partition_generation import normalize_partition_config,resolve_partition_plan


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('mode',['manual','auto'])
def test_manager_dispatches_ming_partition_global_and_unet(tmp_path,monkeypatch,profile,mode):
    from ming_image_01_pipeline_test import manager,params,cpu_only
    from ming_image_01_rewiring_test import config,spec_for,restore_topology
    from core.models.ming_image_01.model_definition import create_transformer
    from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter
    value,c=manager(profile)
    model=create_transformer(config('learned' if profile=='layer_decompose' else 'zero_masked'),device='cpu',dtype=torch.float32).eval()
    if profile=='layer_decompose':
        torch.nn.init.constant_(model.x_pad_token,.02);torch.nn.init.constant_(model.cap_pad_token,.03)
    restore_topology(model,spec_for(model))
    branch=MingPartitionGlobalAdapter(16,24,rank=8,summary_tokens=3)
    torch.nn.init.normal_(branch.out.weight,std=.02)
    model._ming_partition_global_payload={'metadata':branch.metadata(),'state':branch.export_tensors()}
    c.transformer=model;c.artifact.configs['transformer']=dict(model.config)
    value.ming_image_01_components['transformer']=model
    frames=3 if profile=='layer_decompose' else 1
    settings=normalize_partition_config({'mode':mode,'halo_tokens':0,'max_target_tokens':frames*32,
        'layout':{'axis':'x','ratio':.5,'first':None,'second':None} if mode=='manual' else None})
    request=params(width=128,height=128,steps=2,ming_image_01_layers=2,ming_image_01_partition=settings)
    trace=[]
    result,_,_=value.generate_img2img(request,Image.new('RGBA',(128,128),(70,20,90,160)),step_callback=lambda i,t,x:trace.append(x.clone()))
    assert len(trace)==2 and all(x.isfinite().all() for x in trace)
    actual=request['ming_image_01_partition_actual']
    assert actual['target_frames']==frames and actual['global_adapter'] and not actual['kv_cache']
    assert actual['full_target_tokens']==64*frames and actual['largest_input_tokens']<=32*frames
    assert len(actual['regions'])==2 and result.model_info['dit_rewiring_spec']==model._residual_unet_spec
    assert result.groups[0].target_frames==frames
    from utils import image_utils
    monkeypatch.setattr(image_utils.settings,'outputs_dir',str(tmp_path))
    path=image_utils.save_image_with_metadata(result.groups[0].values[0],request,model_info={'type':'ming_image_01'},output_filename='partition.png')
    with Image.open(tmp_path/path) as image:
        assert json.loads(image.info['ming_image_01_partition_actual'])==actual
        assert json.loads(image.info['ming_image_01_partition'])==settings
    cpu_only(c)


def test_partition_public_defaults_multipart_and_openapi():
    import yaml
    from api import routes
    default=GENERATION_DEFAULTS['ming_image_01_partition']
    assert routes.GenerationParams(prompt='test').ming_image_01_partition==default
    root=Path(__file__).resolve().parents[2]
    schema=yaml.safe_load((root/'openapi.yaml').read_text(encoding='utf-8'))
    prop=schema['components']['schemas']['GenerationParams']['properties']['ming_image_01_partition']
    assert prop['default']==default
    for name in ('generate_txt2img','generate_img2img','generate_inpaint','generate_outpaint'):
        form=inspect.signature(getattr(routes,name)).parameters['ming_image_01_partition'].default
        assert json.loads(form.default)==default
    source=(root/'backend/api/routes.py').read_text(encoding='utf-8')
    assert source.count('"ming_image_01_partition": _parse_ming_partition(ming_image_01_partition)')==4
    frontend=(root/'frontend/src/utils/api.ts').read_text(encoding='utf-8')
    assert frontend.count('formData.append("ming_image_01_partition",')==4
    assert normalize_partition_config(json.dumps(default))==default


@pytest.mark.parametrize('value',[{'mode':'bad'},{'halo_tokens':1},{'version':True}])
def test_invalid_partition_request_refused(value):
    from api.routes import GenerationParams,_parse_ming_partition
    from pydantic import ValidationError
    from fastapi import HTTPException
    with pytest.raises(ValidationError):GenerationParams(prompt='test',ming_image_01_partition=value)
    with pytest.raises(HTTPException) as error:_parse_ming_partition(value)
    assert error.value.status_code==400


def test_wrong_arch_and_layer_auto_ceiling():
    from api.routes import _parse_ming_partition
    from fastapi import HTTPException
    assert _parse_ming_partition(None,arch='qwen_image_21')['mode']=='off'
    with pytest.raises(HTTPException):_parse_ming_partition({'mode':'auto'},arch='qwen_image_21')
    assert resolve_partition_plan({'mode':'auto','max_target_tokens':64,'halo_tokens':0},8,8,1) is None
    layout=resolve_partition_plan({'mode':'auto','max_target_tokens':64,'halo_tokens':0},8,8,3)
    assert layout and max(b.tokens*3 for b in layout.inputs)<=64
    with pytest.raises(ValueError,match='target frames'):resolve_partition_plan({'mode':'auto','max_target_tokens':2},8,8,3)


@pytest.mark.parametrize('operation',['inpaint','outpaint'])
def test_partition_design_mask_operations_and_actual_metadata(operation):
    from ming_image_01_pipeline_test import manager,params,cpu_only
    value,c=manager('generation_edit')
    settings=normalize_partition_config({'mode':'manual','halo_tokens':0,
        'layout':{'axis':'x','ratio':.5,'first':None,'second':None}})
    if operation=='inpaint':
        image=Image.new('RGBA',(64,64),(20,80,40,113));mask=Image.new('L',(64,64),0)
        mask.paste(255,(32,0,64,64));request=params(width=64,height=64,ming_image_01_partition=settings)
        result,_,_=value.generate_inpaint(request,image,mask)
        assert result.primary.crop((0,0,32,64)).tobytes()==image.crop((0,0,32,64)).tobytes()
    else:
        image=Image.new('RGBA',(32,32),(20,80,40,113))
        request=params(canvas_width=64,canvas_height=64,place_x=16,place_y=16,place_width=32,place_height=32,
            ming_image_01_partition=settings)
        result,_,_=value.generate_outpaint(request,image)
        assert result.primary.crop((16,16,48,48)).tobytes()==image.tobytes()
    assert len(request['ming_image_01_partition_actual']['regions'])==2
    assert result.primary.mode=='RGBA'
    cpu_only(c)


def test_partition_guide_refusal_before_conditioner(monkeypatch):
    from ming_image_01_pipeline_test import manager,params
    from api.error_handlers import ValidationError
    value,c=manager('generation_edit')
    def forbidden(*a,**k):raise AssertionError('Conditioner should not run for excluded guide')
    monkeypatch.setattr(c.conditioner,'forward',forbidden)
    settings={'mode':'manual','halo_tokens':0,'layout':{'axis':'x','ratio':.5,'first':None,'second':None}}
    guide={'is_reference_guide':True,'image':Image.new('RGBA',(64,64))}
    with pytest.raises(ValidationError,match='excludes Reference Guide'):
        value.generate_txt2img(params(width=64,height=64,ming_image_01_partition=settings,controlnet_images=[guide]))


def test_direct_manager_rejects_ming_partition_on_other_architecture():
    from ming_image_01_pipeline_test import manager
    from api.error_handlers import ValidationError
    value,_=manager('generation_edit');value.is_ming_image_01_model=False
    with pytest.raises(ValidationError,match='requires Ming'):
        value._validate_ming_partition_arch({'ming_image_01_partition':{'mode':'auto'}})
