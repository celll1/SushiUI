"""Offline native prompt protocols/cache/provider lifecycle; no live API or GPU."""
import base64
from dataclasses import replace
import io
import json
import pytest
from PIL import Image
from core.extensions.ming_image_01_prompt_upsampler import MingPromptOptions, MingImage01PromptUpsampler, validate_output
from core.extensions.minimax_h3_prompt_assistant import PromptAssistError
from api.param_defaults import MING_IMAGE_01_PROMPT_UPSAMPLE_DEFAULTS


def image_url(color="red"):
    buffer=io.BytesIO();Image.new("RGB",(2,2),color).save(buffer,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()


def options(**overrides):
    defaults={k:v for k,v in MING_IMAGE_01_PROMPT_UPSAMPLE_DEFAULTS.items() if k not in {"enabled","cache_max_entries","api_key"}}
    defaults.update(prompt="Keep 日本語 exactly",mode="t2i",base_url="http://localhost:1234",model="local-model",width=64,height=64,layers=2)
    defaults.update(overrides)
    return MingPromptOptions(**defaults)


def design():
    return {"canvas_settings":{"aspect_ratio":"1:1","ambient_lighting":"soft","image_style":"poster"},"layers":[{"description":'Text "日本語"',"coordinates":"cx: 0.500, cy: 0.500, w: 1.000, h: 1.000","hierarchy_and_relation":"foreground","color_specs":["#ffffff"]}]}


def layer(count=2):
    return f"Decompose this image into {count} layers with the following specifications:\n\nNumber of layers: {count}\n"+"\n".join(f"Layer {i+1}: {'Text 日本語' if i==0 else 'Background'}" for i in range(count))


@pytest.mark.parametrize("mutation",["keys","canvas","ratio","box","colors","empty","shape"])
def test_design_strict_native_output(mutation):
    value=design()
    if mutation=="keys":value["unexpected"]=True
    elif mutation=="canvas":value["canvas_settings"]["extra"]="x"
    elif mutation=="ratio":value["canvas_settings"]["aspect_ratio"]="2:1"
    elif mutation=="box":value["layers"][0]["coordinates"]="cx: 0.500, cy: 0.500, w: 2.000, h: 1.000"
    elif mutation=="colors":value["layers"][0]["color_specs"]=["red"]
    elif mutation=="empty":value["layers"]=[]
    elif mutation=="shape":value["layers"][0]["coordinates"]={"x":.5}
    with pytest.raises(PromptAssistError):validate_output(json.dumps(value),options())


@pytest.mark.parametrize("raw",[layer(3),layer().replace("Layer 2:","Layer 1:"),layer().replace("Background",""),layer()+"\nLayer 3: extra"])
def test_layer_count_order_and_nonempty_refusal(raw):
    with pytest.raises(PromptAssistError):validate_output(raw,options(mode="layer"))


def test_templates_cache_geometry_and_force_refresh(tmp_path,monkeypatch):
    service=MingImage01PromptUpsampler(8,tmp_path/"cache.sqlite3")
    calls=[]
    def provider(opts,url,system,key):
        calls.append((opts,url,system,key))
        value=design();value["canvas_settings"]["aspect_ratio"]=f"{opts.width}:{opts.height}"
        return json.dumps(value,ensure_ascii=False),[]
    monkeypatch.setattr(service,"_lm_studio",provider)
    first=service.transform(options(),profile="generation_edit",api_key="secret")
    assert not first["cached"] and "日本語" in first["prompt"]
    assert service.transform(options(),profile="generation_edit",api_key="different")["cached"]
    assert len(calls)==1 and "Figma-style" in calls[0][2] and "64:64" in calls[0][2]
    assert service.transform(options(force_refresh=True),profile="generation_edit")["cache_key"]==first["cache_key"]
    assert len(calls)==2
    assert service.transform(options(width=128),profile="generation_edit")["cache_key"]!=first["cache_key"]
    edit=service.transform(options(mode="i2i",images=[image_url()]),profile="generation_edit")
    assert edit["cache_key"]!=first["cache_key"] and calls[-1][0].mode=="i2i"
    changed=service.transform(options(mode="i2i",images=[image_url("blue")]),profile="generation_edit")
    assert changed["cache_key"]!=edit["cache_key"]
    assert service.cache.clear()==4


@pytest.mark.parametrize("overrides,profile",[({"engine":"official"},"generation_edit"),({"mode":"layer","images":[image_url()]},"generation_edit"),({"mode":"i2i"},"generation_edit"),({"images":[image_url()]},"generation_edit"),({"mode":"layer","images":[image_url(),image_url()]},"layer_decompose"),({"base_url":"https://example.com"},"generation_edit"),({"width":65},"generation_edit")])
def test_refuse_before_provider(tmp_path,monkeypatch,overrides,profile):
    service=MingImage01PromptUpsampler(8,tmp_path/"cache.sqlite3")
    monkeypatch.setattr(service,"_lm_studio",lambda *a: pytest.fail("Provider contacted for invalid request"))
    with pytest.raises(PromptAssistError):service.transform(options(**overrides),profile=profile)


@pytest.mark.parametrize("engine",["lm_studio","ollama"])
def test_real_transport_load_release_and_native_layer_text(tmp_path,monkeypatch,engine):
    import core.extensions.qwen_image_21_prompt_upsampler as transport
    calls=[]
    class Response:
        def __init__(self,value):self.value=value
        def raise_for_status(self):pass
        def json(self):return self.value
    def post(url,**kwargs):
        calls.append((url,kwargs))
        if url.endswith('/models/load'):return Response({"instance_id":"instance"})
        if url.endswith('/chat/completions'):return Response({"choices":[{"message":{"content":layer()}}]})
        if url.endswith('/api/chat'):return Response({"message":{"content":layer()}})
        return Response({})
    monkeypatch.setattr(transport.requests,"post",post)
    service=MingImage01PromptUpsampler(8,tmp_path/"cache.sqlite3")
    value=service.transform(options(engine=engine,mode="layer",images=[image_url()]),profile="layer_decompose")
    assert value["prompt"]==layer()
    assert calls[-1][0].endswith('/models/unload' if engine=='lm_studio' else '/api/generate')
    chat=next(kwargs['json'] for url,kwargs in calls if url.endswith('/chat/completions') or url.endswith('/api/chat'))
    assert "Follow the user's plan EXACTLY" in chat['messages'][0]['content']
    if engine=='ollama':assert 'format' not in chat and chat['messages'][1]['images']
    else:assert chat['model']=='instance' and chat['messages'][1]['content'][0]['type']=='image_url'
    assert service.transform(options(engine=engine,mode="layer",images=[image_url()]),profile="layer_decompose")["cached"]
    assert len(calls)==(3 if engine=='lm_studio' else 2)


def test_api_schema_defaults_and_spec_match():
    import yaml
    from pathlib import Path
    from api.routes import MingImage01PromptUpsampleRequest
    from api.schema_routes import get_ming_image_01_prompt_upsample_defaults
    value=MingImage01PromptUpsampleRequest(prompt="rough",mode="t2i",width=64,height=64,layers=2)
    assert get_ming_image_01_prompt_upsample_defaults()==MING_IMAGE_01_PROMPT_UPSAMPLE_DEFAULTS
    spec=yaml.safe_load((Path(__file__).parents[2]/"openapi.yaml").read_text(encoding="utf-8"))
    schema=spec['components']['schemas']['MingImage01PromptUpsampleRequest']
    assert set(schema['required'])=={name for name,field in value.model_fields.items() if field.is_required()}
    for name,field in value.model_fields.items():
        if not field.is_required() and name!='images':assert schema['properties'][name]['default']==getattr(value,name)
    assert set(schema['properties'])==set(value.model_fields)
    assert '/prompt-assist/ming-image-01/transform' in spec['paths']


@pytest.mark.parametrize("engine",["lm_studio","ollama"])
def test_invalid_provider_output_releases_model_without_cache(tmp_path,monkeypatch,engine):
    import core.extensions.qwen_image_21_prompt_upsampler as transport
    calls=[]
    class Response:
        def raise_for_status(self):pass
        def json(self):
            return {"instance_id":"instance","choices":[{"message":{"content":"invalid"}}],"message":{"content":"invalid"}}
    def post(url,**kwargs):calls.append(url);return Response()
    monkeypatch.setattr(transport.requests,"post",post)
    service=MingImage01PromptUpsampler(8,tmp_path/"cache.sqlite3")
    with pytest.raises(PromptAssistError):service.transform(options(engine=engine),profile="generation_edit")
    assert calls[-1].endswith('/models/unload' if engine=='lm_studio' else '/api/generate')
    assert service.cache.clear()==0
