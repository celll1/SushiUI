"""Native packed conditioner loading, strict preflight and actual CPU generation."""
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import load_file,save_file
from core.models.ming_image_01.loader import load_components
from core.models.ming_image_01.checkpoint import save_bundle
from ming_image_01_conditioning_test import write_tiny_package


def source(tmp_path,profile,wide=False):
    root=tmp_path/'source';root.mkdir();write_tiny_package(root,profile=profile)
    if wide:
        from core.models.ming_image_01.vendor.configuration_bailingmm2 import BailingMM2Config
        from core.models.ming_image_01.vendor.modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
        from core.models.ming_image_01.conditioning import MingConditioner
        from transformers import Qwen2Config,Qwen2ForCausalLM
        c=load_components(root,dtype=torch.float32,conditioner_attention='eager')
        config=BailingMM2Config.from_dict(c.artifact.configs['mllm'])
        config.llm_config.hidden_size=256;config.llm_config.head_dim=32
        config.llm_config.num_attention_heads=config.llm_config.num_key_value_heads=8
        config.llm_config.rope_scaling={'type':'default','mrope_section':[4,0,4]}
        config._attn_implementation='eager'
        native=BailingMM2NativeForConditionalGeneration(config)
        cfg=Qwen2Config.from_dict(c.artifact.configs['connector'])
        cfg.hidden_size=256;cfg.num_attention_heads=cfg.num_key_value_heads=8
        cfg.head_dim=32;cfg._attn_implementation='eager'
        conditioner=MingConditioner(native,Qwen2ForCausalLM(cfg),c.artifact.configs['mlp'])
        with torch.no_grad():
            for value in native.query_tokens_dict.values():value.normal_(0,.02)
        excluded=('connector.','query_tokens_dict.','proj_in.','proj_out.','proj_directvlm.')
        native.save_pretrained(root/'mllm',state_dict={k:v for k,v in native.state_dict().items() if not k.startswith(excluded)})
        native.connector.save_pretrained(root/'connector')
        save_file(conditioner.projection_state_dict(),str(root/'mlp/model.safetensors'))
    return root


def pack(root,component,convrot=False):
    path=root/component/'model.safetensors';state=load_file(str(path))
    selected=[key for key in state if key.endswith(('self_attn.q_proj.weight','self_attn.k_proj.weight','attention.query_key_value.weight','attention.dense.weight'))]
    assert len(selected)>=2
    selected=selected[:2]
    from core.models.common.quantized_checkpoint_guard import encode_comfy_quant_marker
    from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight
    for index,key in enumerate(selected):
        stem=key.removesuffix('.weight')
        if convrot:
            q,scale=quantize_int8_convrot_weight(state[key],256)
            state[key]=q.contiguous();state[stem+'.weight_scale']=scale.float().reshape(-1,1).contiguous()
            state[stem+'.comfy_quant']=encode_comfy_quant_marker({'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256})
        else:
            state[key]=(state[key]*64).to(torch.int8 if index==0 else torch.float8_e4m3fn)
            state[stem+'.weight_scale']=torch.full((state[key].shape[0],),1/64.)
    save_file(state,str(path))
    return {key:value for key,value in state.items() if not key.endswith('.bias') and any(key.startswith(name.removesuffix('.weight')+'.') for name in selected)}


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('component',['mllm','connector'])
def test_mixed_conditioner_codes_scales_generation_and_bundle_reload(tmp_path,profile,dtype,component):
    root=source(tmp_path,profile);packed=pack(root,component)
    c=load_components(root,dtype=dtype,conditioner_attention='eager')
    model=c.conditioner.native if component=='mllm' else c.conditioner.native.connector
    for key,value in packed.items():
        actual=model.state_dict()[key]
        assert actual.dtype==value.dtype and torch.equal(actual.view(torch.uint8),value.view(torch.uint8))
    from ming_image_01_pipeline_test import manager,params
    from PIL import Image
    value,_=manager(profile)
    value._install_ming_image_01({'ming_native':c,'transformer':c.transformer,'vae':c.vae,'text_encoder':c.conditioner},model_id='packed-conditioner',source_type='local',source=str(root))
    images=[Image.new('RGBA',(64,64),(70,90,120,180))] if profile=='layer_decompose' else []
    states=[]
    result,_,_=value._ming_image_run(params(width=64,height=64,ming_image_01_layers=1),images=images,
        step_callback=lambda i,t,latents:states.append(bool(torch.isfinite(latents).all())))
    assert states==[True] and result.groups[0].values[0].mode=='RGBA'
    saved=save_bundle(c,tmp_path/'saved')
    restored=load_components(saved,dtype=dtype,conditioner_attention='eager')
    target=restored.conditioner.native if component=='mllm' else restored.conditioner.native.connector
    assert all(torch.equal(v.view(torch.uint8),target.state_dict()[key].view(torch.uint8)) for key,v in packed.items())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('component',['mllm','connector'])
def test_actual_convrot_conditioner_forward_and_fp32_scale_retention(tmp_path,component):
    root=source(tmp_path,'generation_edit',wide=True);packed=pack(root,component,convrot=True)
    c=load_components(root,dtype=torch.bfloat16,conditioner_attention='eager')
    model=c.conditioner.native if component=='mllm' else c.conditioner.native.connector
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
    import torch.nn.functional as F
    for name,child in model.named_modules():
        if not isinstance(child,ConvRotInt8Linear):continue
        assert child.weight.dtype==torch.int8 and child.weight_scale.dtype==torch.float32
        x=torch.randn(2,child.in_features,dtype=torch.bfloat16)
        expected=F.linear(x,dequantize_int8_convrot_weight(child.weight,child.weight_scale[:,None],256).to(x.dtype),child.bias)
        torch.testing.assert_close(child._dequant_forward(x),expected,rtol=1e-3,atol=1e-3)
    from core.models.ming_image_01.prompting import prepare_inputs,encode_inputs
    prompt=prepare_inputs(c.processor,['word5'],references=[[]],width=64,height=64,resolution=64)
    conditions=encode_inputs(c.conditioner,prompt,requires_grad=False)
    assert torch.isfinite(conditions.queries).all() and torch.isfinite(conditions.direct).all()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('component',['mllm','connector'])
@pytest.mark.parametrize('damage',['missing_scale','wrong_scale','marker'])
def test_malformed_conditioner_headers_refused_before_weight_payload(tmp_path,monkeypatch,component,damage):
    root=source(tmp_path,'generation_edit');packed=pack(root,component)
    path=root/component/'model.safetensors';state=load_file(str(path))
    key=next(k for k in packed if k.endswith('.weight'))
    scale=key.removesuffix('.weight')+'.weight_scale'
    if damage=='missing_scale':del state[scale]
    elif damage=='wrong_scale':state[scale]=torch.ones(1)
    else:state[key.removesuffix('.weight')+'.comfy_quant']=torch.ones(1,dtype=torch.uint8)
    save_file(state,str(path))
    import safetensors.torch
    def forbidden(*a,**k):raise AssertionError('Malformed conditioner reached payload loading')
    monkeypatch.setattr(safetensors.torch,'load_file',forbidden)
    with pytest.raises(ValueError,match='Ming'):
        load_components(root,dtype=torch.float32,conditioner_attention='eager')


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_packed_vision_activation_dtype_and_reference_conditioning(tmp_path,dtype):
    root=source(tmp_path,'layer_decompose')
    path=root/'mllm/model.safetensors';state=load_file(str(path))
    key='vision.blocks.0.mlp.down_proj.weight'
    state[key]=(state[key]*64).to(torch.float8_e4m3fn)
    state[key.removesuffix('.weight')+'.weight_scale']=torch.full((state[key].shape[0],),1/64.)
    save_file(state,str(path))
    c=load_components(root,dtype=dtype,conditioner_attention='eager')
    assert c.conditioner.native.vision.get_dtype()==dtype
    assert c.conditioner.native.vision.blocks[0].mlp.down_proj.weight.dtype==torch.float8_e4m3fn
    from PIL import Image
    from core.models.ming_image_01.prompting import prepare_inputs,encode_inputs
    prompt=prepare_inputs(c.processor,['word5'],references=[[Image.new('RGBA',(64,64),(80,40,100,180))]],width=64,height=64,resolution=64)
    output=encode_inputs(c.conditioner,prompt,requires_grad=False)
    assert torch.isfinite(output.queries).all() and torch.isfinite(output.direct).all()
    assert not torch.cuda.is_initialized()


def test_declared_connector_embedding_alias_restored_after_packed_assignment(tmp_path):
    root=source(tmp_path,'generation_edit');pack(root,'connector')
    cfg=root/'connector/config.json';value=json.loads(cfg.read_text());value['tie_word_embeddings']=True
    cfg.write_text(json.dumps(value))
    path=root/'connector/model.safetensors';state=load_file(str(path));state.pop('lm_head.weight')
    save_file(state,str(path))
    c=load_components(root,dtype=torch.float32,conditioner_attention='eager')
    native=c.conditioner.native.connector
    assert native.lm_head.weight is native.model.embed_tokens.weight
    saved=save_bundle(c,tmp_path/'saved')
    restored=load_components(saved,dtype=torch.float32,conditioner_attention='eager')
    assert restored.conditioner.native.connector.lm_head.weight is restored.conditioner.native.connector.model.embed_tokens.weight


def test_convrot_dit_training_export_preserves_prepared_conditioner_bytes(tmp_path):
    from ming_image_01_conversion_test import make_source,digest_tree
    from core.training.ming_full_export import export_ming_full_checkpoint
    root=tmp_path/'run';root.mkdir()
    checkpoint=root/'run_step_000002'
    make_source(checkpoint,'generation_edit')
    packed=pack(checkpoint,'mllm')
    before=digest_tree(checkpoint/'mllm')
    (checkpoint/'training.json').write_text(json.dumps({'schema_version':1,'kind':'full_parameter_weights','profile':'generation_edit'}))
    exported=export_ming_full_checkpoint(checkpoint,root)
    assert digest_tree(exported/'mllm')==before==digest_tree(checkpoint/'mllm')
    c=load_components(exported,dtype=torch.bfloat16,conditioner_attention='eager')
    assert any(key.endswith('.comfy_quant') for key in c.transformer.state_dict())
    assert all(torch.equal(v.view(torch.uint8),c.conditioner.native.state_dict()[key].view(torch.uint8)) for key,v in packed.items())


@pytest.mark.parametrize('text_mode',['swap_onthefly','cpu_prefetch','pre_encoded_cache'])
def test_frozen_packed_conditioner_ordinary_lora_training_and_caches(tmp_path,monkeypatch,text_mode):
    from core.training.lora_trainer import LoRATrainer
    from core.training import latent_cache
    from types import SimpleNamespace
    from PIL import Image
    monkeypatch.setattr(latent_cache,'get_cache_base_dir',lambda:str(tmp_path/'cache'))
    root=source(tmp_path,'generation_edit');pack(root,'mllm');pack(root,'connector')
    owner=LoRATrainer(model_path=str(root),output_dir=str(tmp_path/'run'),lora_rank=2,lora_alpha=2,
        device='cpu',weight_dtype='fp32',training_dtype='fp32',vae_dtype='fp32',mixed_precision=False,
        attention_backend='native',train_unet=True,train_text_encoder=False,
        train_config={'ming_conditioner_groups':[],'text_encoding_mode':text_mode,'latent_encoding_mode':'swap_onthefly','seed':17})
    before={key:value.detach().clone() for key,value in owner.text_encoder.state_dict().items()}
    initial={name:value.detach().clone() for name,value in owner.transformer.named_parameters() if value.requires_grad}
    image=tmp_path/'target.png';Image.new('RGBA',(64,64),(100,70,40,180)).save(image)
    dataset=SimpleNamespace(unique_id='packed-conditioner-lora',dataset_root=tmp_path,
        items=[{'image_path':str(image),'caption':'word5','width':64,'height':64}],reload_for_epoch=lambda **k:None)
    owner.train([dataset],num_epochs=2,total_steps=2,batch_size=1,save_every_n_steps=1,
        sample_every_n_steps=0,enable_bucketing=False,base_resolutions=[64],latent_encoding_mode='swap_onthefly',text_encoding_mode=text_mode)
    current=dict(owner.transformer.named_parameters())
    assert any(not torch.equal(value,current[name]) for name,value in initial.items())
    assert all(torch.isfinite(value).all() for value in current.values())
    assert all(torch.equal(value.view(torch.uint8),owner.text_encoder.state_dict()[key].view(torch.uint8)) for key,value in before.items())
    assert all(not value.requires_grad for value in owner.text_encoder.parameters())
    assert (owner.output_dir/'run_step_000002.safetensors').is_file()
    assert not torch.cuda.is_initialized()


def test_conflicting_declared_floating_embedding_aliases_refused(tmp_path):
    root=source(tmp_path,'generation_edit');pack(root,'connector')
    cfg=root/'connector/config.json';value=json.loads(cfg.read_text());value['tie_word_embeddings']=True
    cfg.write_text(json.dumps(value))
    path=root/'connector/model.safetensors';state=load_file(str(path))
    state['lm_head.weight']=state['model.embed_tokens.weight']+1
    save_file(state,str(path))
    with pytest.raises(ValueError,match='tied floating checkpoint weights disagree'):
        load_components(root,dtype=torch.float32,conditioner_attention='eager')
