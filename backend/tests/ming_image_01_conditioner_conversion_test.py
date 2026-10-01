"""CPU authored conditioner packages: atomic conversion, protected state and native outputs."""
from pathlib import Path
import pytest
import torch
from PIL import Image
from core.models.ming_image_01.conversion import convert_package
from core.models.ming_image_01.loader import load_components
from ming_image_01_conditioner_quantized_test import source
from ming_image_01_conversion_test import digest_tree


@pytest.mark.parametrize('profile',['generation_edit','layer_decompose'])
@pytest.mark.parametrize('format_',['int8','int8_convrot'])
def test_authored_conditioners_preserve_companions_load_generate_and_roundtrip(tmp_path,profile,format_):
    from safetensors.torch import save_file
    from core.models.ming_image_01.residual_unet import restore_topology,model_topology_metadata
    from ming_image_01_rewiring_test import spec_for
    from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter,write_payload
    from core.models.ming_image_01.checkpoint import save_bundle
    root=source(tmp_path,profile,wide=True)
    original=load_components(root,dtype=torch.float32,conditioner_attention='eager')
    topology=spec_for(original.transformer);restore_topology(original.transformer,topology)
    save_file(original.transformer.state_dict(),str(root/'transformer/diffusion_pytorch_model.safetensors'),metadata=model_topology_metadata(original.transformer))
    branch=MingPartitionGlobalAdapter(16,24,rank=4,summary_tokens=2)
    setattr(original.transformer,'ming_partition_global_adapter',branch);write_payload(original.transformer,root)
    before=digest_tree(root);events=[]
    result=convert_package(root,tmp_path/'converted',format_=format_,components=['mllm','connector','mlp'],max_shard_bytes=4096,progress_callback=lambda *x:events.append(x))
    exported=Path(result['manifest']).parent
    assert digest_tree(root)==before
    assert set(result['report']['components'])=={'mllm','connector','mlp'}
    assert all(sum(value['converted'].values())>0 for value in result['report']['components'].values())
    assert events[-1][0]==events[-1][1]
    for component in ['transformer','vae','scheduler']:
        assert digest_tree(root/component)==digest_tree(exported/component)
    assert digest_tree(root/'partition_global')==digest_tree(exported/'partition_global')
    for component in ['mllm','connector','mlp']:
        assert (exported/component/'model.safetensors.index.json').is_file()
        assert (exported/component/'config.json').read_bytes()==(root/component/'config.json').read_bytes()
    converted=load_components(exported,dtype=torch.float32,conditioner_attention='eager')
    assert converted.transformer._residual_unet_spec==topology
    for component,module in [('mllm',converted.conditioner.native),('connector',converted.conditioner.native.connector)]:
        old=original.conditioner.native if component=='mllm' else original.conditioner.native.connector
        for key,tensor in old.state_dict().items():
            if any(fragment in key for fragment in ('embed_tokens','lm_head','norm')) and key in converted.artifact.tensor_headers[component]:
                assert torch.equal(tensor,module.state_dict()[key]),key
    for key,tensor in original.conditioner.projection_state_dict().items():
        if key.startswith('query_tokens_dict.') or '.0.weight' in key:
            assert torch.equal(tensor,converted.conditioner.projection_state_dict()[key]),key
    from ming_image_01_pipeline_test import manager,params
    value,_=manager(profile)
    value._install_ming_image_01({'ming_native':converted,'transformer':converted.transformer,'vae':converted.vae,'text_encoder':converted.conditioner},model_id='authored-conditioners',source_type='local',source=str(exported))
    images=[Image.new('RGBA',(64,64),(20,70,110,180))] if profile=='layer_decompose' else []
    trace=[]
    outputs,_,_=value._ming_image_run(params(width=64,height=64,ming_image_01_layers=1),images=images,step_callback=lambda i,t,x:trace.append(bool(torch.isfinite(x).all())))
    assert trace==[True] and outputs.groups[0].values[0].mode=='RGBA'
    saved=load_components(save_bundle(converted,tmp_path/'saved'),dtype=torch.float32,conditioner_attention='eager')
    for key,tensor in converted.conditioner.state_dict().items():
        assert tensor.dtype==saved.conditioner.state_dict()[key].dtype
        assert torch.equal(tensor.view(torch.uint8),saved.conditioner.state_dict()[key].view(torch.uint8)),key
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('components',[[],['mllm','mllm'],['vae'],['missing'],'mllm',[True]])
def test_invalid_component_selection_refuses_before_source_read(tmp_path,components):
    with pytest.raises(ValueError,match='components'):convert_package(tmp_path/'absent',tmp_path/'out',format_='int8',components=components)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('stop',['cancel','failure'])
def test_conditioner_conversion_failure_after_progress_never_publishes(tmp_path,monkeypatch,stop):
    root=source(tmp_path,'generation_edit',wide=True);before=digest_tree(root);events=[]
    if stop=='failure':
        from core.models.common import convrot_export
        original=convrot_export.rotate_and_quantize
        def fail(tensor):
            if events:raise OSError('conditioner conversion failed')
            return original(tensor)
        monkeypatch.setattr(convrot_export,'rotate_and_quantize',fail)
    with pytest.raises(InterruptedError if stop=='cancel' else OSError):
        convert_package(root,tmp_path/'out',format_='int8_convrot',components=['mllm','connector'],max_shard_bytes=4096,progress_callback=lambda *x:events.append(x),cancelled=lambda:bool(events) if stop=='cancel' else False)
    assert events and digest_tree(root)==before
    assert not (tmp_path/'out').exists() and not list(tmp_path.glob('.out.ming-*'))


def test_cli_component_selection_and_cpu_default(tmp_path,monkeypatch):
    import runpy
    root=source(tmp_path,'generation_edit',wide=True)
    module=runpy.run_path(str(Path(__file__).resolve().parents[2]/'subapps/ming_image_01_convert.py'))
    monkeypatch.setattr('sys.argv',['ming_image_01_convert',str(root),str(tmp_path/'out'),'--format','int8','--components','mllm','connector'])
    result=module['main']()
    assert result['report']['device']=='cpu'
    assert set(result['report']['components'])=={'mllm','connector'}


@pytest.mark.parametrize('format_',['int8','int8_convrot'])
def test_tied_connector_heads_stay_floating_and_alias_is_restored(tmp_path,format_):
    root=source(tmp_path,'generation_edit',wide=True)
    original=load_components(root,dtype=torch.float32,conditioner_attention='eager')
    connector=original.conditioner.native.connector
    connector.config.tie_word_embeddings=True;connector.tie_weights()
    assert connector.lm_head.weight is connector.model.embed_tokens.weight
    connector.save_pretrained(root/'connector')
    result=convert_package(root,tmp_path/'out',format_=format_,components=['connector'])
    restored=load_components(result['manifest'],dtype=torch.float32,conditioner_attention='eager').conditioner.native.connector
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    assert torch.equal(restored.lm_head.weight,connector.lm_head.weight)
    assert not any(row['name'].endswith('lm_head') for row in result['report']['layer_audit'])


def test_conditioner_conversion_can_follow_existing_packed_dit(tmp_path):
    root=source(tmp_path,'generation_edit',wide=True)
    first=Path(convert_package(root,tmp_path/'dit',format_='int8')['manifest']).parent
    before=digest_tree(first)
    second=Path(convert_package(first,tmp_path/'all',format_='int8_convrot',components=['mllm','connector','mlp'])['manifest']).parent
    assert digest_tree(first)==before
    assert digest_tree(first/'transformer')==digest_tree(second/'transformer')
    loaded=load_components(second,dtype=torch.float32,conditioner_attention='eager')
    assert any(h.dtype=='I8' for h in loaded.artifact.tensor_headers['transformer'].values())
    assert any(key.endswith('.comfy_quant') for key in loaded.artifact.tensor_headers['connector'])
    with pytest.raises(ValueError,match='original floating'):convert_package(second,tmp_path/'again',format_='int8',components=['mllm'])
    assert not (tmp_path/'again').exists()


def test_nonfinite_conditioner_weight_never_publishes(tmp_path):
    from safetensors.torch import load_file,save_file
    root=source(tmp_path,'generation_edit',wide=True)
    path=root/'mllm/model.safetensors';state=load_file(str(path))
    key=next(key for key in state if key.endswith('attention.query_key_value.weight'))
    state[key][0,0]=float('nan');save_file(state,str(path));before=digest_tree(root)
    with pytest.raises(ValueError,match='nonfinite'):convert_package(root,tmp_path/'out',format_='int8',components=['mllm'])
    assert digest_tree(root)==before and not (tmp_path/'out').exists()
    assert not list(tmp_path.glob('.out.ming-*'))

@pytest.mark.parametrize('profile,format_',[('generation_edit','int8'),('layer_decompose','int8_convrot')])
def test_authored_all_conditioners_materialize_train_and_restore_floating(tmp_path,profile,format_):
    from ming_image_01_conditioner_full_quantized_test import owner,dataset
    import json
    root=source(tmp_path,profile,wide=True)
    result=convert_package(root,tmp_path/'packed',format_=format_,components=['mllm','connector','mlp'])
    value=owner(Path(result['manifest']).parent,tmp_path/'run',['language','vision','connector','queries','projections'],optin=True)
    assert sum(value.ming_conditioner_full_initialization['formats'].values())==sum(sum(component['converted'].values()) for component in result['report']['components'].values())
    before={name:param.detach().clone() for name,param in value.text_encoder.named_parameters() if param.requires_grad}
    frozen={name:tensor.clone() for name,tensor in value.transformer.state_dict().items()}
    data=dataset(tmp_path,profile)
    value.train([data],num_epochs=1,total_steps=1,batch_size=1,save_every_n_steps=1,sample_every_n_steps=0,enable_bucketing=False,base_resolutions=[64],text_encoding_mode='swap_onthefly',latent_encoding_mode='swap_onthefly')
    current=dict(value.text_encoder.named_parameters())
    assert any(not torch.equal(param,current[name]) for name,param in before.items())
    assert all(torch.isfinite(param).all() for param in current.values())
    assert all(torch.equal(tensor,value.transformer.state_dict()[name]) for name,tensor in frozen.items())
    saved=value.output_dir/'run_step_000001'
    metadata=json.loads((saved/'training.json').read_text())
    assert metadata['ming_conditioner_full_initialization']==value.ming_conditioner_full_initialization
    restored=load_components(saved,dtype=torch.float32,conditioner_attention='eager')
    assert all(torch.equal(tensor,restored.conditioner.state_dict()[name]) for name,tensor in value.text_encoder.state_dict().items())
    assert not any(name.endswith(('.weight_scale','.comfy_quant')) for name in restored.conditioner.state_dict())
    assert not torch.cuda.is_initialized()
