"""CPU contracts for MiniMax Music 3's residual flow block loop."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from core.models.minimax_music3.vendor.transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel
from core.models.minimax_music3.residual_unet import (
    native_config, restore_topology, default_topology, temporal_grid, generation_metadata,
    validate_topology_identity, model_topology_metadata)
from core.rewiring.spec import OPERATORS, build_spec, preview_plan, read_checkpoint_metadata


class MiniMaxMusic3ResidualContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(318)
        self.model=MiniMaxMusic3Transformer1DModel(in_channels=4,condition_dim=6,num_layers=7,
            num_attention_heads=2,attention_head_dim=8,ff_inner_dim=24,rotary_dim=4,
            fourier_embedding_dim=8).float().eval()
        self.regions=[dict(id='outer',stream='main',parent=None,start_block=1,end_block=6,
                           down_factor=[2,1],**OPERATORS),
                      dict(id='inner',stream='main',parent='outer',start_block=2,end_block=5,
                           down_factor=[2,1],**OPERATORS)]
        self.spec=build_spec('minimax_music3',[{'name':'main','block_count':7}],self.regions,
                             native_config(self.model.config))
        self.inputs=dict(hidden_states=torch.randn(2,4,13,requires_grad=True),
            timestep=torch.tensor([.2,.8]),encoder_hidden_states=torch.randn(2,13,6,requires_grad=True),
            return_dict=False)

    def test_native_blocks_heads_gradients_and_parameter_size(self):
        names=set(self.model.state_dict())
        count=sum(p.numel() for p in self.model.parameters())
        restore_topology(self.model,self.spec)
        observed=[]
        hooks=[block.register_forward_pre_hook(lambda m,args:observed.append(
            (args[0].shape[1],args[1][0].shape[0]))) for block in self.model.transformer_blocks]
        try: result=self.model(**self.inputs)[0]
        finally:
            for hook in hooks:hook.remove()
        self.assertEqual(observed,[(n,n) for n in (17,9,5,5,5,9,17)])
        self.assertEqual(result.shape,self.inputs['hidden_states'].shape)
        self.assertTrue(torch.isfinite(result).all())
        result.square().mean().backward()
        for key in ('hidden_states','encoder_hidden_states'):
            self.assertGreater(torch.count_nonzero(self.inputs[key].grad).item(),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for block in self.model.transformer_blocks for p in block.parameters()))
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),count)
        self.assertEqual(set(self.model.state_dict()),names)
        self.assertEqual(read_checkpoint_metadata(model_topology_metadata(self.model)),self.spec)
        actual=generation_metadata(self.model)
        self.assertEqual(actual['native_audio_tokens'],13)
        self.assertEqual(actual['padded_audio_tokens'],16)
        self.assertEqual(actual['prefix_tokens'],1)
        actual['spec']['regions'][0]['id']='detached'
        self.assertEqual(self.model._residual_unet_spec,self.spec)
        self.assertFalse(torch.cuda.is_initialized())

    def test_gradient_checkpointing_matches_native_residual_backward(self):
        restore_topology(self.model,self.spec)
        regular=self.model(**self.inputs)[0]
        regular.square().mean().backward()
        expected=[p.grad.clone() for p in self.model.parameters()]
        input_grads=[self.inputs[k].grad.clone() for k in ('hidden_states','encoder_hidden_states')]
        self.model.zero_grad(set_to_none=True)
        for key in ('hidden_states','encoder_hidden_states'):self.inputs[key].grad=None
        self.model.enable_gradient_checkpointing()
        actual=self.model(**self.inputs)[0]
        torch.testing.assert_close(actual,regular,rtol=0,atol=0)
        actual.square().mean().backward()
        for param,grad in zip(self.model.parameters(),expected):
            torch.testing.assert_close(param.grad,grad,rtol=1e-5,atol=1e-6)
        for key,grad in zip(('hidden_states','encoder_hidden_states'),input_grads):
            torch.testing.assert_close(self.inputs[key].grad,grad,rtol=1e-5,atol=1e-6)
        with torch.no_grad():inference=self.model(**self.inputs)[0]
        torch.testing.assert_close(inference,regular,rtol=2e-5,atol=2e-6)

    def test_zero_block_updates_preserve_native_projection_and_time_token(self):
        class IdentityBlock(torch.nn.Module):
            def forward(self,hidden,rotary):return hidden
        self.model.transformer_blocks=torch.nn.ModuleList([IdentityBlock() for _ in range(7)])
        with torch.no_grad():expected=self.model(**self.inputs)[0]
        restore_topology(self.model,self.spec)
        with torch.no_grad():actual=self.model(**self.inputs)[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        restore_topology(self.model,None)
        self.assertEqual(model_topology_metadata(self.model),{})
        self.assertIsNone(generation_metadata(self.model))
        with torch.no_grad():restored=self.model(**self.inputs)[0]
        torch.testing.assert_close(restored,expected,rtol=0,atol=0)

    def test_refusals_precede_native_projection_and_invalidate_previous_observation(self):
        restore_topology(self.model,self.spec)
        with torch.no_grad():self.model(**self.inputs)
        bad=dict(self.inputs);bad['encoder_hidden_states']=torch.randn(2,12,6)
        with patch.object(self.model.preprocess_conv,'forward',side_effect=AssertionError('projected')):
            with self.assertRaisesRegex(ValueError,'shapes'):self.model(**bad)
        with self.assertRaisesRegex(ValueError,'did not observe'):generation_metadata(self.model)
        for feature in ('_fbcache','_skip_blocks','_tread_router','_block_offloader','_attention_plan'):
            setattr(self.model,feature,object())
            with patch.object(self.model.preprocess_conv,'forward',side_effect=AssertionError('projected')):
                with self.assertRaisesRegex(ValueError,'incompatible'):self.model(**self.inputs)
            delattr(self.model,feature)
        with torch.no_grad():self.model(**self.inputs)
        with patch.object(self.model.proj_out,'forward',side_effect=RuntimeError('head failed')):
            with torch.no_grad(),self.assertRaisesRegex(RuntimeError,'head failed'):self.model(**self.inputs)
        with self.assertRaisesRegex(ValueError,'did not observe'):generation_metadata(self.model)

    def test_native_identity_depth_and_temporal_axis_contract(self):
        config=dict(self.model.config)
        for changes in ({'rotary_dim':3},{'rotary_dim':10},{'fourier_embedding_dim':7},
                        {'num_layers':True},{'sushi_residual_grid_policy':'guessed'}):
            with self.assertRaises(ValueError):native_config({**config,**changes})
        with self.assertRaisesRegex(ValueError,'config mismatch'):
            restore_topology(self.model,build_spec('minimax_music3',[{'name':'main','block_count':7}],
                self.regions,{'fixture':'wrong_native'}))
        spatial=deepcopy(self.regions);spatial[0]['down_factor']=[2,2]
        with self.assertRaisesRegex(ValueError,'temporal axis'):
            validate_topology_identity(build_spec('minimax_music3',[{'name':'main','block_count':7}],
                spatial,native_config(self.model.config)),self.model.config)
        self.model.transformer_blocks=self.model.transformer_blocks[:-1]
        with self.assertRaisesRegex(ValueError,'depth'):restore_topology(self.model,self.spec)
        self.assertFalse(hasattr(self.model,'_residual_unet_spec'))

    def test_temporal_padding_and_default_deepest_four_blocks(self):
        self.assertEqual([temporal_grid(self.spec,n) for n in (1,3,4,13,16,17)], [4,4,4,16,16,20])
        for bad in (0,-1,True,1.5):
            with self.assertRaises(ValueError):temporal_grid(self.spec,bad)
        self.assertIsNone(default_topology(self.model.config))
        config={**dict(self.model.config),'num_layers':36}
        spec=default_topology(config)
        plan=preview_plan(spec,16,1)
        self.assertEqual([sum(b['depth']==depth for b in plan['blocks']) for depth in (0,1,2)],[8,24,4])
        self.assertEqual([level['tokens'] for level in plan['levels']],[8,4])
        self.assertNotIn('minimax_music3',__import__('core.training.arch',fromlist=['ARCH_REGISTRY']).ARCH_REGISTRY)

    def test_native_component_checkpoint_roundtrip_and_early_identity_refusal(self):
        import tempfile
        from safetensors.torch import save_file
        from core.models.minimax_music3.loader import _build_diffusers_component
        from core.models.minimax_music3.residual_unet import export_metadata
        restore_topology(self.model,self.spec)
        with torch.no_grad():expected=self.model(**self.inputs)[0]
        with tempfile.TemporaryDirectory() as folder:
            component=Path(folder)/'transformer'
            self.model.save_config(component)
            source=component/'diffusion_pytorch_model.safetensors'
            save_file(self.model.state_dict(),str(source),metadata=export_metadata(self.model))
            loaded,config=_build_diffusers_component(folder,'transformer',MiniMaxMusic3Transformer1DModel,
                'MiniMaxMusic3Transformer1DModel',torch.float32)
            self.assertEqual(loaded._residual_unet_spec,self.spec)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            self.assertEqual(set(loaded.state_dict()),set(self.model.state_dict()))
            for key,value in loaded.state_dict().items():
                torch.testing.assert_close(value,self.model.state_dict()[key],rtol=0,atol=0)
            with torch.no_grad():actual=loaded(**self.inputs)[0]
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
            wrong=build_spec('minimax_music3',[{'name':'main','block_count':7}],self.regions,
                {**native_config(self.model.config),'condition_dim':8})
            from core.rewiring.spec import checkpoint_metadata
            save_file(self.model.state_dict(),str(source),metadata=checkpoint_metadata(wrong))
            with patch('core.models.common.single_file_format.load_component_state_dict',
                       side_effect=AssertionError('loaded tensors')):
                with self.assertRaisesRegex(ValueError,'config mismatch'):
                    _build_diffusers_component(folder,'transformer',MiniMaxMusic3Transformer1DModel,
                        'MiniMaxMusic3Transformer1DModel',torch.float32)
            save_file(self.model.state_dict(),str(source),metadata=export_metadata(self.model))
            save_file({'unrelated':torch.zeros(1)},str(component/'dense.safetensors'))
            with patch('core.models.common.single_file_format.load_component_state_dict',
                       side_effect=AssertionError('loaded tensors')):
                with self.assertRaisesRegex(ValueError,'declarations disagree'):
                    _build_diffusers_component(folder,'transformer',MiniMaxMusic3Transformer1DModel,
                        'MiniMaxMusic3Transformer1DModel',torch.float32)
        self.assertFalse(torch.cuda.is_initialized())

    def test_flat_safetensors_and_gguf_restore_the_same_native_residual_graph(self):
        import json
        import tempfile
        from safetensors.torch import load_file,save_file
        from tests.minimax_music3_flat_dit_fixture import write_tiny_flat_dit_and_official_tree
        from tests.minimax_music3_gguf_fixture import write_gguf
        from core.models.minimax_music3 import loader
        from core.models.minimax_music3.residual_unet import export_metadata
        with tempfile.TemporaryDirectory() as folder:
            fixture=write_tiny_flat_dit_and_official_tree(folder)
            config_path=Path(fixture['official'])/'transformer'/'config.json'
            config=json.loads(config_path.read_text(encoding='utf-8'));config['num_layers']=7
            config_path.write_text(json.dumps(config),encoding='utf-8')
            flat=load_file(fixture['dit_path'])
            layer_zero={key:value for key,value in flat.items()
                        if key.startswith('diffusion_transformer.transformer.layers.0.')}
            for index in range(1,7):
                for key,value in layer_zero.items():
                    flat[key.replace('.layers.0.',f'.layers.{index}.')]=value.clone()
            save_file(flat,fixture['dit_path'])
            dense,_,condition,_=loader.build_transformer_and_condition_encoder_from_flat_dit(
                fixture['dit_path'],fixture['official'],torch.float32)
            self.assertIsNone(generation_metadata(dense))
            spec=build_spec('minimax_music3',[{'name':'main','block_count':7}],self.regions,
                            native_config(config))
            restore_topology(dense,spec)
            with torch.no_grad():expected=dense(**self.inputs)[0]
            metadata=export_metadata(dense)
            for kind in ('safetensors','gguf'):
                if kind=='safetensors':
                    source=fixture['dit_path']
                    builder=loader.build_transformer_and_condition_encoder_from_flat_dit
                    materializer='core.models.common.single_file_format.read_state_dict'
                    def write(values,md):save_file(values,source,metadata=md)
                else:
                    source=str(Path(folder)/'residual.gguf')
                    builder=loader.build_transformer_and_condition_encoder_from_gguf_dit
                    materializer='core.models.common.gguf_container.GGUFStateDict'
                    def write(values,md):write_gguf(source,values,{'general.architecture':'minimax_music3',**md})
                with self.subTest(format=kind):
                    write(flat,metadata)
                    loaded,_,restored_condition,_=builder(source,fixture['official'],torch.float32)
                    self.assertEqual(loaded._residual_unet_spec,spec)
                    for key,value in loaded.state_dict().items():
                        torch.testing.assert_close(value,dense.state_dict()[key],rtol=0,atol=0)
                    for key,value in restored_condition.state_dict().items():
                        torch.testing.assert_close(value,condition.state_dict()[key],rtol=0,atol=0)
                    with torch.no_grad():actual=loaded(**self.inputs)[0]
                    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
                    bad=dict(metadata);bad['sushi.rewire.spec_sha256']='0'*64
                    write(flat,bad)
                    with patch(materializer,side_effect=AssertionError('materialized tensors')):
                        with self.assertRaisesRegex(ValueError,'hash mismatch'):
                            builder(source,fixture['official'],torch.float32)
                    wrong=build_spec('minimax_music3',[{'name':'main','block_count':7}],self.regions,
                        {**native_config(config),'condition_dim':8})
                    from core.rewiring.spec import checkpoint_metadata
                    write(flat,checkpoint_metadata(wrong))
                    with patch(materializer,side_effect=AssertionError('materialized tensors')):
                        with self.assertRaisesRegex(ValueError,'config mismatch'):
                            builder(source,fixture['official'],torch.float32)
                    missing={key:value for key,value in flat.items() if '.layers.6.' not in key}
                    write(missing,metadata)
                    with patch(materializer,side_effect=AssertionError('materialized tensors')):
                        with self.assertRaisesRegex(ValueError,'complete native'):
                            builder(source,fixture['official'],torch.float32)
                    partial=dict(metadata);partial.pop('sushi.rewire.spec_sha256')
                    write(flat,partial)
                    with patch(materializer,side_effect=AssertionError('materialized tensors')):
                        with self.assertRaisesRegex(ValueError,'Incomplete'):
                            builder(source,fixture['official'],torch.float32)
        self.assertFalse(torch.cuda.is_initialized())

    def flow_pipeline(self):
        from core.models.minimax_music3.pipeline import MiniMaxMusic3Pipeline
        from core.models.minimax_music3.vendor.condition_embedder_minimax_music3 import MiniMaxMusic3ConditionEncoder
        from diffusers import FlowMatchEulerDiscreteScheduler
        pipeline=object.__new__(MiniMaxMusic3Pipeline)
        pipeline.transformer=self.model
        pipeline.condition_encoder=MiniMaxMusic3ConditionEncoder(condition_hidden_dim=3,
            num_condition_layers=2,out_dim=6,input_sampling_rate=10,input_hop_length=1,
            output_sampling_rate=13,output_hop_length=1).float().eval()
        pipeline.scheduler=FlowMatchEulerDiscreteScheduler()
        pipeline._execution_device=torch.device('cpu')
        pipeline.vocoder=pipeline.language_model=pipeline.rvq_depth_decoder=None
        return pipeline

    def test_actual_flow_chunks_capture_cfg_counts_and_real_audio_sidecar(self):
        import json,tempfile
        from core.models.minimax_music3 import pipeline as module
        from core.models.minimax_music3.residual_unet import capture_generation_observation
        from utils import audio_utils
        restore_topology(self.model,self.spec)
        pipeline=self.flow_pipeline()
        inputs=torch.randn(1,22,6)
        with patch.object(module,'CHUNK_FRAMES',10),patch.object(module,'CHUNK_HOP',5), \
             patch.object(module,'OVERLAP_LATENT_LENGTH',2):
            chunks=pipeline.denoise_chunks(inputs,num_inference_steps=3,flow_guidance_scale=1.7,
                generator=torch.Generator().manual_seed(9))
        self.assertEqual([chunk.shape[-1] for chunk in chunks],[13,13,13,9])
        self.assertTrue(all(torch.isfinite(chunk).all() for chunk in chunks))
        actual=pipeline._dit_rewiring_flow_actual
        self.assertEqual(actual['scope'],'flow_input_chunks')
        self.assertEqual(actual['spec'],self.spec)
        self.assertEqual([(r['input_frame_start'],r['input_frame_end']) for r in actual['chunks']],
                         [(0,10),(5,15),(10,20),(15,22)])
        self.assertEqual([r['native_audio_tokens'] for r in actual['chunks']],[13,13,13,9])
        self.assertEqual([r['padded_audio_tokens'] for r in actual['chunks']],[16,16,16,12])
        self.assertTrue(all(r['flow_evaluations']==6 and r['inference_steps']==3 for r in actual['chunks']))
        params={'seed':9,'dit_rewiring_actual':{'stale':True}}
        capture_generation_observation(pipeline,params,self.model)
        self.assertEqual(params['dit_rewiring_actual'],actual)
        params['dit_rewiring_actual']['chunks'][0]['native_audio_tokens']=999
        self.assertEqual(actual['chunks'][0]['native_audio_tokens'],13)
        capture_generation_observation(pipeline,params,self.model)
        with tempfile.TemporaryDirectory() as folder,patch.object(audio_utils.settings,'outputs_dir',folder):
            filename=audio_utils.save_audio_with_metadata(torch.sin(torch.arange(80).float())[None]*.1,
                8000,params,'txt2aud',model_info={'type':'minimax_music3'})
            self.assertGreater((Path(folder)/filename).stat().st_size,0)
            sidecar=json.loads((Path(folder)/Path(filename).with_suffix('.json')).read_text(encoding='utf-8'))
            self.assertEqual(sidecar['dit_rewiring_actual'],actual)
        with patch.object(pipeline.scheduler,'step',side_effect=RuntimeError('flow failed')):
            with self.assertRaisesRegex(RuntimeError,'flow failed'):
                pipeline.denoise_chunks(inputs,3,1.7)
        self.assertFalse(hasattr(pipeline,'_dit_rewiring_flow_actual'))
        with self.assertRaisesRegex(ValueError,'completed flow chunks'):
            capture_generation_observation(pipeline,params,self.model)
        self.assertNotIn('dit_rewiring_actual',params)
        self.assertFalse(torch.cuda.is_initialized())

    def test_flow_graph_change_and_dense_record_absence(self):
        from core.models.minimax_music3.residual_unet import capture_generation_observation
        pipeline=self.flow_pipeline()
        restore_topology(self.model,self.spec)
        changed=deepcopy(self.regions);changed[1]['start_block']=3
        alternate=build_spec('minimax_music3',[{'name':'main','block_count':7}],changed,
                             native_config(self.model.config))
        def change_graph(*args):restore_topology(self.model,alternate)
        with self.assertRaisesRegex(ValueError,'graph changed'):
            pipeline.denoise_chunks(torch.randn(1,10,6),3,1.7,progress_callback=change_graph)
        self.assertFalse(hasattr(pipeline,'_dit_rewiring_flow_actual'))
        restore_topology(self.model,None)
        pipeline.denoise_chunks(torch.randn(1,10,6),3,1.7)
        params={'dit_rewiring_actual':{'stale':True}}
        capture_generation_observation(pipeline,params,self.model)
        self.assertNotIn('dit_rewiring_actual',params)
        self.assertFalse(hasattr(pipeline,'_dit_rewiring_flow_actual'))

    def test_backend_publishes_flow_metadata_only_after_successful_decode(self):
        from types import SimpleNamespace
        from core.models.minimax_music3 import pipeline as module
        from core.pipeline_backends.minimax_music3 import MiniMaxMusic3Mixin
        class Component(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False)
                self.model=SimpleNamespace(layers=[])
                self.config=SimpleNamespace(num_codebooks=2,sampling_rate=8000)
        restore_topology(self.model,self.spec)
        flow=self.flow_pipeline()
        language,depth,vocoder=Component(),Component(),Component()
        class Manager(MiniMaxMusic3Mixin):
            device='cpu'
            is_minimax_music3_model=True
        manager=Manager()
        manager.minimax_music3_components=dict(tokenizer=object(),language_model=language,
            rvq_depth_decoder=depth,transformer=self.model,condition_encoder=flow.condition_encoder,
            scheduler=flow.scheduler,vocoder=vocoder,frame_rate=10,
            rvq_depth_decoder_config={'num_codebooks':2})
        fail_decode=False
        def factory(**components):
            pipeline=object.__new__(type(flow))
            pipeline.__dict__.update(flow.__dict__)
            for key,value in components.items():setattr(pipeline,key,value)
            pipeline.encode_text=lambda *a:torch.zeros(2,4)
            pipeline.generate_ar=lambda *a,**kw:SimpleNamespace(frame_hiddens=torch.randn(1,22,6),
                frame_codes=torch.zeros(22,2,dtype=torch.long),prefix_codes=torch.zeros(1,2,dtype=torch.long))
            def decode(*a,**kw):
                if fail_decode:raise RuntimeError('decode failed')
                return torch.full((1,2,32),.1)
            pipeline.decode=decode
            return pipeline
        params=dict(prompt='CPU flow fixture',lyrics='[verse] test',seed=9,audio_duration=2.2,
                    num_inference_steps=3,flow_guidance_scale=1.7,dit_rewiring_actual={'stale':True})
        with patch.object(module,'MiniMaxMusic3Pipeline',side_effect=factory), \
             patch.object(module,'CHUNK_FRAMES',10),patch.object(module,'CHUNK_HOP',5), \
             patch.object(module,'OVERLAP_LATENT_LENGTH',2):
            result=manager._generate_txt2aud_minimax_music3(params)
            self.assertEqual(result.waveform.shape,(2,32))
            self.assertEqual(params['dit_rewiring_actual']['spec'],self.spec)
            self.assertEqual(len(params['dit_rewiring_actual']['chunks']),4)
            self.assertTrue(all(record['flow_evaluations']==6 for record in params['dit_rewiring_actual']['chunks']))
            fail_decode=True
            with self.assertRaisesRegex(RuntimeError,'decode failed'):
                manager._generate_txt2aud_minimax_music3(params)
            self.assertNotIn('dit_rewiring_actual',params)
        self.assertFalse(torch.cuda.is_initialized())

    def test_loaded_model_info_exposes_native_flow_chunk_geometry(self):
        import tempfile
        from core.models.minimax_music3.residual_unet import model_info,temporal_geometry
        from core import pipeline as pipeline_module
        restore_topology(self.model,self.spec)
        condition=self.flow_pipeline().condition_encoder
        components=dict(type='minimax_music3',transformer=self.model,
            condition_encoder=condition,condition_encoder_config=dict(condition.config),
            text_encoder_path='/fixture/official/language_model')
        info=model_info(components)
        self.assertEqual(info['dit_rewiring_spec'],self.spec)
        self.assertEqual(info['dit_rewiring_temporal_geometry'],
            dict(frame_rate=13.,patch_size=1,preview_frames=260,grid_policy='zero_token_tail_v1'))
        info['dit_rewiring_spec']['regions'][0]['id']='detached'
        self.assertEqual(self.model._residual_unet_spec,self.spec)
        default=temporal_geometry({})
        self.assertEqual(default['frame_rate'],44100/512)
        self.assertEqual(default['preview_frames'],689)
        for bad in (0,-1,True,1.5):
            with self.assertRaisesRegex(ValueError,'rates/hops'):
                temporal_geometry({'input_hop_length':bad})
        with self.assertRaisesRegex(ValueError,'channels disagree'):
            model_info({**components,'condition_encoder_config':{'out_dim':8}})
        with self.assertRaisesRegex(ValueError,'conditioning geometry'):
            model_info({'transformer':self.model})
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(pipeline_module,'LAST_MODEL_CONFIG_FILE',Path(folder)/'last_model.json'), \
             patch.object(pipeline_module.ModelLoader,'load_model',return_value=components):
            manager=pipeline_module.DiffusionPipelineManager()
            manager._load_model_locked('diffusers',str(Path(folder)/'music3'))
            self.assertTrue(manager.is_minimax_music3_model)
            self.assertEqual(manager.current_model_info['dit_rewiring_spec'],self.spec)
            self.assertEqual(manager.current_model_info['dit_rewiring_temporal_geometry'],
                             info['dit_rewiring_temporal_geometry'])
        restore_topology(self.model,None)
        self.assertEqual(model_info(components),{'dit_rewiring_spec':None})
        self.assertEqual(model_info({}),{'dit_rewiring_spec':None})
        self.assertFalse(torch.cuda.is_initialized())

    def test_public_inspection_of_native_flat_and_gguf_without_tensor_loading(self):
        import asyncio,json,tempfile
        from contextlib import ExitStack
        import httpx
        from fastapi import FastAPI
        from safetensors.torch import load_file,save_file
        from tests.minimax_music3_flat_dit_fixture import write_tiny_flat_dit_and_official_tree
        from tests.minimax_music3_gguf_fixture import write_gguf
        from core.models.minimax_music3.flat_remap import apply_flat_dit_state_dict
        from core.models.minimax_music3.residual_unet import native_config,default_topology
        from core.rewiring.inspect import inspect_model
        from core.rewiring.spec import checkpoint_metadata
        from api.rewiring_routes import router
        with tempfile.TemporaryDirectory() as folder:
            fixture=write_tiny_flat_dit_and_official_tree(folder)
            official=Path(fixture['official'])
            config_path=official/'transformer'/'config.json'
            config=json.loads(config_path.read_text(encoding='utf-8'));config['num_layers']=36
            config_path.write_text(json.dumps(config),encoding='utf-8')
            flat=load_file(fixture['dit_path'])
            first={key:value for key,value in flat.items() if '.layers.0.' in key}
            for index in range(1,36):
                for key,value in first.items():flat[key.replace('.layers.0.',f'.layers.{index}.')]=value.clone()
            graph=default_topology(native_config(config))
            metadata={'model_type':'minimax_music3',**checkpoint_metadata(graph)}
            save_file(flat,fixture['dit_path'],metadata=metadata)
            native=official/'transformer'/'diffusion_pytorch_model.safetensors'
            save_file(apply_flat_dit_state_dict(flat)['transformer'],str(native),metadata=metadata)
            gguf=Path(fixture['dit_path']).with_suffix('.gguf')
            write_gguf(str(gguf),flat,{'general.architecture':'minimax_music3',**metadata})
            sources=[Path(folder),official,official/'transformer',native,Path(fixture['dit_path']),gguf]
            with ExitStack() as stack:
                for target in ('core.models.common.single_file_format.read_state_dict',
                               'core.models.common.single_file_format.load_component_state_dict',
                               'core.models.common.gguf_container.GGUFStateDict',
                               'core.models.minimax_music3.vendor.transformer_minimax_music3.MiniMaxMusic3Transformer1DModel.from_config'):
                    stack.enter_context(patch(target,side_effect=AssertionError('loaded/allocated weights')))
                infos=[inspect_model(str(source)) for source in sources]
                self.assertTrue(all(info==infos[0] for info in infos))
                info=infos[0]
                self.assertEqual(info['topology'],graph)
                self.assertEqual(info['default_spec'],graph)
                self.assertEqual(info['temporal_geometry']['preview_frames'],689)
                self.assertEqual(info['temporal_geometry']['frame_rate'],44100/512)
                self.assertEqual(info['initial_spec']['regions'][0]['down_factor'],[2,1])
                app=FastAPI();app.include_router(router,prefix='/api/v1')
                async def exercise():
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://cpu') as client:
                        for source in sources:
                            result=await client.post('/api/v1/models/rewiring/inspect',json={'model_path':str(source)})
                            self.assertEqual(result.status_code,200,result.text)
                            self.assertEqual(result.json(),info)
                        result=await client.post('/api/v1/models/rewiring/validate',json={
                            'spec':info['default_spec'],'token_height':692,'token_width':1})
                        self.assertEqual(result.status_code,200,result.text)
                        self.assertEqual([sum(b['depth']==depth for b in result.json()['plan']['blocks'])
                            for depth in (0,1,2)],[8,24,4])
                        adapter=Path(fixture['dit_path']).with_name('adapter.safetensors')
                        save_file({'lora_down.weight':torch.zeros(1,1)},str(adapter),metadata=metadata)
                        result=await client.post('/api/v1/models/rewiring/inspect',json={'model_path':str(adapter)})
                        self.assertEqual(result.status_code,422)
                        nested_adapter=native.with_name('adapter.safetensors')
                        save_file({'lora_down.weight':torch.zeros(1,1)},str(nested_adapter),metadata=metadata)
                        result=await client.post('/api/v1/models/rewiring/inspect',json={'model_path':str(nested_adapter)})
                        self.assertEqual(result.status_code,422)
                        malformed=official/'condition_encoder'/'config.json'
                        original=malformed.read_text(encoding='utf-8')
                        malformed.write_text('[]',encoding='utf-8')
                        result=await client.post('/api/v1/models/rewiring/inspect',json={'model_path':str(native)})
                        self.assertEqual(result.status_code,422)
                        malformed.write_text(original,encoding='utf-8')
                        native_weights=apply_flat_dit_state_dict(flat)['transformer']
                        native_weights['time_proj.weight']=native_weights['time_proj.weight'].to(torch.int8)
                        save_file(native_weights,str(native),metadata=metadata)
                        result=await client.post('/api/v1/models/rewiring/inspect',json={'model_path':str(native)})
                        self.assertEqual(result.status_code,422)
                        self.assertIn('quantized',result.text)
                asyncio.run(exercise())
        from core.training.arch import ARCH_REGISTRY
        self.assertNotIn('minimax_music3',ARCH_REGISTRY)
        self.assertFalse(torch.cuda.is_initialized())
