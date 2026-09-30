"""CPU contracts for packed MiniMax-H3 spatial residual transport."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core.models.minimax_h3.h3_pipeline_ops import build_packed_layout
from core.models.minimax_h3.packed_residual_layout import prepare_layout, coarse_context
from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import OPERATORS, build_spec


class PackedResidualLayoutContracts(unittest.TestCase):
    def setUp(self):
        self.frames, self.height, self.width = 3, 8, 12
        self.native = build_packed_layout(num_text_tokens=4, num_audio_latents=3,
            num_latent_frames=self.frames, latent_height=16, latent_width=24,
            patch_size=(1, 2, 2), keyframe_anchors=('first',))
        self.target = torch.arange(self.native['sequence_length'] - 288, self.native['sequence_length'])
        self.times = torch.zeros(self.native['sequence_length'], dtype=torch.long)
        self.times[self.native['video_indices'][:96]] = 1
        self.layout = self.prepare()
        self.spec = build_spec('minimax_h3', [{'name':'main', 'block_count':7}], [
            dict(id='outer', stream='main', parent=None, start_block=1, end_block=6,
                 down_factor=[2, 2], **OPERATORS),
            dict(id='inner', stream='main', parent='outer', start_block=2, end_block=5,
                 down_factor=[2, 2], **OPERATORS)], {'fixture': 'packed_native'})

    def prepare(self, **changes):
        args = dict(position_ids=self.native['position_ids'], token_tags=self.native['token_tags'],
                    timestep_indices=self.times, target_indices=self.target,
                    frames=self.frames, height=self.height, width=self.width)
        args.update(changes)
        return prepare_layout(**args)

    def test_arbitrary_native_row_order_and_gradients_restore(self):
        total = self.native['sequence_length']
        permutation = torch.randperm(total)
        inverse = permutation.argsort()
        layout = self.prepare(position_ids=self.native['position_ids'][permutation],
            token_tags=self.native['token_tags'][permutation], timestep_indices=self.times[permutation],
            target_indices=inverse[self.target])
        hidden = torch.randn(2, total, 16, requires_grad=True)
        restored = layout.unpack(layout.pack(hidden))
        self.assertTrue(torch.equal(restored, hidden))
        restored.sum().backward()
        self.assertTrue(torch.equal(hidden.grad, torch.ones_like(hidden)))

    def test_reference_audio_text_rows_and_media_clock_survive_both_levels(self):
        context = self.layout.context
        for h, w in ((4, 6), (2, 3)):
            context = coarse_context(context, torch.zeros(2, self.layout.prefix + 3*h*w, 16),
                                     3*h, w, self.layout.prefix)
            for name in ('position_ids', 'token_tags', 'timestep_indices'):
                self.assertTrue(torch.equal(context[name][:self.layout.prefix],
                                            self.layout.context[name][:self.layout.prefix]))
            clock = context['position_ids'][self.layout.prefix:, 0].reshape(3, h*w)
            native_clock = self.native['position_ids'][self.target, 0].reshape(3, 96)[:, 0]
            self.assertTrue(torch.equal(clock, native_clock[:, None].expand_as(clock)))

    def test_mixed_spatial_pins_refuse_but_whole_frame_pins_survive(self):
        times = self.times.clone(); times[self.target[0]] = 1
        layout = self.prepare(timestep_indices=times)
        with self.assertRaisesRegex(ValueError, 'mix pinned'):
            coarse_context(layout.context, torch.zeros(1, layout.prefix+72, 16), 12, 6, layout.prefix)
        times[self.target[:96]] = 1
        layout = self.prepare(timestep_indices=times)
        ctx = coarse_context(layout.context, torch.zeros(1, layout.prefix+72, 16), 12, 6, layout.prefix)
        self.assertTrue((ctx['timestep_indices'][layout.prefix:layout.prefix+24] == 1).all())
        self.assertTrue((ctx['timestep_indices'][layout.prefix+24:] == 0).all())

    def test_invalid_target_indices_and_nonrectangular_coordinates_refuse(self):
        for target in (self.target[:-1], self.target.flip(0), torch.zeros_like(self.target),
                       self.target + self.native['sequence_length']):
            with self.subTest(target=target[:3].tolist()), self.assertRaises(ValueError):
                self.prepare(target_indices=target)
        positions = self.native['position_ids'].clone(); positions[self.target[1], 0] += 1
        with self.assertRaisesRegex(ValueError, 'rectangular'):
            self.prepare(position_ids=positions)

    def test_native_blocks_restore_rows_and_backpropagate_without_cuda(self):
        from core.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
        torch.manual_seed(208)
        model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
            hidden_size=16, num_layers=7, num_refiner_layers=2, ffn_dim=32,
            in_channels=4, audio_in_channels=6, patch_size=(1,2,2), text_dim=10,
            freq_dim=8, time_embed_hidden_dim=16, time_embed_dim=8, rope_freq_dim=2,
            adaln_curve_grid=33).float().eval()
        with torch.no_grad(): model.adaln_t_table.copy_(torch.randn_like(model.adaln_t_table))
        model._stamp_attention_backend()
        temb = model.adaln_t_table[[5, 24]]
        original = torch.randn(1, self.native['sequence_length'], 16, requires_grad=True)
        observed = []
        def execute(index, hidden, context):
            observed.append(hidden.shape[1])
            labels = context['timestep_indices'] * 3 + context['token_tags']
            return model.transformer_blocks[index](hidden, temb, labels, model.rope(context['position_ids']))
        result = run_rewired_stream(self.layout.pack(original), self.spec, 'main', 24, 12,
            self.layout.prefix, deepcopy(self.layout.context), execute, coarse_context, grid_groups=3)
        restored = self.layout.unpack(result)
        self.assertEqual(observed, [self.layout.prefix+n for n in (288,72,18,18,18,72,288)])
        self.assertEqual(restored.shape, original.shape)
        expected = restored.detach().clone()
        restored.square().mean().backward()
        self.assertTrue(torch.isfinite(original.grad).all())
        self.assertGreater(torch.count_nonzero(original.grad).item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for b in model.transformer_blocks for p in b.parameters()))
        with torch.no_grad():
            inference = run_rewired_stream(self.layout.pack(original.detach()), self.spec, 'main', 24, 12,
                self.layout.prefix, self.layout.context, execute, coarse_context, grid_groups=3)
        torch.testing.assert_close(self.layout.unpack(inference), expected, rtol=2e-5, atol=2e-6)
        self.assertFalse(torch.cuda.is_initialized())

class MiniMaxH3WrapperResidualContracts(unittest.TestCase):
    prepare = PackedResidualLayoutContracts.prepare

    def setUp(self):
        PackedResidualLayoutContracts.setUp(self)
        from core.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
        from core.models.minimax_h3.residual_unet import native_config, wrap_transformer
        torch.manual_seed(209)
        self.inner = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=16,
            hidden_size=16, num_layers=7, num_refiner_layers=2, ffn_dim=32,
            in_channels=4, audio_in_channels=6, patch_size=(1,2,2), text_dim=10,
            freq_dim=8, time_embed_hidden_dim=16, time_embed_dim=8, rope_freq_dim=2,
            adaln_curve_grid=33).float().eval()
        with torch.no_grad(): self.inner.adaln_t_table.copy_(torch.randn_like(self.inner.adaln_t_table))
        self.spec = build_spec('minimax_h3', [{'name':'main', 'block_count':7}],
                              self.spec['regions'], native_config(self.inner.config))
        self.wrapper = wrap_transformer(self.inner, self.spec)
        self.args = dict(hidden_states=torch.randn(2, self.native['video_indices'].numel(), 16, requires_grad=True),
            audio_hidden_states=torch.randn(2, self.native['audio_indices'].numel(), 6, requires_grad=True),
            encoder_hidden_states=torch.randn(2, 4, 10, requires_grad=True),
            timestep=torch.tensor([.3, .9]), timestep_indices=self.times,
            token_tags=self.native['token_tags'], position_ids=self.native['position_ids'],
            video_indices=self.native['video_indices'], audio_indices=self.native['audio_indices'],
            text_indices=self.native['text_indices'], return_dict=False,
            residual_video_grid=dict(frames=3, height=8, width=12, target_indices=self.target))

    def test_wrapper_native_heads_and_checkpointed_gradients(self):
        observed = []
        hooks = [block.register_forward_pre_hook(lambda module, args: observed.append(args[0].shape[1]))
                 for block in self.inner.transformer_blocks]
        try:
            expected = self.wrapper(**self.args)
        finally:
            for handle in hooks: handle.remove()
        self.assertEqual(observed, [106+n for n in (288,72,18,18,18,72,288)])
        self.assertEqual(expected[0].shape, self.args['hidden_states'].shape)
        self.assertEqual(expected[1].shape, self.args['audio_hidden_states'].shape)
        sum(value.square().mean() for value in expected).backward()
        expected_grad = {name:p.grad.clone() for name,p in self.inner.named_parameters() if p.grad is not None}
        input_grad = {key:self.args[key].grad.clone() for key in
                      ('hidden_states', 'audio_hidden_states', 'encoder_hidden_states')}
        self.inner.zero_grad(set_to_none=True)
        for key in input_grad: self.args[key].grad = None
        self.inner.enable_gradient_checkpointing()
        got = self.wrapper(**self.args)
        for a,b in zip(got,expected): torch.testing.assert_close(a,b,rtol=0,atol=0)
        sum(value.square().mean() for value in got).backward()
        for name,p in self.inner.named_parameters():
            if name in expected_grad:
                torch.testing.assert_close(p.grad,expected_grad[name],rtol=0,atol=0)
        for key in input_grad:
            torch.testing.assert_close(self.args[key].grad,input_grad[key],rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_block_swap_runs_every_native_block_and_dense_delegation_restores(self):
        from core.models.minimax_h3.residual_unet import restore_topology
        class Offloader:
            blocks_to_swap = 2
            calls = []
            def wait_for_block(self,index): self.calls.append(('wait',index))
            def submit_move_blocks_forward(self,index): self.calls.append(('submit',index))
        offloader = Offloader()
        self.wrapper.attach_block_offloader(offloader)
        with torch.no_grad(): self.wrapper(**self.args)
        self.assertEqual(offloader.calls, [call for index in range(7)
            for call in (('wait',index),('submit',index))])
        self.wrapper.attach_block_offloader(None)
        restore_topology(self.wrapper, None)
        self.assertFalse(self.wrapper._any_feature_active())
        args = {key:value for key,value in self.args.items() if key != 'residual_video_grid'}
        with torch.no_grad():
            expected = self.inner(**args); got = self.wrapper(**args)
        for a,b in zip(got,expected): self.assertTrue(torch.equal(a,b))

    def test_preflight_refuses_conflicts_and_deep_mixed_pins_before_projection(self):
        from unittest.mock import patch
        with patch.object(self.inner.proj_in, 'forward', side_effect=AssertionError('projection ran')):
            for key,value in (('_fbcache',object()), ('_skip_blocks',frozenset([1])),
                              ('_residual_probe',object())):
                with self.subTest(key=key), patch.object(self.wrapper,key,value), self.assertRaises(ValueError):
                    self.wrapper(**self.args)
            with patch.object(self.inner,'_attention_plan',object(),create=True), self.assertRaises(ValueError):
                self.wrapper(**self.args)
            for override in ({'residual_video_grid':None}, {'attention_kwargs':{'scale':2.}},
                             {'audio_indices':self.args['text_indices']},
                             {'timestep':torch.tensor([.3])}):
                with self.subTest(override=tuple(override)), self.assertRaises(ValueError):
                    self.wrapper(**{**self.args, **override})
            times = self.times.clone()
            # One homogeneous first-level cell, mixed inside the deeper 4x4 cell.
            times[self.target[torch.tensor([0,1,12,13])]] = 1
            with self.assertRaisesRegex(ValueError,'mix pinned'):
                self.wrapper(**{**self.args,'timestep_indices':times})

    def test_topology_identity_cannot_be_transferred_to_another_native_config(self):
        from core.models.minimax_h3.residual_unet import restore_topology
        config = dict(self.inner.config)
        self.inner.register_to_config(rope_theta=config['rope_theta']+1)
        with self.assertRaisesRegex(ValueError,'native config'):
            restore_topology(self.wrapper,self.spec)
        self.inner.register_to_config(**config)
        broken = deepcopy(self.spec); broken['architecture']='ltx2'
        with self.assertRaises(ValueError): restore_topology(self.wrapper,broken)

    def test_unpruned_modulation_and_native_output_object(self):
        from core.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
        from core.models.minimax_h3.residual_unet import native_config, wrap_transformer, GRID_POLICY_KEY
        cfg = native_config(self.inner.config)
        cfg.pop(GRID_POLICY_KEY); cfg['adaln_curve_grid'] = None
        model = MiniMaxH3Transformer3DModel(**cfg).float().eval()
        spec = build_spec('minimax_h3', [{'name':'main','block_count':7}], self.spec['regions'], native_config(model.config))
        wrapper = wrap_transformer(model,spec)
        result = wrapper(**{**self.args, 'return_dict':True})
        self.assertEqual(result.sample.shape,self.args['hidden_states'].shape)
        self.assertEqual(result.audio_sample.shape,self.args['audio_hidden_states'].shape)
        (result.sample.square().mean()+result.audio_sample.square().mean()).backward()
        self.assertTrue(torch.isfinite(model.time_embedder.linear_1.weight.grad).all())
        self.assertGreater(torch.count_nonzero(model.time_embedder.linear_1.weight.grad).item(),0)

    def test_permuted_pinned_frame_keeps_original_video_head_order(self):
        indices = self.args['video_indices']
        permutation = torch.cat((torch.arange(192,288),torch.arange(96),torch.arange(96,192),torch.arange(288,384)))
        args = {**self.args, 'video_indices':indices[permutation],
                'hidden_states':self.args['hidden_states'][:,permutation]}
        times = self.times.clone(); times[self.target[96:192]] = 1
        with torch.no_grad():
            expected = self.wrapper(**{**self.args,'timestep_indices':times})
            got = self.wrapper(**{**args,'timestep_indices':times})
        for a,b in ((got[0],expected[0][:,permutation]),(got[1],expected[1])):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_actual_native_sampler_runs_nested_grid_and_preserves_condition_rows(self):
        from core.models.minimax_h3.h3_pipeline_ops import denoise
        from core.models.minimax_h3.vendor import MiniMaxH3Scheduler
        layout = build_packed_layout(4,3,16,24,3,keyframe_anchors=('first',))
        video = torch.randn(384,16); audio = torch.randn(6,6)
        anchor = video[:96].clone()
        observed = []
        hooks = [b.register_forward_pre_hook(lambda module,args:observed.append(args[0].shape[1]))
                 for b in self.inner.transformer_blocks]
        try:
            with torch.no_grad():
                result = denoise(self.wrapper,MiniMaxH3Scheduler(),MiniMaxH3Scheduler(),
                    prompt_embeds=torch.randn(1,4,10), layout=layout,
                    video_rows=video, audio_rows=audio, num_inference_steps=4, device='cpu')
        finally:
            for hook in hooks: hook.remove()
        self.assertEqual(observed, [106+n for n in (288,72,18,18,18,72,288)]*3)
        self.assertTrue(torch.equal(result[0][:96],anchor))
        self.assertEqual(result[0].shape,video.shape);self.assertEqual(result[1].shape,audio.shape)
        self.assertTrue(all(torch.isfinite(value).all() for value in result))
        self.assertFalse(torch.cuda.is_initialized())

    def test_sampler_rejects_forecasters_and_missing_geometry_before_scheduler_mutation(self):
        from unittest.mock import patch
        from core.models.minimax_h3.h3_pipeline_ops import denoise
        from core.models.minimax_h3.vendor import MiniMaxH3Scheduler
        layout = build_packed_layout(4,3,16,24,3)
        scheduler = MiniMaxH3Scheduler()
        args = dict(prompt_embeds=torch.randn(1,4,10), layout=layout,
            video_rows=torch.randn(288,16),audio_rows=torch.randn(6,6),num_inference_steps=4,device='cpu')
        with patch.object(scheduler,'set_shift',side_effect=AssertionError('scheduler mutated')):
            for params in ({'spectrum_enable':True},{'fbcache_enable':True,'fbcache_threshold':.1}):
                with self.assertRaisesRegex(ValueError,'Spectrum and FBCache'):
                    denoise(self.wrapper,scheduler,MiniMaxH3Scheduler(),**args,spectrum_params=params)
            bad = dict(layout);bad.pop('target_video_grid')
            with self.assertRaisesRegex(ValueError,'target geometry'):
                denoise(self.wrapper,scheduler,MiniMaxH3Scheduler(),**{**args,'layout':bad})

    def test_actual_training_step_runs_native_nested_av_loss_and_updates_weights(self):
        from types import SimpleNamespace
        from core.training.ops.minimax_h3_ops import train_step
        metrics = {}
        trainer = SimpleNamespace(device=torch.device('cpu'),training_dtype=torch.float32,
            transformer=self.wrapper,audio_loss_weight=.7,reconstruction_loss_weight=0.,
            log_extra_metric=lambda name,value:metrics.update({name:float(value)}))
        self.inner.enable_gradient_checkpointing()
        before = self.inner.transformer_blocks[3].attn.to_q.weight.detach().clone()
        optimizer = torch.optim.AdamW(self.inner.parameters(),lr=1e-3)
        loss,pred,recon = train_step(trainer,torch.randn(1,4,3,16,24),torch.randn(1,4,10),
            dict(num_text_tokens=torch.tensor([4]),audio_latents=torch.randn(1,6,6),
                 audio_present=torch.tensor([True]),
                 audio_valid_rows=torch.tensor([[True,True,True,True,False,False]])),
            timesteps=torch.tensor([.4]))
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(loss),0)
        self.assertEqual(float(loss.detach()),float(pred));self.assertEqual(recon,0.)
        self.assertEqual(metrics['h3_audio_present'],1.)
        self.assertGreater(metrics['h3_video_loss'],0);self.assertGreater(metrics['h3_audio_loss'],0)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before,self.inner.transformer_blocks[3].attn.to_q.weight))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for b in self.inner.transformer_blocks for p in b.parameters()))
        self.assertEqual(trainer._minimax_h3_layout_cache[(4,3,16,24,3,'cpu')]['target_video_grid'],
                         dict(frames=3,height=8,width=12,start_row=10))
        self.assertFalse(torch.cuda.is_initialized())

    def test_ref2va_native_target_geometry_excludes_variable_reference_banks(self):
        from core.models.minimax_h3.h3_pipeline_ops import build_ref2va_packed_layout
        from core.models.minimax_h3.residual_unet import forward_kwargs
        layout = build_ref2va_packed_layout([1,1,1,1],[('video',False),('audio',False)],
            [(2,8,12)],[4],3,16,24,3,pinned_video_frames=(1,))
        grid = forward_kwargs(self.wrapper,layout)['residual_video_grid']
        self.assertEqual((grid['frames'],grid['height'],grid['width']),(3,8,12))
        self.assertEqual(grid['target_indices'].numel(),288)
        self.assertEqual(int(grid['target_indices'][0]),layout['sequence_length']-288)
        self.assertTrue(torch.isin(grid['target_indices'],layout['video_indices']).all())
        self.assertNotEqual(int(layout['video_indices'][0]),int(grid['target_indices'][0]))

    def test_real_lora_save_resume_restores_graph_weights_and_native_predictions(self):
        import tempfile
        from types import SimpleNamespace
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        from unittest.mock import patch
        from core.models.minimax_h3.residual_unet import restore_topology
        from core.rewiring.spec import read_checkpoint_metadata, META_VERSION, META_SPEC, META_HASH
        from core.training.adapters.minimax_h3_adapter import MiniMaxH3LoRAAdapter
        from core.training.lora_trainer import LoRATrainer
        from core.training.ops.minimax_h3_ops import train_step
        base = deepcopy(self.inner)
        restore_topology(base,None)
        def trainer_for(model):
            trainer = SimpleNamespace(transformer=model,transformer_original=getattr(model,'transformer',model),
                minimax_h3_components={'transformer':model},config={},is_minimax_h3=True,
                lora_layers={},lora_rank=2,lora_alpha=2,device=torch.device('cpu'),training_dtype=torch.float32,
                audio_loss_weight=1.,reconstruction_loss_weight=0.,log_prefix='[H3 residual CPU]',
                log_extra_metric=lambda name,value:None)
            trainer.transformer.requires_grad_(False)
            trainer.adapter = MiniMaxH3LoRAAdapter(trainer,lora_rank=2,lora_alpha=2)
            trainer.adapter.apply_lora_to_unet(trainer.lora_layers)
            return trainer
        trained = trainer_for(self.wrapper)
        params = [p for layer in trained.lora_layers.values() for p in layer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params,lr=1e-3)
        loss,_,_ = train_step(trained,torch.randn(1,4,3,16,24),torch.randn(1,4,10),
            dict(num_text_tokens=torch.tensor([4]),audio_latents=torch.randn(1,6,6)),timesteps=torch.tensor([.4]))
        loss.backward();optimizer.step()
        self.assertTrue(any(torch.count_nonzero(layer.lora_up.weight).item()>0 for layer in trained.lora_layers.values()))
        with torch.no_grad(): expected = trained.transformer(**self.args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trained.safetensors'
            trained.adapter.save_checkpoint(trained.lora_layers,step=8,epoch=1,output_path=path)
            with safe_open(str(path),framework='pt',device='cpu') as handle:
                metadata = handle.metadata()
            self.assertEqual(read_checkpoint_metadata(metadata),self.spec)
            generation_base = deepcopy(base)
            resumed = trainer_for(base)
            step = LoRATrainer.load_checkpoint(resumed,str(path))
            self.assertEqual(step,8)
            self.assertEqual(resumed.config['dit_rewiring_spec'],self.spec)
            self.assertIs(resumed.minimax_h3_components['transformer'],resumed.transformer)
            self.assertIs(resumed.transformer_original,base)
            with torch.no_grad(): got = resumed.transformer(**self.args)
            for a,b in zip(got,expected): torch.testing.assert_close(a,b,rtol=0,atol=0)
            for name,layer in resumed.lora_layers.items():
                for key,value in layer.branch_tensors().items():
                    self.assertTrue(torch.equal(value,trained.lora_layers[name].branch_tensors()[key]))
            from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin
            from core.adapters import AdapterIncompatible
            class Backend(MiniMaxH3Mixin):
                def __init__(self,model):
                    self.minimax_h3_components = {'transformer':model,'variant':'fl2va'}
                @staticmethod
                def _minimax_h3_resolve_lora_path(path): return str(path)
            backend = Backend(generation_base)
            self.assertGreater(backend._load_lora_minimax_h3([{'path':str(path),'strength':1.}],{}),0)
            callable_model,offloader,probe = backend._ensure_minimax_h3_swap_and_offload({},torch.device('cpu'))
            self.assertIsNone(offloader);self.assertIsNone(probe)
            with torch.no_grad(): generated = callable_model(**self.args)
            for a,b in zip(generated,expected): torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
            with patch.object(backend,'_minimax_h3_move',side_effect=AssertionError('device move ran')):
                for params in ({'spectrum_enable':True},{'_minimax_h3_debug_skip_blocks':[1]},
                               {'_minimax_h3_debug_probe_residuals':True},
                               {'fbcache_enable':True,'fbcache_threshold':.1}):
                    with self.assertRaisesRegex(ValueError,'refuses caches'):
                        backend._ensure_minimax_h3_swap_and_offload(params,torch.device('cpu'))
            self.assertGreater(backend._unload_lora_minimax_h3(),0)
            self.assertIsNone(getattr(generation_base,'_residual_unet_spec',None))
            dense_md = {key:value for key,value in metadata.items() if key not in (META_VERSION,META_SPEC,META_HASH)}
            dense = Path(directory)/'dense.safetensors'
            save_file(load_file(str(path)),str(dense),metadata=dense_md)
            with self.assertRaises(AdapterIncompatible):
                backend._load_lora_minimax_h3([{'path':str(path),'strength':1.},
                                              {'path':str(dense),'strength':1.}],{})
            self.assertIsNone(getattr(generation_base,'_residual_unet_spec',None))
            snapshots = {name:{key:value.clone() for key,value in layer.branch_tensors().items()}
                         for name,layer in resumed.lora_layers.items()}
            with patch('safetensors.torch.load_file',side_effect=AssertionError('tensor load ran')):
                with self.assertRaisesRegex(ValueError,'topology disagree'):
                    LoRATrainer.load_checkpoint(resumed,str(dense))
            self.assertEqual(resumed.config['dit_rewiring_spec'],self.spec)
            for name,layer in resumed.lora_layers.items():
                for key,value in layer.branch_tensors().items():
                    self.assertTrue(torch.equal(value,snapshots[name][key]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_resolution_dry_run_preserves_state_and_requires_wrapper_owner(self):
        from core.models.minimax_h3.residual_unet import configure_training_topology,restore_topology
        base = deepcopy(self.inner);restore_topology(base,None)
        config = {}
        graph = configure_training_topology(base,config,resume_spec=self.spec,install=False)
        self.assertEqual(graph,self.spec);self.assertEqual(config,{})
        self.assertIsNone(getattr(base,'_residual_unet_spec',None))
        with self.assertRaisesRegex(ValueError,'wrapper or trainer'):
            configure_training_topology(base,config,resume_spec=self.spec)
        self.assertIsNone(getattr(base,'_residual_unet_spec',None));self.assertEqual(config,{})
        self.assertEqual(configure_training_topology(self.wrapper,config),self.spec)
        for feature in ('tread_enable','repa_enable','blockskip_enable'):
            with self.subTest(feature=feature),self.assertRaises(ValueError):
                configure_training_topology(self.wrapper,{feature:True})
