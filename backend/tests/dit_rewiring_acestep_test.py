"""CPU contracts for ACE-Step temporal decoder transport."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import OPERATORS, build_spec, read_checkpoint_metadata


class AceStepRuntimeContracts(unittest.TestCase):
    def setUp(self):
        for obj, name, value in ((torch.cuda,"get_device_capability",lambda *a,**k:(8,9)),
                (torch.cuda,"_lazy_init",lambda *a,**k:None),(torch._C,"_cuda_init",lambda *a,**k:None)):
            guard = patch.object(obj,name,value);guard.start();self.addCleanup(guard.stop)
        from core.models.acestep.vendor.configuration_acestep_v15 import AceStepConfig
        from core.models.acestep.vendor.modeling_acestep_v15_turbo import AceStepDiTModel
        from core.models.acestep.residual_unet import native_config
        torch.manual_seed(94)
        config = AceStepConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=7,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, patch_size=2,
            in_channels=12, audio_acoustic_hidden_dim=4, encoder_hidden_size=16,
            sliding_window=3, attention_dropout=0)
        config._attn_implementation = "eager"
        self.model = AceStepDiTModel(config)
        self.spec = build_spec("acestep", [{"name":"main","block_count":7}], [
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,1],**OPERATORS),
            dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,1],**OPERATORS)],native_config(config))
        self.audio = torch.randn(2,48,4,requires_grad=True)
        self.context = torch.randn(2,48,8,requires_grad=True)
        self.encoder = torch.randn(2,5,16,requires_grad=True)

    def forward(self, **overrides):
        args = dict(hidden_states=self.audio, timestep=torch.tensor([.5,.7]),
            timestep_r=torch.tensor([.1,.2]), attention_mask=torch.ones(2,48),
            encoder_hidden_states=self.encoder, encoder_attention_mask=torch.ones(2,5),
            context_latents=self.context, use_cache=False)
        args.update(overrides)
        return self.model(**args)

    def test_training_inheritance_resume_and_atomic_refusals(self):
        from core.models.acestep.residual_unet import configure_training_topology, restore_topology
        restore_topology(self.model, self.spec)
        config = {}
        self.assertEqual(configure_training_topology(self.model, config), self.spec)
        self.assertEqual(config["dit_rewiring_spec"], self.spec)
        untouched = {}
        self.assertEqual(configure_training_topology(self.model, untouched, resuming=True,
            resume_spec=self.spec, install=False), self.spec)
        self.assertEqual(untouched, {})
        changed = deepcopy(self.spec); changed["regions"][1]["end_block"] = 4
        for invalid, options in (({}, dict(resume_spec=None)),
                ({"dit_rewiring_mode":"configure","dit_rewiring_spec":changed}, {}),
                ({"block_skip_rate":0.5}, {}), ({"tread_enable":True}, {})):
            before = deepcopy(invalid)
            with self.assertRaises(ValueError):
                configure_training_topology(self.model, invalid, resuming=True, **options)
            self.assertEqual(invalid, before)
            self.assertEqual(self.model._residual_unet_spec, self.spec)
        self.model.config._attn_implementation = "flash_attention_2"
        with self.assertRaisesRegex(ValueError, "SDPA"):
            configure_training_topology(self.model, {})
        self.model.config._attn_implementation = "eager"
        restore_topology(self.model, None)
        requested = {"dit_rewiring_mode":"inherit","dit_rewiring_spec":self.spec}
        self.assertEqual(configure_training_topology(self.model, requested, resuming=True,
            resume_spec=self.spec), self.spec)
        restore_topology(self.model, None)
        config = {}
        self.assertIsNone(configure_training_topology(self.model, config))
        self.assertIsNone(config["dit_rewiring_spec"])
        self.assertFalse(hasattr(self.model, "_residual_unet_spec"))
        owner = torch.nn.Module()
        owner.decoder = self.model
        owner.config = deepcopy(self.model.config)
        self.model.config.intermediate_size += 8
        try:
            with self.assertRaisesRegex(ValueError, "owner/decoder"):
                configure_training_topology(owner, {}, resume_spec=self.spec, install=False)
            self.assertFalse(hasattr(owner, "_residual_unet_spec"))
        finally:
            self.model.config.intermediate_size -= 8

    def test_real_lora_resume_restores_graph_before_native_prediction(self):
        import tempfile
        from types import SimpleNamespace
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        from core.models.acestep.residual_unet import restore_topology
        from core.training.adapters.acestep_adapter import AceStepLoRAAdapter
        from core.training.lora_trainer import LoRATrainer
        owner = torch.nn.Module()
        owner.decoder = self.model
        owner.config = self.model.config
        trainer = SimpleNamespace(transformer=owner, config={}, is_acestep=True,
            log_prefix="[CPU]", lora_rank=2, lora_alpha=2, lora_layers={})
        adapter = AceStepLoRAAdapter(trainer, 2, 2, torch.float32)
        trainer.adapter = adapter
        adapter.apply_lora_to_unet(trainer.lora_layers)
        restore_topology(owner, self.spec)
        with torch.no_grad():
            for layer in trainer.lora_layers.values():
                for tensor in layer.branch_tensors().values():
                    tensor.fill_(.03)
        expected = self.forward()[0].detach()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"audio.safetensors"
            adapter.save_checkpoint(trainer.lora_layers, 3, 1, path)
            with safe_open(str(path), framework="pt") as handle:
                metadata = handle.metadata()
            before = {name:{key:value.detach().clone() for key,value in layer.branch_tensors().items()}
                      for name,layer in trainer.lora_layers.items()}
            bad = Path(temp)/"dense.safetensors"
            dense_metadata = {key:value for key,value in metadata.items() if not key.startswith("sushi.rewire.")}
            save_file(load_file(str(path)), str(bad), metadata=dense_metadata)
            with self.assertRaisesRegex(ValueError, "topology disagree"):
                LoRATrainer.load_checkpoint(trainer, str(bad))
            self.assertEqual(owner._residual_unet_spec, self.spec)
            for name,layer in trainer.lora_layers.items():
                for key,value in layer.branch_tensors().items():
                    torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)
            restore_topology(owner, None)
            with torch.no_grad():
                for layer in trainer.lora_layers.values():
                    for tensor in layer.branch_tensors().values():
                        tensor.zero_()
            self.assertEqual(LoRATrainer.load_checkpoint(trainer, str(path)), 3)
            self.assertEqual(owner._residual_unet_spec, self.spec)
            self.assertEqual(self.model._residual_unet_spec, self.spec)
            torch.testing.assert_close(self.forward()[0], expected, rtol=0, atol=0)

    def test_generation_lora_topology_scope_and_native_prediction(self):
        import tempfile
        from types import SimpleNamespace
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        from core.adapters import AdapterIncompatible, CompositeAdapterLayer
        from core.models.acestep.residual_unet import restore_topology
        from core.pipeline_backends.acestep import AceStepMixin, _restores_acestep_lora
        from core.training.adapters.acestep_adapter import AceStepLoRAAdapter
        class Backend(AceStepMixin):
            def __init__(self, model):
                self.acestep_components = {"dit":model}
                self.is_acestep_model = True
                self.device = "cpu"
        owner = torch.nn.Module()
        owner.decoder = self.model
        owner.config = self.model.config
        generation = deepcopy(owner)
        def predict(model):
            return model.decoder(self.audio,torch.tensor([.5,.7]),torch.tensor([.1,.2]),
                torch.ones(2,48),self.encoder,torch.ones(2,5),self.context,use_cache=False)[0]
        dense = predict(generation).detach()
        adapter = AceStepLoRAAdapter(SimpleNamespace(transformer=owner,config={}), 2, 2, torch.float32)
        layers = {}
        adapter.apply_lora_to_unet(layers)
        with torch.no_grad():
            for layer in layers.values():
                for tensor in layer.branch_tensors().values(): tensor.fill_(.03)
        restore_topology(owner,self.spec)
        expected = predict(owner).detach()
        backend = Backend(generation)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"rewired.safetensors"
            adapter.save_checkpoint(layers,3,1,path)
            with safe_open(str(path),framework="pt") as handle: metadata=handle.metadata()
            dense_path = Path(temp)/"dense.safetensors"
            save_file(load_file(str(path)),str(dense_path),
                metadata={key:value for key,value in metadata.items() if not key.startswith("sushi.rewire.")})
            configs = [{"path":str(path),"strength":1.0}]
            original = generation.decoder.layers[0].self_attn.q_proj
            with self.assertRaises(AdapterIncompatible) as rejected:
                backend._load_lora_acestep(configs+[{"path":str(dense_path),"strength":1.0}])
            self.assertEqual(rejected.exception.code,"lora_rewiring_incompatible")
            self.assertIs(generation.decoder.layers[0].self_attn.q_proj,original)
            self.assertFalse(hasattr(generation,"_residual_unet_spec"))
            self.assertGreater(backend._load_lora_acestep(configs),0)
            self.assertEqual(generation._residual_unet_spec,self.spec)
            self.assertEqual(generation.decoder._residual_unet_spec,self.spec)
            torch.testing.assert_close(predict(generation),expected,rtol=2e-5,atol=2e-6)
            backend._unload_lora_acestep()
            self.assertFalse(hasattr(generation,"_residual_unet_spec"))
            self.assertFalse(hasattr(generation.decoder,"_residual_unet_spec"))
            self.assertIs(generation.decoder.layers[0].self_attn.q_proj,original)
            torch.testing.assert_close(predict(generation),dense,rtol=0,atol=0)
            restore_topology(generation,self.spec)
            with self.assertRaises(AdapterIncompatible):
                backend._load_lora_acestep([{"path":str(dense_path),"strength":1.0}])
            self.assertEqual(generation._residual_unet_spec,self.spec)
            backend._load_lora_acestep(configs)
            backend._unload_lora_acestep()
            self.assertEqual(generation._residual_unet_spec,self.spec)
            restore_topology(generation,None)
            @_restores_acestep_lora
            def failed_request(instance):
                instance._load_lora_acestep(configs)
                raise RuntimeError("denoise failed")
            with self.assertRaisesRegex(RuntimeError,"denoise failed"):
                failed_request(backend)
            self.assertFalse(hasattr(generation,"_residual_unet_spec"))
            self.assertFalse(any(isinstance(module,CompositeAdapterLayer) for module in generation.modules()))
            @_restores_acestep_lora
            def zero_targets(instance):
                with patch.object(instance._acestep_lora_session,"load",return_value=SimpleNamespace(applied=0)):
                    instance._load_lora_acestep(configs)
                self.assertEqual(generation._residual_unet_spec,self.spec)
            zero_targets(backend)
            self.assertFalse(hasattr(generation,"_residual_unet_spec"))
            replacement = deepcopy(generation)
            restore_topology(replacement,self.spec)
            backend._load_lora_acestep(configs)
            backend.acestep_components={"dit":replacement}
            backend._unload_lora_acestep()
            self.assertFalse(hasattr(generation,"_residual_unet_spec"))
            self.assertEqual(replacement._residual_unet_spec,self.spec)
            self.assertFalse(any(isinstance(module,CompositeAdapterLayer) for module in replacement.modules()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_shapes_positions_masks_hooks_and_dense_restore(self):
        from core.models.acestep.residual_unet import restore_topology, native_config, model_topology_metadata
        dense = self.forward()[0].detach()
        restore_topology(self.model,self.spec)
        self.assertEqual(native_config(self.model.config),native_config(native_config(self.model.config)))
        identities = {name:id(value) for name,value in self.model.named_parameters()}
        seen = []
        def observe(layer,args):
            hidden, rotary, timestep, mask, positions = args[:5]
            seen.append((hidden.shape[1],positions.clone(),None if mask is None else mask.clone()))
            self.assertIs(args[9],encoder_projection[0])
            self.assertFalse(args[7]);self.assertIsNone(args[5])
        encoder_projection = []
        handles = [self.model.condition_embedder.register_forward_hook(lambda m,a,out:encoder_projection.append(out))]
        handles += [layer.register_forward_pre_hook(observe) for layer in self.model.layers]
        try:
            with patch("core.models.acestep.vendor.modeling_acestep_v15_turbo.create_4d_mask",
                       side_effect=AssertionError("Dense masks must not allocate in residual execution")):
                output, cache = self.forward()
        finally:
            for handle in handles: handle.remove()
        self.assertIsNone(cache);self.assertEqual(output.shape,self.audio.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual([item[0] for item in seen],[24,12,6,6,6,12,24])
        for index, center in ((0,0.),(1,.5),(2,1.5),(5,.5),(6,0.)):
            self.assertEqual(seen[index][1][0,0].item(),center)
        inner_mask = seen[2][2][0,0]
        self.assertEqual(inner_mask[0,0].item(),0)
        self.assertLess(inner_mask[0,1].item(),-1e10)
        self.assertEqual(identities,{name:id(value) for name,value in self.model.named_parameters()})
        self.assertEqual(read_checkpoint_metadata(model_topology_metadata(self.model)),self.spec)
        restore_topology(self.model,None)
        torch.testing.assert_close(self.forward()[0],dense,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_checkpoint_values_and_gradients_and_sdpa_parity(self):
        from core.models.acestep.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        values = [self.audio,self.context,self.encoder,*self.model.parameters()]
        direct = self.forward()[0]
        before = torch.autograd.grad(direct.square().mean(),values,allow_unused=True)
        self.model.gradient_checkpointing_enable()
        checked = self.forward()[0]
        after = torch.autograd.grad(checked.square().mean(),values,allow_unused=True)
        torch.testing.assert_close(checked,direct,rtol=0,atol=0)
        for a,b in zip(before,after):
            self.assertEqual(a is None,b is None)
            if a is not None:
                torch.testing.assert_close(a,b);self.assertTrue(torch.isfinite(b).all())
        named = dict(zip(["audio","context","encoder",*dict(self.model.named_parameters())],after))
        for name in ("audio","context","encoder","layers.0.self_attn.q_proj.weight",
                     "layers.0.cross_attn.k_proj.weight","proj_in.1.weight","proj_out.1.weight"):
            self.assertTrue(torch.count_nonzero(named[name]),name)
        self.model.set_attn_implementation("sdpa")
        torch.testing.assert_close(self.forward()[0],checked,rtol=2e-5,atol=2e-6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_writer_native_loader_and_lora_metadata_roundtrip(self):
        import json, tempfile
        from types import SimpleNamespace
        from safetensors.torch import load_file, save_file
        from core.models.acestep.vendor import AceStepConfig, AceStepConditionGenerationModel
        from core.models.acestep.residual_unet import native_config, restore_topology, preflight_checkpoint
        from core.models.acestep.loader import _build_dit
        from core.training.adapters.acestep_adapter import AceStepFullParameterAdapter, AceStepLoRAAdapter
        from core.rewiring.spec import checkpoint_metadata
        raw = self.model.config.to_dict()
        raw.update(num_lyric_encoder_hidden_layers=1,num_timbre_encoder_hidden_layers=1,
            num_attention_pooler_hidden_layers=1,num_audio_decoder_hidden_layers=1,
            text_hidden_dim=8,timbre_hidden_dim=4,fsq_dim=8,fsq_input_levels=[2,2],pool_window_size=2)
        config = AceStepConfig(**raw)
        config._attn_implementation = "sdpa"
        owner = AceStepConditionGenerationModel(config).eval()
        restore_topology(owner,self.spec)
        def predict(model):
            return model.decoder(self.audio,torch.tensor([.5,.7]),torch.tensor([.1,.2]),
                torch.ones(2,48),self.encoder,torch.ones(2,5),self.context,use_cache=False)[0]
        expected = predict(owner).detach()
        trainer = SimpleNamespace(transformer=owner)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"full.safetensors"
            AceStepFullParameterAdapter(trainer).write_checkpoint(3,1,path)
            spec, embedded = preflight_checkpoint(path)
            self.assertEqual(spec,self.spec);self.assertEqual(native_config(embedded),native_config(config))
            restored, restored_config = _build_dit(str(path),torch.float32)
            self.assertEqual(restored._residual_unet_spec,self.spec)
            self.assertEqual(restored.decoder._residual_unet_spec,self.spec)
            self.assertEqual(set(owner.state_dict()),set(restored.state_dict()))
            for name,value in owner.state_dict().items():
                torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
            torch.testing.assert_close(predict(restored),expected,rtol=0,atol=0)
            lora = AceStepLoRAAdapter(trainer,2,2)
            self.assertEqual(read_checkpoint_metadata(lora.checkpoint_metadata({},3,1)),self.spec)
            saved = load_file(str(path))
            plain = {key.removeprefix("net."):value for key,value in saved.items()}
            plain_path = Path(folder)/"plain.safetensors"
            save_file(plain,str(plain_path),metadata={"acestep_config":json.dumps(embedded),**checkpoint_metadata(self.spec)})
            plain_owner,_ = _build_dit(str(plain_path),torch.float32)
            torch.testing.assert_close(predict(plain_owner),expected,rtol=0,atol=0)
            broken = Path(folder)/"bad.safetensors"
            cases = [(dict(saved,null_condition_emb=plain["null_condition_emb"].clone()),
                    {"acestep_config":json.dumps(embedded),**checkpoint_metadata(self.spec)}),
                (saved,checkpoint_metadata(self.spec)),
                ({"lora_unet_decoder_layers_0_self_attn_q_proj.lora_down.weight":torch.ones(2,16)},
                    {"acestep_config":json.dumps(embedded),**checkpoint_metadata(self.spec)}),
                (saved,{"acestep_config":json.dumps(dict(embedded,patch_size=4)),**checkpoint_metadata(self.spec)}),
                (saved,{"acestep_config":json.dumps(embedded),**checkpoint_metadata(self.spec),"sushi.rewire.spec_sha256":"0"*64})]
            with patch("core.models.acestep.vendor.AceStepConditionGenerationModel",side_effect=AssertionError("Model allocation forbidden")):
                for tensors,metadata in cases:
                    save_file(tensors,str(broken),metadata=metadata)
                    with self.assertRaises(ValueError):_build_dit(str(broken),torch.float32)
            del saved["net.decoder.layers.0.self_attn.q_proj.weight"]
            save_file(saved,str(broken),metadata={"acestep_config":json.dumps(embedded),**checkpoint_metadata(self.spec)})
            with self.assertRaises(RuntimeError):_build_dit(str(broken),torch.float32)
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_low_precision_backward_and_cache_disable(self):
        from core.models.acestep.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        self.model.eval()
        output, cache = self.forward(use_cache=True)
        self.assertIsNone(cache);self.assertTrue(torch.isfinite(output).all())
        for dtype in (torch.bfloat16,torch.float16):
            self.model.to(dtype).train()
            self.model.zero_grad(set_to_none=True)
            audio = self.audio.detach().to(dtype).requires_grad_()
            result = self.forward(hidden_states=audio,context_latents=self.context.detach().to(dtype),
                                  encoder_hidden_states=self.encoder.detach().to(dtype),
                                  timestep=torch.tensor([.5,.7],dtype=dtype),
                                  timestep_r=torch.tensor([.1,.2],dtype=dtype),
                                  encoder_attention_mask=torch.tensor([[1,1,0,0,0],[1,0,1,0,1]]))[0]
            self.assertEqual(result.dtype,dtype);self.assertTrue(torch.isfinite(result).all())
            result.float().square().mean().backward()
            self.assertTrue(torch.isfinite(audio.grad).all())
            self.assertTrue(torch.count_nonzero(audio.grad))
            self.assertTrue(torch.isfinite(self.model.layers[0].self_attn.q_proj.weight.grad).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_conditioning_padding_is_masked_at_every_resolution(self):
        from core.models.acestep.residual_unet import restore_topology
        restore_topology(self.model, self.spec)
        mask = torch.tensor([[1,1,0,0,0],[1,0,1,0,1]])
        changed = self.encoder.detach().clone()
        changed[~mask.bool()] += 100
        observations = []
        handles = [layer.register_forward_pre_hook(lambda module,args:
            observations.append((args[0].shape[1],args[10]))) for layer in self.model.layers]
        try:
            expected = self.forward(encoder_attention_mask=mask)[0]
        finally:
            for handle in handles: handle.remove()
        self.assertEqual([count for count,_ in observations], [24,12,6,6,6,12,24])
        shared = observations[0][1]
        self.assertEqual(shared.shape, (2,1,1,5))
        self.assertTrue(all(value is shared for _,value in observations))
        torch.testing.assert_close(shared[:,0,0] == 0, mask.bool())
        torch.testing.assert_close(self.forward(encoder_hidden_states=changed,
            encoder_attention_mask=mask)[0], expected, rtol=0, atol=0)
        expected.square().mean().backward()
        self.assertEqual(torch.count_nonzero(self.encoder.grad[~mask.bool()]).item(), 0)
        self.assertGreater(torch.count_nonzero(self.encoder.grad[mask.bool()]).item(), 0)
        gradients = {name:value.grad.detach().clone() for name,value in self.model.named_parameters()
                     if value.grad is not None}
        self.model.zero_grad(set_to_none=True)
        self.model.gradient_checkpointing_enable()
        checkpointed = self.forward(encoder_attention_mask=mask)[0]
        torch.testing.assert_close(checkpointed, expected, rtol=0, atol=0)
        checkpointed.square().mean().backward()
        for name,value in self.model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(value.grad, gradients[name], rtol=0, atol=0)
        self.model.gradient_checkpointing_disable()
        self.model.set_attn_implementation("sdpa")
        torch.testing.assert_close(self.forward(encoder_attention_mask=mask)[0], expected,
                                   rtol=2e-5, atol=2e-6)
        invalid = [torch.ones(1,5), torch.ones(2,6), torch.full((2,5), .5),
                   torch.full((2,5), float("nan")), torch.tensor([[1,1,1,1,1],[0,0,0,0,0]])]
        with patch.object(self.model.proj_in,"forward",side_effect=AssertionError("Projection must not run")):
            for value in invalid:
                with self.subTest(mask=value), self.assertRaises(ValueError):
                    self.forward(encoder_attention_mask=value)
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_conditioner_flow_training_updates_full_and_lora(self):
        from types import SimpleNamespace
        from core.models.acestep.vendor import AceStepConfig, AceStepConditionGenerationModel
        from core.models.acestep.residual_unet import native_config, restore_topology, configure_training_topology
        from core.training.ops.acestep_ops import train_step
        from core.training.adapters.acestep_adapter import AceStepLoRAAdapter
        raw = self.model.config.to_dict()
        raw.update(in_channels=192, audio_acoustic_hidden_dim=64,
            num_lyric_encoder_hidden_layers=1, num_timbre_encoder_hidden_layers=1,
            num_attention_pooler_hidden_layers=1, num_audio_decoder_hidden_layers=1,
            text_hidden_dim=8, timbre_hidden_dim=64, timbre_fix_frame=48,
            fsq_dim=16, fsq_input_levels=[2,2], pool_window_size=2)
        for method in ("full", "lora"):
            with self.subTest(method=method):
                config = AceStepConfig(**raw)
                config._attn_implementation = "sdpa"
                owner = AceStepConditionGenerationModel(config).train()
                spec = build_spec("acestep", self.spec["streams"], self.spec["regions"], native_config(config))
                restore_topology(owner, spec)
                owner.requires_grad_(False)
                trainer = SimpleNamespace(transformer=owner, config={}, device=torch.device("cpu"),
                    training_dtype=torch.float32, mixed_precision=False, reconstruction_loss_weight=.25,
                    acestep_silence_latent=torch.zeros(1,48,64))
                configure_training_topology(owner, trainer.config)
                if method == "full":
                    owner.decoder.requires_grad_(True)
                else:
                    adapter = AceStepLoRAAdapter(trainer, 2, 2, torch.float32)
                    layers = {}
                    adapter.apply_lora_to_unet(layers)
                owner.gradient_checkpointing_enable()
                trainable = [value for value in owner.parameters() if value.requires_grad]
                snapshots = [value.detach().clone() for value in trainable]
                optimizer = torch.optim.AdamW(trainable, lr=.001)
                observed = []
                hook = owner.decoder.register_forward_pre_hook(lambda module,args,kwargs:
                    observed.append(kwargs["encoder_attention_mask"].detach().clone()), with_kwargs=True)
                try:
                    loss, prediction, reconstruction = train_step(trainer, torch.randn(2,48,64),
                        torch.randn(2,5,8), dict(text_attention_mask=torch.tensor([[1,1,1,1,1],[1,1,0,0,0]]),
                            lyric_hidden_states=torch.randn(2,4,8),
                            lyric_attention_mask=torch.tensor([[1,1,1,1],[1,0,0,0]])),
                        timesteps=torch.tensor([.5,.7]))
                finally:
                    hook.remove()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(prediction.item(), 0)
                self.assertGreater(reconstruction.item(), 0)
                self.assertTrue((observed[0] == 0).any())
                self.assertTrue(observed[0].bool().any(dim=1).all())
                loss.backward()
                self.assertTrue(all(value.grad is None or torch.isfinite(value.grad).all() for value in trainable))
                self.assertTrue(any(value.grad is not None and torch.count_nonzero(value.grad) for value in trainable))
                self.assertTrue(all(value.grad is None for value in owner.encoder.parameters()))
                optimizer.step()
                self.assertTrue(any(not torch.equal(before,after) for before,after in zip(snapshots,trainable)))
                self.assertEqual(owner._residual_unet_spec, spec)
                self.assertEqual(owner.decoder._residual_unet_spec, spec)
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_multistep_generation_avoids_residual_cache(self):
        from core.models.acestep.vendor import AceStepConfig, AceStepConditionGenerationModel
        from core.models.acestep.residual_unet import native_config, restore_topology
        raw = self.model.config.to_dict()
        raw.update(in_channels=192, audio_acoustic_hidden_dim=64,
            num_lyric_encoder_hidden_layers=1, num_timbre_encoder_hidden_layers=1,
            num_attention_pooler_hidden_layers=1, num_audio_decoder_hidden_layers=1,
            text_hidden_dim=8, timbre_hidden_dim=64, timbre_fix_frame=48,
            fsq_dim=16, fsq_input_levels=[2,2], pool_window_size=2)
        config = AceStepConfig(**raw)
        config._attn_implementation = "sdpa"
        owner = AceStepConditionGenerationModel(config).eval()
        spec = build_spec("acestep", self.spec["streams"], self.spec["regions"], native_config(config))
        restore_topology(owner,spec)
        args = dict(text_hidden_states=torch.randn(2,5,8),
            text_attention_mask=torch.tensor([[1,1,1,1,1],[1,1,0,0,0]]),
            lyric_hidden_states=torch.randn(2,4,8),
            lyric_attention_mask=torch.tensor([[1,1,1,1],[1,0,0,0]]),
            refer_audio_acoustic_hidden_states_packed=torch.zeros(2,48,64),
            refer_audio_order_mask=torch.arange(2), src_latents=torch.zeros(2,48,64),
            chunk_masks=torch.ones(2,48,64),is_covers=torch.zeros(2,dtype=torch.bool),
            silence_latent=torch.zeros(1,48,64),seed=94,infer_steps=3,dcw_enabled=False)
        from core.models.acestep.residual_unet import GRID_POLICY_KEY
        legacy_config=native_config(owner.config);legacy_config.pop(GRID_POLICY_KEY)
        legacy=build_spec("acestep",spec["streams"],spec["regions"],legacy_config)
        restore_topology(owner,legacy)
        with patch.object(owner,"prepare_condition",side_effect=AssertionError("No conditioning on invalid grid")):
            with self.assertRaisesRegex(ValueError,"not divisible"):
                owner.generate_audio(**{**args,"src_latents":args["src_latents"][:,:45],
                                        "chunk_masks":args["chunk_masks"][:,:45]})
        restore_topology(owner,spec)
        for sampler, method, calls in (("euler","ode",3),("heun","ode",5),("euler","sde",3)):
            with self.subTest(sampler=sampler,method=method):
                seen = []
                def observe(module, positional, kwargs):
                    self.assertIsNone(kwargs["past_key_values"])
                    self.assertFalse(kwargs["use_cache"])
                    seen.append(kwargs["hidden_states"].shape)
                hook = owner.decoder.register_forward_pre_hook(observe,with_kwargs=True)
                try:
                    with patch("core.models.acestep.vendor.modeling_acestep_v15_turbo.EncoderDecoderCache",
                               side_effect=AssertionError("Residual sampler must not allocate KV cache")):
                        result = owner.generate_audio(**args,sampler_mode=sampler,infer_method=method)
                finally:
                    hook.remove()
                self.assertEqual(len(seen),calls)
                self.assertEqual(result["target_latents"].shape,(2,48,64))
                self.assertTrue(torch.isfinite(result["target_latents"]).all())
                self.assertEqual(owner.decoder._residual_unet_spec,spec)
        with patch("core.models.acestep.vendor.modeling_acestep_v15_turbo.EncoderDecoderCache",
                   side_effect=AssertionError("No residual KV cache")):
            extended=owner.generate_audio(**{**args,"src_latents":torch.zeros(2,250,64),
                "chunk_masks":torch.ones(2,250,64)},sampler_mode="euler",infer_method="ode")
        self.assertEqual(extended["target_latents"].shape,(2,250,64))
        self.assertTrue(torch.isfinite(extended["target_latents"]).all())
        restore_topology(owner,None)
        dense_seen = []
        def observe_dense(module, positional, kwargs):
            self.assertIsNotNone(kwargs["past_key_values"])
            self.assertTrue(kwargs["use_cache"])
            dense_seen.append(kwargs["hidden_states"].shape)
        hook = owner.decoder.register_forward_pre_hook(observe_dense,with_kwargs=True)
        try:
            result = owner.generate_audio(**args,sampler_mode="euler",infer_method="ode")
        finally:
            hook.remove()
        self.assertEqual(len(dense_seen),3)
        self.assertTrue(torch.isfinite(result["target_latents"]).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_generation_grid_preflight_and_audio_sidecar(self):
        import json, tempfile
        from core.models.acestep.residual_unet import restore_topology, generation_metadata
        from core.pipeline_backends.acestep import AceStepMixin
        from api.error_handlers import ValidationError
        from core.rewiring.spec import digest
        class Backend(AceStepMixin):
            def __init__(self, owner):
                self.acestep_components={"dit":owner,"vae":torch.nn.Linear(1,1),
                                         "text_encoder":torch.nn.Linear(1,1),"tokenizer":object()}
                self.is_acestep_model=True
                self.device=torch.device("cpu")
        backend=Backend(self.model)
        restore_topology(self.model,self.spec)
        actual=generation_metadata(self.model,47)
        self.assertEqual(actual["token_grid"],[24,1])
        self.assertEqual(actual["input_frames"],47)
        self.assertEqual(actual["spec_sha256"],digest(self.spec))
        params={"seed":94}
        backend._acestep_rewiring_preflight(params,48)
        self.assertEqual(params["dit_rewiring_actual"]["input_frames"],48)
        with self.assertRaises(ValidationError): backend._acestep_rewiring_preflight({},0)
        self.assertEqual(self.model._residual_unet_spec,self.spec)
        from core.models.acestep.residual_unet import native_config, GRID_POLICY_KEY
        legacy_config=native_config(self.model.config);legacy_config.pop(GRID_POLICY_KEY)
        legacy=build_spec("acestep",self.spec["streams"],self.spec["regions"],legacy_config)
        restore_topology(self.model,legacy)
        with patch.object(backend,"_acestep_ensure_silence_latent",side_effect=AssertionError("No VAE staging")), \
                patch.object(backend,"_acestep_move",side_effect=AssertionError("No model staging")), \
                self.assertRaises(ValidationError):
            backend._generate_txt2aud_acestep({"audio_duration":1.8,"seed":94})
        restore_topology(self.model,self.spec)
        import utils.audio_utils as audio
        with tempfile.TemporaryDirectory() as directory, patch.object(audio.settings,"outputs_dir",directory):
            filename=audio.save_audio_with_metadata(torch.zeros(2,480),48000,params,"txt2aud")
            sidecar=json.loads((Path(directory)/Path(filename).with_suffix(".json")).read_text(encoding="utf-8"))
            self.assertEqual(sidecar["dit_rewiring_actual"],params["dit_rewiring_actual"])
            self.assertTrue((Path(directory)/filename).is_file())
        restore_topology(self.model,None)
        backend._acestep_rewiring_preflight(params,45)
        self.assertNotIn("dit_rewiring_actual",params)
        self.assertIsNone(generation_metadata(self.model,45))
        self.assertFalse(torch.cuda.is_initialized())

    def test_transport_tail_padding_restores_length_gradients_and_legacy_identity(self):
        from core.models.acestep.residual_unet import restore_topology, generation_metadata, native_config, GRID_POLICY_KEY
        restore_topology(self.model,self.spec)
        for length, tokens in ((45,24),(250,128),(1,4)):
            with self.subTest(length=length):
                audio=torch.randn(2,length,4,requires_grad=True)
                context=torch.randn(2,length,8,requires_grad=True)
                seen=[]
                hooks=[layer.register_forward_pre_hook(lambda module,args:seen.append(args[0].shape[1]))
                       for layer in self.model.layers]
                try:
                    result=self.forward(hidden_states=audio,context_latents=context,
                        attention_mask=torch.ones(2,length))[0]
                finally:
                    for hook in hooks:hook.remove()
                self.assertEqual(result.shape,(2,length,4))
                self.assertEqual(seen,[tokens,tokens//2,tokens//4,tokens//4,tokens//4,tokens//2,tokens])
                self.assertEqual(generation_metadata(self.model,length)["token_grid"],[tokens,1])
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(audio.grad).all())
                self.assertGreater(torch.count_nonzero(audio.grad).item(),0)
                self.model.zero_grad(set_to_none=True)
        args=dict(hidden_states=self.audio[:,:45],context_latents=self.context[:,:45],attention_mask=torch.ones(2,45))
        before=self.forward(**args)[0]
        before.square().mean().backward()
        gradients={name:value.grad.detach().clone() for name,value in self.model.named_parameters() if value.grad is not None}
        self.model.zero_grad(set_to_none=True)
        self.model.gradient_checkpointing_enable()
        after=self.forward(**args)[0]
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        after.square().mean().backward()
        for name,value in self.model.named_parameters():
            if name in gradients: torch.testing.assert_close(value.grad,gradients[name],rtol=0,atol=0)
        self.model.gradient_checkpointing_disable()
        handles=[patch.object(layer,"forward",side_effect=lambda hidden,*a,**k:(hidden,)) for layer in self.model.layers]
        for handle in handles:handle.start()
        try:
            args=dict(hidden_states=self.audio[:,:45],context_latents=self.context[:,:45],attention_mask=torch.ones(2,45))
            restored=self.forward(**args)[0]
            restore_topology(self.model,None)
            dense=self.forward(**args)[0]
            torch.testing.assert_close(restored,dense,rtol=2e-5,atol=1e-6)
        finally:
            for handle in handles:handle.stop()
        legacy_config=native_config(self.model.config);legacy_config.pop(GRID_POLICY_KEY)
        legacy=build_spec("acestep",self.spec["streams"],self.spec["regions"],legacy_config)
        restore_topology(self.model,legacy)
        with self.assertRaisesRegex(ValueError,"not divisible"): generation_metadata(self.model,45)
        self.assertEqual(generation_metadata(self.model,48)["grid_policy"],"require_divisible")
        self.assertEqual(self.model._residual_unet_spec,legacy)
        self.assertFalse(torch.cuda.is_initialized())

    def test_identity_decoder_padding_and_atomic_refusals(self):
        from core.models.acestep.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        masks = [dict(attention_mask=torch.zeros(2,48)),dict(encoder_attention_mask=torch.zeros(2,5)),
                 dict(past_key_values=object()),dict(output_attentions=True),dict(return_hidden_states=1),
                 dict(custom_layers_config={0:1}),dict(enable_early_exit=True),
]
        with patch.object(self.model.proj_in,"forward",side_effect=AssertionError("Projection must not run")):
            for args in masks:
                with self.subTest(args=list(args)),self.assertRaises(ValueError):self.forward(**args)
        with self.assertRaises(ValueError):self.forward(position_ids=torch.zeros(1,24))
        original = self.model.config._attn_implementation
        self.model.config._attn_implementation = "flash_attention_2"
        try:
            with self.assertRaisesRegex(ValueError,"eager or SDPA"):self.forward()
        finally:self.model.config._attn_implementation = original
        wrong = deepcopy(self.spec);wrong["regions"][0]["down_factor"] = [2,2]
        with self.assertRaisesRegex(ValueError,"temporal axis"):restore_topology(self.model,wrong)
        self.assertEqual(self.model._residual_unet_spec,self.spec)
        handles = [patch.object(layer,"forward",side_effect=lambda hidden,*a,**k:(hidden,)) for layer in self.model.layers]
        for handle in handles:handle.start()
        try:
            head_inputs = []
            hook = self.model.norm_out.register_forward_pre_hook(lambda m,a:head_inputs.append(a[0].detach().clone()))
            try:
                rewired = self.forward()[0]
                restore_topology(self.model,None)
                ordinary = self.forward()[0]
            finally:hook.remove()
            torch.testing.assert_close(head_inputs[0],head_inputs[1],rtol=0,atol=0)
            torch.testing.assert_close(rewired,ordinary,rtol=2e-5,atol=1e-6)
            restore_topology(self.model,self.spec)
            padded = self.forward(hidden_states=self.audio[:,:47],context_latents=self.context[:,:47],
                                  attention_mask=torch.ones(2,47))[0]
            self.assertEqual(padded.shape,(2,47,4))
        finally:
            for handle in handles:handle.stop()
        self.assertFalse(torch.cuda.is_initialized())
