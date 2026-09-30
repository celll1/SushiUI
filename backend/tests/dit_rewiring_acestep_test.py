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
        for invalid, options in (({}, dict(resume_spec=None)),
                ({"dit_rewiring_mode":"configure","dit_rewiring_spec":self.spec}, {}),
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
                                  timestep_r=torch.tensor([.1,.2],dtype=dtype))[0]
            self.assertEqual(result.dtype,dtype);self.assertTrue(torch.isfinite(result).all())
            result.float().square().mean().backward()
            self.assertTrue(torch.isfinite(audio.grad).all())
            self.assertTrue(torch.count_nonzero(audio.grad))
            self.assertTrue(torch.isfinite(self.model.layers[0].self_attn.q_proj.weight.grad).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_identity_decoder_padding_and_atomic_refusals(self):
        from core.models.acestep.residual_unet import restore_topology
        restore_topology(self.model,self.spec)
        masks = [dict(attention_mask=torch.zeros(2,48)),dict(encoder_attention_mask=torch.zeros(2,5)),
                 dict(past_key_values=object()),dict(output_attentions=True),dict(return_hidden_states=1),
                 dict(custom_layers_config={0:1}),dict(enable_early_exit=True),
                 dict(hidden_states=self.audio[:,:45],context_latents=self.context[:,:45])]
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
            padded = self.forward(hidden_states=self.audio[:,:47],context_latents=self.context[:,:47])[0]
            self.assertEqual(padded.shape,(2,47,4))
        finally:
            for handle in handles:handle.stop()
        self.assertFalse(torch.cuda.is_initialized())
