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
