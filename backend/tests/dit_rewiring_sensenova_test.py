"""CPU contracts for SenseNova generation transport and immutable prefix KV."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import OPERATORS, build_spec


class SenseNovaRuntimeContracts(unittest.TestCase):
    def setUp(self):
        for target, name, value in ((torch.cuda, "get_device_capability", lambda *a, **k: (8, 9)),
                (torch.cuda, "_lazy_init", lambda *a, **k: None), (torch._C, "_cuda_init", lambda *a, **k: None)):
            guard = patch.object(target, name, value); guard.start(); self.addCleanup(guard.stop)
        from core.models.sensenova.vendor.configuration_neo_chat import NEOChatConfig
        from core.models.sensenova.vendor.modeling_qwen3 import Qwen3Model, create_block_causal_mask
        from core.models.sensenova.residual_unet import native_config
        from core.training.ops.sensenova_ops import forward_und_prefix_layers
        torch.manual_seed(150)
        config = NEOChatConfig(vision_config=dict(patch_size=16), use_pixel_head=True,
            llm_config=dict(architectures=["Qwen3ForCausalLM"], vocab_size=32, hidden_size=16,
                intermediate_size=24, num_hidden_layers=7, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8, attention_dropout=0.0))
        config.llm_config._attn_implementation = "eager"
        self.model = Qwen3Model(config.llm_config)
        self.owner = SimpleNamespace(config=config, language_model=SimpleNamespace(model=self.model))
        self.spec = build_spec("sensenova", [{"name": "main", "block_count": 7}], [
            dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
                down_factor=[2, 2], **OPERATORS),
            dict(id="inner", stream="main", parent="outer", start_block=2, end_block=5,
                down_factor=[2, 2], **OPERATORS)], native_config(config))
        self.hidden = torch.randn(1, 96, 16, requires_grad=True)
        self.indexes = torch.stack((torch.full((96,), 4), torch.arange(8).repeat_interleave(12),
                                   torch.arange(12).repeat(8)))
        self.text = torch.tensor([[2, 5, 7, 3]])
        self.text_indexes = torch.stack((torch.arange(4), torch.zeros(4), torch.zeros(4))).long()
        self.text_mask = {"full_attention": create_block_causal_mask(self.text_indexes[0])}
        with torch.no_grad():
            self.prefix = forward_und_prefix_layers(self.model, self.text, self.text_indexes,
                self.text_mask, checkpoint_layers=False)

    def forward(self, **overrides):
        args = dict(inputs_embeds=self.hidden, indexes=self.indexes,
            image_gen_indicators=torch.ones((1, 96), dtype=torch.bool),
            attention_mask={"full_attention": None}, past_key_values=self.prefix,
            use_cache=True, update_cache=False)
        args.update(overrides)
        return self.model(**args)

    def test_native_transport_positions_hooks_parameter_identity_and_cache(self):
        from core.models.sensenova.residual_unet import restore_topology, model_topology_metadata
        from core.rewiring.spec import read_checkpoint_metadata
        dense = self.forward().last_hidden_state.detach()
        restore_topology(self.owner, self.spec)
        counts = []; positions = []; events = []
        identities = {name: id(value) for name, value in self.model.named_parameters()}
        saved = [(layer.keys.clone(), layer.values.clone()) for layer in self.prefix.layers]
        def observe(module, args, kwargs):
            counts.append(args[0].shape[1]); positions.append(kwargs["indexes"].clone())
            self.assertFalse(kwargs["update_cache"]); self.assertFalse(kwargs["use_cache"])
            self.assertIs(kwargs["past_key_values"], self.prefix)
        handles = [layer.register_forward_pre_hook(observe, with_kwargs=True) for layer in self.model.layers]
        handles += [layer.register_forward_hook(lambda m, a, out, i=i: events.append(i))
                    for i, layer in enumerate(self.model.layers)]
        try: output = self.forward()
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(counts, [96, 24, 6, 6, 6, 24, 96]); self.assertEqual(events, list(range(7)))
        torch.testing.assert_close(positions[1][:, 0], torch.tensor([4., 0.5, 0.5]))
        torch.testing.assert_close(positions[2][:, 0], torch.tensor([4., 1.5, 1.5]))
        torch.testing.assert_close(positions[-1], self.indexes.float(), rtol=0, atol=0)
        self.assertEqual(output.last_hidden_state.shape, self.hidden.shape)
        self.assertIs(output.past_key_values, self.prefix)
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())
        self.assertEqual(identities, {name: id(value) for name, value in self.model.named_parameters()})
        for layer, (keys, values) in zip(self.prefix.layers, saved):
            torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
            torch.testing.assert_close(layer.values, values, rtol=0, atol=0)
        self.assertEqual(read_checkpoint_metadata(model_topology_metadata(self.owner)), self.spec)
        restore_topology(self.owner, None)
        torch.testing.assert_close(self.forward().last_hidden_state, dense, rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_checkpoint_gradient_parity_and_understanding_prefix(self):
        from core.models.sensenova.residual_unet import restore_topology
        from core.training.ops.sensenova_ops import forward_gen_decoder_layers, forward_und_prefix_layers
        restore_topology(self.owner, self.spec)
        values = [self.hidden, *self.model.parameters()]
        def training(checkpointed):
            prefix = forward_und_prefix_layers(self.model, self.text, self.text_indexes,
                self.text_mask, checkpoint_layers=checkpointed)
            saved = [(layer.keys.detach().clone(), layer.values.detach().clone()) for layer in prefix.layers]
            result = forward_gen_decoder_layers(self.model, self.hidden, indexes=self.indexes,
                prefix_cache=prefix, checkpoint_layers=checkpointed, trainable_prefix=True)
            grads = torch.autograd.grad(result.square().mean(), values, allow_unused=True)
            for layer, (keys, vals) in zip(prefix.layers, saved):
                torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
                torch.testing.assert_close(layer.values, vals, rtol=0, atol=0)
            return result, grads
        direct, before = training(False); checked, after = training(True)
        torch.testing.assert_close(direct, checked, rtol=0, atol=0)
        for a, b in zip(before, after):
            self.assertEqual(a is None, b is None)
            if a is not None:
                self.assertTrue(torch.isfinite(b).all()); torch.testing.assert_close(a, b)
        by_name = dict(zip(["input", *dict(self.model.named_parameters())], after))
        for name in ("input", "layers.0.self_attn.q_proj_mot_gen.weight",
                     "layers.0.self_attn.k_proj.weight", "embed_tokens.weight"):
            self.assertTrue(torch.count_nonzero(by_name[name]), name)
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_model_checkpointing_and_zero_delta_identity(self):
        from core.models.sensenova.residual_unet import restore_topology
        restore_topology(self.owner, self.spec)
        self.model.train()
        normal = self.forward().last_hidden_state
        before = torch.autograd.grad(normal.square().mean(), self.hidden)[0]
        self.model.gradient_checkpointing = True
        checked = self.forward().last_hidden_state
        after = torch.autograd.grad(checked.square().mean(), self.hidden)[0]
        torch.testing.assert_close(normal, checked, rtol=0, atol=0)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.model.gradient_checkpointing = False
        handles = [patch.object(layer, "forward", side_effect=lambda states, **kw: states)
                   for layer in self.model.layers]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for handle in handles: stack.enter_context(handle)
            identity = self.forward().last_hidden_state
        torch.testing.assert_close(identity, self.model.norm_mot_gen(self.hidden), rtol=0, atol=0)

    def test_understanding_forward_is_dense_and_prepared_flash_prefix_is_immutable(self):
        from core.models.sensenova.residual_unet import restore_topology
        args = dict(input_ids=self.text, indexes=self.text_indexes, attention_mask=self.text_mask,
                    use_cache=True)
        dense = self.model(**args)
        restore_topology(self.owner, self.spec)
        with patch("core.models.sensenova.residual_unet.run_rewired_generation",
                   side_effect=AssertionError("understanding must stay dense")):
            checked = self.model(**args)
        torch.testing.assert_close(dense.last_hidden_state, checked.last_hidden_state, rtol=0, atol=0)
        for a, b in zip(dense.past_key_values.layers, checked.past_key_values.layers):
            torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
            torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)
        baseline = self.forward().last_hidden_state.detach()
        for layer in self.prefix.layers:
            layer.flash_prefix_len = 4
            layer.flash_k_cache = torch.empty(1, 100, 1, 8)
            layer.flash_v_cache = torch.empty_like(layer.flash_k_cache)
            layer.flash_k_cache[:, :4].copy_(layer.keys.transpose(1, 2))
            layer.flash_v_cache[:, :4].copy_(layer.values.transpose(1, 2))
        with torch.no_grad(): prepared = self.forward().last_hidden_state
        torch.testing.assert_close(prepared, baseline, rtol=0, atol=0)
        for layer in self.prefix.layers:
            torch.testing.assert_close(layer.flash_k_cache[:, :4], layer.keys.transpose(1, 2), rtol=0, atol=0)
            torch.testing.assert_close(layer.flash_v_cache[:, :4], layer.values.transpose(1, 2), rtol=0, atol=0)

    def test_refusals_precede_native_blocks_and_config_restore_is_atomic(self):
        from core.models.sensenova.residual_unet import restore_topology
        restore_topology(self.owner, self.spec)
        with patch.object(self.model.layers[0], "forward", side_effect=AssertionError("block executed")):
            for overrides in (dict(update_cache=True), dict(attention_mask={"full_attention": torch.zeros(1, 1, 96, 100)}),
                              dict(packed_gen=object())):
                with self.assertRaises(ValueError): self.forward(**overrides)
            bad = self.indexes.clone(); bad[1, 10] += 1
            for indexes in (bad, self.indexes[:, :-1], self.indexes.flip(1)):
                with self.assertRaises(ValueError): self.forward(indexes=indexes)
            self.model._repa_tap_depth = 0
            with self.assertRaisesRegex(ValueError, "REPA"): self.forward()
            del self.model._repa_tap_depth
            self.model.layers[3].self_attn._style_ctx = object()
            with self.assertRaisesRegex(ValueError, "style"): self.forward()
            self.model.layers[3].self_attn._style_ctx = None
            short = SimpleNamespace(layers=self.prefix.layers[:-1], get_seq_length=lambda: 4)
            with self.assertRaisesRegex(ValueError, "every native"): self.forward(past_key_values=short)
            wrong_kv = deepcopy(self.prefix)
            wrong_kv.layers[-1].keys = torch.zeros(1, 2, 4, 8)
            wrong_kv.layers[-1].values = torch.zeros(1, 2, 4, 8)
            with self.assertRaisesRegex(ValueError, "batch/head"): self.forward(past_key_values=wrong_kv)
        other = deepcopy(self.owner.config); other.gen_patch_size = 16
        from core.models.sensenova.residual_unet import native_config
        wrong = build_spec("sensenova", self.spec["streams"], self.spec["regions"], native_config(other))
        with self.assertRaises(ValueError): restore_topology(self.owner, wrong)
        self.assertEqual(self.owner._residual_unet_spec, self.spec)
        self.assertEqual(self.model._residual_unet_spec, self.spec)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
