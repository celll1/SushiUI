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

    def test_training_resolution_and_atomic_lora_resume(self):
        import tempfile
        from safetensors.torch import save_file
        from core.models.sensenova.residual_unet import configure_training_topology, restore_topology
        from core.rewiring.spec import checkpoint_metadata
        from core.training.lora_trainer import LoRATrainer
        from core.adapters import LoRALinearLayer
        restore_topology(self.owner, self.spec)
        config = {}
        self.assertEqual(configure_training_topology(self.owner,config),self.spec)
        for key in ("repa_enable","tread_enable","blockskip_enable","dit_partition_training_enabled"):
            with self.assertRaisesRegex(ValueError,key):
                configure_training_topology(self.owner,dict(config,**{key:True}))
        with self.assertRaisesRegex(ValueError,"batch_size"):
            configure_training_topology(self.owner,dict(config,batch_size=2))
        restore_topology(self.owner,None)
        settings = {}
        self.assertEqual(configure_training_topology(self.owner,settings,resuming=True,
            resume_spec=self.spec,install=False),self.spec)
        self.assertEqual(settings,{})
        self.assertFalse(hasattr(self.model,"_residual_unet_spec"))
        self.model.requires_grad_(False)
        layers = {}
        for attr in ("q_proj_mot_gen","q_proj","k_proj","v_proj"):
            name = "language_model.model.layers.0.self_attn."+attr
            base = getattr(self.model.layers[0].self_attn,attr)
            layer = LoRALinearLayer(base,2,2,name)
            setattr(self.model.layers[0].self_attn,attr,layer);layers[name] = layer
        trainer = SimpleNamespace(transformer=self.owner,config=settings,is_sensenova=True,
            log_prefix="[SenseNovaResumeTest]",lora_rank=2,lora_alpha=2,
            adapter_algorithm="lora",weight_decompose=False,adapter_config={},lora_layers=layers)
        before = {f"{name}.{key}":tensor.detach().clone() for name,layer in layers.items()
                  for key,tensor in layer.branch_tensors().items()}
        tensors = {key:value+0.05 for key,value in before.items()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"resume.safetensors"
            metadata = dict(checkpoint_metadata(self.spec),model_type="sensenova",step="17")
            save_file({"unrelated":torch.zeros(1)},str(path),metadata=metadata)
            with self.assertRaisesRegex(ValueError,"missing required tensor"):
                LoRATrainer.load_checkpoint(trainer,str(path))
            self.assertEqual(settings,{})
            self.assertFalse(hasattr(self.owner,"_residual_unet_spec"))
            for name,layer in layers.items():
                for key,tensor in layer.branch_tensors().items():
                    torch.testing.assert_close(tensor,before[f"{name}.{key}"],rtol=0,atol=0)
            save_file(tensors,str(path),metadata=metadata)
            last = list(layers.values())[-1]
            def fail(values):
                with torch.no_grad():last.lora_down.weight.fill_(123)
                raise RuntimeError("injected resume failure")
            with patch.object(last,"load_tensors",side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,"injected resume failure"):
                    LoRATrainer.load_checkpoint(trainer,str(path))
            self.assertEqual(settings,{})
            self.assertFalse(hasattr(self.owner,"_residual_unet_spec"))
            for name,layer in layers.items():
                for key,tensor in layer.branch_tensors().items():
                    torch.testing.assert_close(tensor,before[f"{name}.{key}"],rtol=0,atol=0)
            self.assertEqual(LoRATrainer.load_checkpoint(trainer,str(path)),17)
            self.assertEqual(settings["dit_rewiring_spec"],self.spec)
            self.assertEqual(self.owner._residual_unet_spec,self.spec)
            self.assertEqual(self.model._residual_unet_spec,self.spec)
            for name,layer in layers.items():
                for key,tensor in layer.branch_tensors().items():
                    torch.testing.assert_close(tensor,tensors[f"{name}.{key}"],rtol=0,atol=0)
            save_file(tensors,str(path),metadata={"model_type":"sensenova"})
            with patch("safetensors.torch.load_file",side_effect=AssertionError("weights read")):
                with self.assertRaisesRegex(ValueError,"disagree"):
                    LoRATrainer.load_checkpoint(trainer,str(path))
        from core.training.ops.sensenova_ops import forward_und_prefix_layers,forward_gen_decoder_layers
        prefix = forward_und_prefix_layers(self.model,self.text,self.text_indexes,self.text_mask,
                                          checkpoint_layers=True)
        result = forward_gen_decoder_layers(self.model,self.hidden,indexes=self.indexes,prefix_cache=prefix,
                                            checkpoint_layers=True,trainable_prefix=True)
        (result-torch.randn_like(result)).square().mean().backward()
        for layer in layers.values():
            self.assertIsNotNone(layer.lora_up.weight.grad)
            self.assertTrue(torch.isfinite(layer.lora_up.weight.grad).all())
            self.assertTrue(torch.count_nonzero(layer.lora_up.weight.grad))
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_loader_resolves_after_geometry_before_freeze(self):
        from core.models.sensenova.residual_unet import restore_topology
        from core.training.ops import sensenova_ops
        from contextlib import ExitStack
        root = torch.nn.Module()
        root.config = self.owner.config
        root.language_model = torch.nn.Module(); root.language_model.model = self.model
        root.fm_modules = torch.nn.ModuleDict({"fm_head":torch.nn.Identity()})
        restore_topology(root,self.spec)
        events = []
        trainer = SimpleNamespace(model_path="checkpoint.safetensors",weight_dtype=torch.float32,
            device=torch.device("cpu"),attention_backend="native",config={},blocks_to_swap=0,
            arch=SimpleNamespace(resolve_latent_refiner=lambda owner:events.append("refiner")))
        components = dict(transformer=root,config=root.config,tokenizer=object(),config_dict={})
        original = root.requires_grad_
        def freeze(flag):
            self.assertEqual(trainer.config["dit_rewiring_spec"],self.spec)
            events.append("freeze");return original(flag)
        with ExitStack() as stack:
            stack.enter_context(patch("core.models.sensenova.loader.load_sensenova_from_path",return_value=components))
            stack.enter_context(patch.object(sensenova_ops,"_assert_supported_quantized_training_base"))
            stack.enter_context(patch.object(sensenova_ops,"_assert_pixel_head_fm_decoder"))
            stack.enter_context(patch("core.training.vae_swap.apply_latent_space",side_effect=lambda *a:events.append("geometry")))
            stack.enter_context(patch.object(root,"requires_grad_",side_effect=freeze))
            stack.enter_context(patch.object(sensenova_ops,"setup_attention_backend"))
            stack.enter_context(patch("core.models.common.quantized_frozen_training.frozen_training_fused_requested",return_value=False))
            sensenova_ops.load_components(trainer)
        self.assertEqual(events,["geometry","refiner","freeze"])
        self.assertTrue(root.training)
        self.assertFalse(any(parameter.requires_grad for parameter in root.parameters()))
        self.assertEqual(root._residual_unet_spec,self.spec)
        self.assertEqual(self.model._residual_unet_spec,self.spec)
        self.assertFalse(torch.cuda.is_initialized())

    @staticmethod
    def full_native_config():
        return dict(vision_config=dict(patch_size=16, hidden_size=16, llm_hidden_size=16,
                    max_position_embeddings_vision=256),
            llm_config=dict(architectures=["Qwen3ForCausalLM"], vocab_size=32, hidden_size=16,
                intermediate_size=24, num_hidden_layers=42, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8, attention_dropout=0.0),
            template="neo1_0", fm_head_layers=2, use_pixel_head=True, concat_time_token_num=0,
            noise_scale=1., noise_scale_mode="none", noise_scale_base_image_seq_len=1024,
            add_noise_scale_embedding=False, noise_scale_max_value=1., time_schedule="linear",
            time_shift_type="none", base_shift=0., max_shift=1., base_image_seq_len=256,
            max_image_seq_len=4096, t_eps=0.05)

    def test_full_native_artifacts_restore_through_production_loader(self):
        import tempfile, json
        from safetensors import safe_open
        from safetensors.torch import save_file
        from core.models.sensenova.vendor.configuration_neo_chat import NEOChatConfig
        from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel
        from core.models.sensenova.residual_unet import (native_config, restore_topology,
            preflight_checkpoint, model_topology_metadata)
        from core.models.sensenova import loader
        from core.models.common.single_file_format import read_state_dict
        from core.training.ops.sensenova_ops import forward_und_prefix_layers
        from core.rewiring.spec import read_checkpoint_metadata, META_SPEC
        from core.model_loader import ModelLoader
        raw = self.full_native_config()
        model = NEOChatModel(NEOChatConfig(**raw)).to(torch.bfloat16).eval()
        spec = build_spec("sensenova", [{"name":"main", "block_count":42}], [
            dict(id="outer", stream="main", parent=None, start_block=4, end_block=38,
                 down_factor=[2,2], **OPERATORS),
            dict(id="inner", stream="main", parent="outer", start_block=19, end_block=23,
                 down_factor=[2,2], **OPERATORS)], native_config(model.config))
        restore_topology(model, spec)
        hidden = self.hidden.detach().to(torch.bfloat16)
        z = torch.randn(1, 96, 3*32*32, dtype=torch.bfloat16)
        def predict(owner):
            prefix = forward_und_prefix_layers(owner.language_model.model, self.text,
                self.text_indexes, self.text_mask, checkpoint_layers=False)
            return owner._t2i_predict_v(hidden, self.indexes, {"full_attention":None}, prefix,
                torch.tensor(0.4), z, 96, token_hw=(8,12))
        with torch.no_grad(): expected = predict(model)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for stem, limit, save_format in (("single", 10**8, "bf16"), ("sharded", 20000, "bf16"),
                                             ("int8", 10**8, "int8")):
                written, census = loader.save_sensenova_full_finetune_checkpoint(model,
                    str(root/(stem+".safetensors")), branch="both", save_format=save_format,
                    raw_config=raw, config=model.config, max_shard_bytes=limit,
                    extra_metadata={META_SPEC:"spoofed"})
                self.assertEqual(census["trained_int8" if save_format == "int8" else "trained_bf16"], 588)
                self.assertEqual(preflight_checkpoint(written), spec)
                state, metadata = read_state_dict(written)
                self.assertEqual(read_checkpoint_metadata(metadata), spec)
                if written.endswith(".index.json"):
                    index = json.loads(Path(written).read_text(encoding="utf-8"))
                    for name in set(index["weight_map"].values()):
                        with safe_open(str(root/name), framework="pt", device="cpu") as handle:
                            self.assertEqual(read_checkpoint_metadata(handle.metadata()), spec)
                requested = str(root/(stem+".safetensors"))
                with patch.object(loader, "_load_sensenova_tokenizer", return_value=object()) as tokenizer:
                    loaded = ModelLoader.load_sensenova_from_path(requested)["transformer"]
                tokenizer.assert_called_once()
                self.assertEqual(model_topology_metadata(loaded), model_topology_metadata(model))
                self.assertEqual(loaded.language_model.model._residual_unet_spec, spec)
                for key, value in loaded.state_dict().items():
                    torch.testing.assert_close(value, state["transformer."+key], rtol=0, atol=0)
                with torch.no_grad(): actual = predict(loaded)
                if save_format == "bf16":
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                else:
                    from core.models.ideogram4.vendor.int8_linear import Int8Linear
                    quantized = [module for module in loaded.modules() if type(module) is Int8Linear]
                    self.assertEqual(len(quantized), 588)
                    self.assertTrue(all(not module._allow_int8_mm for module in quantized))
                    self.assertTrue(torch.isfinite(actual).all())
                del loaded
            bad_config = deepcopy(raw); bad_config["llm_config"]["num_hidden_layers"] = 41
            with self.assertRaises(ValueError):
                loader.save_sensenova_full_finetune_checkpoint(model, str(root/"bad-save.safetensors"),
                    branch="both", save_format="bf16", raw_config=raw,
                    extra_metadata={"sensenova_config":json.dumps(bad_config)})
            self.assertFalse((root/"bad-save.safetensors").exists())
            # A missing native weight refuses before tokenizer or VAE module loading.
            single_state, single_md = read_state_dict(str(root/"single.safetensors"))
            single_state.pop("transformer.language_model.model.layers.41.mlp_mot_gen.down_proj.weight")
            broken = root/"missing-weight.safetensors"
            save_file(single_state, str(broken), metadata=single_md)
            with patch.object(loader, "_load_sensenova_tokenizer", side_effect=AssertionError("tokenizer loaded")):
                with self.assertRaisesRegex(RuntimeError, "native weights incomplete"):
                    loader.load_sensenova_from_path(str(broken))
        self.assertFalse(torch.cuda.is_initialized())

    def test_header_refusals_refiner_dependencies_and_lora_metadata(self):
        import tempfile, json
        from safetensors.torch import save_file
        from core.models.sensenova.residual_unet import restore_topology, preflight_checkpoint
        from core.models.sensenova import loader
        from core.rewiring.spec import checkpoint_metadata, META_HASH, META_SPEC
        from core.training.adapters.sensenova_adapter import SenseNovaLoRAAdapter
        from core.models.common.single_file_format import save_single_file_state
        restore_topology(self.owner, self.spec)
        trainer = SimpleNamespace(transformer=self.owner, config={})
        adapter = SenseNovaLoRAAdapter(trainer, 2, 2)
        lora_md = adapter.checkpoint_metadata({"native_target":object()}, 4, 1)
        self.assertEqual(json.loads(lora_md[META_SPEC]), self.spec)
        from core.models.sensenova.loader import _serializable_sensenova_config
        raw = _serializable_sensenova_config(self.owner.config)
        # Preserve original scalar vision fields rather than the vendor's tuple serialization.
        raw["vision_config"]["downsample_ratio"] = 0.5
        raw["vision_config"]["llm_hidden_size"] = 2048
        metadata = dict(checkpoint_metadata(self.spec), model_type="sensenova", sensenova_config=json.dumps(raw))
        state = {"transformer.language_model.model."+key: value.detach().clone()
                 for key,value in self.model.state_dict().items()}
        state["transformer.fm_modules.fm_head.conv1.weight"] = torch.zeros(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); base = root/"base.safetensors"
            save_file(state, str(base), metadata=metadata)
            self.assertEqual(preflight_checkpoint(base), self.spec)
            dense = root/"dense.safetensors"
            save_file(state, str(dense), metadata={"model_type":"sensenova"})
            cases = [(dict(metadata, **{META_HASH:"0"*64}), state),
                     ({key:value for key,value in metadata.items() if key!="sensenova_config"}, state),
                     (dict(metadata, sensenova_config=json.dumps(dict(raw, gen_patch_size=16))), state),
                     (metadata, {"transformer.lora_target.lora_up.weight":torch.zeros(2,2)})]
            for number, (md, weights) in enumerate(cases):
                path = root/f"bad-{number}.safetensors"; save_file(weights, str(path), metadata=md)
                with patch.object(loader,"read_state_dict",side_effect=AssertionError("tensors read")), \
                     patch.object(loader,"NEOChatModel",side_effect=AssertionError("model built")):
                    with self.assertRaises(ValueError):loader.load_sensenova_from_path(str(path))
            delta = root/"delta.safetensors"
            delta_md = dict(metadata, sensenova_checkpoint_kind=loader.SENSENOVA_REFINER_DELTA_KIND,
                            sensenova_refiner_base_model_path=base.name)
            save_file({"transformer.fm_modules.fm_refiner.gate":torch.zeros(1)}, str(delta), metadata=delta_md)
            self.assertEqual(preflight_checkpoint(delta), self.spec)
            from core.models.sensenova.latent_refiner import LatentRefiner, REFINER_INPUTS, REFINER_NORM
            self.owner.fm_modules = torch.nn.ModuleDict({"fm_refiner":LatentRefiner(3,16,1)})
            delta_config = dict(raw, gen_in_channels=3, gen_refiner=dict(version=1,width=16,depth=1,
                inputs=list(REFINER_INPUTS),norm=REFINER_NORM,detach_anchor_step=None,detach_steps=None,detach_accum=None))
            exported = loader.save_sensenova_refiner_delta_checkpoint(self.owner,str(root/"exported-delta.safetensors"),
                base_model_path=str(base),raw_config=delta_config,extra_metadata={"step":"4",META_SPEC:"spoofed"})
            self.assertEqual(preflight_checkpoint(exported),self.spec)
            with self.assertRaisesRegex(ValueError,"base residual topologies"):
                loader.save_sensenova_refiner_delta_checkpoint(self.owner,str(root/"wrong-delta.safetensors"),
                    base_model_path=str(dense),raw_config=delta_config,extra_metadata={"step":"4"})
            self.assertFalse((root/"wrong-delta.safetensors").exists())
            for bad_md in (dict(delta_md, sensenova_refiner_base_model_path=dense.name),
                           {key:value for key,value in delta_md.items() if not key.startswith("sushi.rewire.")},
                           dict(delta_md, sensenova_refiner_base_model_path=delta.name)):
                save_file({"refiner":torch.zeros(1)},str(delta),metadata=bad_md)
                with patch.object(loader,"read_state_dict",side_effect=AssertionError("tensors read")):
                    with self.assertRaises(ValueError):loader.load_sensenova_from_path(str(delta))
            index_path = save_single_file_state(state, metadata, str(root/"shards.safetensors"), max_shard_bytes=3000)
            index = json.loads(Path(index_path).read_text(encoding="utf-8"))
            first = root/next(iter(index["weight_map"].values()))
            from core.models.common.single_file_format import read_state_dict
            shard, shard_md = read_state_dict(str(first))
            save_file(shard,str(first),metadata={key:value for key,value in shard_md.items() if not key.startswith("sushi.rewire.")})
            with patch.object(loader,"read_state_dict",side_effect=AssertionError("tensors read")):
                with self.assertRaisesRegex(ValueError,"index and shard"):loader.load_sensenova_from_path(index_path)
        self.assertFalse(torch.cuda.is_initialized())



if __name__ == "__main__":
    unittest.main()
