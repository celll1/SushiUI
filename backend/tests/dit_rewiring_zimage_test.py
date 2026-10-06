"""Z-Image native image-prefix, auxiliary tokens and training-wrapper contracts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from core.rewiring.spec import OPERATORS, build_spec


class ZImageRuntimeContracts(unittest.TestCase):
    def setUp(self):
        for target, name, value in ((torch.cuda, "get_device_capability", lambda *a, **k: (8, 9)),
                (torch.cuda, "_lazy_init", lambda *a, **k: None), (torch._C, "_cuda_init", lambda *a, **k: None)):
            guard = patch.object(target, name, value); guard.start(); self.addCleanup(guard.stop)
        from core.models.zimage_transformer import ZImageTransformer2DModel
        from core.models.zimage_residual_unet import native_config, restore_topology
        torch.manual_seed(39)
        self.model = ZImageTransformer2DModel(in_channels=4, dim=24, n_layers=7,
            n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=12,
            axes_dims=[4, 4, 4], axes_lens=[128, 64, 64])
        with torch.no_grad():
            for parameter in self.model.parameters():
                if parameter.ndim >= 2: parameter.normal_(0, 0.02)
        self.spec = build_spec("zimage", [{"name": "main", "block_count": 7}], [
            dict(id="outer", stream="main", parent=None, start_block=1,
                 end_block=6, down_factor=[2, 2], **OPERATORS),
            dict(id="inner", stream="main", parent="outer", start_block=2,
                 end_block=5, down_factor=[2, 2], **OPERATORS)], native_config(self.model))
        restore_topology(self.model, self.spec)
        self.images = torch.randn(2, 4, 1, 16, 24, requires_grad=True)
        self.captions = [torch.randn(3, 12, requires_grad=True), torch.randn(35, 12, requires_grad=True)]
        self.times = torch.tensor([0.5, 0.7])

    def forward(self):
        output, _ = self.model(list(self.images.unbind()), self.times, self.captions)
        return torch.stack(output)




    def test_generation_lora_topology_scope_and_metadata(self):
        import tempfile
        from copy import deepcopy
        from types import SimpleNamespace
        from safetensors.torch import save_file
        from core.pipeline_backends.zimage import ZImageMixin
        from core.models.zimage_residual_unet import restore_topology
        from core.adapters import CompositeAdapterLayer, AdapterIncompatible
        from core.rewiring.spec import checkpoint_metadata, digest
        class Manager(ZImageMixin):
            def __init__(manager):
                manager.zimage_components = {"transformer": self.model,
                    "vae": SimpleNamespace(config=SimpleNamespace(block_out_channels=[1,2,3,4]))}
            def _zimage_resolve_lora_path(manager, path):
                return str(path) if Path(path).is_file() else None
        manager = Manager()
        restore_topology(self.model, None)
        original = self.model.layers[0].attention.to_q
        with tempfile.TemporaryDirectory() as directory:
            def write(name, topology, wrong=False):
                path = Path(directory)/f"{name}.safetensors"
                save_file({
                    "lora_transformer_layers_0_attention_to_q.lora_down.weight": torch.randn(2, 23 if wrong else 24),
                    "lora_transformer_layers_0_attention_to_q.lora_up.weight": torch.randn(24, 2)}, str(path),
                    metadata={"model_type": "zimage", **({} if topology is None else checkpoint_metadata(topology))})
                return {"path": str(path), "strength": 1.0}
            good, dense, bad = write("good", self.spec), write("dense", None), write("bad", self.spec, True)
            self.assertEqual(manager._load_lora_zimage([good]), 1)
            self.assertIsInstance(self.model.layers[0].attention.to_q, CompositeAdapterLayer)
            self.assertEqual(self.model._residual_unet_spec, self.spec)
            params = {}
            manager._zimage_rewiring_metadata(params, self.model, 128, 192)
            self.assertEqual(params["dit_rewiring_actual"]["spec_sha256"], digest(self.spec))
            self.assertTrue(torch.isfinite(self.forward()).all())
            for options, message in (({"style_transfer": {"enabled": True}}, "style"),
                    ({"fbcache_enable": True, "fbcache_threshold": 0.1}, "FBCache"),
                    ({"nag_enable": True, "nag_scale": 2, "nag_negative_prompt": "blurry"}, "NAG"),
                    ({"prompt": "(blurry:-1)"}, "NegPip")):
                with self.assertRaisesRegex(Exception, message):
                    manager._zimage_rewiring_metadata(options, self.model, 128, 192)
            with self.assertRaisesRegex(Exception, "divisible"):
                manager._zimage_rewiring_metadata({}, self.model, 144, 192)
            manager._unload_lora_zimage()
            self.assertIs(self.model.layers[0].attention.to_q, original)
            self.assertFalse(hasattr(self.model, "_residual_unet_spec"))
            manager._zimage_rewiring_metadata(params, self.model, 128, 192)
            self.assertNotIn("dit_rewiring_actual", params)
            for files in ([good, dense], [good, bad]):
                with self.assertRaises(RuntimeError): manager._load_lora_zimage(files)
                self.assertIs(self.model.layers[0].attention.to_q, original)
                self.assertFalse(hasattr(self.model, "_residual_unet_spec"))
            restore_topology(self.model, self.spec)
            with self.assertRaises(AdapterIncompatible): manager._load_lora_zimage([dense])
            manager._load_lora_zimage([good]); manager._unload_lora_zimage()
            self.assertEqual(self.model._residual_unet_spec, self.spec)
            # A model swap must not inherit a previous model's restore snapshot.
            restore_topology(self.model, None)
            manager._load_lora_zimage([good])
            replacement = deepcopy(self.model)
            restore_topology(replacement, None)
            replacement.layers[0].attention.to_q = deepcopy(original)
            manager.zimage_components["transformer"] = replacement
            manager._load_lora_zimage([good]); manager._unload_lora_zimage()
            self.assertFalse(hasattr(replacement, "_residual_unet_spec"))
            self.assertFalse(hasattr(self.model, "_residual_unet_spec"))
        self.assertFalse(torch.cuda.is_initialized())


    def test_all_generation_entries_select_and_restore_topology_before_staging(self):
        import tempfile
        from types import SimpleNamespace
        from PIL import Image
        from safetensors.torch import save_file
        from core.pipeline_backends.zimage import ZImageMixin
        from core.models.zimage_residual_unet import restore_topology
        from core.rewiring.spec import checkpoint_metadata
        restore_topology(self.model, None)
        class Manager(ZImageMixin):
            device = "cpu"
            current_attention_type = "native"
            def __init__(manager):
                manager.zimage_components = {"transformer": self.model,
                    "vae": SimpleNamespace(config=SimpleNamespace(block_out_channels=[1,2,3,4])),
                    "text_encoder": object(), "tokenizer": object(), "scheduler": SimpleNamespace()}
                manager.seen = []; manager.cleaned = []
            def _get_zimage_scheduler(manager, sampler): return SimpleNamespace()
            def _zimage_resolve_lora_path(manager, path): return str(path)
            def _zimage_rewiring_metadata(manager, params, transformer, height, width):
                super()._zimage_rewiring_metadata(params, transformer, height, width)
                manager.seen.append(params["dit_rewiring_actual"]["spec"])
                raise RuntimeError("CPU stop before staging")
            def _zimage_cleanup(manager, **options):
                manager.cleaned.append(getattr(self.model, "_residual_unet_spec", None))
        manager = Manager()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"rewired.safetensors"
            save_file({"lora_transformer_layers_0_attention_to_q.lora_down.weight": torch.randn(2,24),
                       "lora_transformer_layers_0_attention_to_q.lora_up.weight": torch.randn(24,2)},
                      str(path), metadata={"model_type": "zimage", **checkpoint_metadata(self.spec)})
            image = Image.new("RGB", (192,128)); mask = Image.new("L", (192,128))
            original = self.model.layers[0].attention.to_q
            for entry, args in ((manager._generate_txt2img_zimage, ()),
                                (manager._generate_img2img_zimage, (image,)),
                                (manager._generate_inpaint_zimage, (image,mask))):
                params = {"loras": [{"path": str(path), "strength": 1.0}],
                          "height": 128, "width": 192, "seed": 42, "attention_type": "native"}
                with self.assertRaisesRegex(RuntimeError, "CPU stop before staging"):
                    entry(params, *args)
                self.assertFalse(hasattr(self.model, "_residual_unet_spec"))
                self.assertIs(self.model.layers[0].attention.to_q, original)
                self.assertEqual(params["dit_rewiring_actual"]["spec"], self.spec)
            self.assertEqual(manager.seen, [self.spec]*3)
            self.assertEqual(manager.cleaned, [None]*3)
        self.assertFalse(torch.cuda.is_initialized())

    def test_generation_finally_restores_even_when_offload_fails(self):
        from core.pipeline_backends.zimage import ZImageMixin
        from core.models.zimage_residual_unet import restore_topology
        events = []
        class Manager(ZImageMixin):
            def __init__(manager): manager.zimage_components = {"transformer": self.model}
            def _zimage_cleanup(manager, **options):
                events.append(options)
                raise RuntimeError("offload failed")
        manager = Manager()
        restore_topology(self.model, None)
        manager._load_lora_zimage([])
        restore_topology(self.model, self.spec)
        with self.assertRaisesRegex(RuntimeError, "offload failed"):
            manager._zimage_finish_generation(gen_succeeded=False)
        self.assertFalse(hasattr(self.model, "_residual_unet_spec"))
        self.assertEqual(events, [{"gen_succeeded": False}])
        with patch.object(manager, "_unload_lora_zimage", side_effect=RuntimeError("restore failed")), \
             patch.object(manager, "_zimage_cleanup") as cleanup:
            with self.assertRaisesRegex(RuntimeError, "restore failed"):
                manager._zimage_finish_generation(gen_succeeded=False)
            cleanup.assert_called_once_with(gen_succeeded=False)

    def test_component_loading_inherits_after_latent_space(self):
        from types import SimpleNamespace
        from core.model_loader import ModelLoader
        from core.training.ops import zimage_ops
        components = {"transformer": self.model, "vae": torch.nn.Linear(4, 4),
                      "text_encoder": torch.nn.Linear(12, 12), "tokenizer": object(),
                      "scheduler": SimpleNamespace()}
        components["vae"].config = SimpleNamespace(latent_channels=4)
        trainer = SimpleNamespace(log_prefix="[ZImageLoadTest]", model_path="base.safetensors",
            weight_dtype=torch.float32, vae_dtype=torch.float32, config={},
            use_flash_attention=False, gradient_checkpointing=False, blocks_to_swap=1,
            device="cpu", resume_from_checkpoint="adapter.safetensors", _manages_own_resume=True)
        order = []
        def latent_space(owner, declared):
            self.assertIs(owner.transformer.transformer, self.model)
            self.assertEqual(owner.config, {})
            order.append("latent_space")
        with patch.object(ModelLoader, "load_zimage_from_diffusers", return_value=components), \
             patch.object(zimage_ops, "_apply_latent_space", side_effect=latent_space):
            zimage_ops.load_components(trainer)
        self.assertEqual(order, ["latent_space"])
        self.assertEqual(trainer.config["dit_rewiring_spec"], self.spec)
        self.assertIs(trainer.transformer_original, self.model)
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.transformer.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_training_inheritance_and_lora_resume_atomicity(self):
        import tempfile
        from types import SimpleNamespace
        from safetensors.torch import save_file
        from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
        from core.models.zimage_residual_unet import configure_training_topology, restore_topology
        from core.training.adapters.zimage_adapter import ZImageLoRAAdapter
        from core.training.lora_trainer import LoRATrainer
        from core.rewiring.training import validate_training_rewiring
        from core.rewiring.spec import checkpoint_metadata
        wrapper = BatchedZImageWrapperOptimized(self.model)
        config = {}
        self.assertEqual(configure_training_topology(wrapper, config), self.spec)
        self.assertEqual(configure_training_topology(wrapper, config, resuming=True), self.spec)
        self.assertEqual(validate_training_rewiring(config, "zimage"), ("inherit", self.spec))
        self.assertEqual(configure_training_topology(wrapper, {"dit_rewiring_mode": "configure",
                                                   "dit_rewiring_spec": self.spec}), self.spec)
        changed = {**self.spec, "regions": [self.spec["regions"][0]]}
        with self.assertRaisesRegex(ValueError, "without edits"):
            configure_training_topology(wrapper, {"dit_rewiring_mode": "configure",
                "dit_rewiring_spec": changed}, resuming=True)
        with self.assertRaisesRegex(ValueError, "block_skip_rate"):
            configure_training_topology(wrapper, {**config, "block_skip_rate": 0.1})
        self.assertEqual(self.model._residual_unet_spec, self.spec)
        restore_topology(wrapper, None)
        trainer = SimpleNamespace(transformer=wrapper, config={}, is_zimage=True,
            log_prefix="[ZImageResumeTest]", lora_rank=2, lora_alpha=2,
            adapter_algorithm="lora", weight_decompose=False, adapter_config={})
        adapter = ZImageLoRAAdapter(trainer, 2, 2)
        layers = {}; adapter.apply_lora_to_unet(layers); trainer.lora_layers = layers
        self.assertGreater(len(layers), 0)
        tensors = {f"{name}.{key}": value.detach().clone()+0.01
                   for name, layer in layers.items() for key, value in layer.branch_tensors().items()}
        before = {f"{name}.{key}": value.detach().clone()
                  for name, layer in layers.items() for key, value in layer.branch_tensors().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"resume.safetensors"
            metadata = {"model_type": "zimage", "step": "17", **checkpoint_metadata(self.spec)}
            save_file({"unrelated": torch.ones(1)}, str(path), metadata=metadata)
            with self.assertRaisesRegex(ValueError, "missing required tensor"):
                LoRATrainer.load_checkpoint(trainer, str(path))
            self.assertFalse(hasattr(self.model, "_residual_unet_spec"))
            self.assertEqual(trainer.config, {})
            for name, layer in layers.items():
                for key, value in layer.branch_tensors().items():
                    torch.testing.assert_close(value, before[f"{name}.{key}"], rtol=0, atol=0)
            save_file(tensors, str(path), metadata=metadata)
            self.assertEqual(LoRATrainer.load_checkpoint(trainer, str(path)), 17)
            self.assertEqual(self.model._residual_unet_spec, self.spec)
            for name, layer in layers.items():
                for key, value in layer.branch_tensors().items():
                    torch.testing.assert_close(value, tensors[f"{name}.{key}"], rtol=0, atol=0)
            self.assertEqual(configure_training_topology(wrapper, trainer.config, resuming=True), self.spec)
            save_file(tensors, str(path), metadata={"model_type": "zimage"})
            with self.assertRaisesRegex(ValueError, "disagree"):
                LoRATrainer.load_checkpoint(trainer, str(path))
            self.assertEqual(self.model._residual_unet_spec, self.spec)
        self.assertFalse(torch.cuda.is_initialized())


    def test_default_preset_public_training_request_and_native_geometry(self):
        from collections import Counter
        from types import SimpleNamespace
        from core.models.zimage_residual_unet import native_config, default_topology, configure_training_topology, generation_token_stride
        from core.rewiring.spec import preview_plan
        from api.routes import TrainingRunCreateRequest
        from core.training.training_config import _build_train_section
        config = native_config({})
        model = SimpleNamespace(**config, config=config)
        request = TrainingRunCreateRequest(training_method="lora", base_model_path="zimage",
                                          total_steps=1, dit_rewiring_mode="configure")
        section = _build_train_section(request.model_dump(), total_steps=1, epochs=None,
                                      train_unet=True, train_text_encoder=False, arch="zimage")
        spec = configure_training_topology(model, section)
        self.assertEqual(spec, default_topology(config))
        self.assertEqual(Counter(b["tokens"] for b in preview_plan(spec,64,64)["blocks"]),
                         {4096:8, 1024:18, 256:4})
        self.assertIsNone(default_topology(dict(config,n_layers=7)))
        with self.assertRaisesRegex(ValueError,"No rewiring preset"):
            configure_training_topology(self.model, {"dit_rewiring_mode":"configure"})
        self.assertEqual(generation_token_stride(SimpleNamespace(config=SimpleNamespace(block_out_channels=[1,2,3]))),8)
        with self.assertRaisesRegex(ValueError,"block_skip_rate"):
            TrainingRunCreateRequest(training_method="lora", base_model_path="zimage",
                total_steps=1, dit_rewiring_mode="configure", block_skip_rate=0.1)

    def test_native_shapes_auxiliary_masks_rope_and_offloader(self):
        count = sum(p.numel() for p in self.model.parameters())
        events = []
        class Offloader:
            def wait_for_block(self, index): events.append(("wait", index))
            def submit_move_blocks_forward(self, index): events.append(("submit", index))
        self.model._block_offloader = Offloader()
        for h, w, image_count, extras, lengths in ((16, 24, 96, 64, [160, 88, 70, 70, 70, 88, 160]),
                                                   (8, 8, 16, 80, [96, 84, 81, 81, 81, 84, 96])):
            seen = []; first = []
            def observe(module, args):
                hidden, mask, rotary, modulation = args
                current_images = hidden.shape[1] - extras
                if not first: first.extend([mask[:, image_count:].clone(), rotary[:, image_count:].clone()])
                torch.testing.assert_close(mask[:, current_images:], first[0], rtol=0, atol=0)
                torch.testing.assert_close(rotary[:, current_images:], first[1], rtol=0, atol=0)
                self.assertTrue(mask[:, :current_images].all())
                self.assertEqual(rotary.shape[:2], hidden.shape[:2])
                seen.append(hidden.shape[1])
            handles = [layer.register_forward_pre_hook(observe) for layer in self.model.layers]
            try:
                images = self.images if h == 16 else torch.randn(2, 4, 1, h, w)
                outputs, _ = self.model(list(images.unbind()), self.times, self.captions)
            finally:
                for handle in handles: handle.remove()
            self.assertEqual(seen, lengths)
            self.assertTrue(all(torch.isfinite(output).all() for output in outputs))
            self.assertTrue(all(output.shape == image.shape for output, image in zip(outputs, images)))
        self.assertEqual(events, [(name, index) for _ in range(2) for index in range(7) for name in ("wait", "submit")])
        self.assertEqual(count, sum(p.numel() for p in self.model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_checkpointed_native_output_and_gradients(self):
        parameters = [self.images, *self.captions, *self.model.parameters()]
        expected = self.forward()
        before = torch.autograd.grad(expected.square().mean(), parameters, allow_unused=True)
        self.model.enable_gradient_checkpointing()
        actual = self.forward()
        after = torch.autograd.grad(actual.square().mean(), parameters, allow_unused=True)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(before, after):
            if a is None: self.assertIsNone(b)
            else:
                self.assertTrue(torch.isfinite(b).all())
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        self.assertGreater(float(after[1].abs().sum()), 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_optimized_training_wrapper_uses_native_rewired_semantics(self):
        from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
        from core.models.zimage_residual_unet import restore_topology, model_topology_metadata
        from core.rewiring.spec import read_checkpoint_metadata
        wrapper = BatchedZImageWrapperOptimized(self.model)
        captions = torch.zeros(2, 64, 12)
        mask = torch.zeros(2, 64, dtype=torch.bool)
        for index, value in enumerate(self.captions):
            captions[index, :len(value)] = value.detach(); mask[index, :len(value)] = True
        captions.requires_grad_()
        expected = torch.stack(self.model(list(self.images.unbind()), 1-self.times, self.captions)[0])
        with patch.object(wrapper, "batched_patchify", side_effect=AssertionError("Dense optimized path forbidden")):
            actual, _ = wrapper(self.images, self.times, captions, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(captions.grad).all())
        self.assertEqual(float(captions.grad[~mask].abs().sum()), 0)
        self.assertGreater(float(captions.grad[mask].abs().sum()), 0)
        self.assertEqual(read_checkpoint_metadata(model_topology_metadata(wrapper)), self.spec)
        restore_topology(wrapper, None)
        with patch.object(wrapper, "batched_patchify", wraps=wrapper.batched_patchify) as fast:
            dense, _ = wrapper(self.images.detach(), self.times, captions.detach(), mask)
        fast.assert_called_once()
        self.assertEqual(dense.shape, self.images.shape)
        self.assertTrue(torch.isfinite(dense).all())
        self.assertEqual(model_topology_metadata(wrapper), {})
        self.assertFalse(torch.cuda.is_initialized())

    def test_file_loaded_alias_keeps_native_geometry_and_guidance_refusal(self):
        import importlib.util
        from core.models.zimage_residual_unet import native_config, restore_topology
        name = "zimage_rewiring_cpu_fixture"
        loader = importlib.util.spec_from_file_location(name,
            Path(__file__).resolve().parents[1]/"core"/"models"/"zimage_transformer.py")
        alias = importlib.util.module_from_spec(loader)
        with patch.dict(sys.modules, {name: alias}):
            loader.loader.exec_module(alias)
            model = alias.ZImageTransformer2DModel(**native_config(self.model))
            model.load_state_dict(self.model.state_dict())
            restore_topology(model, self.spec)
            self.assertEqual(model.config, self.model.config)
            with torch.no_grad():
                actual, _ = model(list(self.images.unbind()), self.times, self.captions)
                expected = self.forward()
            torch.testing.assert_close(torch.stack(actual), expected, rtol=0, atol=0)
            self.assertIsNot(type(model.layers[0].attention), type(self.model.layers[0].attention))
            alias.ZImageAttention._nag_ctx = {}
            with self.assertRaisesRegex(ValueError, "inactive attention"):
                model(list(self.images.unbind()), self.times, self.captions)
            alias.ZImageAttention._nag_ctx = None
        self.assertFalse(torch.cuda.is_initialized())

    def test_auxiliary_token_order_and_residual_identity_delta(self):
        from core.models.zimage_residual_unet import run_rewired_zimage
        hidden = torch.randn(2, 160, 24)
        ids = [self.model.create_coordinate_grid((1, 8, 12), start=(offset, 0, 0)).flatten(0, 2)
               for offset in (33, 65)]
        rotary = torch.ones(2, 160, 6, dtype=torch.complex64)
        mask = torch.ones(2, 160, dtype=torch.bool); mask[0, 128:] = False
        events = []
        class AddBlock(torch.nn.Module):
            def __init__(block, index): super().__init__(); block.index = index
            def forward(block, value, current_mask, frequencies, modulation):
                if block.index == 0: torch.testing.assert_close(value, hidden, rtol=0, atol=0)
                events.append(value.shape[1])
                return value + (block.index+1)/8
        original = self.model.layers
        self.model.layers = torch.nn.ModuleList([AddBlock(i) for i in range(7)])
        try:
            actual = run_rewired_zimage(self.model, hidden, torch.zeros(2, 24), rotary, mask,
                [(1, 16, 24)]*2, ids, 2, self.spec)
        finally:
            self.model.layers = original
        torch.testing.assert_close(actual, hidden+3.5, rtol=1e-6, atol=1e-6)
        self.assertEqual(events, [160, 88, 70, 70, 70, 88, 160])
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_writer_native_loader_single_shards_and_preflight(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from safetensors import safe_open
        from safetensors.torch import save_file
        from core.model_loader import ModelLoader
        from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
        from core.models.common.single_file_format import read_state_dict, save_single_file_state
        from core.models.zimage_residual_unet import native_config, preflight_artifact
        from core.training.adapters.zimage_adapter import ZImageFullParameterAdapter, ZImageLoRAAdapter
        from core.rewiring.spec import read_checkpoint_metadata
        config = native_config(self.model)
        wrapper = BatchedZImageWrapperOptimized(self.model)
        trainer = SimpleNamespace(transformer=wrapper, train_unet=True, train_text_encoder=False,
            text_encoder=None, vae=None, config={}, bundle_vae=False)
        adapter = ZImageFullParameterAdapter(trainer)
        self.assertEqual(read_checkpoint_metadata(ZImageLoRAAdapter(trainer, 2, 2).checkpoint_metadata({}, 1, 0)), self.spec)
        with torch.no_grad(): expected = self.forward()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/"model.safetensors"
            with patch("core.training.vae_swap.swap_metadata", return_value=(None, False, {})):
                self.assertEqual(adapter.write_checkpoint(1, 0, path), path)
            raw, metadata = read_state_dict(str(path))
            self.assertEqual(json.loads(metadata["transformer_config"]), config)
            self.assertEqual(preflight_artifact(path), (self.spec, config))
            def load(source):
                weights, _ = read_state_dict(str(source))
                state, _, _, layout = ModelLoader._normalize_zimage_state_dict(weights)
                result = ModelLoader._build_zimage_transformer_from_state(config,
                    config["n_layers"], config["in_channels"], state, layout, torch.float32, path=str(source))
                self.assertEqual(result._residual_unet_spec, self.spec)
                self.assertEqual(native_config(result), config)
                for key, value in self.model.state_dict().items():
                    torch.testing.assert_close(value, result.state_dict()[key], rtol=0, atol=0)
                with torch.no_grad(): actual, _ = result(list(self.images.unbind()), self.times, self.captions)
                torch.testing.assert_close(torch.stack(actual), expected, rtol=1e-5, atol=2e-7)
            load(path)
            sharded = root/"sharded.safetensors"
            save_single_file_state(raw, metadata, str(sharded), max_shard_bytes=120000)
            self.assertFalse(sharded.exists())
            self.assertEqual(preflight_artifact(sharded), (self.spec, config))
            load(sharded)
            index = Path(str(sharded)+".index.json")
            with patch.object(ModelLoader, "load_zimage_from_comfy_safetensors", return_value={}) as dispatch:
                ModelLoader.load_zimage_from_diffusers(str(sharded), device="cpu", torch_dtype=torch.float32)
            self.assertEqual(dispatch.call_args.kwargs["file_path"], str(index))
            fake_base = root/"base"/"transformer"; fake_base.mkdir(parents=True)
            (fake_base/"config.json").write_text(json.dumps(dict(config, dim=48, n_layers=30)), encoding="utf-8")
            built = []
            def stop_after_config(*args, **kwargs):
                built.append(args[:3]); raise RuntimeError("Stop before companion loading")
            with patch("huggingface_hub.snapshot_download", return_value=str(fake_base.parent)), \
                 patch.object(ModelLoader, "_build_zimage_transformer_from_state", side_effect=stop_after_config):
                with self.assertRaisesRegex(RuntimeError, "Stop before companion"):
                    ModelLoader.load_zimage_from_comfy_safetensors(str(path), device="cpu", torch_dtype=torch.float32)
            self.assertEqual(built[0], (config, 7, 4))
            from core.models.zimage_transformer import ZImageTransformer2DModel
            multi_config = dict(config, all_patch_size=[2, 4], all_f_patch_size=[2, 1])
            multi_model = ZImageTransformer2DModel(**multi_config)
            multi_path = root/"multibank.safetensors"
            save_file({"model.diffusion_model."+key: value for key, value in multi_model.state_dict().items()},
                str(multi_path), metadata={"model_type": "zimage", "transformer_config": json.dumps(multi_config)})
            with patch("huggingface_hub.snapshot_download", return_value=str(fake_base.parent)), \
                 patch.object(ModelLoader, "_build_zimage_transformer_from_state", side_effect=stop_after_config):
                with self.assertRaisesRegex(RuntimeError, "Stop before companion"):
                    ModelLoader.load_zimage_from_comfy_safetensors(str(multi_path), device="cpu", torch_dtype=torch.float32)
            self.assertEqual(built[1], (multi_config, 7, 4))
            damaged = root/"bad.safetensors"
            save_file(raw, str(damaged), metadata={**metadata,
                "transformer_config": json.dumps(dict(config, cap_feat_dim=13))})
            with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("Downloads forbidden")):
                with self.assertRaisesRegex(ValueError, "native config mismatch"):
                    ModelLoader.load_zimage_from_comfy_safetensors(str(damaged), device="cpu")
            state, _, _, layout = ModelLoader._normalize_zimage_state_dict(raw)
            with patch("core.model_loader.torch.device", side_effect=AssertionError("Allocation forbidden")):
                with self.assertRaisesRegex(ValueError, "native config mismatch"):
                    ModelLoader._build_zimage_transformer_from_state(dict(config, cap_feat_dim=13),
                        7, 4, state, layout, torch.float32, path=str(path))
            legacy = root/"legacy.safetensors"
            save_file(raw, str(legacy), metadata={"model_type": "zimage"})
            self.assertEqual(preflight_artifact(legacy), (None, None))
            state, _, _, layout = ModelLoader._normalize_zimage_state_dict(raw)
            dense = ModelLoader._build_zimage_transformer_from_state(config, 7, 4, state, layout,
                torch.float32, path=str(legacy))
            self.assertFalse(hasattr(dense, "_residual_unet_spec"))
            save_file(raw, str(damaged), metadata={k: v for k, v in metadata.items() if k != "transformer_config"})
            with self.assertRaisesRegex(ValueError, "requires native config"):
                preflight_artifact(damaged)
        self.assertFalse(torch.cuda.is_initialized())

    def test_feature_grid_and_identity_refusal(self):
        from core.models.zimage_residual_unet import restore_topology, native_config
        for name in ("_nag_request", "_negpip_request", "_style_ctx", "_fbcache", "_tread_config", "_block_skip_config", "_repa_tap_depth"):
            setattr(self.model, name, 0 if name == "_repa_tap_depth" else {})
            with self.assertRaisesRegex(ValueError, "incompatible"): self.forward()
            setattr(self.model, name, None)
        with self.assertRaisesRegex(ValueError, "single-frame"):
            self.model([self.images[0].repeat(1, 2, 1, 1)], self.times[:1], self.captions[:1])
        with self.assertRaisesRegex(ValueError, "one image grid"):
            self.model([self.images[0], torch.randn(4, 1, 8, 8)], self.times, self.captions)
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.model([torch.randn(4, 1, 18, 24)], self.times[:1], self.captions[:1])
        with self.assertRaisesRegex(ValueError, "native config mismatch"):
            restore_topology(self.model, dict(self.spec, base_config_sha256="0"*64))
        self.assertEqual(self.model._residual_unet_spec, self.spec)
        self.assertEqual(native_config(self.model)["n_refiner_layers"], 1)
        self.assertEqual(native_config(self.model)["cap_feat_dim"], 12)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
