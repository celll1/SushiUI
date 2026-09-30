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
        expected = self.forward()
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
